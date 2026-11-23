"""Glass-box rare-disease retrieval and evidence-grounded explanation pipeline.

End-to-end, single file. Deterministic where possible; the only stochastic
component is the local LLM, which is asked for strict JSON at temperature 0.

    case (text + CPU image captions)
        -> frozen fusion-v4-SHAP8 retriever -> top-10 + trace   [no LLM]
        -> CALL 1  plan: atomic findings [C#] + per-candidate checks
        -> evidence fetch: ZebraMap profile sentences [Z-x-n],
                           OptimusKG triples [K-x-n],
                           retriever trace [R-x-n]              [no LLM]
        -> CALL 2  auditor: 10 shufffled candidates, ranks hidden
        -> CALL 3  explanation writer: final top-5, cited claims
        -> validator: IDs, citation rules, verbatim quotes          [no LLM]
        -> CALL 4  adjudicator: keep / soften / drop each claim
        -> report + metrics (retrieval vs complete system)

The retriever is the saved model set in rerank/: no retraining happens here.

Run (CPU caption env, see README at bottom of this file):

    /tmp/opencode/capenv/bin/python glassbox_pipeline.py --n-cases 250 --workers 4

Resumable: every case is checkpointed to rerank/glassbox/cases/<CaseID>.json.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
SPLIT = REPO / "splits" / "disjoint_multilabel"
RERANK = REPO / "rerank"
BOW = RERANK / "bow"
LEX = RERANK / "lexical"
V4 = RERANK / "full_v4"
MODELS = RERANK / "models"
DATA = REPO / "Zebramap_DBS"
OUT = BASE
CASE_DIR = OUT / "cases"
CAPTION_CACHE = OUT / "caption_cache.json"
KG_CACHE = OUT / "cache" / "optimuskg"
LOG_PATH = OUT / "run.log"

PIPELINE_VERSION = "glassbox-v2"
SEED = 42
N_CASES_DEFAULT = 250
K_RETRIEVE = 10
K_EXPLAIN = 5
MAX_IMAGES_PER_CASE = 4
MAX_CASE_CHARS = 6000
CAPTION_MODEL = "WafaaFraih/blip-roco-radiology-captioning"

LLM_BASE = os.environ.get("LLM_BASE", "http://127.0.0.1:8080/v1")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "900"))

# Fusion v4 SHAP-8 selected features (kept in the saved order of gbm_v4_shap8.txt).
V4_FEATURES = [
    "disease_svd_0", "rank_min", "tfidf_z", "tfidf_minmax",
    "tfidf_rank", "svd_cosine", "tfidf_x_bow", "tfidf_recip_rank",
]

PROFILE_FIELDS = [
    "Clinical description", "Etiology", "Diagnostic methods",
    "Differential diagnosis", "Epidemiology", "Management and treatment",
    "Prognosis", "Genetic counseling", "Antenatal diagnosis",
]

STOP = set("""
a an the and or of to in on for with without from by is are was were be been being
this that these those it its as at into during after before between both all any
patient case year old years male female man woman girl boy presented showed revealed
not no but if then than also very more most less least may might can could would
which who whom whose what when where while about above below over under again
""".split())


# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------

def now() -> float:
    return time.time()


LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    """Console + file logging; thread-safe so worker lines never interleave."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with LOG_LOCK:
        print(line, flush=True)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def content_terms(text: str) -> set[str]:
    return {t for t in norm_text(text).split() if len(t) >= 4 and t not in STOP}


def split_sentences(text: str, min_len: int = 35) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < min_len:
            continue
        if len(p) > 240:
            p = p[:237].rsplit(" ", 1)[0] + "..."
        out.append(p)
    return out


def stable_rank_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    rank = np.empty_like(order, dtype=np.int32)
    rank[order] = np.arange(1, len(scores) + 1, dtype=np.int32)
    return rank


def rank_metrics(order: np.ndarray, gold_idx: int) -> dict:
    r = int(np.where(order == gold_idx)[0][0]) + 1
    return {"mrr": 1.0 / r, "top1": float(r == 1), "top5": float(r <= 5),
            "top10": float(r <= 10), "rank": r}


def bootstrap_ci(delta: np.ndarray, n_boot: int = 2000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = delta[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney AUROC; labels 1 = positive."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ----------------------------------------------------------------------------
# LLM client
# ----------------------------------------------------------------------------

class LLM:
    def __init__(self, base: str = LLM_BASE):
        self.base = base.rstrip("/")
        self.model = self._discover_model()
        self._local = threading.local()
        self.lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = 0
        self.seconds = 0.0

    def _discover_model(self) -> str:
        try:
            with urllib.request.urlopen(self.base + "/models", timeout=20) as r:
                data = json.load(r)
            return data["data"][0]["id"]
        except Exception:
            return "local-model"

    def _session(self):
        import requests
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def chat(self, messages: list[dict], max_tokens: int, tag: str = "",
             json_mode: bool = True, retries: int = 3) -> tuple[str | None, dict]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": SEED,
            "max_tokens": int(max_tokens),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last = None
        for attempt in range(retries):
            t0 = now()
            try:
                r = self._session().post(
                    self.base + "/chat/completions", json=payload, timeout=LLM_TIMEOUT
                )
                if r.status_code != 200:
                    body = r.text[:500]
                    # smaller/older servers may not support these optional fields
                    if r.status_code == 400 and "response_format" in payload and (
                            "response_format" in body or "json" in body.lower()):
                        payload.pop("response_format", None)
                        log(f"LLM {tag}: server rejected response_format, retrying without it")
                        continue
                    if r.status_code == 400 and "chat_template_kwargs" in payload and (
                            "chat_template" in body or "template" in body):
                        payload.pop("chat_template_kwargs", None)
                        log(f"LLM {tag}: server rejected chat_template_kwargs, retrying without it")
                        continue
                    raise RuntimeError(f"HTTP {r.status_code}: {body}")
                data = r.json()
                usage = data.get("usage", {})
                with self.lock:
                    self.calls += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens", 0))
                    self.completion_tokens += int(usage.get("completion_tokens", 0))
                    self.seconds += now() - t0
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                log(f"LLM {tag or 'call':<12} {now()-t0:6.1f}s "
                    f"in={int(usage.get('prompt_tokens', 0)):>6} "
                    f"out={int(usage.get('completion_tokens', 0)):>5}")
                return content, {"usage": usage, "reasoning_chars": len(reasoning),
                                 "tag": tag, "seconds": now() - t0}
            except Exception as exc:  # server hiccup, timeout, malformed body
                last = str(exc)
                with self.lock:
                    self.failures += 1
                time.sleep(min(2 ** attempt, 10))
        return None, {"error": last, "tag": tag}

    def json_chat(self, messages: list[dict], max_tokens: int, required: list[str],
                  tag: str = "") -> tuple[dict | None, dict]:
        content, meta = self.chat(messages, max_tokens, tag=tag)
        if content is None:
            return None, meta
        obj = extract_json(content)
        if obj is not None and all(k in obj for k in required):
            meta["parsed"] = True
            return obj, meta
        meta["parsed"] = False
        meta["raw"] = (content or "")[:1500]
        log(f"LLM {tag} invalid JSON, repairing")
        # one repair round, still temperature 0
        repair = messages + [
            {"role": "assistant", "content": content[:4000]},
            {"role": "user", "content":
             "Return ONLY valid JSON matching the requested schema. "
             f"Required top-level keys: {', '.join(required)}. No prose, no code fence."},
        ]
        content2, meta2 = self.chat(repair, max_tokens, tag=tag + ":repair")
        meta["repair"] = meta2
        obj2 = extract_json(content2 or "")
        if obj2 is not None and all(k in obj2 for k in required):
            meta["parsed"] = True
            return obj2, meta
        meta["raw2"] = (content2 or "")[:1500]
        return None, meta


def extract_json(text: str):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # first balanced object
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    return None
    return None


# ----------------------------------------------------------------------------
# corpus, retriever, trace
# ----------------------------------------------------------------------------

class Retriever:
    """Frozen fusion-v4-SHAP8 retriever plus a decomposable BoW trace."""

    def __init__(self):
        meta = json.load((V4 / "meta.json").open())
        names = meta["feature_names"]
        self.sel = [names.index(f) for f in V4_FEATURES]
        self.booster = lgb.Booster(model_file=str(MODELS / "gbm_v4_shap8.txt"))
        self.cand = np.load(RERANK / "candidates_test_full_top100.npz", allow_pickle=True)
        lex = np.load(LEX / "full_scores_test.npz", allow_pickle=True)
        self.bow = lex["bow"]
        self.tfidf = lex["tfidf"]
        self.gold_idx = lex["gold_idx"]
        self.diseases = [str(d) for d in lex["diseases"]]
        self.case_ids = lex["case_ids"].astype(str)
        self.disease_to_idx = {d: i for i, d in enumerate(self.diseases)}
        self.gold_ids = np.array([self.diseases[g] for g in self.gold_idx])

        self.vectorizer = joblib.load(BOW / "vectorizer_mass80.pkl")
        self.centroids = sparse.load_npz(BOW / "disease_centroids_mass80.npz").tocsr()
        with (BOW / "prune_sets.json").open(encoding="utf-8") as f:
            self.removed = set(json.load(f)["mass80"])
        self.vocab = self.vectorizer.get_feature_names_out()

        # row-order of candidate pools (case-major, 100 rows per case)
        self.cids = np.array(
            [[self.disease_to_idx[str(d)] for d in row] for row in self.cand["candidate_ids"]],
            dtype=np.int32,
        )
        self.tfidf_scores = self.cand["candidate_scores"].astype(np.float32)

    def prune(self, text: str) -> str:
        def core(t):
            return re.sub(r"[^a-z0-9]", "", t.lower())
        return " ".join(t for t in str(text).split() if core(t) not in self.removed)

    def scores(self, case_idx: np.ndarray, Xs: np.ndarray) -> np.ndarray:
        """Fusion score matrix over the full catalog for the given cases."""
        pred = self.booster.predict(
            np.ascontiguousarray(Xs).reshape(-1, Xs.shape[-1])
        ).reshape(-1, K_RETRIEVE * 10)
        n, n_dis = len(pred), len(self.diseases)
        mm = np.full((n, n_dis), np.nan, dtype=np.float32)
        for i, gi in enumerate(case_idx):
            v = pred[i]
            mm[i, self.cids[gi]] = (v - v.min()) / max(float(v.max() - v.min()), 1e-9)
        bow = self.bow[case_idx]
        mmb = (bow - bow.min(1, keepdims=True)) / np.maximum(
            bow.max(1, keepdims=True) - bow.min(1, keepdims=True), 1e-9
        )
        return 0.6 * mmb + 0.4 * (np.where(np.isnan(mm), -0.3, mm) + 1e-4 * mmb)

    def trace(self, case_idx: int, disease_idx: int, case_text: str,
              findings: list[dict]) -> dict:
        """Exact decomposition of the retriever signals for one candidate."""
        did = self.diseases[disease_idx]
        tf_row = self.tfidf[case_idx]
        bow_row = self.bow[case_idx]
        t_rank = int((tf_row > tf_row[disease_idx]).sum()) + 1
        b_rank = int((bow_row > bow_row[disease_idx]).sum()) + 1

        # BoW centroid contribution decomposition: tfidf(case) * centroid(disease)
        q = self.vectorizer.transform([self.prune(case_text)])
        contrib = q.multiply(self.centroids[disease_idx]).tocsr()
        order = np.argsort(-contrib.data, kind="stable")[:8]
        matched = []
        case_norm = " " + norm_text(case_text) + " "
        for k in order:
            term = str(self.vocab[contrib.indices[k]])
            term_norm = norm_text(term)
            present = bool(term_norm) and f" {term_norm} " in case_norm
            matched.append({"term": term, "weight": float(contrib.data[k]),
                            "present_in_case": present})
        matched = [m for m in matched if m["present_in_case"]][:5]

        # exact GB M contribution for candidates that are in the TF-IDF top-100 pool
        pool_pos = np.where(self.cids[case_idx] == disease_idx)[0]
        gbm = None
        if len(pool_pos):
            row = (np.asarray(self.X[case_idx, pool_pos[0]], dtype=np.float64))[None, :]
            c = self.booster.predict(row, pred_contrib=True)[0]
            parts = [{"feature": f, "contribution": float(v)}
                     for f, v in zip(V4_FEATURES, c[:-1])]
            parts.sort(key=lambda x: -abs(x["contribution"]))
            gbm = {"in_tfidf_pool": True, "top_contributions": parts[:3],
                   "bias": float(c[-1])}
        else:
            gbm = {"in_tfidf_pool": False, "top_contributions": []}

        return {
            "tfidf_score": float(tf_row[disease_idx]),
            "bow_score": float(bow_row[disease_idx]),
            "tfidf_rank": t_rank,
            "bow_rank": b_rank,
            "in_tfidf_top100": bool(len(pool_pos)),
            "matched_terms": matched,
            "gbm": gbm,
        }


# ----------------------------------------------------------------------------
# images: CPU BLIP captioning (cached)
# ----------------------------------------------------------------------------

class Captioner:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.cache = {}
        if CAPTION_CACHE.exists():
            try:
                self.cache = json.load(CAPTION_CACHE.open(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.model = None
        self.processor = None
        self.lock = threading.Lock()
        self.new = 0

    def _load(self):
        if self.model is not None or not self.enabled:
            return
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        torch.set_num_threads(max(1, min(16, os.cpu_count() or 4)))
        log(f"caption: loading {CAPTION_MODEL} on CPU (first run only)")
        self.processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).eval()
        self.torch = torch
        log("caption: model ready")

    def caption(self, path: Path) -> str | None:
        key = str(path)
        if key in self.cache:
            return self.cache[key].get("caption")
        if not self.enabled or path is None or not path.exists():
            return None
        try:
            self._load()
            if self.model is None:
                return None
            from PIL import Image
            im = Image.open(path).convert("RGB")
            inp = self.processor(im, return_tensors="pt")
            with self.torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=36, num_beams=1)
            text = self.processor.decode(out[0], skip_special_tokens=True).strip()
            self.cache[key] = {"caption": text, "model": CAPTION_MODEL}
            with self.lock:
                self.new += 1
                if self.new % 25 == 0:
                    write_json(CAPTION_CACHE, self.cache)
            return text
        except Exception as exc:
            self.cache[key] = {"caption": None, "model": CAPTION_MODEL,
                               "error": str(exc)[:200]}
            return None

    def flush(self):
        write_json(CAPTION_CACHE, self.cache)


def pick_images(structured: dict, disease_id: str, case_id: str,
                image_count: int) -> list[dict]:
    entry = structured.get(str(disease_id), {}).get(case_id, {})
    imgs = entry.get("ImageData", []) or []
    picked, seen = [], set()
    for im in imgs:
        subtype = im.get("ImageSubType") or im.get("ImageType") or "image"
        if subtype in seen:
            continue
        seen.add(subtype)
        picked.append(im)
        if len(picked) >= MAX_IMAGES_PER_CASE:
            break
    out = []
    for im in picked:
        rel = im.get("ImagePath") or ""
        out.append({
            "path": DATA / rel if rel else None,
            "modality": im.get("ImageType") or "unknown",
            "subtype": im.get("ImageSubType") or "unknown",
            "file": im.get("FileID") or (Path(rel).name if rel else ""),
        })
    return out


def image_findings(case_row, structured, cap: Captioner) -> tuple[list[dict], list[dict]]:
    """Caption images; findings get C-ids later, so they share one finding namespace."""
    imgs = pick_images(structured, case_row.DiseaseID, case_row.CaseID, case_row.ImageCount)
    findings, meta = [], []
    for i, im in enumerate(imgs, 1):
        caption = cap.caption(im["path"]) if im["path"] else None
        meta.append({
            "image_id": f"I{i}",
            "modality": im["modality"],
            "subtype": im["subtype"],
            "file": im["file"],
            "caption": caption,
            "caption_model": CAPTION_MODEL if caption else None,
        })
        text = f"{im['subtype']} image"
        if caption:
            text += f"; automatic caption: {caption}"
        findings.append({"text": text, "type": "imaging", "source": "image",
                         "quote": caption or "", "quote_verified": bool(caption)})
    return findings, meta


# ----------------------------------------------------------------------------
# ZebraMap profile + OptimusKG evidence
# ----------------------------------------------------------------------------

class EvidenceBase:
    def __init__(self, needed_disease_ids: set[str], profiles: dict | None = None):
        self.profiles = profiles if profiles is not None else json.load(
            (DATA / "augmented_disease_data.json").open(encoding="utf-8"))
        names = {}
        for zid in needed_disease_ids:
            prof = self.profiles.get(str(zid)) or {}
            nm = (prof.get("DiseaseInfo") or {}).get("Disease Name")
            if nm:
                names[str(zid)] = str(nm)
        self.kg = KGIndex(needed_disease_ids, names)
        self.needed = needed_disease_ids

    def zebramap_sentences(self, disease_id: str, query_terms: set[str],
                           limit: int = 6) -> list[dict]:
        prof = self.profiles.get(str(disease_id)) or {}
        info = prof.get("DiseaseInfo") or {}
        summary = prof.get("DiseaseSummary") or {}
        pool: list[tuple[str, str, float]] = []

        def add(field, text, bonus=0.0, allow_short=False):
            text = str(text or "").strip()
            if not text:
                return
            sents = split_sentences(text, min_len=4 if allow_short else 35)
            if allow_short and not sents:
                sents = [text[:240]]
            for s in sents:
                toks = content_terms(s)
                if not toks:
                    continue
                overlap = len(toks & query_terms)
                score = overlap / math.sqrt(len(toks)) + bonus
                pool.append((field, s, score))

        add("Definition", info.get("Disease Definition", ""), 1.0)
        add("Prevalence", info.get("Prevalence", ""), 0.3, allow_short=True)
        add("Inheritance", info.get("Inheritance", ""), 0.3, allow_short=True)
        add("Onset", info.get("Age of onset", ""), 0.2, allow_short=True)
        for f in PROFILE_FIELDS:
            add(f, summary.get(f, ""))

        # always keep the definition first, then fill by relevance with field diversity
        pool.sort(key=lambda x: -x[2])
        chosen, per_field = [], defaultdict(int)
        defs = [s for f, s, _ in pool if f == "Definition"]
        if defs:
            chosen.append(("Definition", defs[0]))
            per_field["Definition"] = 1
        for field, s, score in pool:
            if field == "Definition":
                continue
            if per_field[field] >= 2:
                continue
            chosen.append((field, s))
            per_field[field] += 1
            if len(chosen) >= limit:
                break
        return [{"field": f, "text": s} for f, s in chosen[:limit]]

    def kg_triples(self, disease_id: str, query_terms: set[str],
                   limit: int = 7) -> list[dict]:
        return self.kg.triples(disease_id, query_terms, limit)


class KGIndex:
    """Small, exact OptimusKG slice: Orphanet id first, then name/synonym."""

    FILES = [
        "nodes/disease.parquet", "nodes/phenotype.parquet", "nodes/gene.parquet",
        "edges/disease_phenotype.parquet", "edges/disease_gene.parquet",
        "edges/disease_disease.parquet",
    ]

    def __init__(self, needed: set[str], names_by_id: dict[str, str] | None = None):
        self.available = False
        self.map: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.pheno: dict[str, list] = defaultdict(list)
        self.genes: dict[str, list] = defaultdict(list)
        self.parents: dict[str, list] = defaultdict(list)
        self.names_by_id = names_by_id or {}
        try:
            self._ensure_files()
            self._build(needed)
            self.available = True
        except Exception as exc:
            log(f"[kg] disabled: {exc}")

    def _ensure_files(self):
        marker = KG_CACHE / "nodes" / "disease.parquet"
        if marker.exists():
            return
        import zipfile
        KG_CACHE.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(REPO / "optimuskg.zip") as z:
            for name in self.FILES:
                z.extract(name, KG_CACHE)

    def _build(self, needed: set[str]):
        d = pd.read_parquet(KG_CACHE / "nodes" / "disease.parquet")
        by_orpha, by_name = {}, {}
        for row in d.itertuples(index=False):
            props = row.properties
            node = str(row.id)
            name = props.get("name")
            if name:
                self.names[node] = str(name)
                by_name.setdefault(norm_text(name), node)
            for key in ("exact_synonyms", "related_synonyms", "narrow_synonyms",
                        "broad_synonyms"):
                vals = props.get(key)
                if vals is not None:
                    for s in vals:
                        by_name.setdefault(norm_text(s), node)
            xrefs = props.get("xrefs")
            if xrefs is not None:
                for x in xrefs:
                    m = re.fullmatch(r"Orphanet:(\d+)", str(x))
                    if m:
                        by_orpha.setdefault(m.group(1), node)
        for zid in needed:
            node = by_orpha.get(str(zid))
            if node is None:
                nm = self.names_by_id.get(str(zid), "")
                if nm:
                    node = by_name.get(norm_text(nm))
            if node:
                self.map[str(zid)] = node
        nodes = set(self.map.values())
        if not nodes:
            return

        # phenotype edges (small)
        ph = pd.read_parquet(
            KG_CACHE / "edges" / "disease_phenotype.parquet",
            filters=[("from", "in", list(nodes))],
        )
        phn = pd.read_parquet(KG_CACHE / "nodes" / "phenotype.parquet")
        ph_name = {}
        for row in phn.itertuples(index=False):
            p = row.properties
            ph_name[str(row.id)] = str(p.get("name") or row.id)
        for frm, to, props in zip(ph["from"], ph["to"], ph["properties"]):
            freq = None
            try:
                f = (props or {}).get("frequency")
                freq = str(f[0]) if f is not None and len(f) else None
            except Exception:
                freq = None
            self.pheno[str(frm)].append({
                "name": ph_name.get(str(to), str(to)),
                "frequency": freq,
            })

        # gene edges (large; predicate pushdown keeps it small)
        eg = pd.read_parquet(
            KG_CACHE / "edges" / "disease_gene.parquet",
            columns=["from", "to", "properties"],
            filters=[("from", "in", list(nodes))],
        )
        gn = pd.read_parquet(KG_CACHE / "nodes" / "gene.parquet")
        sym = {}
        for row in gn.itertuples(index=False):
            p = row.properties
            sym[str(row.id)] = str(p.get("symbol") or row.id)
        for frm, to, props in zip(eg["from"], eg["to"], eg["properties"]):
            props = props or {}
            score = props.get("evidence_score") or 0.0
            try:
                score = float(score)
            except Exception:
                score = 0.0
            self.genes[str(frm)].append({"symbol": sym.get(str(to), str(to)),
                                         "score": score})
        for k in self.genes:
            self.genes[k].sort(key=lambda x: -x["score"])

        dd = pd.read_parquet(
            KG_CACHE / "edges" / "disease_disease.parquet",
            filters=[("from", "in", list(nodes))],
        )
        for frm, to in zip(dd["from"], dd["to"]):
            self.parents[str(frm)].append({"name": self.names.get(str(to), str(to))})

    def triples(self, disease_id: str, query_terms: set[str], limit: int) -> list[dict]:
        node = self.map.get(str(disease_id))
        if not node:
            return []
        scored = []
        for p in self.pheno.get(node, []):
            toks = content_terms(p["name"])
            overlap = len(toks & query_terms)
            scored.append((overlap, 1 if p.get("frequency") else 0, "has_phenotype",
                           p["name"] + (f" (frequency {p['frequency']})"
                                        if p.get("frequency") else "")))
        for g in self.genes.get(node, [])[:15]:
            scored.append((0, 0, "associated_gene", g["symbol"]))
        for p in self.parents.get(node, [])[:4]:
            scored.append((0, 0, "is_a", p["name"]))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [{"relation": rel, "object": obj} for _, _, rel, obj in scored[:limit]]


# ----------------------------------------------------------------------------
# LLM prompts
# ----------------------------------------------------------------------------

PLAN_SYSTEM = """You are a clinical case analyst. Extract atomic findings from one rare-disease case report.

Rules:
- 6-14 findings. Each finding is ONE short clinical fact (max 18 words).
- Only facts stated in the case text or the supplied image metadata/captions.
- quote must be an exact substring of the case text (max 25 words). For an image finding, quote may be the caption.
- Do not diagnose, do not name or suggest diseases in findings.
- checks: one short thing to verify for each candidate disease name provided.
- Be terse. No commentary.

Return JSON only:
{"findings":[{"text":"...","type":"symptom|sign|lab|imaging|history|demographic","quote":"..."}],
 "checks":[{"candidate":"<name>","check":"..."}]}"""

AUDIT_SYSTEM = """You are a diagnostic evidence auditor. Candidates are anonymised A-J and unordered.

For each candidate compare the case findings [C#] with the supplied evidence:
- [Z-*] ZebraMap/Orphanet disease profile sentences
- [K-*] OptimusKG triples (phenotypes, genes, parents)
- [R-*] retriever trace (matched terms, ranks, model contributions)

Score:
- plausibility 0-100: how completely the findings are explained by this disease
- verdict: supported | uncertain | contradicted
- supports: finding IDs explained by cited evidence; cite only that candidate's IDs
- contradictions: findings that conflict with, or are unexpectedly absent from, this disease
- missing: the single most important expected feature not documented

Rules:
- Use ONLY the supplied evidence. No outside medical knowledge.
- Never invent an ID. Cite only IDs shown for that candidate.
- When a K triple directly matches a case finding, cite it; K is structured
  phenotype/gene evidence, not prose.
- Retriever rank is hidden and irrelevant; do not guess it.
- Judge each candidate independently. Several may be plausible.

Return JSON only:
{"candidates":[{"id":"A","plausibility":87,"verdict":"supported",
 "supports":[{"finding":"C1","evidence":["Z-A-1","K-A-2"],"reason":"max 12 words"}],
 "contradictions":[{"finding":"C3","evidence":["Z-A-4"],"reason":"max 12 words"}],
 "missing":"max 12 words"}]}"""

WRITE_SYSTEM = """You write short evidence-grounded explanations for ranked rare-disease candidates.

For each candidate write 3-5 claims. Rules:
- Every claim cites IDs. Support claims MUST cite at least one case finding C# AND at least one disease evidence item Z-* or K-*.
- Prefer a K citation when a triple directly supports the claim; otherwise use Z.
- contrast/caveat claims may cite fewer items but at least one.
- A claim must be directly supported by the cited text. No outside knowledge.
- Max 22 words per claim. No diagnostic certainty beyond what evidence shows.

Return JSON only:
{"claims":[{"candidate":"A","kind":"support|contrast|caveat","claim":"...","cites":["C2","Z-A-3"]}]}"""

ADJUDICATE_SYSTEM = """You are a strict citation adjudicator. For each claim, read the exact cited texts.

Verdicts:
- keep: fully supported by the cited text
- soften: partly supported; give weaker revised wording
- drop: not supported, citation mismatch, or adds unsupported detail

Return JSON only:
{"judgements":[{"i":0,"verdict":"keep|soften|drop","reason":"max 15 words","revised":"..."}]}"""

LLM_ONLY_SYSTEM = """You are a rare-disease clinician. You see a case summary and ten candidate diseases (A-J, unordered).

You have NO retrieval scores, NO ZebraMap profile, NO OptimusKG evidence. Use only the case summary and your medical knowledge.

Rank ALL ten candidates from most to least likely, and give a one-line reason for the top five.

Return JSON only:
{"order":["A","B","C","D","E","F","G","H","I","J"],
 "top5":[{"id":"A","reason":"max 12 words"}]}"""


def fmt_llm_only_input(row, findings, candidates) -> str:
    fl = "\n".join(f"[{f['id']}] ({f['type']}) {f['text']}" for f in findings)
    cl = "\n".join(f"- {c['audit_label']}: {c['name']}" for c in candidates)
    return (
        f"CASE SUMMARY ({row.CaseID})\nAge: {row.Age}; Gender: {row.Gender}\n\n"
        f"FINDINGS:\n{fl}\n\nCANDIDATES (unordered):\n{cl}\n\n"
        "Rank all ten and return the JSON object."
    )


def fmt_case_for_plan(row, text: str, images: list[dict], candidates: list[dict]) -> str:
    img_lines = []
    for im in images:
        cap = im.get("caption")
        img_lines.append(
            f"- {im['image_id']}: modality={im['subtype']}; "
            + (f"auto-caption=\"{cap}\" (machine-generated, may be wrong)" if cap
               else "no caption")
        )
    cand_lines = [f"- {c['audit_label']}: {c['name']}" for c in candidates]
    return (
        f"CASE {row.CaseID}\nAge: {row.Age}; Gender: {row.Gender}\n\n"
        f"IMAGE METADATA (metadata is reliable; captions are not):\n"
        + ("\n".join(img_lines) if img_lines else "- none")
        + f"\n\nCANDIDATE NAMES (unordered, no ranks):\n" + "\n".join(cand_lines)
        + f"\n\nCASE TEXT:\n{text}"
    )


def fmt_candidate_evidence(c: dict, ev: dict) -> str:
    lab = c["audit_label"]
    lines = [f"=== CANDIDATE {lab}: {c['name']} ==="]
    lines.append("ZEBRAMAP PROFILE:")
    if ev["z"]:
        for i, s in enumerate(ev["z"], 1):
            lines.append(f"[Z-{lab}-{i}] ({s['field']}) {s['text']}")
    else:
        lines.append("- none")
    lines.append("OPTIMUSKG:")
    if ev["kg_available"]:
        if ev["k"]:
            for i, t in enumerate(ev["k"], 1):
                lines.append(f"[K-{lab}-{i}] {t['object']} [{t['relation']}]")
        else:
            lines.append("- linked but no usable triples")
    else:
        lines.append("- no OptimusKG link for this disease")
    tr = ev["trace"]
    lines.append(
        f"RETRIEVER TRACE: ranks tfidf={tr['tfidf_rank']} bow={tr['bow_rank']}; "
        f"tfidf={tr['tfidf_score']:.3f} bow={tr['bow_score']:.3f}; "
        f"in_tfidf_pool={tr['in_tfidf_top100']}"
    )
    terms = ", ".join(m["term"] for m in tr["matched_terms"])
    lines.append(f"[R-{lab}-1] matched case terms: {terms or 'none'}")
    if tr["gbm"]["top_contributions"]:
        parts = ", ".join(
            f"{x['feature']}={x['contribution']:+.3f}"
            for x in tr["gbm"]["top_contributions"]
        )
        lines.append(f"[R-{lab}-2] model contributions: {parts}")
    return "\n".join(lines)


def fmt_audit_input(findings, candidates, evidence) -> str:
    fl = "\n".join(f"[{f['id']}] ({f['type']}) {f['text']}" for f in findings)
    blocks = [fmt_candidate_evidence(c, evidence[c["audit_label"]]) for c in candidates]
    return (
        "CASE FINDINGS:\n" + fl + "\n\n"
        + "\n\n".join(blocks)
        + "\n\nAudit every candidate A-J and return the JSON object."
    )


def fmt_write_input(final5, findings, evidence) -> str:
    fl = "\n".join(f"[{f['id']}] ({f['type']}) {f['text']}" for f in findings)
    blocks = []
    for c in final5:
        lab = c["audit_label"]
        ev = evidence[lab]
        z = "\n".join(f"[Z-{lab}-{i}] {s['text']}"
                      for i, s in enumerate(ev["z"], 1))
        k = "\n".join(f"[K-{lab}-{i}] {t['object']} [{t['relation']}]"
                      for i, t in enumerate(ev["k"], 1))
        blocks.append(
            f"=== CANDIDATE {lab}: {c['name']} ===\n"
            f"ZEBRAMAP:\n{z or '- none'}\nOPTIMUSKG:\n{k or '- none'}"
        )
    return ("CASE FINDINGS:\n" + fl + "\n\n" + "\n\n".join(blocks)
            + "\n\nWrite claims for these 5 candidates and return the JSON object.")


# ----------------------------------------------------------------------------
# deterministic validator
# ----------------------------------------------------------------------------

def validate_claims(claims: list[dict], findings: list[dict],
                    evidence: dict, case_norm: str) -> tuple[list[dict], dict]:
    c_ids = {f["id"]: f for f in findings}
    valid, reasons = [], []
    for cl in claims:
        label = str(cl.get("candidate", ""))
        ev = evidence.get(label, {})
        z_ids = {f"Z-{label}-{i}" for i in range(1, len(ev.get("z", [])) + 1)}
        k_ids = {f"K-{label}-{i}" for i in range(1, len(ev.get("k", [])) + 1)}
        r_ids = {f"R-{label}-{i}" for i in range(1, 3)}
        allowed = set(c_ids) | z_ids | k_ids | r_ids
        cites = [str(x) for x in (cl.get("cites") or [])]
        errs = []
        if not str(cl.get("claim", "")).strip():
            errs.append("empty claim")
        if not cites:
            errs.append("no citations")
        bad = [x for x in cites if x not in allowed]
        if bad:
            errs.append("unknown: " + ",".join(bad))
        kind = str(cl.get("kind", "support"))
        if kind == "support":
            if not any(x in c_ids for x in cites):
                errs.append("support claim lacks case finding")
            if not any(x in z_ids or x in k_ids for x in cites):
                errs.append("support claim lacks disease evidence")
        # retriever-term claims must point at terms that really occur in the case
        for x in cites:
            if x in r_ids and x.endswith("-1"):
                terms = [m["term"] for m in ev.get("trace", {}).get("matched_terms", [])]
                if not terms or not all(norm_text(t) in case_norm for t in terms):
                    errs.append("retriever term not in case text")
                    break
        cl = dict(cl)
        cl["validator_ok"] = not errs
        cl["validator_errors"] = errs
        valid.append(cl)
        reasons.extend(errs)

    cited = set()
    for cl in valid:
        if cl["validator_ok"]:
            cited.update(str(x) for x in cl.get("cites", []))
    metrics = {
        "claims": len(valid),
        "valid_claims": sum(1 for c in valid if c["validator_ok"]),
        "valid_rate": (sum(1 for c in valid if c["validator_ok"]) / len(valid)) if valid else None,
        "errors": len(reasons),
        "unique_evidence_cited": len(cited),
    }
    return valid, metrics


# ----------------------------------------------------------------------------
# one case through the pipeline
# ----------------------------------------------------------------------------

def process_case(rec: dict, llm: LLM, ev_base: EvidenceBase, ret: Retriever,
                 force: bool = False) -> dict:
    case_id = rec["case_id"]
    out_path = CASE_DIR / f"{case_id}.json"
    if out_path.exists() and not force:
        try:
            old = json.load(out_path.open(encoding="utf-8"))
            if (old.get("pipeline_version") == PIPELINE_VERSION
                    and old.get("llm_model") == llm.model):
                log(f"CASE cache {case_id} (model match, skipped)")
                return old
            log(f"CASE cache {case_id} ignored "
                f"(model {old.get('llm_model')} != {llm.model})")
        except Exception:
            pass

    t_start = now()
    case_idx = rec["case_idx"]
    row = rec["row"]
    text_full = str(row.Text)
    text = text_full[:MAX_CASE_CHARS]
    if len(text_full) > MAX_CASE_CHARS:
        text += "\n[... case text truncated for the planning call ...]"

    retr_order = rec["retriever_order"]
    gold_retr_rank = int(np.where(retr_order == int(rec["gold_idx"]))[0][0]) + 1
    log(f"CASE start {case_id} | gold={rec['gold_name']} | retrieval_rank={gold_retr_rank} "
        f"| text_chars={len(text_full)}")
    top10 = retr_order[:K_RETRIEVE]
    cand_meta = []
    for j, didx in enumerate(top10):
        cand_meta.append({
            "label": chr(ord("A") + j),
            "disease_idx": int(didx),
            "disease_id": ret.diseases[didx],
            "name": rec["name_by_idx"][int(didx)],
        })

    # ---- shuffling hides retriever order from the auditor
    rng = np.random.default_rng(SEED + case_idx)
    audit_meta = list(cand_meta)
    rng.shuffle(audit_meta)
    for pos, c in enumerate(audit_meta):
        c["audit_label"] = chr(ord("A") + pos)
    label_map = {c["audit_label"]: c for c in audit_meta}

    # Evidence is fetched before shuffling; IDs use the audit label.
    evidence = {}
    for c in audit_meta:
        evidence[c["audit_label"]] = {
            "disease_id": c["disease_id"],
            "name": c["name"],
            "z": [],
            "k": [],
            "kg_available": False,
            "trace": {},
        }

    # ---- images (cached captions)
    img_findings, img_meta = image_findings(row, rec["structured"], rec["captioner"])

    # ---- CALL 1: plan
    plan_prompt = fmt_case_for_plan(row, text, img_meta, audit_meta)
    plan, plan_meta = llm.json_chat(
        [{"role": "system", "content": PLAN_SYSTEM},
         {"role": "user", "content": plan_prompt}],
        max_tokens=1400, required=["findings"], tag="plan",
    )
    findings = []
    if plan:
        for f in (plan.get("findings") or [])[:16]:
            t = str(f.get("text", "")).strip()
            if not t:
                continue
            q = str(f.get("quote", "")).strip()
            quote_ok = bool(q) and norm_text(q) in norm_text(text_full)
            findings.append({
                "id": f"C{len(findings)+1}",
                "text": t,
                "type": str(f.get("type", "sign")),
                "quote": q,
                "quote_verified": quote_ok,
                "source": "text",
            })
    if not [f for f in findings if f["source"] == "text"]:
        # deterministic fallback so a failed call does not kill the case
        for s in split_sentences(text)[:10]:
            findings.append({"id": f"C{len(findings)+1}", "text": s[:160],
                             "type": "sign", "quote": s, "quote_verified": True,
                             "source": "text"})
    for im in img_findings:
        findings.append({"id": f"C{len(findings)+1}", **im})

    checks = {}
    for ck in (plan or {}).get("checks", []) or []:
        nm = norm_text(str(ck.get("candidate", "")))
        if nm:
            checks[nm] = str(ck.get("check", ""))
    query_terms = set()
    for f in findings:
        query_terms |= content_terms(f["text"])
    query_terms |= content_terms(" ".join(checks.values()))

    # ---- deterministic evidence fetch
    for c in audit_meta:
        ev = evidence[c["audit_label"]]
        check = checks.get(norm_text(c["name"]), "")
        q = query_terms | content_terms(check)
        ev["z"] = ev_base.zebramap_sentences(c["disease_id"], q, limit=6)
        ev["k"] = ev_base.kg_triples(c["disease_id"], q, limit=7)
        ev["kg_available"] = bool(ev_base.kg.map.get(str(c["disease_id"])))
        ev["trace"] = ret.trace(case_idx, c["disease_idx"], text_full, findings)
        ev["check"] = check

    # ---- CALL 2b: LLM-only baseline (no evidence, no ranks)
    lo_obj, lo_meta = llm.json_chat(
        [{"role": "system", "content": LLM_ONLY_SYSTEM},
         {"role": "user", "content": fmt_llm_only_input(row, findings, audit_meta)}],
        max_tokens=900, required=["order"], tag="llm_only",
    )
    valid_labels = [c["audit_label"] for c in audit_meta]
    lo_order = []
    if lo_obj:
        for lab in lo_obj.get("order") or []:
            lab = str(lab).strip().upper()[:1]
            if lab in label_map and lab not in lo_order:
                lo_order.append(lab)
    lo_ok = len(lo_order) == len(valid_labels)
    if not lo_ok:
        lo_order = [c["audit_label"] for c in sorted(
            audit_meta, key=lambda x: int(np.where(retr_order == x["disease_idx"])[0][0]))]
    lo_tail = [int(x) for x in retr_order[K_RETRIEVE:]
               if int(x) not in {label_map[l]["disease_idx"] for l in lo_order}]
    llm_only = {
        "ok": lo_ok,
        "order": lo_order,
        "top5": [
            {"label": str(x.get("id", "")).upper()[:1],
             "name": label_map.get(str(x.get("id", "")).upper()[:1], {}).get("name", ""),
             "reason": str(x.get("reason", ""))}
            for x in (lo_obj or {}).get("top5", [])[:5]
        ],
        "metrics": rank_metrics(
            np.array([label_map[l]["disease_idx"] for l in lo_order] + lo_tail),
            int(rec["gold_idx"]),
        ),
        "call_meta": lo_meta,
    }

    # ---- CALL 2: audit
    audit_input = fmt_audit_input(findings, audit_meta, evidence)
    audit, audit_meta_call = llm.json_chat(
        [{"role": "system", "content": AUDIT_SYSTEM},
         {"role": "user", "content": audit_input}],
        max_tokens=2600, required=["candidates"], tag="audit",
    )
    audit_by_label = {}
    if audit:
        for c in audit.get("candidates", []) or []:
            lab = str(c.get("id", "")).strip().upper()[:1]
            if lab in label_map:
                try:
                    p = float(c.get("plausibility"))
                except Exception:
                    p = 50.0
                audit_by_label[lab] = {
                    "label": lab,
                    "plausibility": max(0.0, min(100.0, p)),
                    "verdict": str(c.get("verdict", "uncertain")),
                    "supports": c.get("supports") or [],
                    "contradictions": c.get("contradictions") or [],
                    "missing": c.get("missing", ""),
                }
    audit_ok = len(audit_by_label) == len(audit_meta)
    if not audit_ok:
        # fallback: retrieval order expressed as plausibility
        for c in audit_meta:
            r = int(np.where(retr_order == c["disease_idx"])[0][0]) + 1
            audit_by_label.setdefault(c["audit_label"], {
                "label": c["audit_label"], "plausibility": float(max(1, 101 - 10 * r)),
                "verdict": "uncertain", "supports": [], "contradictions": [],
                "missing": "", "fallback": True,
            })

    # ---- final ordering: audit plausibility, retriever rank as tie-break
    def retr_pos(c):
        return int(np.where(retr_order == c["disease_idx"])[0][0])

    final_order = sorted(
        cand_meta,
        key=lambda c: (-audit_by_label[c["audit_label"]]["plausibility"], retr_pos(c)),
    )
    final5 = final_order[:K_EXPLAIN]
    for c in cand_meta:
        c["retriever_rank"] = retr_pos(c) + 1
        c["final_rank"] = final_order.index(c) + 1
        c["plausibility"] = audit_by_label[c["audit_label"]]["plausibility"]
    moved = sum(1 for c in cand_meta if c["final_rank"] != c["retriever_rank"])
    log(f"CASE audit  {case_id} | parsed={audit_ok} | positions_changed={moved} "
        f"| new_top1={final_order[0]['name']} (retr {final_order[0]['retriever_rank']})")

    # ---- CALL 3: explanations for final top-5
    write_input = fmt_write_input(final5, findings, evidence)
    written, write_meta = llm.json_chat(
        [{"role": "system", "content": WRITE_SYSTEM},
         {"role": "user", "content": write_input}],
        max_tokens=2200, required=["claims"], tag="write",
    )
    claims = []
    if written:
        for cl in (written.get("claims") or [])[:30]:
            lab = str(cl.get("candidate", "")).strip().upper()[:1]
            if lab in label_map:
                claims.append({
                    "candidate": lab,
                    "disease": label_map[lab]["name"],
                    "kind": str(cl.get("kind", "support")),
                    "claim": str(cl.get("claim", "")).strip(),
                    "cites": [str(x) for x in (cl.get("cites") or [])],
                })
    case_norm = " " + norm_text(text_full) + " "
    claims, valid_metrics = validate_claims(claims, findings, evidence, case_norm)

    # ---- CALL 4: adjudicate each claim against exact cited text
    adj_input, claim_index = [], []
    for idx, cl in enumerate(claims):
        lab = cl["candidate"]
        ev = evidence[lab]
        ctx = []
        for cid in cl["cites"]:
            txt = evidence_text(cid, findings, ev)
            if txt:
                ctx.append(f"{cid}: {txt}")
        adj_input.append({
            "i": idx,
            "claim": cl["claim"],
            "kind": cl["kind"],
            "cited_text": ctx,
        })
        claim_index.append(idx)
    judgements = {}
    if claims:
        adj, adj_meta = llm.json_chat(
            [{"role": "system", "content": ADJUDICATE_SYSTEM},
             {"role": "user", "content":
              "Judge each claim against its cited texts.\n"
              + json.dumps(adj_input, ensure_ascii=False)
              + "\nReturn the JSON object."}],
            max_tokens=1600, required=["judgements"], tag="adjudicate",
        )
        if adj:
            for j in adj.get("judgements", []) or []:
                try:
                    i = int(j.get("i"))
                except Exception:
                    continue
                judgements[i] = {
                    "verdict": str(j.get("verdict", "keep")).lower(),
                    "reason": str(j.get("reason", "")),
                    "revised": str(j.get("revised", "")),
                }
    for idx, cl in enumerate(claims):
        j = judgements.get(idx, {"verdict": "keep", "reason": "no adjudication",
                                 "revised": ""})
        if j["verdict"] not in ("keep", "soften", "drop"):
            j["verdict"] = "keep"
        cl["adjudication"] = j

    rec_out = {
        "pipeline_version": PIPELINE_VERSION,
        "llm_model": llm.model,
        "case_id": case_id,
        "case_idx": int(case_idx),
        "gold_disease_id": rec["gold_id"],
        "gold_disease_name": rec["gold_name"],
        "gold_index": int(rec["gold_idx"]),
        "retriever": {
            "system": "fusion_v4_shap8",
            "order": [int(x) for x in retr_order[:K_RETRIEVE]],
            "metrics": rank_metrics(retr_order, int(rec["gold_idx"])),
            "top10": [
                {"rank": c["retriever_rank"], "label": c["label"],
                 "disease_id": c["disease_id"], "name": c["name"],
                 "score": float(rec["scores"][c["disease_idx"]])}
                for c in sorted(cand_meta, key=lambda x: x["retriever_rank"])
            ],
        },
        "case_text_chars": len(text_full),
        "case_text_truncated": len(text_full) > MAX_CASE_CHARS,
        "images": img_meta,
        "findings": findings,
        "plans": {"plan_meta": plan_meta, "checks": checks},
        "evidence": evidence,
        "audit": {
            "call_meta": audit_meta_call,
            "ok": audit_ok,
            "candidates": [audit_by_label[c["audit_label"]] for c in audit_meta],
        },
        "audit_label_map": {c["audit_label"]: c["disease_idx"] for c in audit_meta},
        "candidates": [
            {"label": c["label"], "audit_label": c["audit_label"],
             "disease_id": c["disease_id"], "name": c["name"],
             "plausibility": c["plausibility"], "retriever_rank": c["retriever_rank"],
             "final_rank": c["final_rank"]}
            for c in cand_meta
        ],
        "final_order": [
            {"label": c["label"], "audit_label": c["audit_label"],
             "disease_id": c["disease_id"], "name": c["name"],
             "plausibility": c["plausibility"], "retriever_rank": c["retriever_rank"],
             "final_rank": c["final_rank"]}
            for c in final_order
        ],
        "final_metrics": rank_metrics(
            np.array([c["disease_idx"] for c in final_order]
                     + [int(x) for x in retr_order[K_RETRIEVE:]]),
            int(rec["gold_idx"]),
        ),
        "claims": claims,
        "claim_validation": valid_metrics,
        "llm_only_baseline": llm_only,
        "call_meta": {"write": write_meta},
        "elapsed_s": round(now() - t_start, 2),
    }
    write_json(out_path, rec_out)
    log(f"CASE done  {case_id} | retr_rank={gold_retr_rank} "
        f"final_rank={rec_out['final_metrics']['rank']} "
        f"| MRR {rec_out['retriever']['metrics']['mrr']:.3f} -> "
        f"{rec_out['final_metrics']['mrr']:.3f} "
        f"| claims={len(claims)} valid={valid_metrics['valid_claims']} "
        f"| {rec_out['elapsed_s']:.1f}s")
    return rec_out


def evidence_text(cid: str, findings: list[dict], ev: dict) -> str | None:
    if cid.startswith("C"):
        for f in findings:
            if f["id"] == cid:
                return f["text"]
        return None
    m = re.fullmatch(r"([ZKR])-[A-J]-(\d+)", cid)
    if not m:
        return None
    kind, n = m.group(1), int(m.group(2))
    if kind == "Z" and 1 <= n <= len(ev.get("z", [])):
        return ev["z"][n - 1]["text"]
    if kind == "K" and 1 <= n <= len(ev.get("k", [])):
        t = ev["k"][n - 1]
        return f"{t['object']} [{t['relation']}]"
    if kind == "R" and n == 1:
        return "matched case terms: " + ", ".join(
            m["term"] for m in ev.get("trace", {}).get("matched_terms", []))
    if kind == "R" and n == 2:
        return "; ".join(
            f"{x['feature']}={x['contribution']:+.3f}"
            for x in ev.get("trace", {}).get("gbm", {}).get("top_contributions", []))
    return None


# ----------------------------------------------------------------------------
# evaluation and report
# ----------------------------------------------------------------------------

def evaluate(results: list[dict]) -> dict:
    if not results:
        return {}
    ret_mrr = np.array([r["retriever"]["metrics"]["mrr"] for r in results])
    fin_mrr = np.array([r["final_metrics"]["mrr"] for r in results])
    delta = fin_mrr - ret_mrr
    lo, hi = bootstrap_ci(delta)

    # plausibility separation: gold vs the nine non-gold audited candidates
    pos, neg = [], []
    for r in results:
        for c in r["candidates"]:
            (pos if c["disease_id"] == r["gold_disease_id"] else neg).append(c["plausibility"])
    pl_auc = auroc(np.array(pos + neg),
                   np.array([1] * len(pos) + [0] * len(neg)))

    claims = [c for r in results for c in r["claims"]]
    judged = [c for c in claims if c.get("adjudication")]
    verdicts = defaultdict(int)
    for c in judged:
        verdicts[c["adjudication"]["verdict"]] += 1

    def zk(cites):
        return any(str(x).startswith(("Z-", "K-")) for x in cites)

    def k_only(cites):
        return any(str(x).startswith("K-") for x in cites)

    kg_claims = [c for c in claims if k_only(c["cites"])]
    non_kg = [c for c in claims if not k_only(c["cites"])]

    def keep_rate(cs):
        cs = [c for c in cs if c.get("adjudication")]
        if not cs:
            return None
        return sum(c["adjudication"]["verdict"] != "drop" for c in cs) / len(cs)

    # KG linkage availability across audited candidates
    kg_cov = []
    for r in results:
        for lab, ev in r["evidence"].items():
            kg_cov.append(bool(ev.get("kg_available")))

    # trace terms validated
    trace_terms = 0
    trace_present = 0
    for r in results:
        for ev in r["evidence"].values():
            for m in ev["trace"].get("matched_terms", []):
                trace_terms += 1
                trace_present += int(m.get("present_in_case", False))

    val = [r["claim_validation"] for r in results]
    n_claims = sum(v.get("claims", 0) for v in val)
    n_valid = sum(v.get("valid_claims", 0) for v in val)

    # where does the LLM help? bucket by retriever rank of the gold
    buckets = {"rank_1": [], "rank_2_5": [], "rank_6_10": [], "outside_10": []}
    for r in results:
        rr = r["retriever"]["metrics"]["rank"]
        key = ("rank_1" if rr == 1 else "rank_2_5" if rr <= 5
               else "rank_6_10" if rr <= 10 else "outside_10")
        buckets[key].append(r)
    by_rank = {}
    for key, rows in buckets.items():
        if not rows:
            continue
        by_rank[key] = {
            "n": len(rows),
            "retrieval_mrr": float(np.mean([r["retriever"]["metrics"]["mrr"] for r in rows])),
            "complete_mrr": float(np.mean([r["final_metrics"]["mrr"] for r in rows])),
            "top5_delta": float(np.mean([
                r["final_metrics"]["top5"] - r["retriever"]["metrics"]["top5"] for r in rows
            ])),
        }
    rescued_top5 = sum(
        1 for r in results
        if r["retriever"]["metrics"]["top5"] == 0 and r["final_metrics"]["top5"] == 1
    )
    lost_top5 = sum(
        1 for r in results
        if r["retriever"]["metrics"]["top5"] == 1 and r["final_metrics"]["top5"] == 0
    )

    lo_rows = [r.get("llm_only_baseline") or {} for r in results]
    lo_mrr = np.array([x.get("metrics", {}).get("mrr", np.nan) for x in lo_rows], dtype=float)
    lo_mask = ~np.isnan(lo_mrr)
    llm_only_metrics = None
    if lo_mask.any():
        llm_only_metrics = {
            "mrr": float(np.nanmean(lo_mrr)),
            "top1": float(np.mean([x["metrics"]["top1"] for x, m in zip(lo_rows, lo_mask) if m])),
            "top5": float(np.mean([x["metrics"]["top5"] for x, m in zip(lo_rows, lo_mask) if m])),
            "top10": float(np.mean([x["metrics"]["top10"] for x, m in zip(lo_rows, lo_mask) if m])),
            "delta_mrr_vs_retrieval": float(np.nanmean(lo_mrr - ret_mrr)),
            "improved": int(np.nansum(lo_mrr - ret_mrr > 1e-9)),
            "harmed": int(np.nansum(lo_mrr - ret_mrr < -1e-9)),
            "parse_ok_rate": float(np.mean([bool(x.get("ok")) for x in lo_rows])),
        }

    return {
        "n_cases": len(results),
        "retrieval": {
            "mrr": float(ret_mrr.mean()),
            "top1": float(np.mean([r["retriever"]["metrics"]["top1"] for r in results])),
            "top5": float(np.mean([r["retriever"]["metrics"]["top5"] for r in results])),
            "top10": float(np.mean([r["retriever"]["metrics"]["top10"] for r in results])),
        },
        "complete_system": {
            "mrr": float(fin_mrr.mean()),
            "top1": float(np.mean([r["final_metrics"]["top1"] for r in results])),
            "top5": float(np.mean([r["final_metrics"]["top5"] for r in results])),
            "top10": float(np.mean([r["final_metrics"]["top10"] for r in results])),
        },
        "llm_only": llm_only_metrics,
        "delta": {
            "mrr": float(delta.mean()),
            "mrr_ci95": [lo, hi],
            "top1": float(np.mean([r["final_metrics"]["top1"] - r["retriever"]["metrics"]["top1"]
                                   for r in results])),
            "top5": float(np.mean([r["final_metrics"]["top5"] - r["retriever"]["metrics"]["top5"]
                                   for r in results])),
            "improved": int((delta > 1e-9).sum()),
            "harmed": int((delta < -1e-9).sum()),
            "tied": int((np.abs(delta) <= 1e-9).sum()),
            "rescued_top5": rescued_top5,
            "lost_top5": lost_top5,
            "by_retriever_rank": by_rank,
        },
        "plausibility": {
            "gold_mean": float(np.mean(pos)) if pos else None,
            "nongold_mean": float(np.mean(neg)) if neg else None,
            "auroc_gold_vs_nongold": pl_auc,
            "auditor_top1_accuracy": float(np.mean([
                r["final_order"][0]["disease_id"] == r["gold_disease_id"] for r in results
            ])),
        },
        "evidence": {
            "kg_candidate_coverage": float(np.mean(kg_cov)) if kg_cov else None,
            "kg_available_candidates": int(sum(kg_cov)),
            "audited_candidates": len(kg_cov),
            "claims_total": len(claims),
            "claims_citing_kg": len(kg_claims),
            "claims_citing_kg_rate": (len(kg_claims) / len(claims)) if claims else None,
            "keep_or_soften_rate_kg": keep_rate(kg_claims),
            "keep_or_soften_rate_non_kg": keep_rate(non_kg),
            "trace_terms": trace_terms,
            "trace_terms_in_case": trace_present,
            "trace_term_rate": (trace_present / trace_terms) if trace_terms else None,
            "zebra_available_candidates": int(sum(
                1 for r in results for ev in r["evidence"].values() if ev.get("z"))),
        },
        "citation": {
            "claims": n_claims,
            "valid_claims": n_valid,
            "valid_rate": (n_valid / n_claims) if n_claims else None,
            "verdicts": dict(verdicts),
            "claims_per_case": (n_claims / len(results)) if results else None,
        },
        "llm": {},
    }


def render_case(r: dict) -> str:
    gold_rank = r["retriever"]["metrics"]["rank"]
    lines = [
        f"## {r['case_id']}",
        f"Gold: **{r['gold_disease_name']}** (`{r['gold_disease_id']}`) · "
        f"retriever rank {gold_rank} · final rank {r['final_metrics']['rank']}",
        "",
        "| retr | final | disease | plausibility |",
        "|---:|---:|---|---:|",
    ]
    for c in r["final_order"]:
        lines.append(
            f"| {c['retriever_rank']} | {c['final_rank']} | {c['name']} | {c['plausibility']:.0f} |"
        )
    lines.append("")
    by_label = {c["label"]: c for c in r["candidates"]}
    verdict_icon = {"keep": "KEEP", "soften": "SOFTEN", "drop": "DROP"}
    for c in r["final_order"][:K_EXPLAIN]:
        if c["final_rank"] > K_EXPLAIN:
            continue
        lines.append(f"### {c['final_rank']}. {c['name']} — plausibility {c['plausibility']:.0f}")
        mine = [x for x in r["claims"] if x["candidate"] == c["audit_label"]]
        kept = [x for x in mine if x["adjudication"]["verdict"] != "drop"]
        dropped = [x for x in mine if x["adjudication"]["verdict"] == "drop"]
        for x in kept:
            v = x["adjudication"]["verdict"]
            text = x["claim"]
            if v == "soften" and x["adjudication"].get("revised"):
                text = f"{text} → *{x['adjudication']['revised']}*"
            flag = "" if x["validator_ok"] else " ⚠validator"
            lines.append(f"- [{verdict_icon[v]}]{flag} {text} ({', '.join(x['cites'])})")
        if dropped:
            lines.append(f"- struck: {len(dropped)} claim(s)")
            for x in dropped:
                lines.append(f"  - ~~{x['claim']}~~ ({x['adjudication']['reason']})")
        if not mine:
            lines.append("- no claims produced")
        lines.append("")
    return "\n".join(lines)


def write_report(results, metrics, llm: LLM, wall_s: float) -> None:
    lines = [
        "# Glass-box explanation pipeline — report",
        "",
        f"Cases: **{metrics.get('n_cases', 0)}** · retriever: **fusion v4 SHAP-8** · "
        f"LLM: `{llm.model}` (temperature 0) · seed {SEED}",
        "",
        "## Headline metrics",
        "",
        "| System | MRR | Top-1 | Top-5 | Top-10 |",
        "|---|---:|---:|---:|---:|",
    ]
    rm, cm = metrics["retrieval"], metrics["complete_system"]
    lines.append(f"| Retrieval alone | {rm['mrr']:.4f} | {rm['top1']:.4f} | "
                 f"{rm['top5']:.4f} | {rm['top10']:.4f} |")
    lm = metrics.get("llm_only")
    if lm:
        lines.append(f"| LLM-only (10 candidates, no evidence) | {lm['mrr']:.4f} | "
                     f"{lm['top1']:.4f} | {lm['top5']:.4f} | {lm['top10']:.4f} |")
    lines.append(f"| Complete system (LLM audit order) | {cm['mrr']:.4f} | "
                 f"{cm['top1']:.4f} | {cm['top5']:.4f} | {cm['top10']:.4f} |")
    d = metrics["delta"]
    ci_lo, ci_hi = d["mrr_ci95"][0], d["mrr_ci95"][1]
    lines += [
        "",
        f"Δ MRR **{d['mrr']:+.4f}** (95% CI {ci_lo:+.4f} … {ci_hi:+.4f}); "
        f"wins {d['improved']}, losses {d['harmed']}, "
        f"ties {d['tied']}. Δ Top-1 {d['top1']:+.4f}; Δ Top-5 {d['top5']:+.4f}.",
        "",
        "Top-10 is unchanged by construction: the LLM only sees and reorders the "
        "retriever’s top-10. Only Top-5 and MRR can move.",
        "",
        "### Where the LLM changes the ranking",
        "",
        "| gold retrieval rank | n | retrieval MRR | complete MRR | Δ top-5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("rank_1", "1"), ("rank_2_5", "2–5"),
                       ("rank_6_10", "6–10"), ("outside_10", ">10")):
        b = d["by_retriever_rank"].get(key)
        if not b:
            continue
        lines.append(f"| {label} | {b['n']} | {b['retrieval_mrr']:.4f} | "
                     f"{b['complete_mrr']:.4f} | {b['top5_delta']:+.4f} |")
    lines += [
        "",
        f"Rescued into top-5: **{d['rescued_top5']}** · lost from top-5: "
        f"**{d['lost_top5']}**.",
        "",
        "## Does the LLM add anything?",
        "",
        f"- Auditor top-1 accuracy: **{metrics['plausibility']['auditor_top1_accuracy']:.4f}** "
        f"(retrieval top-1 {rm['top1']:.4f}).",
        f"- Gold plausibility {fmt_num(metrics['plausibility']['gold_mean'])} vs non-gold "
        f"{fmt_num(metrics['plausibility']['nongold_mean'])}; AUROC "
        f"**{fmt_num(metrics['plausibility']['auroc_gold_vs_nongold'], 4)}**.",
    ]
    if lm:
        lines += [
            f"- LLM-only ranking of the same ten candidates: MRR {lm['mrr']:.4f} "
            f"({lm['delta_mrr_vs_retrieval']:+.4f} vs retrieval), top-1 {lm['top1']:.4f}, "
            f"top-5 {lm['top5']:.4f}; wins {lm['improved']}, losses {lm['harmed']}. "
            f"JSON parse rate {lm['parse_ok_rate']:.1%}.",
            "- That baseline has no retrieved evidence and no retriever ranks, so it isolates "
            "the contribution of evidence-grounded auditing from the LLM's prior alone.",
        ]
    ev = metrics["evidence"]
    lines += [
        "",
        "## Is the evidence useful?",
        "",
        f"- OptimusKG linked for **{ev['kg_available_candidates']}/{ev['audited_candidates']}** "
        f"audited candidates ({ev['kg_candidate_coverage']:.1%}).",
        f"- Claims citing OptimusKG: **{ev['claims_citing_kg']}/{ev['claims_total']}** "
        f"({(ev['claims_citing_kg_rate'] or 0):.1%}).",
        f"- Keep-or-soften rate, claims citing KG: "
        f"**{fmt_pct(ev['keep_or_soften_rate_kg'])}** vs ZebraMap-only "
        f"**{fmt_pct(ev['keep_or_soften_rate_non_kg'])}**.",
        f"- Retriever trace terms verified in case text: "
        f"**{ev['trace_terms_in_case']}/{ev['trace_terms']}** "
        f"({(ev['trace_term_rate'] or 0):.1%}).",
        f"- ZebraMap profile available for {ev['zebra_available_candidates']} "
        f"of {ev['audited_candidates']} audited candidates.",
        "",
        "## Citation quality",
        "",
        f"- Claims: **{metrics['citation']['claims']}** "
        f"({fmt_num(metrics['citation']['claims_per_case'], 1)}/case).",
        f"- Deterministic validator pass rate: **{fmt_pct(metrics['citation']['valid_rate'])}**.",
        f"- Adjudication: " + ", ".join(
            f"{k} {v}" for k, v in sorted(metrics["citation"]["verdicts"].items())
        ),
        "",
        f"Wall time {wall_s/60:.1f} min · LLM calls {llm.calls} · "
        f"tokens in/out {llm.prompt_tokens:,}/{llm.completion_tokens:,} · "
        f"parse/build failures {llm.failures}.",
        "",
        "## Example cases",
        "",
    ]
    lines.append(render_case(results[0]))
    lines.append("")
    if len(results) > 1:
        lines.append(render_case(results[len(results) // 2]))
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    with (OUT / "cases_report.md").open("w", encoding="utf-8") as f:
        f.write("# Per-case glass-box reports\n\n")
        for r in sorted(results, key=lambda x: x["case_id"]):
            f.write(render_case(r))
            f.write("\n\n")


def fmt_pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def fmt_num(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=N_CASES_DEFAULT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    if args.force:
        LOG_PATH.write_text("", encoding="utf-8")
    t0 = now()
    log("=" * 78)
    log(f"START pipeline={PIPELINE_VERSION} n_cases={args.n_cases} workers={args.workers} "
        f"force={args.force} captions={not args.no_caption}")
    log(f"      repo={REPO}  output={OUT}")

    log("[1/6] loading retriever artifacts")
    ret = Retriever()
    n_test = len(ret.case_ids)

    log("[2/6] scoring full test set with frozen fusion v4 SHAP-8")
    X = np.load(V4 / "X_test.npy", mmap_mode="r")
    Xs = np.ascontiguousarray(X[:, ret.sel]).reshape(n_test, 100, len(ret.sel))
    all_idx = np.arange(n_test)
    S = ret.scores(all_idx, Xs)
    orders = np.argsort(-S, axis=1, kind="stable")
    ranks = np.array([np.where(orders[i] == ret.gold_idx[i])[0][0] + 1 for i in range(n_test)])
    full_metrics = {"mrr": float(np.mean(1.0 / ranks)),
                    "top1": float(np.mean(ranks == 1)),
                    "top5": float(np.mean(ranks <= 5)),
                    "top10": float(np.mean(ranks <= 10))}
    log(f"      full test: MRR {full_metrics['mrr']:.4f}  "
        f"top1 {full_metrics['top1']:.4f}  top5 {full_metrics['top5']:.4f}  "
        f"top10 {full_metrics['top10']:.4f}")

    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(n_test, min(args.n_cases, n_test), replace=False))
    ret.X = Xs  # used by trace()

    log("[3/6] loading cases, images, ZebraMap profiles")
    df = pd.read_csv(SPLIT / "test.csv", keep_default_na=False)
    df = df.iloc[sample].reset_index(drop=True)
    structured = json.load((DATA / "structured_cases.json").open(encoding="utf-8"))
    profiles = json.load((DATA / "augmented_disease_data.json").open(encoding="utf-8"))
    catalog_names = {}
    for zid, prof in profiles.items():
        nm = (prof.get("DiseaseInfo") or {}).get("Disease Name")
        if nm:
            catalog_names[str(zid)] = str(nm)

    caper = Captioner(enabled=not args.no_caption)

    records = []
    needed = set()
    for i, row in df.iterrows():
        gi = int(sample[i])
        order = orders[gi]
        for d in order[:K_RETRIEVE]:
            needed.add(ret.diseases[int(d)])
        records.append({
            "case_idx": gi,
            "case_id": str(row.CaseID),
            "row": row,
            "gold_id": str(ret.gold_ids[gi]),
            "gold_name": str(row.DiseaseName),
            "gold_idx": int(ret.gold_idx[gi]),
            "retriever_order": order,
            "scores": S[gi],
            "structured": structured,
            "captioner": caper,
            "name_by_idx": {int(j): catalog_names.get(ret.diseases[int(j)],
                                                      ret.diseases[int(j)])
                            for j in range(len(ret.diseases))},
        })

    n_img = int(df.ImageCount.sum())
    log(f"[4/6] captions for {len(df)} sampled cases ({n_img} images, "
        f"max {MAX_IMAGES_PER_CASE}/case)")
    if not args.no_images:
        for k, rec in enumerate(records, 1):
            image_findings(rec["row"], structured, caper)
            if k % 25 == 0 or k == len(records):
                log(f"      caption pass {k}/{len(records)} "
                    f"({caper.new} new captions, {len(caper.cache)} cached)")
        caper.flush()
        log(f"      captioned new images: {caper.new}")
    else:
        caper.enabled = False

    log("[5/6] loading evidence bases")
    evb = EvidenceBase(needed, profiles)
    log(f"      OptimusKG links: {len(evb.kg.map)}/{len(needed)} candidate diseases "
        f"({(len(evb.kg.map)/max(1,len(needed))):.1%}); "
        f"phenotype edges for {len(evb.kg.pheno)}, gene edges for {len(evb.kg.genes)}")
    llm = LLM()
    log(f"      LLM model: {llm.model}  base={LLM_BASE}")

    log(f"[6/6] running {len(records)} cases with {args.workers} workers")
    results, done, failed = [], 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process_case, r, llm, evb, ret, args.force): r for r in records}
        for fut in as_completed(futs):
            rec = futs[fut]
            try:
                out = fut.result()
                results.append(out)
            except Exception as exc:
                import traceback
                failed += 1
                log(f"FAILED {rec['case_id']}: {exc}")
                traceback.print_exc()
            done += 1
            if done % 5 == 0 or done == len(records):
                elapsed = now() - t0
                eta = (len(records) - done) * elapsed / max(1, done) / 60
                log(f"PROGRESS {done}/{len(records)} | ok={len(results)} failed={failed} "
                    f"| {elapsed/60:.1f} min elapsed | ETA {eta:.1f} min "
                    f"| LLM calls={llm.calls} tokens={llm.prompt_tokens:,}/"
                    f"{llm.completion_tokens:,}")

    # order results by sample position
    order_key = {str(r["case_id"]): i for i, r in enumerate(records)}
    results.sort(key=lambda r: order_key.get(r["case_id"], 1e9))

    log("[final] evaluating and writing reports")
    metrics = evaluate(results)
    metrics["full_test_retrieval"] = full_metrics
    if not results:
        log("No cases completed; check the LLM server.")
        write_json(OUT / "summary.json", {"error": "no completed cases",
                                          "llm": {"failures": llm.failures}})
        return
    metrics["config"] = {
        "pipeline_version": PIPELINE_VERSION,
        "seed": SEED,
        "n_cases_requested": args.n_cases,
        "n_cases_completed": len(results),
        "k_retrieve": K_RETRIEVE,
        "k_explain": K_EXPLAIN,
        "caption_model": CAPTION_MODEL,
        "llm_model": llm.model,
        "llm_base": LLM_BASE,
        "sample_indices": [int(x) for x in sample],
    }
    metrics["llm"] = {
        "calls": llm.calls,
        "prompt_tokens": llm.prompt_tokens,
        "completion_tokens": llm.completion_tokens,
        "failures": llm.failures,
    }
    wall = now() - t0
    metrics["wall_minutes"] = round(wall / 60, 2)

    write_json(OUT / "summary.json", metrics)

    # JSONL, one row per case, compact
    with (OUT / "cases.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            slim = {
                "case_id": r["case_id"],
                "gold": r["gold_disease_name"],
                "retriever": r["retriever"]["metrics"],
                "final": r["final_metrics"],
                "llm_only": (r.get("llm_only_baseline") or {}).get("metrics"),
                "final5": [c["name"] for c in r["final_order"][:5]],
                "claims": [
                    {"disease": c["disease"], "kind": c["kind"], "claim": c["claim"],
                     "cites": c["cites"], "validator_ok": c["validator_ok"],
                     "verdict": c["adjudication"]["verdict"],
                     "reason": c["adjudication"]["reason"]}
                    for c in r["claims"]
                ],
                "evidence_counts": {
                    "z": sum(len(e["z"]) for e in r["evidence"].values()),
                    "k": sum(len(e["k"]) for e in r["evidence"].values()),
                },
            }
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    write_report(results, metrics, llm, wall)
    log("SUMMARY " + json.dumps(
        {k: metrics[k] for k in ("retrieval", "llm_only", "complete_system", "delta")},
        ensure_ascii=False))
    log(f"Wrote {OUT/'report.md'}, {OUT/'cases_report.md'}, "
        f"{OUT/'summary.json'}, {OUT/'cases.jsonl'}")
    log(f"DONE wall {wall/60:.1f} min")


if __name__ == "__main__":
    main()
