"""AI diagnosis service built on Aarav's handoff bundle (handoff/).

Two levels:
  rank(text)     - local fusion v4 SHAP-8 retriever + catalog evidence. Free, no LLM.
  explain(text)  - the four-call evidence-grounded pipeline (plan, audit, write,
                   adjudicate) via DeepSeek's OpenAI-compatible API.

The retriever's order is always the primary ranking (the bundle's evaluation found
LLM reordering hurts correct top-1 cases); LLM output is shown as supporting
evidence and a plausibility score only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HANDOFF = ROOT / "handoff"
sys.path.insert(0, str(HANDOFF / "code"))

import glassbox_pipeline as g  # noqa: E402
from live_engine import LiveEngine  # noqa: E402
from reference_retriever import Retriever  # noqa: E402

g.LOG_PATH = ROOT / "ai.log"


def orphanet_url(disease_id: str) -> str:
    return f"https://www.orpha.net/en/disease/detail/{disease_id}"


# ----------------------------------------------------------------------------
# DeepSeek client (OpenAI-compatible, with API key)
# ----------------------------------------------------------------------------
class DeepSeekLLM(g.LLM):
    def __init__(self, base: str, api_key: str, model: str):
        self.api_key = api_key
        self._model_name = model
        super().__init__(base)

    def _discover_model(self) -> str:
        return self._model_name

    def chat(self, messages, max_tokens, tag="", json_mode=True, retries=3):
        import requests
        payload = {"model": self.model, "messages": messages, "temperature": 0.0,
                   "max_tokens": int(max_tokens), "stream": False}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = requests.post(self.base + "/chat/completions", json=payload,
                                  headers=headers, timeout=180)
                if r.status_code in (401, 402, 403):
                    # bad key / no balance: retrying will not help
                    raise PermissionError({
                        401: "DeepSeek rejected the API key. Check DEEPSEEK_API_KEY in .env.",
                        402: "The DeepSeek account has no credit left. Top up at platform.deepseek.com, then try again.",
                    }.get(r.status_code, f"DeepSeek refused the request (HTTP {r.status_code})."))
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                usage = data.get("usage", {})
                with self.lock:
                    self.calls += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens", 0))
                    self.completion_tokens += int(usage.get("completion_tokens", 0))
                    self.seconds += time.time() - t0
                g.log(f"LLM {tag:<12} {time.time()-t0:6.1f}s in={usage.get('prompt_tokens')} "
                      f"out={usage.get('completion_tokens')}")
                return data["choices"][0]["message"].get("content") or "", {"usage": usage}
            except PermissionError:
                raise
            except Exception as exc:
                last = str(exc)
                with self.lock:
                    self.failures += 1
                time.sleep(min(2 ** attempt, 10))
        return None, {"error": last, "tag": tag}


# ----------------------------------------------------------------------------
class _Engine(LiveEngine):
    """LiveEngine without its built-in local LLM client (we attach DeepSeek)."""

    def __init__(self, handoff_dir: Path):
        root = Path(handoff_dir).resolve()
        self.root = root
        self.ret = Retriever(root / "weights" / "retriever")
        cat = root / "weights" / "catalog"
        self.profiles = json.load((cat / "augmented_disease_data.json").open(encoding="utf-8"))
        index = json.load((cat / "disease_index.json").open(encoding="utf-8"))["diseases"]
        self.name = {d["id"]: d["name"] for d in index}
        self.kg = json.load((cat / "optimuskg_catalog.json").open(encoding="utf-8"))
        self.llm = None


class AIService:
    def __init__(self):
        self.engine: _Engine | None = None
        self.status = "not loaded"
        self.error = ""
        self._lock = threading.Lock()

    # --- loading -----------------------------------------------------------
    def load_async(self) -> None:
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        with self._lock:
            if self.engine is not None:
                return
            self.status = "loading"
            t0 = time.time()
            try:
                self.engine = _Engine(HANDOFF)
                self.status = "ready"
                g.log(f"AI model loaded in {time.time()-t0:.1f}s")
            except Exception as exc:  # pragma: no cover
                self.status = "error"
                self.error = str(exc)
                g.log(f"AI model failed to load: {exc}")

    @property
    def ready(self) -> bool:
        return self.engine is not None

    @property
    def llm_configured(self) -> bool:
        return bool(os.environ.get("DEEPSEEK_API_KEY"))

    _avail_cache: tuple[float, bool, str] = (0.0, False, "")

    def llm_available(self, force: bool = False) -> bool:
        """Configured AND able to answer. For DeepSeek, the free /user/balance endpoint says
        whether the account can be used; the answer is cached for 10 minutes."""
        if not self.llm_configured:
            AIService._avail_cache = (time.time(), False, "No API key configured.")
            return False
        t, ok, _ = AIService._avail_cache
        if not force and time.time() - t < 600:
            return ok
        base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        ok, why = True, ""
        if "deepseek.com" in base:
            try:
                import requests
                r = requests.get(base.split("/v1")[0] + "/user/balance", timeout=10,
                                 headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
                if r.status_code == 200:
                    ok = bool(r.json().get("is_available"))
                    why = "" if ok else "The DeepSeek account has no credit."
                elif r.status_code in (401, 403):
                    ok, why = False, "DeepSeek rejected the API key."
            except Exception as exc:  # network trouble: assume usable, calls fall back on failure
                why = f"Balance check failed: {exc}"
        AIService._avail_cache = (time.time(), ok, why)
        return ok

    @property
    def llm_unavailable_reason(self) -> str:
        return AIService._avail_cache[2]

    def mark_llm_failed(self, why: str) -> None:
        AIService._avail_cache = (time.time(), False, why)

    def llm(self) -> "DeepSeekLLM":
        return self._llm()

    def _llm(self) -> DeepSeekLLM:
        return DeepSeekLLM(os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                           os.environ["DEEPSEEK_API_KEY"],
                           os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))

    # --- level 1: free local ranking --------------------------------------
    def rank(self, text: str, k: int = 10) -> dict:
        e = self.engine
        assert e is not None, "AI model is still loading"
        ranked = e.ret.rank(text, k=k)
        by_id = {t["disease_id"]: t for t in ranked["trace"]}
        query = g.content_terms(text)
        out = []
        for row in ranked["top"]:
            did = row["disease_id"]
            tr = by_id[did]
            prof = (e.profiles.get(did) or {}).get("DiseaseInfo") or {}
            if isinstance(prof, str):
                prof = {}
            out.append({
                "rank": row["rank"], "disease_id": did, "name": row["name"],
                "score": row["score"],
                "matched_terms": [m["term"] for m in tr["matched_terms"]
                                  if g.content_terms(m["term"])],
                "drivers": tr["gbm_contributions"],
                "tfidf_rank": tr["tfidf_rank"], "bow_rank": tr["bow_rank"],
                "evidence": e._zebra(did, query, 4),
                "kg": e._kg(did, query, 6),
                "inheritance": str(prof.get("Inheritance", "") or ""),
                "onset": str(prof.get("Age of onset", "") or ""),
                "prevalence": str(prof.get("Prevalence", "") or ""),
                "url": orphanet_url(did),
            })
        return {"candidates": out, "catalog_size": ranked["catalog_size"]}

    # --- level 2: DeepSeek explanation (4 LLM calls) -----------------------
    def explain(self, text: str, case_id: str = "case", progress=None,
                images: list[dict] | None = None) -> dict:
        """Adapted from handoff/code/live_engine.py:analyze.

        Differences: skips the LLM-only baseline call, returns the evidence and
        label map so citations can be displayed, keeps retriever order primary.
        """
        e = self.engine
        assert e is not None, "AI model is still loading"
        llm = self._llm()

        def step(name):
            if progress:
                progress(name)

        ranked = e.ret.rank(text, k=10)
        by_id = {t["disease_id"]: t for t in ranked["trace"]}
        cand = [{"label": chr(65 + i), "disease_idx": e.ret.index[r["disease_id"]],
                 "disease_id": r["disease_id"], "name": r["name"], "retriever_rank": r["rank"]}
                for i, r in enumerate(ranked["top"])]
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        audit = list(cand)
        np.random.default_rng(seed).shuffle(audit)
        for pos, c in enumerate(audit):
            c["audit_label"] = chr(65 + pos)
        label_map = {c["audit_label"]: c for c in audit}

        age = re.search(r"(\d{1,3})[- ]year[- ]old", text, re.I)
        gender = re.search(r"\b(male|female|man|woman|boy|girl)\b", text, re.I)
        row = SimpleNamespace(CaseID=case_id, Age=age.group(1) if age else "unknown",
                              Gender=gender.group(1) if gender else "unknown")

        # images (app/images.py:pipeline_meta), at most 4 as in the handoff pipeline
        from .images import image_findings
        img_meta = list(images or [])[:4]

        # CALL 1: plan
        step("Extracting findings (1/4)")
        plan, _ = llm.json_chat(
            [{"role": "system", "content": g.PLAN_SYSTEM},
             {"role": "user", "content": g.fmt_case_for_plan(row, text[:6000], img_meta, audit)}],
            max_tokens=1400, required=["findings"], tag="plan")
        findings = []
        for f in ((plan or {}).get("findings") or [])[:16]:
            t = str(f.get("text", "")).strip()
            if not t:
                continue
            q = str(f.get("quote", "")).strip()
            findings.append({"id": f"C{len(findings)+1}", "text": t,
                             "type": str(f.get("type", "sign")), "quote": q,
                             "quote_verified": bool(q) and g.norm_text(q) in g.norm_text(text)})
        if not findings:
            for s in g.split_sentences(text, min_len=10)[:10]:
                findings.append({"id": f"C{len(findings)+1}", "text": s[:160],
                                 "type": "sign", "quote": s, "quote_verified": True})
        for f in findings:
            f.setdefault("source", "text")
        findings += image_findings(img_meta, len(findings))
        checks = {}
        for ck in (plan or {}).get("checks", []) or []:
            nm = g.norm_text(str(ck.get("candidate", "")))
            if nm:
                checks[nm] = str(ck.get("check", ""))
        query = set()
        for f in findings:
            query |= g.content_terms(f["text"])
        query |= g.content_terms(" ".join(checks.values()))

        # deterministic evidence fetch
        evidence = {}
        for c in audit:
            lab = c["audit_label"]
            q = query | g.content_terms(checks.get(g.norm_text(c["name"]), ""))
            tr = by_id[c["disease_id"]]
            evidence[lab] = {
                "disease_id": c["disease_id"], "name": c["name"],
                "z": e._zebra(c["disease_id"], q, 6), "k": e._kg(c["disease_id"], q, 7),
                "kg_available": c["disease_id"] in e.kg,
                "trace": {
                    "tfidf_score": tr["tfidf_score"], "bow_score": tr["bow_score"],
                    "tfidf_rank": tr["tfidf_rank"], "bow_rank": tr["bow_rank"],
                    "in_tfidf_top100": tr["pooled"],
                    "matched_terms": [{"term": m["term"], "weight": m["weight"],
                                       "present_in_case": True} for m in tr["matched_terms"]],
                    "gbm": {"in_tfidf_pool": tr["pooled"],
                            "top_contributions": tr["gbm_contributions"]},
                },
            }

        # CALL 2: audit (candidates anonymised and shuffled; ranks hidden)
        step("Auditing candidates against evidence (2/4)")
        a, _ = llm.json_chat(
            [{"role": "system", "content": g.AUDIT_SYSTEM},
             {"role": "user", "content": g.fmt_audit_input(findings, audit, evidence)}],
            max_tokens=2600, required=["candidates"], tag="audit")
        audit_by = {}
        for c in (a or {}).get("candidates", []) or []:
            lab = str(c.get("id", "")).strip().upper()[:1]
            if lab in label_map:
                try:
                    p = float(c.get("plausibility"))
                except Exception:
                    p = 50.0
                audit_by[lab] = {"plausibility": max(0.0, min(100.0, p)),
                                 "verdict": str(c.get("verdict", "uncertain")),
                                 "supports": c.get("supports") or [],
                                 "contradictions": c.get("contradictions") or [],
                                 "missing": str(c.get("missing", "") or "")}
        audit_ok = len(audit_by) == len(audit)

        # explain the retriever's top 5 (retriever order stays primary)
        top5 = sorted(cand, key=lambda c: c["retriever_rank"])[:5]

        # CALL 3: writer
        step("Writing cited explanations (3/4)")
        w, _ = llm.json_chat(
            [{"role": "system", "content": g.WRITE_SYSTEM},
             {"role": "user", "content": g.fmt_write_input(top5, findings, evidence)}],
            max_tokens=2200, required=["claims"], tag="write")
        claims = []
        for cl in ((w or {}).get("claims") or [])[:30]:
            lab = str(cl.get("candidate", "")).strip().upper()[:1]
            if lab in label_map:
                claims.append({"candidate": lab, "disease": label_map[lab]["name"],
                               "kind": str(cl.get("kind", "support")),
                               "claim": str(cl.get("claim", "")).strip(),
                               "cites": [str(x) for x in (cl.get("cites") or [])]})
        case_norm = " " + g.norm_text(text) + " "
        claims, valid = g.validate_claims(claims, findings, evidence, case_norm)

        # CALL 4: adjudicator
        step("Checking every citation (4/4)")
        adj_input = []
        for i, cl in enumerate(claims):
            ev = evidence[cl["candidate"]]
            ctx = [f"{cid}: {t}" for cid in cl["cites"]
                   if (t := g.evidence_text(cid, findings, ev))]
            adj_input.append({"i": i, "claim": cl["claim"], "kind": cl["kind"], "cited_text": ctx})
        verdicts = {}
        if claims:
            adj, _ = llm.json_chat(
                [{"role": "system", "content": g.ADJUDICATE_SYSTEM},
                 {"role": "user", "content": "Judge each claim against its cited texts.\n"
                  + json.dumps(adj_input, ensure_ascii=False) + "\nReturn the JSON object."}],
                max_tokens=1600, required=["judgements"], tag="adjudicate")
            for j in (adj or {}).get("judgements", []) or []:
                try:
                    verdicts[int(j.get("i"))] = {
                        "verdict": str(j.get("verdict", "keep")).lower(),
                        "reason": str(j.get("reason", "")),
                        "revised": str(j.get("revised", ""))}
                except Exception:
                    continue
        for i, cl in enumerate(claims):
            cl["adjudication"] = verdicts.get(i, {"verdict": "unchecked", "reason": "", "revised": ""})
            # resolve cited ids to text for display
            ev = evidence[cl["candidate"]]
            cl["cited"] = [{"id": cid, "text": g.evidence_text(cid, findings, ev) or "(unknown id)"}
                           for cid in cl["cites"]]
            # what a doctor should read: drop -> hidden, soften -> revised wording
            adj = cl["adjudication"]
            cl["show"] = cl["validator_ok"] and adj["verdict"] != "drop"
            cl["display"] = adj["revised"] if adj["verdict"] == "soften" and adj["revised"] else cl["claim"]

        per_disease = {}
        for c in cand:
            lab = c["audit_label"]
            au = audit_by.get(lab, {})
            per_disease[c["disease_id"]] = {
                "label": lab, "name": c["name"], "retriever_rank": c["retriever_rank"],
                "plausibility": au.get("plausibility"), "verdict": au.get("verdict"),
                "missing": au.get("missing", ""),
                "supports": [s for s in au.get("supports", []) if isinstance(s, dict)],
                "contradictions": [s for s in au.get("contradictions", []) if isinstance(s, dict)],
                "claims": [cl for cl in claims if cl["candidate"] == lab],
                "evidence": evidence[lab],
            }
        return {
            "model": llm.model,
            "findings": findings,
            "images": img_meta,
            "audit_ok": audit_ok,
            "diseases": per_disease,
            "order": [c["disease_id"] for c in sorted(cand, key=lambda c: c["retriever_rank"])],
            "claim_validation": valid,
            "usage": {"calls": llm.calls, "prompt_tokens": llm.prompt_tokens,
                      "completion_tokens": llm.completion_tokens, "seconds": round(llm.seconds, 1)},
        }


AI = AIService()
