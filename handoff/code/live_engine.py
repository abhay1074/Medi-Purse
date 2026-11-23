"""Live single-case engine for the handoff bundle.

Uses the shipped retriever weights and the shipped catalog evidence. Returns the
same kind of report as the batch pipeline, but takes raw case text instead of a
benchmark row.

    from live_engine import LiveEngine
    engine = LiveEngine("..", llm_base="http://127.0.0.1:8080/v1")
    report = engine.analyze(case_text, case_id="case-001")
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import glassbox_pipeline as g
from reference_retriever import Retriever


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


class LiveEngine:
    def __init__(self, handoff_dir: str | Path,
                 llm_base: str = "http://127.0.0.1:8080/v1"):
        root = Path(handoff_dir).resolve()
        self.root = root
        self.ret = Retriever(root / "weights" / "retriever")
        self.profiles = json.load((root / "weights" / "catalog" /
                                   "augmented_disease_data.json").open(encoding="utf-8"))
        index = json.load((root / "weights" / "catalog" /
                           "disease_index.json").open(encoding="utf-8"))["diseases"]
        self.name = {d["id"]: d["name"] for d in index}
        self.kg = json.load((root / "weights" / "catalog" /
                             "optimuskg_catalog.json").open(encoding="utf-8"))
        g.LOG_PATH = root / "live.log"
        self.llm = g.LLM(llm_base)

    # ------------------------------------------------------------------
    # evidence selection (same policy as the batch pipeline)
    # ------------------------------------------------------------------
    def _zebra(self, disease_id: str, query: set[str], limit: int = 6) -> list[dict]:
        prof = self.profiles.get(str(disease_id)) or {}
        info = prof.get("DiseaseInfo") or {}
        summary = prof.get("DiseaseSummary") or {}
        pool: list[tuple[str, str, float]] = []

        def add(field, text, bonus=0.0, allow_short=False):
            text = str(text or "").strip()
            if not text:
                return
            sents = g.split_sentences(text, min_len=4 if allow_short else 35)
            if allow_short and not sents:
                sents = [text[:240]]
            for s in sents:
                toks = g.content_terms(s)
                if not toks:
                    continue
                score = len(toks & query) / math.sqrt(len(toks)) + bonus
                pool.append((field, s, score))

        add("Definition", info.get("Disease Definition", ""), 1.0)
        add("Prevalence", info.get("Prevalence", ""), 0.3, True)
        add("Inheritance", info.get("Inheritance", ""), 0.3, True)
        add("Onset", info.get("Age of onset", ""), 0.2, True)
        for field in g.PROFILE_FIELDS:
            add(field, summary.get(field, ""))

        pool.sort(key=lambda x: -x[2])
        chosen, per = [], {}
        defs = [s for f, s, _ in pool if f == "Definition"]
        if defs:
            chosen.append(("Definition", defs[0]))
            per["Definition"] = 1
        for field, s, _ in pool:
            if field == "Definition" or per.get(field, 0) >= 2:
                continue
            chosen.append((field, s))
            per[field] = per.get(field, 0) + 1
            if len(chosen) >= limit:
                break
        return [{"field": f, "text": s} for f, s in chosen[:limit]]

    def _kg(self, disease_id: str, query: set[str], limit: int = 7) -> list[dict]:
        entry = self.kg.get(str(disease_id))
        if not entry:
            return []
        scored = []
        for p in entry.get("phenotypes", []):
            toks = g.content_terms(p.get("name", ""))
            scored.append((len(toks & query), 1 if p.get("frequency") else 0,
                           "has_phenotype",
                           p["name"] + (f" (frequency {p['frequency']})"
                                        if p.get("frequency") else "")))
        for gene in entry.get("genes", [])[:10]:
            scored.append((0, 0, "associated_gene", gene))
        for parent in entry.get("parents", [])[:4]:
            scored.append((0, 0, "is_a", parent))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [{"relation": r, "object": o} for _, _, r, o in scored[:limit]]

    # ------------------------------------------------------------------
    def analyze(self, text: str, case_id: str = "live",
                images: list[dict] | None = None,
                progress=None) -> dict:
        def step(name, pct):
            if progress:
                progress(name, pct)

        text = str(text)
        step("retrieved", 0.05)
        ranked = self.ret.rank(text, k=10)
        by_id = {t["disease_id"]: t for t in ranked["trace"]}
        cand = []
        for i, row in enumerate(ranked["top"]):
            cand.append({"label": chr(65 + i), "disease_idx": self.ret.index[row["disease_id"]],
                         "disease_id": row["disease_id"], "name": row["name"]})
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        audit = list(cand)
        rng.shuffle(audit)
        for pos, c in enumerate(audit):
            c["audit_label"] = chr(65 + pos)
        label_map = {c["audit_label"]: c for c in audit}

        age = re.search(r"(\d{1,3})[- ]year[- ]old", text, re.I)
        gender = re.search(r"\b(male|female|man|woman|boy|girl)\b", text, re.I)
        row = SimpleNamespace(CaseID=case_id, Age=age.group(1) if age else "unknown",
                              Gender=gender.group(1) if gender else "unknown")

        img_meta = []
        for i, im in enumerate((images or [])[:4], 1):
            img_meta.append({"image_id": f"I{i}", "subtype": im.get("subtype", "image"),
                             "caption": im.get("caption"), "modality": im.get("modality", ""),
                             "file": im.get("file", "")})

        # CALL 1
        plan, _ = self.llm.json_chat(
            [{"role": "system", "content": g.PLAN_SYSTEM},
             {"role": "user", "content": g.fmt_case_for_plan(row, text[:6000], img_meta, audit)}],
            max_tokens=1400, required=["findings"], tag="plan")
        findings = []
        if plan:
            for f in (plan.get("findings") or [])[:16]:
                t = str(f.get("text", "")).strip()
                if not t:
                    continue
                q = str(f.get("quote", "")).strip()
                findings.append({"id": f"C{len(findings)+1}", "text": t,
                                 "type": str(f.get("type", "sign")), "quote": q,
                                 "quote_verified": bool(q) and g.norm_text(q) in g.norm_text(text),
                                 "source": "text"})
        if not findings:
            for s in g.split_sentences(text)[:10]:
                findings.append({"id": f"C{len(findings)+1}", "text": s[:160],
                                 "type": "sign", "quote": s, "quote_verified": True,
                                 "source": "text"})
        for im in img_meta:
            txt = f"{im['subtype']} image"
            if im.get("caption"):
                txt += f"; automatic caption: {im['caption']}"
            findings.append({"id": f"C{len(findings)+1}", "text": txt, "type": "imaging",
                             "quote": im.get("caption") or "", "quote_verified": bool(im.get("caption")),
                             "source": "image"})

        checks = {}
        for ck in (plan or {}).get("checks", []) or []:
            nm = g.norm_text(str(ck.get("candidate", "")))
            if nm:
                checks[nm] = str(ck.get("check", ""))
        query = set()
        for f in findings:
            query |= g.content_terms(f["text"])
        query |= g.content_terms(" ".join(checks.values()))
        step("planned", 0.25)

        # deterministic evidence fetch
        evidence = {}
        for c in audit:
            lab = c["audit_label"]
            q = query | g.content_terms(checks.get(g.norm_text(c["name"]), ""))
            tr = by_id[c["disease_id"]]
            evidence[lab] = {
                "disease_id": c["disease_id"], "name": c["name"],
                "z": self._zebra(c["disease_id"], q, 6),
                "k": self._kg(c["disease_id"], q, 7),
                "kg_available": c["disease_id"] in self.kg,
                "trace": {
                    "tfidf_score": tr["tfidf_score"], "bow_score": tr["bow_score"],
                    "tfidf_rank": tr["tfidf_rank"], "bow_rank": tr["bow_rank"],
                    "in_tfidf_top100": tr["pooled"],
                    "matched_terms": [{"term": m["term"], "weight": m["weight"],
                                       "present_in_case": True}
                                      for m in tr["matched_terms"]],
                    "gbm": {"in_tfidf_pool": tr["pooled"],
                            "top_contributions": tr["gbm_contributions"]},
                },
            }
        step("evidence", 0.35)

        # LLM-only baseline
        lo, _ = self.llm.json_chat(
            [{"role": "system", "content": g.LLM_ONLY_SYSTEM},
             {"role": "user", "content": g.fmt_llm_only_input(row, findings, audit)}],
            max_tokens=900, required=["order"], tag="llm_only")
        lo_order = []
        if lo:
            for lab in lo.get("order") or []:
                lab = str(lab).strip().upper()[:1]
                if lab in label_map and lab not in lo_order:
                    lo_order.append(lab)
        lo_ok = len(lo_order) == len(audit)
        if not lo_ok:
            lo_order = [c["audit_label"] for c in cand]
        llm_only = {"ok": lo_ok, "order": lo_order,
                    "top5": (lo or {}).get("top5", [])}

        # CALL 2 audit
        a, _ = self.llm.json_chat(
            [{"role": "system", "content": g.AUDIT_SYSTEM},
             {"role": "user", "content": g.fmt_audit_input(findings, audit, evidence)}],
            max_tokens=2600, required=["candidates"], tag="audit")
        audit_by = {}
        if a:
            for c in a.get("candidates", []) or []:
                lab = str(c.get("id", "")).strip().upper()[:1]
                if lab in label_map:
                    try:
                        p = float(c.get("plausibility"))
                    except Exception:
                        p = 50.0
                    audit_by[lab] = {"label": lab, "plausibility": max(0.0, min(100.0, p)),
                                     "verdict": str(c.get("verdict", "uncertain")),
                                     "supports": c.get("supports") or [],
                                     "contradictions": c.get("contradictions") or [],
                                     "missing": c.get("missing", "")}
        audit_ok = len(audit_by) == len(audit)
        pos = {row["disease_id"]: i + 1 for i, row in enumerate(ranked["top"])}
        if not audit_ok:
            for c in cand:
                audit_by.setdefault(c["audit_label"], {
                    "label": c["audit_label"],
                    "plausibility": float(max(1, 101 - 10 * pos[c["disease_id"]])),
                    "verdict": "uncertain", "supports": [], "contradictions": [],
                    "missing": "", "fallback": True})
        step("audited", 0.65)

        final = sorted(cand, key=lambda c: (-audit_by[c["audit_label"]]["plausibility"],
                                            pos[c["disease_id"]]))
        for c in cand:
            c["retriever_rank"] = pos[c["disease_id"]]
            c["final_rank"] = final.index(c) + 1
            c["plausibility"] = audit_by[c["audit_label"]]["plausibility"]

        # CALL 3 writer
        w, _ = self.llm.json_chat(
            [{"role": "system", "content": g.WRITE_SYSTEM},
             {"role": "user", "content": g.fmt_write_input(final[:5], findings, evidence)}],
            max_tokens=2200, required=["claims"], tag="write")
        claims = []
        if w:
            for cl in (w.get("claims") or [])[:30]:
                lab = str(cl.get("candidate", "")).strip().upper()[:1]
                if lab in label_map:
                    claims.append({"candidate": lab, "disease": label_map[lab]["name"],
                                   "kind": str(cl.get("kind", "support")),
                                   "claim": str(cl.get("claim", "")).strip(),
                                   "cites": [str(x) for x in (cl.get("cites") or [])]})
        case_norm = " " + _norm(text) + " "
        claims, valid = g.validate_claims(claims, findings, evidence, case_norm)
        step("written", 0.8)

        # CALL 4 adjudicator
        adj_input = []
        for i, cl in enumerate(claims):
            ev = evidence[cl["candidate"]]
            ctx = []
            for cid in cl["cites"]:
                txt = g.evidence_text(cid, findings, ev)
                if txt:
                    ctx.append(f"{cid}: {txt}")
            adj_input.append({"i": i, "claim": cl["claim"], "kind": cl["kind"],
                              "cited_text": ctx})
        verdicts = {}
        if claims:
            adj, _ = self.llm.json_chat(
                [{"role": "system", "content": g.ADJUDICATE_SYSTEM},
                 {"role": "user", "content":
                  "Judge each claim against its cited texts.\n"
                  + json.dumps(adj_input, ensure_ascii=False) + "\nReturn the JSON object."}],
                max_tokens=1600, required=["judgements"], tag="adjudicate")
            if adj:
                for j in adj.get("judgements", []) or []:
                    try:
                        i = int(j.get("i"))
                    except Exception:
                        continue
                    verdicts[i] = {"verdict": str(j.get("verdict", "keep")).lower(),
                                   "reason": str(j.get("reason", "")),
                                   "revised": str(j.get("revised", ""))}
        for i, cl in enumerate(claims):
            cl["adjudication"] = verdicts.get(i, {"verdict": "keep",
                                                  "reason": "no adjudication",
                                                  "revised": ""})
        step("adjudicated", 1.0)

        return {
            "case_id": case_id,
            "retrieval": ranked,
            "findings": findings,
            "images": img_meta,
            "llm_only": llm_only,
            "audit": {"ok": audit_ok,
                      "candidates": [audit_by[c["audit_label"]] for c in audit]},
            "final_order": [
                {"rank": c["final_rank"], "disease_id": c["disease_id"],
                 "name": c["name"], "plausibility": c["plausibility"],
                 "retriever_rank": c["retriever_rank"]}
                for c in final
            ],
            "claims": claims,
            "claim_validation": valid,
        }
