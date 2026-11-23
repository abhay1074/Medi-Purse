"""End-to-end evaluation of MediPurse AI diagnosis on synthetic rare and common cases.

What is measured is what a doctor sees after "Find likely conditions":
  1. the fusion v4 SHAP-8 retriever ranks 1,464 rare diseases (top 10 shown);
  2. the realist agent (rule mode) decides common_first / both / rare_ok and lists
     up to 5 common conditions;
  3. the page shows common conditions first when the verdict is common_first,
     otherwise the rare list first and common conditions after it.

Usage (from the repo root, app venv):
    python evaluation/run_eval.py --reps 5 --http-reps 3 --http http://localhost:8001
Writes evaluation/results.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def rr(rank: int | None, k: int = 10) -> float:
    return 1.0 / rank if rank and rank <= k else 0.0


def ndcg(rank: int | None, k: int = 10) -> float:
    return 1.0 / math.log2(rank + 1) if rank and rank <= k else 0.0


def summarise_ranks(ranks: list[int | None], k: int = 10) -> dict:
    n = len(ranks)
    if not n:
        return {}
    out = {"n": n, f"mrr@{k}": sum(rr(r, k) for r in ranks) / n, f"ndcg@{k}": sum(ndcg(r, k) for r in ranks) / n}
    for c in (1, 3, 5, 10):
        if c <= k:
            out[f"recall@{c}"] = sum(1 for r in ranks if r and r <= c) / n
    found = sorted(r for r in ranks if r)
    out["median_rank_if_found"] = statistics.median(found) if found else None
    out["not_found"] = sum(1 for r in ranks if not r)
    out["mrr_ci95"] = bootstrap_ci([rr(r, k) for r in ranks])
    return out


def bootstrap_ci(values: list[float], b: int = 2000, seed: int = 42) -> list[float]:
    rnd = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rnd.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return [round(means[int(0.025 * b)], 4), round(means[int(0.975 * b) - 1], 4)]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def lat(xs: list[float]) -> dict:
    return {"n": len(xs), "mean_ms": round(statistics.mean(xs), 1), "p50_ms": round(pct(xs, 50), 1),
            "p90_ms": round(pct(xs, 90), 1), "p99_ms": round(pct(xs, 99), 1), "max_ms": round(max(xs), 1)}


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random positive > score of a random negative)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


# ----------------------------------------------------------------------------
# system under test
# ----------------------------------------------------------------------------
def load_system():
    t0 = time.perf_counter()
    from app.ai import AI
    from app import realist
    from app.symptoms import INDEX
    AI._load()
    INDEX.load()
    return AI, realist, (time.perf_counter() - t0) * 1000


def displayed(rare_top: list[dict], real: dict) -> list[tuple[str, str]]:
    """The order a doctor reads on the results page: (kind, label)."""
    rare = [("rare", c["disease_id"]) for c in rare_top[:10]]
    common = [("common", c["name"]) for c in real.get("common", [])]
    return common + rare if real["verdict"] == "common_first" and common else rare + common


def first_rank(items, ok) -> int | None:
    for i, it in enumerate(items, 1):
        if ok(it):
            return i
    return None


def run_case(AI, realist, case: dict, kind: str) -> dict:
    t0 = time.perf_counter()
    ranking = AI.rank(case["text"], k=10)
    t1 = time.perf_counter()
    real = realist.assess_rules(case["text"], ranking)
    t2 = time.perf_counter()
    rare = ranking["candidates"]
    disp = displayed(rare, real)
    if kind == "rare":
        gold_ok = lambda it: it == ("rare", case["gold_id"])  # noqa: E731
        rare_rank = first_rank(rare, lambda c: c["disease_id"] == case["gold_id"])
        common_rank = None
    else:
        names = {case["gold"], *case.get("also_ok", [])}
        gold_ok = lambda it: it[0] == "common" and it[1] == case["gold"]  # noqa: E731
        rare_rank = None
        common_rank = first_rank(real.get("common", []), lambda c: c["name"] == case["gold"])
        common_rank_lenient = first_rank(real.get("common", []), lambda c: c["name"] in names)
    lenient_ok = gold_ok if kind == "rare" else (lambda it: it[0] == "common" and it[1] in names)  # noqa: E731
    out = {
        "id": case["id"], "kind": kind, "difficulty": case["difficulty"], "gold": case["gold"],
        "verdict": real["verdict"], "rare_rank": rare_rank, "common_rank": common_rank,
        "e2e_rank": first_rank(disp, gold_ok), "e2e_rank_lenient": first_rank(disp, lenient_ok),
        "top1": rare[0]["name"] if rare else None, "top1_score": rare[0]["score"] if rare else None,
        "common_top": [c["name"] for c in real.get("common", [])][:5],
        "ms_retriever": (t1 - t0) * 1000, "ms_realist": (t2 - t1) * 1000, "ms_total": (t2 - t0) * 1000,
        "words": len(case["text"].split()),
    }
    if kind == "common":
        out["common_rank_lenient"] = common_rank_lenient
    if kind == "rare":   # deeper recall from the retriever alone
        r100 = AI.engine.ret.rank(case["text"], k=100)["top"]
        out["rare_rank_100"] = first_rank(r100, lambda c: c["disease_id"] == case["gold_id"])
    return out


# ----------------------------------------------------------------------------
# HTTP end-to-end latency through the running web app
# ----------------------------------------------------------------------------
def http_latency(base: str, cases: list[dict], reps: int) -> dict:
    import requests
    s = requests.Session()
    s.post(base + "/login", data={"email": "dr.mehta@medipurse.test", "password": "demo1234"})
    pid = 1
    times, errors = [], 0
    for _ in range(reps):
        for c in cases:
            t0 = time.perf_counter()
            r = s.post(f"{base}/patients/{pid}/ai/new", data={"case_text": c["text"], "symptoms": "[]"})
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code == 200 and "/ai/" in r.url:
                times.append(dt)
            else:
                errors += 1
    return {**lat(times), "errors": errors,
            "what": "POST /patients/1/ai/new -> redirect -> rendered results page (retriever + realist + DB write + HTML)"}


# ----------------------------------------------------------------------------
# real ZebraMap rare-disease case reports, held out from router fitting
# ----------------------------------------------------------------------------
def real_heldout(AI, realist, n: int) -> dict:
    from app.router import model
    sys.path.insert(0, str(ROOT / "scripts"))
    from fit_router import real_cases
    tuned = set((model() or {}).get("tuning_articles", []))
    cases = real_cases(n, seed=11, exclude_articles=tuned)
    rows = []
    for c in cases:
        ranking = AI.rank(c["text"], k=10)
        real = realist.assess_rules(c["text"], ranking)
        rare_rank = first_rank(ranking["candidates"], lambda x: x["disease_id"] == c["gold_id"])
        disp = displayed(ranking["candidates"], real)
        rows.append({"verdict": real["verdict"], "rare_rank": rare_rank,
                     "e2e_rank": first_rank(disp, lambda it: it == ("rare", c["gold_id"])),
                     "words": len(c["text"].split())})
    return {"n": len(rows), "note": "real case reports from ZebraMap; about 69% come from the shipped "
            "retriever's training split, so retriever-only scores are optimistic. The routing numbers "
            "are the point: real rare cases should not be sent common-first.",
            "retriever_only": summarise_ranks([r["rare_rank"] for r in rows]),
            "end_to_end": summarise_ranks([r["e2e_rank"] for r in rows]),
            "verdicts": {v: sum(1 for r in rows if r["verdict"] == v) for v in ("rare_ok", "both", "common_first")},
            "median_words": statistics.median(r["words"] for r in rows)}


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5, help="in-process latency repetitions per case")
    ap.add_argument("--http", default="", help="base URL of a running app for HTTP latency (use a test database)")
    ap.add_argument("--http-reps", type=int, default=3)
    ap.add_argument("--real", type=int, default=0,
                    help="also score N real ZebraMap case reports from articles the router was not fitted on")
    a = ap.parse_args()

    rare = json.load(open(HERE / "synthetic_rare.json", encoding="utf-8"))["cases"]
    common = json.load(open(HERE / "synthetic_common.json", encoding="utf-8"))["cases"]
    AI, realist, load_ms = load_system()

    # warm-up (first call pays for lazy imports and caches)
    for c in rare[:3] + common[:3]:
        AI.rank(c["text"], k=10)
        realist.assess_rules(c["text"])

    rows = [run_case(AI, realist, c, "rare") for c in rare] + [run_case(AI, realist, c, "common") for c in common]

    # repeated timing for stable percentiles
    t_ret, t_real, t_tot = [], [], []
    for _ in range(a.reps):
        for c in rare + common:
            t0 = time.perf_counter(); rk = AI.rank(c["text"], k=10); t1 = time.perf_counter()
            realist.assess_rules(c["text"], rk); t2 = time.perf_counter()
            t_ret.append((t1 - t0) * 1000); t_real.append((t2 - t1) * 1000); t_tot.append((t2 - t0) * 1000)

    R = [r for r in rows if r["kind"] == "rare"]
    C = [r for r in rows if r["kind"] == "common"]
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "cases": {"rare": len(R), "common": len(C)},
           "model_load_ms": round(load_ms, 0)}

    # ---- rare set
    res["rare"] = {
        "retriever_only": summarise_ranks([r["rare_rank"] for r in R]),
        "retriever_recall@100": sum(1 for r in R if r["rare_rank_100"]) / len(R),
        "end_to_end": summarise_ranks([r["e2e_rank"] for r in R]),
        "verdicts": {v: sum(1 for r in R if r["verdict"] == v) for v in ("rare_ok", "both", "common_first")},
        "by_difficulty": {d: summarise_ranks([r["rare_rank"] for r in R if r["difficulty"] == d])
                          for d in sorted({r["difficulty"] for r in R})},
    }
    # ---- common set
    res["common"] = {
        "realist_common_list_strict": summarise_ranks([r["common_rank"] for r in C], k=5),
        "realist_common_list_lenient": summarise_ranks([r["common_rank_lenient"] for r in C], k=5),
        "end_to_end": summarise_ranks([r["e2e_rank"] for r in C]),
        "end_to_end_lenient": summarise_ranks([r["e2e_rank_lenient"] for r in C]),
        "verdicts": {v: sum(1 for r in C if r["verdict"] == v) for v in ("rare_ok", "both", "common_first")},
        "by_difficulty": {d: summarise_ranks([r["common_rank"] for r in C if r["difficulty"] == d], k=5)
                          for d in sorted({r["difficulty"] for r in C})},
    }
    # ---- combined
    res["combined"] = {
        "end_to_end": summarise_ranks([r["e2e_rank"] for r in rows]),
        "end_to_end_lenient": summarise_ranks([r["e2e_rank_lenient"] for r in rows]),
    }
    # ---- routing: does the realist send each case down the right path?
    tp = sum(1 for r in C if r["verdict"] == "common_first")
    fp = sum(1 for r in R if r["verdict"] == "common_first")
    fn = len(C) - tp
    res["routing"] = {
        "definition": "positive = realist says 'start with common causes' (common_first)",
        "common_cases_sent_common_first": tp / len(C),
        "rare_cases_sent_common_first": fp / len(R),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / len(C),
        "f1": (2 * tp / (2 * tp + fp + fn)) if tp else 0.0,
        "accuracy": (tp + (len(R) - fp)) / len(rows),
        # common_first only reorders the page when there is a common list to put first
        "rare_gold_pushed_below_common_list": sum(1 for r in R if r["verdict"] == "common_first"
                                                  and r["rare_rank"] and r["common_top"]),
        "common_cases_listed_after_rare": sum(1 for r in C if r["verdict"] != "common_first" and r["common_rank"]),
    }
    # ---- can the retriever's own confidence tell rare from common?
    res["retriever_confidence"] = {
        "auroc_top1_score_rare_vs_common": auroc([r["top1_score"] for r in R], [r["top1_score"] for r in C]),
        "mean_top1_score_rare": statistics.mean(r["top1_score"] for r in R),
        "mean_top1_score_common": statistics.mean(r["top1_score"] for r in C),
    }
    # ---- latency
    res["latency_in_process"] = {"retriever": lat(t_ret), "realist_rules": lat(t_real), "total": lat(t_tot),
                                 "reps_per_case": a.reps}
    if a.http:
        res["latency_http"] = http_latency(a.http, rare + common, a.http_reps)
        res["latency_http"]["reps_per_case"] = a.http_reps
    if a.real:
        res["real_heldout"] = real_heldout(AI, realist, a.real)
    res["rows"] = rows
    out = HERE / "results.json"
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, default=float)

    def f(x):
        return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    print(f"model load {res['model_load_ms']:.0f} ms")
    for k in ("retriever_only", "end_to_end"):
        s = res["rare"][k]
        print(f"RARE {k:15s} MRR@10 {f(s['mrr@10'])} R@1 {f(s['recall@1'])} R@5 {f(s['recall@5'])} R@10 {f(s['recall@10'])}")
    print("RARE R@100", f(res["rare"]["retriever_recall@100"]), "verdicts", res["rare"]["verdicts"])
    s = res["common"]["realist_common_list_strict"]; l = res["common"]["realist_common_list_lenient"]
    print(f"COMMON list strict MRR@5 {f(s['mrr@5'])} R@1 {f(s['recall@1'])} R@5 {f(s['recall@5'])} | lenient R@5 {f(l['recall@5'])}")
    s = res["common"]["end_to_end"]
    print(f"COMMON e2e MRR@10 {f(s['mrr@10'])} R@10 {f(s['recall@10'])} verdicts {res['common']['verdicts']}")
    s = res["combined"]["end_to_end"]
    print(f"COMBINED e2e MRR@10 {f(s['mrr@10'])} R@10 {f(s['recall@10'])} CI {s['mrr_ci95']}")
    print("ROUTING", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["routing"].items() if k != "definition"})
    print("AUROC top1 score", round(res["retriever_confidence"]["auroc_top1_score_rare_vs_common"], 3))
    print("LATENCY in-process total", res["latency_in_process"]["total"])
    if a.http:
        print("LATENCY http", res["latency_http"])
    if a.real:
        rh = res["real_heldout"]
        print(f"REAL held-out {rh['n']}: retriever MRR@10 {rh['retriever_only']['mrr@10']:.3f} e2e {rh['end_to_end']['mrr@10']:.3f} verdicts {rh['verdicts']}")
    print("written", out)


if __name__ == "__main__":
    main()
