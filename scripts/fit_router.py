"""Fit the realist router (app/router.py) and save app/data/router.json.

Fitting data (never the evaluation sets):
  - evaluation/tuning_cases.json: 36 synthetic rare cases (diseases not in the evaluation set)
    and 51 synthetic everyday cases (one per common condition);
  - real ZebraMap rare-disease case reports (data/zebramap/ZebraMap.json), sampled by source
    article with seed 7. Articles used here are excluded from the hold-out sample in
    evaluation/run_eval.py --real.

    python scripts/fit_router.py --real 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TUNE = ROOT / "evaluation" / "tuning_cases.json"
ZEBRA = ROOT / "data" / "zebramap" / "ZebraMap.json"
CACHE = ROOT / "evaluation" / "router_tuning_features.json"


def real_cases(n: int, seed: int, exclude_articles: set[str] = frozenset()) -> list[dict]:
    """n real case reports, at most one per source article, gold in the 1,464 catalog."""
    z = json.loads(ZEBRA.read_text(encoding="utf-8"))
    pool = []
    for did, v in z.items():
        for cid, c in v["Cases"].items():
            art = c.get("SourceArticle") or cid.rsplit("_", 1)[0]
            if art in exclude_articles or len((c.get("Text") or "").split()) < 20:
                continue
            pool.append({"id": cid, "gold_id": did, "article": art, "text": c["Text"]})
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    out, seen = [], set()
    for c in pool:
        if c["article"] in seen:
            continue
        seen.add(c["article"])
        out.append(c)
        if len(out) >= n:
            break
    return out


def case_row(AI, realist, case: dict, kind: str) -> dict:
    from app import router
    ranking = AI.rank(case["text"], k=10)
    s = realist._signals(case["text"])
    feat = router.features(case["text"], s, ranking, realist.common_fit(s["keys"]))
    common = realist._score_common(s["keys"])
    row = {"id": case["id"], "kind": kind, "feat": feat, "n_common": len(common)}
    if kind == "common":
        row["common_rank"] = next((i for i, c in enumerate(common, 1) if c["name"] == case["gold"]), None)
    else:
        row["rare_rank"] = next((i for i, c in enumerate(ranking["candidates"], 1) if c["disease_id"] == case["gold_id"]), None)
    return row


def e2e(row, v: str) -> float:
    """Reciprocal rank in the shown list (top 10) for a given verdict."""
    common_first = v == "common_first" and row["n_common"] > 0
    if row["kind"] == "common":
        r = row.get("common_rank")
        r = None if r is None else (r if common_first else r + 10)
    else:
        r = row.get("rare_rank")
        r = None if r is None else (r + (min(row["n_common"], 5) if common_first else 0))
    return 1 / r if r and r <= 10 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=200)
    ap.add_argument("--reuse", action="store_true", help="reuse cached features")
    a = ap.parse_args()
    from app import realist
    from app.router import FEATURES, MODEL_FILE

    if a.reuse and CACHE.exists():
        rows = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        from app.ai import AI
        from app.symptoms import INDEX
        AI._load(); INDEX.load()
        t = json.loads(TUNE.read_text(encoding="utf-8"))
        real = real_cases(a.real, seed=7)
        t0 = time.time()
        rows = [case_row(AI, realist, c, "rare") for c in t["rare"]]
        rows += [case_row(AI, realist, c, "common") for c in t["common"]]
        rows += [dict(case_row(AI, realist, c, "rare"), real=True, article=c["article"]) for c in real]
        print(f"features for {len(rows)} cases in {time.time() - t0:.0f} s")
        CACHE.write_text(json.dumps(rows), encoding="utf-8")

    X = np.array([[r["feat"][f] for f in FEATURES] for r in rows])
    y = np.array([1 if r["kind"] == "common" else 0 for r in rows])
    # balance: everyday cases vs synthetic rare vs real rare count equally
    groups = ["common" if r["kind"] == "common" else ("real" if r.get("real") else "synth") for r in rows]
    gw = {g: len(rows) / (3 * groups.count(g)) for g in set(groups)}
    w = np.array([gw[g] for g in groups])

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    C = 0.5
    # cross-validated AUC on the tuning data (honest estimate of separability)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
        m = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict_proba(Z[te])[:, 1]
    cv_auc = roc_auc_score(y, oof, sample_weight=w)
    lr = LogisticRegression(C=C, max_iter=2000).fit(Z, y, sample_weight=w)
    p = lr.predict_proba(Z)[:, 1]

    # thresholds: maximise balanced end-to-end MRR@10 on the out-of-fold probabilities
    best = None
    for tl in np.arange(0.05, 0.61, 0.025):
        for th in np.arange(max(tl + 0.05, 0.3), 0.96, 0.025):
            def v(pp):
                return "common_first" if pp >= th else ("rare_ok" if pp <= tl else "both")
            score = {g: [] for g in ("common", "synth", "real")}
            for row, g, pp in zip(rows, groups, oof):
                score[g].append(e2e(row, v(pp)))
            obj = sum(np.mean(score[g]) for g in score) / 3
            if best is None or obj > best[0] + 1e-9:
                best = (obj, float(tl), float(th))
    obj, tl, th = best
    model = {"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
             "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
             "t_low": round(tl, 3), "t_high": round(th, 3), "C": C,
             "fitted_on": {"synthetic_rare": groups.count("synth"), "synthetic_common": groups.count("common"),
                           "real_rare_zebramap": groups.count("real")},
             "cv_auc": round(float(cv_auc), 4), "tuning_objective_balanced_mrr10": round(float(obj), 4),
             "tuning_articles": sorted({r["article"] for r in rows if r.get("real")}),
             "fitted": time.strftime("%Y-%m-%d %H:%M")}
    MODEL_FILE.write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(f"cv AUC {cv_auc:.3f}   thresholds low {tl:.3f} high {th:.3f}   balanced MRR@10 (oof) {obj:.3f}")
    for f, c in sorted(zip(FEATURES, lr.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {f:14s} {c:+.3f}")
    print("saved", MODEL_FILE)


if __name__ == "__main__":
    main()
