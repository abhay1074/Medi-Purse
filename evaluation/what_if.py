"""What-if: how the same model outputs would score under different page-ordering policies.

No model is re-run or tuned: this re-orders the stored rare top-10 and realist common list
from results.json. It isolates how much of the end-to-end loss comes from routing/ordering.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
res = json.load(open(HERE / "results.json", encoding="utf-8"))
rows = res["rows"]


def e2e_rank(r, common_first: bool):
    """Rank of the gold in the shown list under a given ordering decision."""
    n_common = len(r["common_top"])
    if r["kind"] == "rare":
        if not r["rare_rank"]:
            return None
        return r["rare_rank"] + (n_common if common_first and n_common else 0)
    if not r["common_rank"]:
        return None
    return r["common_rank"] + (0 if common_first else 10)


policies = {
    "current (common first only for common_first)": lambda r: r["verdict"] == "common_first",
    "common first for common_first and both": lambda r: r["verdict"] in ("common_first", "both"),
    "always rare first": lambda r: False,
    "oracle routing (upper bound)": lambda r: r["kind"] == "common",
}


def mrr(ranks, k=10):
    return sum((1 / x) if x and x <= k else 0 for x in ranks) / len(ranks)


def rec(ranks, k=10):
    return sum(1 for x in ranks if x and x <= k) / len(ranks)


out = {}
for name, pol in policies.items():
    ranks = {kind: [e2e_rank(r, pol(r)) for r in rows if r["kind"] == kind] for kind in ("rare", "common")}
    allr = ranks["rare"] + ranks["common"]
    out[name] = {"rare_mrr@10": round(mrr(ranks["rare"]), 3), "common_mrr@10": round(mrr(ranks["common"]), 3),
                 "combined_mrr@10": round(mrr(allr), 3), "combined_recall@10": round(rec(allr), 3)}
res["policy_what_if"] = out
json.dump(res, open(HERE / "results.json", "w", encoding="utf-8"), indent=1)
for k, v in out.items():
    print(f"{k:48s} {v}")
