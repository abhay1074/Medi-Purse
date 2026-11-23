"""Realist router: decides whether a case should start with common causes or rare diseases.

A small logistic model combines the rare-disease retriever's own confidence with the
text signals the realist already extracts. It outputs p_common, the probability that the
case is an everyday presentation; two thresholds turn it into the verdict:

    p_common >= t_high          -> common_first  (common conditions shown first)
    t_low < p_common < t_high   -> both          (rare list first, common causes after)
    p_common <= t_low           -> rare_ok

Weights and thresholds are fitted by scripts/fit_router.py on evaluation/tuning_cases.json
plus real ZebraMap rare-disease case reports, and stored in app/data/router.json. The
evaluation cases (evaluation/synthetic_*.json) are never used for fitting.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

MODEL_FILE = Path(__file__).parent / "data" / "router.json"

# specialised investigations: their presence suggests a work-up beyond everyday care
TESTS_RX = re.compile(
    r"biops|antibod|\banti-[a-z0-9]|mutation|variant|genetic|sequenc|karyotyp|electrophores|"
    r"\bmri\b|\bct\b|angiogra|electromyogra|\bemg\b|\beeg\b|echocardiogra|fundus|fundoscop|slit[- ]lamp|"
    r"histolog|patholog|immunohisto|flow cytometr|marrow|enzyme activity|\bcsf\b|culture|"
    r"electrocardiogra|\becg\b|ultrason|x-ray|radiograph|scan\b|level[s]? (?:was|were|of)|serum",
    re.I)

FEATURES = ["top1_score", "top1_margin", "n_specific", "rare_cue", "n_everyday", "everyday_frac",
            "common_fit", "n_tests", "log_words", "infant"]


def features(text: str, signals: dict, ranking: dict | None, common_fit: float) -> dict:
    cands = (ranking or {}).get("candidates") or []
    top1 = cands[0]["score"] if cands else 0.9
    top2 = cands[1]["score"] if len(cands) > 1 else top1
    n_spec = len(signals["specific"])
    n_every = len(signals["keys"])
    return {
        "top1_score": float(top1),
        "top1_margin": float(top1 - top2),
        "n_specific": math.log1p(n_spec),
        "rare_cue": 1.0 if signals["cues"] else 0.0,
        "n_everyday": math.log1p(n_every),
        "everyday_frac": n_every / (n_every + n_spec + 1),
        "common_fit": float(common_fit),
        "n_tests": math.log1p(len(TESTS_RX.findall(text))),
        "log_words": math.log1p(signals["words"]),
        "infant": 1.0 if signals["infant"] else 0.0,
    }


_model: dict | None = None


def model() -> dict | None:
    global _model
    if _model is None and MODEL_FILE.exists():
        _model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    return _model


def p_common(feat: dict, m: dict | None = None) -> tuple[float, list[tuple[str, float]]]:
    """Probability the case is an everyday one, and each feature's contribution (log-odds)."""
    m = m or model()
    z = m["intercept"]
    contrib = []
    for name, w, mu, sd in zip(m["features"], m["coef"], m["mean"], m["std"]):
        c = w * (feat[name] - mu) / (sd or 1.0)
        contrib.append((name, c))
        z += c
    return 1 / (1 + math.exp(-z)), sorted(contrib, key=lambda x: -abs(x[1]))


def verdict(p: float, m: dict | None = None) -> str:
    m = m or model()
    if p >= m["t_high"]:
        return "common_first"
    if p <= m["t_low"]:
        return "rare_ok"
    return "both"


EXPLAIN = {
    "top1_score": ("the rare-disease ranking is unsure about its top match", "the rare-disease ranking is confident about its top match"),
    "top1_margin": ("no rare disease clearly stands out", "one rare disease clearly stands out"),
    "n_specific": ("there are few specific findings", "there are specific findings"),
    "rare_cue": ("there are no rare-disease cues", "there are rare-disease cues"),
    "n_everyday": ("everyday symptoms dominate", "few everyday symptoms"),
    "everyday_frac": ("most findings are everyday symptoms", "most findings are not everyday symptoms"),
    "common_fit": ("the symptoms fit a common condition well", "the symptoms do not fit a common condition well"),
    "n_tests": ("no specialised tests are mentioned", "specialised tests are mentioned"),
    "log_words": ("the summary is short", "the summary is detailed"),
    "infant": ("", "the patient is an infant"),
}


def explain(contrib: list[tuple[str, float]], n: int = 2) -> str:
    """Plain-language reasons from the largest contributions (positive = towards common)."""
    out = []
    for name, c in contrib[:n]:
        towards_common, towards_rare = EXPLAIN.get(name, ("", ""))
        phrase = towards_common if c > 0 else towards_rare
        if phrase and phrase not in out:
            out.append(phrase)
    return " and ".join(out)
