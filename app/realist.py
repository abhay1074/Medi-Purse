"""Realist agent: a pragmatic generalist check on every case summary.

The rare-disease retriever always ranks rare diseases, even for "fever and sore throat
for two days". The realist reads the case summary and decides whether an upfront
rare-disease workup is reasonable. When it is not (very simple summaries, everyday
complaints, no rare-disease cues) it proposes the five most likely common conditions,
which the doctor sees first in the suggestions list; the rare list stays below.

Two modes, same output:
  rules  deterministic: phenotype detection (OptimusKG/HPO index), rare-disease cues,
         and symptom overlap with clinical.COMMON_CONDITIONS. Always available.
  ai     one LLM call (REALIST_SYSTEM) when an LLM is configured and has credit;
         falls back to the rules result on any failure.

Output:
  {"verdict": "common_first" | "both" | "rare_ok",
   "reason": str, "signals": [str], "source": "rules" | "ai",
   "common": [{"name", "why", "next", "matched": [labels]}]}   (up to 5)
"""

from __future__ import annotations

import json
import re

from .clinical import COMMON_BY_KEY, COMMON_CONDITIONS, RARE_CUES
from .symptoms import INDEX, common_keys_in

SPECIFIC_MAX_DISEASES = 60   # an HPO term linked to <= this many diseases is "specific"


def _signals(text: str) -> dict:
    t = text.lower()
    words = len(re.findall(r"[a-zA-Z]+", text))
    found = INDEX.find_in_text(text)
    keys = common_keys_in(text)
    common_ids = {COMMON_BY_KEY[k][4] for k in keys if COMMON_BY_KEY[k][4]}
    specific = [f for f in found if not f["common"] and f["id"] not in common_ids
                and f["id"].startswith("HP:") and 0 < f["diseases"] <= SPECIFIC_MAX_DISEASES]
    cues = []
    for needle, label in RARE_CUES:
        if needle in t and label not in cues:
            cues.append(label)
    age = re.search(r"(\d{1,3})[- ](?:year|yr)s?[- ]old", t)
    months = re.search(r"(\d{1,2})[- ]month[- ]old", t)
    infant = bool(months) or bool(re.search(r"\b(newborn|neonat|infant)\b", t))
    return {"words": words, "found": found, "keys": keys, "specific": specific, "cues": cues,
            "age": int(age.group(1)) if age else None, "infant": infant}


def _score_common(keys: set[str], limit: int = 5) -> list[dict]:
    scored = []
    for c in COMMON_CONDITIONS:
        ck = c["keys"]
        hits = [k for k in ck if k in keys]
        if not hits:
            continue
        # typical symptoms listed first weigh more; reward explaining more of the case
        weight = sum(1.0 / (1 + 0.15 * ck.index(k)) for k in hits)
        coverage = len(hits) / max(1, len(keys))
        scored.append(((weight + coverage) * c.get("prior", 1.0), c, hits))
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, c, hits in scored[:limit]:
        labels = [COMMON_BY_KEY[k][1].lower() for k in hits]
        out.append({"name": c["name"], "why": "Fits " + ", ".join(labels) + ".",
                    "next": c["next"], "matched": labels})
    return out


def assess_rules(text: str) -> dict:
    s = _signals(text)
    n_found = len({f["id"] for f in s["found"]} | {COMMON_BY_KEY[k][4] or k for k in s["keys"]})
    signals = []
    if s["cues"]:
        signals.append("Rare-disease cues: " + ", ".join(s["cues"]))
    if s["specific"]:
        signals.append("Specific findings: " + ", ".join(f["name"] for f in s["specific"][:5]))
    if s["keys"]:
        signals.append("Everyday symptoms: " + ", ".join(sorted(COMMON_BY_KEY[k][1].lower() for k in s["keys"]))[:200])
    if s["infant"]:
        signals.append("Infant or newborn")

    strong_rare = bool(s["cues"]) or len(s["specific"]) >= 2 or (s["infant"] and n_found >= 2)
    systems = {COMMON_BY_KEY[k][3] for k in s["keys"]}
    multisystem = len(systems) >= 3 and n_found >= 5
    if multisystem:
        signals.append("Findings span " + str(len(systems)) + " body systems")
    simple = n_found <= 4 and s["words"] < 80
    if strong_rare:
        verdict = "rare_ok"
        reason = ("The summary has features that justify considering rare diseases early ("
                  + "; ".join(s["cues"][:3] + [f["name"] for f in s["specific"][:3]]) + ").")
    elif not s["specific"] and not multisystem and (simple or n_found <= 3):
        verdict = "common_first"
        reason = ("This is a short summary of everyday symptoms with no features that point to a rare "
                  "disease. Common causes should be considered first; the rare-disease list is shown "
                  "below for reference only.")
    else:
        verdict = "both"
        reason = ("Several findings across body systems" if multisystem else "Some findings are more specific") + (
            ", but there is no strong rare-disease cue. Rule out common causes alongside the "
            "rare-disease suggestions.")
    common = _score_common(s["keys"]) if verdict != "rare_ok" or s["keys"] else []
    if verdict == "rare_ok":
        common = common[:3]
    return {"verdict": verdict, "reason": reason, "signals": signals, "source": "rules", "common": common}


# ----------------------------------------------------------------------------
# LLM mode
# ----------------------------------------------------------------------------
REALIST_SYSTEM = """You are the realist on a diagnostic team: an experienced general physician.
A rare-disease search engine has already proposed rare diseases for the case below. Your job is
to stop the team from jumping to a rare diagnosis when common conditions are far more likely.

Decide:
- "common_first": the summary is very simple or describes an everyday presentation, and it is
  not advisable to diagnose a rare disease upfront.
- "both": common causes should be ruled out alongside the rare candidates.
- "rare_ok": the case has features that justify an early rare-disease workup (consanguinity,
  developmental delay or regression, congenital or multisystem findings, very specific signs,
  affected relatives, refractory course, unusual age of onset).

Then list the 5 most likely COMMON conditions for this presentation (fewer only if the verdict
is rare_ok), most likely first, taking age, sex and setting into account. For each give one
short reason grounded in the case text and the first sensible next step. Do not list rare
diseases. Do not invent findings that are not in the case.

Return ONLY JSON:
{"verdict": "common_first|both|rare_ok", "reason": "one or two sentences",
 "common": [{"name": "...", "why": "...", "next": "..."}]}"""


def assess_llm(llm, text: str, rare_names: list[str], rules: dict) -> dict:
    user = ("CASE SUMMARY:\n" + text[:5000] + "\n\nRARE-DISEASE ENGINE SUGGESTIONS (for context):\n"
            + "\n".join(f"- {n}" for n in rare_names[:10])
            + "\n\nDETERMINISTIC SIGNALS:\n" + "\n".join(f"- {x}" for x in rules["signals"] or ["none"])
            + "\n\nReturn the JSON object.")
    obj, meta = llm.json_chat([{"role": "system", "content": REALIST_SYSTEM},
                               {"role": "user", "content": user}],
                              max_tokens=900, required=["verdict", "common"], tag="realist")
    if not obj:
        raise RuntimeError(meta.get("error") or "realist returned no valid JSON")
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("common_first", "both", "rare_ok"):
        verdict = rules["verdict"]
    common = []
    for c in (obj.get("common") or [])[:5]:
        if isinstance(c, dict) and str(c.get("name", "")).strip():
            common.append({"name": str(c["name"]).strip()[:160], "why": str(c.get("why", "")).strip()[:300],
                           "next": str(c.get("next", "")).strip()[:300], "matched": []})
    return {"verdict": verdict, "reason": str(obj.get("reason", "")).strip()[:600] or rules["reason"],
            "signals": rules["signals"], "source": "ai", "common": common or rules["common"]}


def dumps(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)
