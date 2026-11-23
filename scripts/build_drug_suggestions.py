"""Build app/data/disease_drugs.json: medicines linked to each rare disease.

Sources:
  - ZebraMap structured_cases.json (CC BY 4.0): the "Treatment" field of each published
    case. Drug names are picked out of the free text with the OptimusKG drug lexicon,
    and counted once per case: "penicillamine, used in 46 of 130 cases with a treatment".
  - OptimusKG drug_disease.parquet: INDICATION / OFF_LABEL_USE / CONTRAINDICATION edges
    for the disease's OptimusKG node (mapping from handoff/weights/catalog/optimuskg_catalog.json).

Run with the OptimusKG environment (needs polars):
    .venv-kg/Scripts/python scripts/build_drug_suggestions.py
"""

from __future__ import annotations

import ast
import collections
import json
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "data" / "disease_drugs.json"
STOP = {"insulin glargine"}  # nothing yet; kept for manual exclusions
# generic words that are also drug synonyms in ChEMBL and would create noise
NOISE = {"oxygen", "water", "saline", "sodium", "potassium", "calcium", "iron", "zinc", "vitamin", "glucose",
         "dextrose", "alcohol", "ethanol", "air", "salt", "sugar", "protein", "diet", "fluids", "gold", "silver",
         "copper", "magnesium", "phosphate", "chloride", "albumin", "heparin sodium", "blood", "plasma", "nitrogen",
         "carbon", "milk", "honey", "caffeine", "nicotine", "acetate", "citrate", "sulfate", "lactate", "urea"}
# keep a few that are real treatments even though they look generic
KEEP_GENERIC = {"zinc acetate", "sodium bicarbonate", "potassium citrate", "iron sucrose", "ferrous sulfate",
                "calcium carbonate", "magnesium sulfate", "potassium chloride", "sodium chloride"}
ALIASES = {  # common case-report wording -> display name
    "ivig": "Immunoglobulin (IVIG)", "intravenous immunoglobulin": "Immunoglobulin (IVIG)",
    "immunoglobulin": "Immunoglobulin (IVIG)", "gamma globulin": "Immunoglobulin (IVIG)",
    "steroids": None, "corticosteroids": None, "antibiotics": None, "chemotherapy": None,
    "enzyme replacement therapy": "Enzyme replacement therapy", "ert": None,
    "plasma exchange": "Plasma exchange", "plasmapheresis": "Plasma exchange",
}


def props(v):
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except Exception:
        return ast.literal_eval(v)


SALTS = {"hydrochloride", "hcl", "hydrobromide", "anhydrous", "monohydrate", "dihydrate", "trihydrate", "sodium",
         "potassium", "calcium", "mesylate", "maleate", "tartrate", "besylate", "succinate", "fumarate", "citrate",
         "sulfate", "phosphate", "acetate", "hyclate", "bromide", "disodium", "dipropionate", "propionate", "magnesium"}


def title(name: str) -> str:
    """Display name without salt forms: 'ZINC ACETATE ANHYDROUS' -> 'Zinc'."""
    words = name.split()
    while len(words) > 1 and words[-1].lower() in SALTS:
        words.pop()
    out = " ".join(w.capitalize() if w.isupper() or w.islower() else w for w in words)
    return "Immunoglobulin (IVIG)" if out.lower() in ("human immunoglobulin g", "immune globulin") else out


def main():
    drugs = pl.read_parquet(ROOT / "data/optimuskg/drug.parquet")
    lex: dict[str, str] = {}
    names: dict[str, str] = {}
    for row in drugs.iter_rows(named=True):
        p = props(row["properties"])
        name = (p.get("name") or "").strip()
        if not name:
            continue
        canon = title(name.lower())
        names[row["id"]] = canon
        for s in [name] + list(p.get("synonyms") or []):
            k = " ".join(re.findall(r"[a-z0-9]+", s.lower()))
            if len(k) < 5 or k in NOISE or re.search(r"\d{3,}", k) or k.count(" ") > 3:
                continue
            if any(w in NOISE for w in k.split()) and k not in KEEP_GENERIC:
                continue
            lex.setdefault(k, canon)
    max_n = 4
    print("lexicon", len(lex))

    def find(text: str) -> set[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        out, i = set(), 0
        while i < len(toks):
            hit = None
            for n in range(min(max_n, len(toks) - i), 0, -1):
                k = " ".join(toks[i:i + n])
                if k in ALIASES:
                    hit = (n, ALIASES[k])
                    break
                if k in lex:
                    hit = (n, lex[k])
                    break
            if hit:
                if hit[1]:
                    out.add(hit[1])
                i += hit[0]
            else:
                i += 1
        return out

    cases = json.loads((ROOT / "data/zebramap/structured_cases.json").read_text(encoding="utf-8"))
    used: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_treated = collections.Counter()
    for did, cs in cases.items():
        for c in cs.values():
            t = ((c.get("CaseData") or {}).get("ActualDiagnosis") or {}).get("Treatment") or ""
            if not t.strip() or t.strip().lower() in ("unknown", "none", "not mentioned", "n/a", "not specified"):
                continue
            n_treated[did] += 1
            for d in find(t):
                used[did][d] += 1

    # OptimusKG indications / contraindications via the handoff orpha -> node mapping
    cat = json.loads((ROOT / "handoff/weights/catalog/optimuskg_catalog.json").read_text(encoding="utf-8"))
    node_to_orpha = collections.defaultdict(list)
    for orpha, v in cat.items():
        if v.get("node"):
            node_to_orpha[v["node"]].append(orpha)
    dd = pl.read_parquet(ROOT / "data/optimuskg/drug_disease.parquet")
    ind = collections.defaultdict(set)
    contra = collections.defaultdict(set)
    for row in dd.iter_rows(named=True):
        for orpha in node_to_orpha.get(row["to"], []):
            nm = names.get(row["from"])
            if not nm or re.search(r"\d|chembl", nm.lower()) or len(nm) > 40:
                continue
            pr = props(row["properties"]) or {}
            direct = ((pr.get("sources") or {}).get("direct")) or []
            if row["relation"] != "CONTRAINDICATION" and "DRUG_CENTRAL" not in direct:
                continue   # approved-label indications only, not trial-only links
            (contra if row["relation"] == "CONTRAINDICATION" else ind)[orpha].add(nm)

    out = {}
    for did in set(n_treated) | set(ind) | set(contra):
        n = n_treated[did]
        top = [{"name": d, "cases": c, "share": round(c / n, 3)} for d, c in used[did].most_common(12)
               if c >= 2 or n < 5]
        out[did] = {"treated_cases": n, "used": top[:10], "indicated": sorted(ind[did])[:12],
                    "contraindicated": sorted(contra[did])[:25]}
    meta = {"sources": ["ZebraMap structured_cases.json (Treatment field, CC BY 4.0)",
                        "OptimusKG drug_disease (INDICATION, OFF_LABEL_USE, CONTRAINDICATION)"],
            "diseases": len(out), "with_case_drugs": sum(1 for v in out.values() if v["used"]),
            "with_indications": sum(1 for v in out.values() if v["indicated"])}
    OUT.write_text(json.dumps({"meta": meta, "diseases": out}, ensure_ascii=False), encoding="utf-8")
    print(meta)
    for k in ("905", "589", "342", "2331", "586"):
        v = out.get(k, {})
        print(k, [f"{d['name']} {d['cases']}" for d in v.get("used", [])][:6], "| ind", v.get("indicated", [])[:4])


if __name__ == "__main__":
    main()
