"""Export the full OptimusKG phenotype list for the app's symptom search.

Run with the Python 3.12 environment that has the `optimuskg` client installed:

    uv venv -p 3.12 .venv-kg && uv pip install -p .venv-kg optimuskg
    .venv-kg/Scripts/python scripts/build_phenotypes.py

Input: OptimusKG gold-layer tables (Harvard Dataverse, doi:10.7910/DVN/IYNGEV):
    nodes/phenotype.parquet, edges/phenotype_phenotype.parquet,
    edges/disease_phenotype.parquet
They are read from the optimuskg cache when the client can reach Dataverse, or from
data/optimuskg/ (download them there with the Dataverse access API if the client
cannot, e.g. `curl -L -o data/optimuskg/phenotype_nodes.parquet
https://dataverse.harvard.edu/api/access/datafile/<id>`).

Output: app/data/phenotypes.json, a compact list used by /api/symptoms:
    {"source": ..., "count": N, "items": [[id, name, [synonyms], category, n_diseases], ...]}

Kept: every HPO term under "Phenotypic abnormality" (HP:0000118) plus MedDRA
symptom terms not already covered by an HPO name or synonym. Dropped: HPO's
non-symptom branches (mode of inheritance, frequency, clinical modifier,
clinical course, past medical history, blood group) and the root.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "data" / "optimuskg"
OUT = ROOT / "app" / "data" / "phenotypes.json"
PHENOTYPIC_ABNORMALITY = "HP_0000118"


def load(kg_path: str, local_name: str) -> pl.DataFrame:
    local = LOCAL / f"{local_name}.parquet"
    if local.exists():
        return pl.read_parquet(local)
    import optimuskg  # noqa: PLC0415  (only needed when not using local copies)
    return optimuskg.load_parquet(kg_path)


def norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").split())


def main() -> None:
    nodes = load("nodes/phenotype.parquet", "phenotype_nodes").unnest("properties")
    pp = load("edges/phenotype_phenotype.parquet", "phenotype_phenotype_edges")
    dp = load("edges/disease_phenotype.parquet", "disease_phenotype_edges")

    # PARENT edges point parent -> child
    children = defaultdict(list)
    for parent, child in pp.select("from", "to").iter_rows():
        children[parent].append(child)

    # top-level organ-system category for every term under Phenotypic abnormality
    names = dict(nodes.select("id", "name").iter_rows())
    category: dict[str, str] = {}
    for top in sorted(children[PHENOTYPIC_ABNORMALITY], key=lambda i: names.get(i, "")):
        label = names.get(top, "").removeprefix("Abnormality of the ").removeprefix("Abnormality of ")
        label = label[:1].upper() + label[1:]
        q = deque([top])
        while q:
            n = q.popleft()
            if n in category:
                continue
            category[n] = label
            q.extend(children[n])

    n_dis = dict(dp.group_by("to").agg(pl.col("from").n_unique().alias("n")).iter_rows())

    items, covered = [], set()
    for row in nodes.iter_rows(named=True):
        pid, name = row["id"], (row["name"] or "").strip()
        if not name or not pid.startswith("HP_") or pid not in category:
            continue
        syn = []
        for key in ("exact_synonyms", "related_synonyms"):
            for s in row.get(key) or []:
                s = s.strip()
                if s and norm(s) != norm(name) and norm(s) not in {norm(x) for x in syn}:
                    syn.append(s)
        covered.add(norm(name))
        covered.update(norm(s) for s in syn)
        items.append([pid.replace("_", ":"), name, syn[:6], category[pid], int(n_dis.get(pid, 0))])

    meddra = 0
    for row in nodes.iter_rows(named=True):
        pid, name = row["id"], (row["name"] or "").strip()
        if pid.startswith("meddra:") and name and norm(name) not in covered:
            items.append([pid.upper().replace("MEDDRA:", "MedDRA:"), name, [], "Symptom (MedDRA)", 0])
            covered.add(norm(name))
            meddra += 1

    items.sort(key=lambda x: (-x[4], x[1].lower()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "OptimusKG (Harvard Dataverse doi:10.7910/DVN/IYNGEV): HPO phenotypic "
                  "abnormality branch plus MedDRA symptom terms",
        "count": len(items),
        "categories": sorted({x[3] for x in items}),
        "items": items,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} with {len(items)} terms ({len(items) - meddra} HPO, {meddra} MedDRA), "
          f"{len(payload['categories'])} categories, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
