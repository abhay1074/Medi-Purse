# Glass-box rare-disease retrieval and explanation — shared retriever bundle

This folder contains everything needed to run the frozen **fusion v4 SHAP-8**
retriever and to build the live evidence-grounded agent around it.

**Contributors.** This is part of the MediPurse team project. The retriever and explanation
engine were built by Animesh Pathak and Aarav Singh, and are committed here through Aarav's
account. Abhay Yadav built the frontend; Animesh Pathak and Abhay Yadav then built the
backend and the backend–frontend–LLM glue. See [`../CONTRIBUTORS.md`](../CONTRIBUTORS.md).

Start with **`docs/architecture.pdf`** (LaTeX source in `docs/architecture.tex`).
It covers the data split, preprocessing, retrieval training, agent architecture,
execution flow, prompts, evaluation, and a deployment guide for a chat frontend
and backend.

## What is validated

Retrieval, full test set (10,702 cases, 1,464-disease catalog):

| System | MRR | Top-1 | Top-5 | Top-10 | R@100 |
|---|---:|---:|---:|---:|---:|
| TF-IDF top-100 | 0.4203 | -- | 0.5344 | 0.6113 | 0.8213 |
| BoW mass80 centroid | 0.4695 | -- | 0.5871 | 0.6533 | 0.8367 |
| GBM v4 140 | 0.4614 | -- | 0.5933 | 0.6663 | 0.8213 |
| Fusion v4 min-max | 0.4955 | -- | 0.6180 | 0.6824 | 0.8444 |
| **Fusion v4 SHAP-8 (shipped)** | **0.4960** | **0.3911** | **0.6199** | **0.6848** | **0.8437** |

The live reference implementation in `code/reference_retriever.py` reproduces the
cached top-10 exactly on 100 sampled test cases.

The LLM layer was evaluated on 62 sampled cases with Gemma-4-E4B. Its ranking effect
is mixed: it improves cases whose gold was ranked 2-5 (MRR 0.378 -> 0.612) but harms
correct top-1 cases (1.000 -> 0.707), for a net MRR change of -0.055 (95% CI
-0.138 to +0.028). The recommended default is retriever ranking plus grounded
evidence display; LLM reordering should be optional until a rank-prior blend is
validated. Details and exact numbers are in `results/agent_interim_62.json` and
the PDF.

## Quick start: retriever

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy scipy scikit-learn lightgbm joblib

python - <<'PY'
import sys; sys.path.insert(0, "code")
from reference_retriever import Retriever
r = Retriever("weights/retriever")
out = r.rank("A 46-year-old woman with progressive vision loss and angioid streaks.", k=10)
for row in out["top"]:
    print(row["rank"], row["disease_id"], row["name"], round(row["score"], 4))
PY
```

`rank()` also returns a trace with TF-IDF/BoW ranks, matched case terms, and GBM
feature contributions for each top candidate.

## Quick start: live engine (all four LLM calls)

A llama.cpp OpenAI-compatible server is required (default
`http://127.0.0.1:8080/v1`).

```bash
python - <<'PY'
import sys; sys.path.insert(0, "code")
from live_engine import LiveEngine
engine = LiveEngine(".", llm_base="http://127.0.0.1:8080/v1")
report = engine.analyze("A 3-year-old boy with left arm jerking ...",
                        case_id="demo-1", progress=print)
print(report["final_order"])
print(report["claims"])
PY
```

The engine uses the shipped catalog evidence (`weights/catalog/`), hides retriever
ranks from the auditor, validates every citation, and adjudicates each claim.

## Quick start: batch pipeline

`code/glassbox_pipeline.py` is the batch reference. It expects the benchmark
repository layout (splits, cached feature matrices, `Zebramap_DBS/`). It is included
for reproducibility and for its exact prompts and fallbacks; the live engine is the
recommended integration point.

## File map

```
docs/architecture.pdf / .tex   full architecture and deployment guide
weights/retriever/             production retriever weights
  gbm_v4_shap8.txt             8-feature LambdaRank, 241 trees (production)
  gbm_v4_140.txt               full 140-feature LambdaRank (alternative)
  tfidf_vectorizer.joblib      50k-feature TF-IDF
  tfidf_classifier.joblib      SGD log-loss, 1,464 classes (280 MB)
  tfidf_config.json            TF-IDF/scaler hyperparameters
  bow_vectorizer_mass80.pkl    mass80 TF-IDF vectorizer
  bow_centroids_mass80.npz     1,464 L2-normalised disease centroids
  prune_sets_mass80.json       2,075 pruned word forms
  svd32.joblib                 32-component TruncatedSVD
  disease_svd32.npy            1,464 x 32 disease SVD rows
  diseases.json                catalog order and names
  retriever_config.json        fusion weights, feature order, metrics
weights/catalog/
  disease_index.json           id/name/index for 1,464 diseases
  augmented_disease_data.json  full ZebraMap/Orphanet profiles
  optimuskg_catalog.json       phenotype/gene/parent slice, 1,244 diseases
code/
  reference_retriever.py       minimal live retriever
  live_engine.py               live four-call explanation engine
  live_backend_example.py      FastAPI session/SSE skeleton
  glassbox_pipeline.py         batch reference pipeline
  prompts.md                   prompt appendix
results/
  retrieval_full_test.json     full-test retrieval numbers
  agent_interim_62.json        interim agent evaluation and claim statistics
SHA256SUMS                     integrity check
```

## Important notes

- **Catalog order is a model contract.** `diseases.json` is the sorted training
  disease list. All centroids, SVD rows, classifier classes, and GBM features use
  this index. Do not reorder it.
- **Feature order is a model contract.** `gbm_v4_shap8.txt` expects the eight
  features in the order listed in `retriever_config.json`:
  `disease_svd_0, rank_min, tfidf_z, tfidf_minmax, tfidf_rank, svd_cosine,
  tfidf_x_bow, tfidf_recip_rank`.
- **TF-IDF scores are SGD probabilities.** The candidate pool is the top 100 of the
  1,464-class classifier, not a centroid score.
- **BoW scores are mass80 centroid cosines** over all 1,464 diseases.
- **Fusion formula:** `0.6 * minmax(bow_full) + 0.4 * (minmax(gbm_pool) or -0.3)
  + 4e-5 * minmax(bow_full)`. The pad applies to diseases outside the TF-IDF pool.
- **Deterministic ties:** descending score, stable sort, catalog index as final
  tie-break. Earlier union-pool v5 numbers that used tie-optimistic fallback
  scoring are retracted; v5 is not part of this bundle.
- **Integrity:** the SVD refit was verified against the cached feature matrices with
  a maximum absolute difference of 0.
- **Scope:** research prototype, not a diagnostic device. Do not use it for clinical
  decisions.

## Reproducing the weights

The weights were trained on `splits/disjoint_multilabel/train.csv` (47,858 cases)
with seed 42. The training scripts are in the source repository; the key settings
are in the PDF and `weights/retriever/retriever_config.json`. The fitted SVD
transformer is shipped as `svd32.joblib`; it can be refit from the shipped mass80
vectorizer and prune set given the source training split (not included here).
