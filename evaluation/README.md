# MediPurse AI diagnosis: usage and evaluation report

Updated 2026-11-24. This report has two halves:

- **Part A: real-world usage.** How the AI diagnosis behaves when people use it: speed, how it holds
  up when several doctors use it at once, memory, AI cost, and what the suggestions are built on.
  Measured with `usage_bench.py`.
- **Part B: accuracy verification.** Checks that the system ranks the right answer, on synthetic
  rare and everyday cases and on real held-out case reports. Measured with `run_eval.py`.

**Research prototype, not a diagnostic device.**

---

## At a glance

| | |
|---|---|
| Submit a case → results page (one doctor, p50 / p99) | **0.33 s / 0.41 s** |
| Analyses per second (one worker, CPU) | **≈2.9** (saturates with 4 doctors at once) |
| Cold start to "AI ready" / memory | **8.9 s** / **571 MB** (peak 860 MB under load) |
| Optional "Explain with AI" (4 DeepSeek calls) | **≈17 s**, ≈8.8k tokens in + 3.8k out per case, **97%** of claims pass the citation checks |
| AI cost | ≈**₹1 per explanation**; the background AI review is ≈₹0.003 per analysis |
| Rare-disease ranking on real held-out case reports (300) | MRR@10 **0.71**, Recall@10 **0.90** (optimistic: ~69% come from the training articles) |
| Rare-disease ranking on the official test split (10,702) | MRR **0.496**, Recall@10 **0.685**: the realistic estimate |
| **All 92 synthetic cases, as shown on the page** | **MRR@10 0.832**, Recall@10 **0.924** (old rules: 0.518 / 0.565) |
| Everyday cases correctly sent "common first" / rare cases wrongly sent | **87%** / **9%** (old rules: 13% / 7%) |

---

# Part A: real-world usage

## A1. Latency for one doctor

Time from pressing **Find likely conditions** to the rendered results page. This includes the
session lookup, the rare-disease ranking, the router, writing to the database and rendering.
Measured on a development laptop (Windows 11, CPU only, one uvicorn worker) with the 92 evaluation
cases, after warm-up.

| Path | n | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| HTTP end to end, AI review off | 12 | 332 ms | 377 ms | 409 ms | 412 ms |
| HTTP end to end, AI review on (runs in the background) | 8 | 282 ms | 307 ms | 339 ms | 342 ms |
| HTTP end to end, evaluation run (3 × 92 cases) | 276 | 310 ms | 350 ms | 416 ms | 1,550 ms * |
| In process: rare-disease ranking | 460 | 281 ms | 321 ms | 358 ms | 414 ms |
| In process: realist router | 460 | 3.8 ms | 7.8 ms | 9.8 ms | 11 ms |

\* The one slow request was the first one, which checks the DeepSeek account balance over the
network. The answer is cached for 10 minutes.

**Takeaway:** results appear in about a third of a second. Almost all of that time is the
rare-disease ranking. The router costs under 10 ms. When DeepSeek has credit, the realist's AI
review runs after the page has returned and refines the verdict a few seconds later, so it never
slows the page.

## A2. Several doctors at once

Each simulated doctor submits cases back to back, with no pause between them. This is a stress
test; real use has minutes between cases.

| Doctors at once | Analyses/s | p50 | p90 | p99 | Errors |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.93 | 332 ms | 377 ms | 409 ms | 0 |
| 2 | 2.51 | 584 ms | 663 ms | 893 ms | 0 |
| 4 | 2.84 | 1.16 s | 1.42 s | 1.67 s | 0 |
| 8 | 2.89 | 2.48 s | 2.94 s | 3.08 s | 0 |
| 4, with the AI review on | 2.47 | 1.27 s | 1.63 s | 1.92 s | 0 |

**Takeaway:**
- One worker tops out at about 3 analyses per second, which is CPU-bound ranking.
- That's roughly 10,000 analyses an hour. A clinic where each doctor runs a few analyses an hour
  will never notice queueing.
- At 8 doctors submitting at exactly the same moment, the slowest answer is about 3 s.
- To scale, add workers (`uvicorn --workers N`). Each worker loads its own copy of the model,
  about 0.6 GB of memory.

## A3. Start-up, memory and size

| | |
|---|---|
| Time from launching the app to the AI being ready | **8.9 s** (6–21 s depending on disk cache) |
| Memory after start | **571 MB** |
| Peak memory under 8 concurrent doctors | **860 MB** |
| Retriever model files | 300 MB (most of it is the 280 MB SGD classifier) |

The app loads the model in the background. Until it's ready, the AI page shows "starting up";
everything else works.

## A4. The AI (LLM) steps

Measured with DeepSeek (`deepseek-chat`) on 4 cases (2 rare, 2 everyday). Spend is the actual
account balance before minus after.

| Step | When it runs | Time (p50 / max) | Tokens per use | Notes |
|---|---|---|---|---|
| **Explain with AI** | doctor clicks the button | **16.8 s** / 18.9 s | 8.8k in, 3.8k out (4 calls) | 16.8 claims per case; **97%** survive the citation validator and the adjudicator |
| Realist AI review | automatically after each analysis | 3.2 s / 3.5 s | ≈1–2k | in the background; agreed with the router on **4/4** cases |
| History-agent reply (AI mode) | each patient message | 2.8 s / 2.8 s | small | the guided mode replies instantly |

**Cost:**
- The whole benchmark cost **₹4.04**: 4 explanations, 4 reviews and 3 history turns. That's about
  **₹1 per explanation**.
- 42 analyses with the background review cost about ₹0.14.
- So the current credit (about ₹70) covers roughly 70 explanations, or about 20,000 background reviews.
- DeepSeek bills this account in yuan; amounts are converted at ₹13.95 per yuan (open.er-api.com,
  23 Nov 2026). `usage_bench.py` fetches the current rate each run.
- With no credit, every AI step falls back to its rule-based or guided mode automatically.

## A5. What the suggestions are built on (coverage)

| | Count |
|---|---:|
| Rare diseases the ranking can suggest | 1,464 |
| …with an OptimusKG profile (phenotypes, genes) for the evidence panel | 1,244 |
| …with medicines used in published ZebraMap case reports (for prescribing suggestions) | **1,006** |
| …with approved indications from OptimusKG / DrugCentral | 158 |
| Searchable symptoms and signs (OptimusKG / HPO) | 18,555 |
| Everyday symptoms with plain-language labels | 81 |
| Everyday conditions the realist can list, each with first-line medicines | 51 |

**Medicine suggestions:**
- For a rare diagnosis, the prescription form lists the medicines actually used in published cases
  for that disease, with how often. For example, Wilson disease: penicillamine in 49 of 130 cases,
  zinc 12, trientine 10. These come from ZebraMap's `Treatment` field for 48,628 cases, with drug
  names picked out using OptimusKG's 18,640-drug lexicon.
- For an everyday diagnosis, it lists usual first-line options.
- OptimusKG contraindications for the patient's diagnoses and the patient's own **medicines to
  avoid** (recorded reactions) show as warnings. Prescribing an avoided medicine needs an
  explicit confirmation.

---

# Part B: accuracy verification

## B1. What is scored

What a doctor sees after "Find likely conditions":

1. the rare-disease ranking (top 10 of 1,464);
2. the realist router's verdict and up to 5 common conditions;
3. the page order:
   - verdict "start with common causes" (`common_first`) → common list first;
   - verdict "rule out common causes too" (`both`) or "rare workup is reasonable" (`rare_ok`) →
     rare list first, common list after it.

**"End to end"** scores the position of the right answer in that combined list.

Metrics:
- **MRR@k:** mean of 1/rank, counted only if the right answer is in the top k.
- **Recall@k:** the share of cases whose right answer is in the top k.
- **NDCG@k:** with one right answer, the mean of 1/log2(rank+1).

95% confidence intervals come from a 2,000-sample bootstrap.

## B2. Test data

| Set | Cases | What it is |
|---|---:|---|
| `synthetic_rare.json` (v2) | 46 | 40 case-report narratives over 40 rare diseases, plus 6 one-line notes |
| `synthetic_common.json` (v2) | 46 | 40 case-report narratives over 40 everyday conditions, plus 6 short notes |
| Real held-out | 300 | ZebraMap case reports from articles **not** used to fit the router, one per article |
| Official test split | 10,702 | real case reports, article-disjoint split, seed 42 |

**How the synthetic cases were written (v2).** ZebraMap notes are single-paragraph case reports:
- median 275 words, middle half 131–506;
- they open with "The patient was…", "A previously healthy…" or "We report…";
- then history, examination, lab values, imaging (90% mention a scan), confirmatory tests,
  treatment and follow-up.

Both synthetic sets follow that structure **with matched lengths**:
- rare median 143 words, everyday median 140;
- 6 short notes in each set, mirroring the ~10% of ZebraMap notes under 72 words.

Nothing is copied from ZebraMap or Orphanet. The first version (v1, short clinic notes, everyday
median 24 words) is kept in `*_v1.json`. Section B6 explains why it was replaced.

## B3. Results

### Rare cases (46)

| | MRR@10 (95% CI) | NDCG@10 | R@1 | R@3 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|---:|
| Ranking alone | 0.957 (0.913–0.989) | 0.968 | 0.913 | 1.000 | 1.000 | 1.000 |
| End to end, router | 0.905 (0.825–0.971) | 0.927 | 0.870 | 0.913 | 0.913 | 1.000 |
| End to end, old rules | 0.920 (0.851–0.978) | 0.940 | 0.870 | 0.957 | 0.957 | 1.000 |

By difficulty (ranking alone): typical 0.969, partial 0.938, one-line 0.917.

### Everyday cases (46)

| | MRR (95% CI) | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|
| Realist's common list (top 5) | 0.833 @5 (0.745–0.911) | 0.739 | 0.978 | — |
| End to end, router | **0.759** @10 (0.649–0.866) | 0.696 | 0.848 | 0.848 |
| End to end, old rules | 0.116 @10 (0.029–0.217) | 0.109 | 0.130 | 0.130 |

### All 92 cases, end to end

| | MRR@10 (95% CI) | NDCG@10 | R@1 | R@3 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|---:|
| **Router** | **0.832** (0.758–0.897) | 0.854 | 0.783 | 0.870 | 0.880 | **0.924** |
| Old rules | 0.518 (0.415–0.614) | 0.530 | 0.489 | 0.543 | 0.543 | 0.565 |
| Perfect routing (upper bound) | 0.895 | — | — | — | — | 0.989 |

### Routing

| | Old rules | Router |
|---|---:|---:|
| Everyday cases sent "common first" | 13% | **87%** |
| Rare cases sent "common first" | 7% | 9% |
| Precision / recall / F1 | 0.67 / 0.13 / 0.22 | **0.91 / 0.87 / 0.89** |
| Accuracy | 0.53 | **0.89** |
| Rare answer pushed below a common list | 2 | 4 |
| Everyday answer listed after the 10 rare diseases | 39 | 6 |
| **Real held-out case reports sent "common first"** (300) | 11.3% | **5.7%** |

### Real held-out case reports (300, rare)

| | MRR@10 | R@1 | R@10 | Verdicts (rare_ok / both / common_first) |
|---|---:|---:|---:|---|
| Ranking alone | 0.710 | 0.590 | 0.897 | — |
| End to end, router | 0.678 | — | 0.893 | 202 / 81 / 17 |
| End to end, old rules | 0.679 | — | 0.883 | 227 / 39 / 34 |

These are real ZebraMap notes, median 290 words. About 69% of all ZebraMap cases are in the
retriever's training split, so the ranking numbers here are optimistic. The point of this check is
routing: the router sends half as many real rare cases to "common first" as the old rules did.

### What-if: page ordering (same model outputs, different ordering rules)

| Ordering rule | Rare | Everyday | Combined MRR@10 | Combined R@10 |
|---|---:|---:|---:|---:|
| **Current** (common first only for `common_first`) | 0.905 | 0.759 | **0.832** | 0.924 |
| Common first for `common_first` and `both` | 0.635 | 0.833 | 0.734 | 0.989 |
| Always rare first | 0.957 | 0.000 | 0.478 | 0.500 |
| Perfect routing | 0.957 | 0.833 | 0.895 | 0.989 |

## B4. Baseline: shipped ranking on the official test split

| System | MRR | Top-1 | Top-5 | Top-10 | R@100 |
|---|---:|---:|---:|---:|---:|
| TF-IDF top-100 | 0.420 | — | 0.534 | 0.611 | 0.821 |
| BoW mass80 centroid | 0.470 | — | 0.587 | 0.653 | 0.837 |
| GBM v4 140 | 0.461 | — | 0.593 | 0.666 | 0.821 |
| **Fusion v4 SHAP-8 (shipped)** | **0.496** | **0.391** | **0.620** | **0.685** | **0.844** |

These are 10,702 real case reports the model never saw (validation split: MRR 0.529). This remains
**the honest estimate of real-world ranking accuracy**. The synthetic sets above are much easier
(0.957) because they're clean and complete.

## B5. The router

`app/router.py`, fitted by `scripts/fit_router.py`, stored in `app/data/router.json`.

**Model.** A logistic model gives `p_common`, the probability that the case is an everyday one.
Two thresholds turn it into the verdict:
- `p ≥ 0.575` → common first;
- `p ≤ 0.05` → rare workup is reasonable;
- otherwise → both.

**Features** (standardised, with their weights). Positive weights point towards everyday:

| Feature | Weight |
|---|---:|
| specialised tests mentioned (biopsy, antibodies, MRI, genetic…) | −2.06 |
| how well the everyday symptoms fit a common condition | +1.65 |
| rare-disease cue (consanguinity, regression, from birth…) | −1.14 |
| infant | −0.55 |
| share of findings that are everyday symptoms | +0.50 |
| ranking: top-1 margin | −0.49 |
| length of the note | +0.47 |
| number of everyday symptoms | +0.46 |
| ranking: top-1 confidence | −0.41 |
| specific phenotypes | +0.06 |

**Fitting data (never the evaluation sets):**
- `tuning_cases.json`: 36 synthetic rare cases (diseases not in the evaluation set) and 51 everyday
  cases (one per condition). 30 of the everyday cases are case-report narratives, 21 are short.
- 200 real ZebraMap rare case reports, one per article, seed 7. Their articles are excluded from
  the 300-case held-out check.

**Thresholds** were chosen on out-of-fold predictions to maximise balanced end-to-end MRR@10.
Cross-validated AUC on the tuning data is 0.958.

**Explanations.** The router's two biggest contributions become the plain-language reason shown on
the results page, for example "specialised tests are mentioned".

**Supporting changes to the realist's text reading:**
- **Negation:** "no fever or back pain" no longer counts fever or back pain. "But" ends a negation.
- **Everyday wording:** about 40 phrase patterns (bowels, big toe, pimples, chest tightness,
  blood-pressure readings…). Some of these phrases were added after seeing v1 evaluation failures;
  they are general clinical wording, but this is disclosed.
- **A new everyday sign:** "High blood pressure".

## B6. Why the test cases were rewritten (v1 → v2)

The first router was fitted with short everyday notes (v1). Its strongest feature became note
length, a shortcut. On v1 it scored 0.825, but that tested length as much as medicine.

The cases were rewritten in ZebraMap structure with matched lengths, and the router refitted:
- Length dropped to a minor feature.
- The old rules collapsed on realistic everyday notes: only 13% were sent "common first", because
  the rules had also relied on everyday notes being short.
- The router holds at 0.832.

The v1 results are kept for reference:
- `results_v1_rules.json`: old rules, v1 cases;
- `results_v2_router.json`: first router, v1 cases;
- `results_v2cases_rules.json`: old rules, v2 cases.

## B7. Remaining errors

- **Ranking:** every rare synthetic case has its answer in the top 10; 4 are not first. On real
  reports, Recall@10 is 0.90.
- **Rare cases sent "common first" (4 of 46):** typical presentations full of everyday symptoms with
  few named tests.
- **Everyday cases not sent "common first" (6 of 46):** mostly case reports describing several
  investigations (CT, culture, ECG), which look like a rare workup. These get "both", which lists
  the common causes after the rare list.
- **One everyday case has no matching common condition** in the realist's list of 5.

---

## Data and downloads

| Source | Used for | How it's obtained |
|---|---|---|
| ZebraMap (Diagnostics 2025, doi 10.3390/diagnostics16010107, [Zenodo 17623607](https://zenodo.org/records/17623607), CC BY 4.0) | real case reports for routing checks; treatments for medicine suggestions | `scripts/zebramap_fetch.py` reads the 12.6 GB zip remotely with HTTP range requests and downloads only `ZebraMap.json` (73 MB) and `structured_cases.json` (16 MB) with 16–32 parallel connections: about 1 minute, no images. Files go to `data/zebramap/` (not in git). |
| OptimusKG (Harvard Dataverse doi:10.7910/DVN/IYNGEV) | phenotype list; drug lexicon; indications and contraindications | `drug.parquet`, `drug_disease.parquet`, phenotype tables, in `data/optimuskg/` (not in git) |
| Fusion v4 SHAP-8 weights | rare-disease ranking | `handoff/weights/retriever/` |

All 69,146 ZebraMap cases are present: exactly the train + validation + test split (47,858 +
10,586 + 10,702), 1,464 diseases, 36,131 source articles.

## Retraining recipe (not run yet)

The data is now available, so retraining on all 69,146 cases is possible; it hasn't been run.
These are the retriever settings (architecture document, `retriever_config.json`). With no held-out set
left, use the iteration counts chosen on the split.

1. **mass80 pruning** for BoW and SVD (reuse `prune_sets_mass80.json`); TF-IDF uses raw text.
2. **TF-IDF:** 1–2 grams, English stop words, `min_df=2`, 50,000 features, `sublinear_tf`.
3. **SGD:** `log_loss`, L2 `alpha=1e-5`, 2,000 iterations, `tol=1e-3`, `class_weight=balanced`, seed 42.
   The top 100 classes form the pool.
4. **BoW centroids:** the mean mass80 vector per disease, L2-normalised. **SVD-32:** seed 42,
   `n_iter=5`.
5. **LambdaRank on the 8 SHAP features:**
   - `truncation_level=3`, `learning_rate=0.02`, `num_leaves=31`, `min_data_in_leaf=50`;
   - feature and bagging fraction 0.8, `lambda_l2=1.0`, `max_bin=63`;
   - **241 rounds**;
   - *recommended:* build the reranker's training features out of fold.
6. **Fusion** unchanged.

After retraining, real ZebraMap cases are all in training. Check for regressions with these
synthetic sets, and keep the test-split numbers above as the real-world estimate.

## Reproduce

```bash
# data (about 1 minute)
python scripts/zebramap_fetch.py --get structured_cases.json ZebraMap.json --threads 24
.venv-kg/Scripts/python scripts/build_drug_suggestions.py      # medicine suggestions
python scripts/fit_router.py --real 200                          # router (about 2 minutes)

# accuracy: start a throw-away app for HTTP timing, then evaluate (about 6 minutes)
HMS_DB=/tmp/eval.db SMTP_HOST= python -m uvicorn app.main:app --port 8001 &
python evaluation/run_eval.py --reps 5 --http http://localhost:8001 --http-reps 3 --real 300
python evaluation/what_if.py

# real-world usage: starts its own app on port 8002; uses about ₹4 of DeepSeek credit
python evaluation/usage_bench.py --clients 1,2,4,8 --per-client 12 --llm 4
```

Outputs: `results.json` (accuracy, with one row per case) and `usage.json` (usage).
`/ai/performance` in the app reads both.

## Limitations and next steps

- **Synthetic cases are cleaner than real notes.** Use them to compare variants. The official test
  split is the real-world ranking estimate.
- **No real everyday-disease notes yet.** PMC-Patients (167k case summaries from PubMed Central)
  would give realistic everyday cases in the same style, but it needs diagnosis labelling.
- **The router was fitted on 287 cases.** Refit it when real clinic notes are available.
- **One measured machine.** Latency scales roughly with CPU single-core speed.
- **Retraining on all data** (recipe above) is ready to run.
