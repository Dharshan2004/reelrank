# ReelRank improvement work log

## 1 November 2026 — correctness, evaluation and interview evidence

**Objective:** strengthen the full retrieval → safety filter → ranking → serving project for an internship application, with reproducible evidence rather than unsupported performance claims.

**Starting point:** HEAD `e8a80cd`, with existing uncommitted user changes. Earlier audits identified missing user inputs in ranking, train/serve feature mismatch, test-tuned fusion, future training histories, unequal baseline policies, weak safety validation and incomplete run provenance. The pre-change cascade's +9.00% / +12.27% exploratory lifts reproduced, but paired bootstrap intervals included zero.

**Preservation:** copied existing model/artifact files and results into `artifacts/archive/pre-correctness-20261101T072925Z/` before retraining. Existing user edits are being retained. No commits, pushes or paid label-generation jobs have been initiated.

**Implemented:**

- Shared training-fitted ranking feature transformation and real user/item inputs; standard validation for checkpoint/fusion selection.
- Strictly earlier event histories for retrieval training, duplicate-positive handling, aligned retrieval cohort and repeat policies.
- Shared offline/API scoring, symmetric safety filtering, exact FAISS serving, paired uncertainty and per-user evaluation records.
- Immutable result records and source/input fingerprints; clearer safety-label provenance and serving response validation.

The code, models and evaluations were rebuilt locally. Changes and results follow; no online effect or hiring outcome is claimed.

### Ranking

- Added `ranking/features.py`: a serialized transformer fitted only to training impressions, including user/video IDs, categories and user numeric bins. Unknown values use zero; validation/API never refit bins. Entity tables are indexed once per transform batch.
- Reworked LR to use sparse one-hot encoded categorical fields. DeepFM and MMoE receive the same information. Stabilized DeepFM initialization and added clipping to neural training.
- Standard validation selects neural checkpoints by log loss and MMoE fusion by long-view AUC. Random data is loaded for evaluation only after choices are fixed, and marked previously inspected exploratory reuse.
- Chose LR for serving on primary standard-validation long-view GAUC; retained MMoE behind `RANKER=mmoe` / `REELRANK_RANKER=mmoe`. This is a baseline decision, not a claim that multi-task learning won.
- LR remains additive: user effects shift its probabilities but alone do not personalize item order. Retrieval supplies personalized candidate sets; neural rankers support learned user–item interactions.

### Retrieval

- Added `data/history.py`: per-example histories strictly precede timestamps; simultaneous events cannot enter one another's history.
- Masked duplicate positive candidate IDs in in-batch softmax. LogQ uses the selected positive-event distribution.
- Aligned training/relevance on long_view, default cohorts at 2,000 seed-42 users, and allow-repeat policy across popularity, ItemCF and two-tower.
- Cached entity tensors and vectorized history construction instead of per-example pandas lookup. A local diagnostic built 384,121 training examples in 2.983 seconds and collated 20 batches in 0.120 seconds; this is a local diagnostic, not a general benchmark.

### Evaluation and serving

- Added shared `ranking/inference.py` for LR/MMoE scoring, serialized feature validation, stable ordering and a common safety filter.
- Both candidate generators now receive the same safety policy. Serving uses exact FAISS Flat.
- Added paired user bootstrap intervals (2,000 resamples), positive-hit counts, and archived per-user recommendations/scores. LR/MMoE and safety on/off have distinct run IDs.
- Load generators now reject empty/duplicate recommendations, invalid probabilities and invalid timings. HTTP success alone is insufficient.
- Added configurable `RANKER` and `EVAL_USERS` Make variables; defaults are LR and 2,000 users. Reference YAML is documented as reference defaults rather than an automatically loaded experiment specification.

### Safety and provenance

- Removed `is_hate` weak-label generation and dislike-enriched annotation selection. New runs use explicit rule-enriched sampling, not representative prevalence estimation.
- Label cache keys include policy, model and source. API fallback labels do not prevent subsequent retries; current-run API calls are counted separately from cache hits.
- Existing human annotation files are preserved, with new samples written separately. Existing safety labels were not regenerated and remain weak-label prototype evidence.
- New result records preserve source ZIP snapshots, source/data/model hashes, resolved run arguments, dirty-worktree status and immutable run directories. Per-user cascade outputs are copied into their immutable record. Latest JSON aliases remain convenient for reports.
- Old cascade files are preserved but excluded from the current generated table. Earlier audits are marked historical.

### Executed experiments

Training commands used one OpenMP/MKL thread and the existing environment:

```bash
python -m reelrank.ranking.train --epochs 3
python -m reelrank.retrieval.train --epochs 8
python -m reelrank.retrieval.baselines
python -m reelrank.eval.pipeline_eval --ranker lr --max-users 2000
python -m reelrank.eval.pipeline_eval --ranker mmoe --max-users 2000
```

Ranking used 400,000 sampled training impressions and 80,000 standard-validation impressions; evaluation used 80,000 random-log impressions. Neural training retained the best validation checkpoint, rather than automatically selecting the final epoch. All runs used seed 42.

| Ranker | Validation long-view GAUC | Validation log loss | Random exploratory GAUC | Random exploratory log loss |
|---|---:|---:|---:|---:|
| LR | 0.64845 | 0.55251 | 0.59627 | 0.44595 |
| DeepFM | 0.64390 | 0.54930 | 0.59893 | 0.41032 |
| MMoE | 0.61726 | 0.57985 | 0.58518 | 0.37213 |

Different metrics favor different models. LR won the declared validation ordering criterion; DeepFM had better validation log loss, and MMoE had lower random-log loss. No post-hoc selection was made on the random table.

| Retriever | Recall@200 | Recall@10 | HitRate@200 |
|---|---:|---:|---:|
| Popularity | 0.05842 | 0.00375 | 0.2395 |
| ItemCF | 0.05350 | 0.00142 | 0.2220 |
| Two-tower | 0.06956 | 0.00421 | 0.2560 |

Two-tower's matched-cohort Recall@200 point estimate is about 19.1% above popularity. This is one training seed, and earlier results used different feature/protocol choices; the before/after change cannot be attributed to one fix.

| Ranker / safety | Two-tower NDCG@10 | Popularity NDCG@10 | Relative lift | Absolute difference 95% paired interval |
|---|---:|---:|---:|---|
| LR, on | 0.0035869 | 0.0027904 | +28.54% | [-0.0002649, +0.0018916] |
| LR, off | 0.0035324 | 0.0025866 | +36.56% | [-0.0001095, +0.0020535] |
| MMoE, on | 0.0030068 | 0.0026426 | +13.78% | [-0.0007932, +0.0014402] |
| MMoE, off | 0.0029688 | 0.0025483 | +16.50% | [-0.0006974, +0.0014496] |

All intervals cross zero. These exploratory point estimates do **not** establish reliable end-to-end superiority. With LR and safety, only 48/2,000 two-tower users and 45/2,000 popularity users have observed positive hits. The useful improvement is the controlled measurement and explicit uncertainty.

### Verification

- **28 tests passed** after integration: chronology, timestamp ties, duplicate negatives, cohort/repeat policy, transformer round-trip and unknowns, LR/MMoE offline/API parity, safety filtering, legacy checkpoints, bootstrap behavior, invalid API parameters, load-test response validation, safety fallback/cache provenance and immutable run snapshots.
- Isolated synthetic ranking smoke trained all three model families and emitted all six split-labeled results before the real run.
- Real localhost smoke returned 10 valid items for users 1, 42, 100 and unknown user 99999999; LR/Flat artifacts loaded successfully.
- The first localhost attempt was blocked by the filesystem/network sandbox (bind/connect operation not permitted), producing an all-error load-test record. It is retained as a failed infrastructure attempt, not a model-performance measurement. Localhost execution was then approved and the actual API/load test rerun.
- The demo GIF was regenerated from the corrected LR service. Analysis notebook now has an executable generator using existing dependencies, replacing unexecuted aggregate-chart cells.
- The completed approved 60-second Locust run sent **2,890 requests**, **zero failures**, **48.36 QPS**, **31 ms p50**, **79 ms p95**, **100 ms p99** at eight clients on macOS ARM, with 50–200 ms think time. This is achieved throughput, not a capacity ceiling. Immutable record: `results/runs/20261101T073944Z-serving_loadtest-2ee00139/result.json`.
- The analysis generator executed **seven cells with zero error outputs**. The five-frame GIF was refreshed and a representative frame visually checked; it identifies LR and displays actual API scores/timings.
- `git diff --check` passed. The only test warning is the environment's existing Starlette/httpx deprecation warning.

### Remaining limitations and next experiments

- Random-log exposure from earlier development cannot be undone; independent confirmation is still needed. No online A/B test was performed.
- Bootstrap intervals cover user sampling, not multiple training seeds. Run a predeclared seed sweep and independent confirmation before claiming reliable lift.
- Pure histories are partial, and static feature snapshot timing is not fully resolved. Numeric normalization for retrieval and categorical field separation remain controlled follow-up experiments.
- Safety is caption-only, with weak classifier performance and no demonstrated unsafe recall from the all-safe review set. A new independently annotated set is needed; none was fabricated or purchased during this work.
- LR's additive structure limits personalized reordering. A user–item interaction model should earn deployment through validation and a controlled cascade comparison.
- No clean-room Docker build, remote deployment or production-capacity claim was made.

## Repository polish — 2026-11-01

- Reorganized the README around the project, quickstart, results, reproduction and repository map. Moved detailed methodology/safety limitations and interview notes to dedicated documents without changing recorded metrics.
- Added a results index that labels historical cascade files and current aliases; preserved immutable records and existing filenames.
- Corrected installation to `uv sync --extra dev --locked`; added the user-selected MIT license and package metadata.
- Moved informational YAML defaults into documentation because no executable config loader used them.
- Added `make example`, a synthetic ranking/filtering walkthrough using shared inference code, with no downloads, artifact writes, or benchmark claims.
- Added storage environment overrides and external-path support in run logging; installed packages default storage to the working directory instead of site-packages.
- Updated Docker to install locked dependencies and mount data/artifacts; added an allowlist-style `.dockerignore`.
- Added `docs/COMMIT_PLAN.md` covering existing and new changes in four dependency-aware phases. No staging, commits, history rewrites or pushes performed. No CI or contributing guidelines added.
- Validation: 29 tests passed; synthetic example ran successfully; wheel and source distribution built. Docker runtime validation was unavailable because the Docker daemon was not running. One existing Starlette/httpx deprecation warning remains.
- Authorship check: no Codex author or co-author found in local history; the initial commit is authored by Cursor Agent. Existing history was left unchanged.

## Local commit execution — 2026-11-01

The user subsequently authorized the four planned local commits, explicitly prohibiting a push and Codex contributor attribution. Commits use the configured Dharshan2004 identity with no co-author trailers. Isolated staged snapshots passed 27 tests for phase 1 and 29 for phase 2. Existing history, including the initial Cursor Agent author, remains unchanged.

## Retrieval seed sweep — 2026-11-01

- Added an independent `--cohort-seed` to retrieval training and checksummed per-user recall evidence to neural and baseline evaluation.
- Added a runner with predeclared seeds 42–46, fixed eight-epoch settings, isolated artifact/result roots, an exact seed-42 reproduction gate, and before/after protection of the original top-level artifacts/results. No serving model replacement or hyperparameter tuning occurred.
- All five runs completed: Recall@200 0.069556, 0.065714, 0.064123, 0.066361, 0.067564; mean 0.066664, sample SD 0.002039. Popularity 0.058418; ItemCF 0.053501.
- Mean relative lifts: +14.11% versus popularity, +24.60% versus ItemCF. Seed-averaged paired user-bootstrap absolute-difference intervals: [0.000924, 0.015759] and [0.005487, 0.020763]. Three individual-seed intervals versus popularity cross zero; all five versus ItemCF exclude zero. These are conditional user intervals, not full training uncertainty.
- Updated the README with one sourced results table, analytical uniform-random reference, seed variation, interval scope and unchanged end-to-end limitations. Preserved clearly labeled historical audits and run records instead of rewriting history.
- Added integration coverage running two tiny training jobs with different training seeds, a fixed cohort, separate output roots and verified per-user evidence; added mismatched-pair and tampered-evidence checks.
- Source snapshots for the five full training runs match. The seed-42 rerun reproduces the prior recall exactly; the integrity record confirms protected files were unchanged. No paid API calls or remote training services used.
- Final verification: 33 tests passed and the synthetic example ran; all 22 original artifact files remain byte-identical. README/documentation links and the exact seed-42 reproduction gate were checked. Changes are recorded as local implementation and results/documentation commits on `codex/retrieval-seed-sweep`, with the configured user identity and no co-author trailers; nothing pushed.
