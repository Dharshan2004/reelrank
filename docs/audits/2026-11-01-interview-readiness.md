# ReelRank interview-readiness audit — 1 November 2026

> Historical pre-repair audit. Subsequent implementation, retraining and checks are documented in [the work log](../WORK_LOG.md); the current README and generated results supersede this snapshot.

> Recheck update: subsequent November 1 results supersede the negative end-to-end numbers below. Current logged lift is +9.00% without safety and +12.27% with safety after applying selected fusion weights. The checkpoint still has no user ranking features, and the feature-transform, test-selection, training-history and evaluation-policy issues remain. The original audit below is retained as a historical snapshot, not the latest metrics table.

**Verdict: relevant internship-project foundation, but the current performance claims are not interview-ready.** There is no defensible hiring probability or metric threshold that guarantees an offer. The strongest next step is to repair evaluation and feature correctness, demonstrate a controlled improvement, and explain the resulting tradeoffs. Keep the full project scope: retrieval, ranking, safety filter, and serving.

This audit combines three subagent investigations (metric validity, recommender research, and interview/engineering readiness), source inspection, local data diagnostics, and the existing test suite. Repository HEAD: `e8a80cd0f5cfcd9c22b035b36e25b9c78ceedb7f`; the working tree already contained modified code/results. Treat this as an audit of that working tree, not a reproduction of the logged runs. No model was retrained and no improved performance is claimed. Five existing tests passed. Only this report was added.

The intended role is [TikTok Machine Learning Engineer Project Intern (Recommendation), Singapore, A163230A](https://lifeattiktok.com/search/7686439502455212293), verified on the audit date. It emphasizes recommendation, user behavior, classifiers and content security; its minimum qualifications also include algorithms, programming, communication and teamwork. Project quality complements those requirements. It does not replace them.

## What the recorded numbers actually say

| Evidence | Current record | Interpretation |
|---|---|---|
| Full cascade, no safety | NDCG@10 0.001894 vs popularity→MMoE 0.002038; **−7.03%** | Does not support README's +59% claim |
| Full cascade, with safety | Recorded lift **−3.45%** | Also negative; filter is applied asymmetrically |
| Retrieval Recall@200 | Two-tower 0.06219; ItemCF 0.05860; popularity 0.05447 | Apparent gain, but cohorts and eligibility differ |
| Retrieval Recall@10 | Two-tower 0.00158; ItemCF 0.00471; popularity 0.00370 | Top of the retrieval list is weak |
| Long-view GAUC | LR **0.60429**; MMoE 0.58233; DeepFM 0.51167 | Neural architecture has not earned its complexity; inputs differ |
| Long-view log loss | LR **0.32156**; MMoE 0.51109; DeepFM 1.47451 | Probability quality needs attention, not just ordering |
| FAISS p99 | Flat **0.0882 ms**; HNSW 0.2378 ms | Exact search is faster in this benchmark |
| HNSW neighbor recall | 0.98544 at K=100 | Approximation loses accuracy without a measured speed benefit here |
| Safety classifier F1 | **0.1081**, against weak labels | Not validated content-safety performance |
| Safety spot-check | 200 safe labels; agreement 97%, kappa 0 | Cannot establish unsafe-content recall or annotation provenance |
| Serving | 1,455 requests, 8 clients, ~49 QPS, p99 85 ms, macOS ARM | Useful short demo run; not maximum capacity or production scale |

Sources: `results/*.json`, `results/RESULTS_TABLE.md`. README's safety F1, hardware and latency statements are also stale. Generate the public headline from a single selected run manifest rather than maintaining a second hand-written table.

## Correctness blockers and their remedies

1. **The neural ranker has no user inputs.** The actual `data/processed/id_maps.json` has `user_cat_idx_cols=[]`. `reelrank/ranking/train.py:237–259` uses user categorical fields, video fields and video bins for DeepFM/MMoE; user numerics are supplied only to LR. Neither user IDs nor histories enter the neural ranker. A fixed video therefore receives the same neural ranking score for every user. Retrieval supplies the current personalization. Add an explicit feature schema, including user/item IDs and meaningful user features, then compare models on matched information. Verify that distinct user features can change a fixed item's score.

2. **Feature encodings disagree across paths.** `ranking/train.py:248–250` fits quantile bins separately on train and evaluation. `eval/pipeline_eval.py:127–129` and `serve/app.py:163–165` use floored logarithms instead. The audit reproduced 97.88% duration-bin disagreement and 84.37% popularity-bin disagreement on the seed-42 training sample. Fit one transformer on training data, serialize it, and reuse it everywhere. Test exact transformed-feature equality and fixed-model score agreement across training, offline inference and API inference.

3. **The nominal random test set has been used for selection.** `ranking/train.py:235` loads it as validation; lines 188–200 tune fusion on it. The subsequent cascade evaluates the same log. Existing results are exploratory. Freeze a development/evaluation protocol before more experiments, use separate development data for checkpoint/fusion/calibration choices, and disclose previous test exposure. Repartitioning already inspected data does not retroactively create a pristine test. A genuinely new external confirmation set is valuable after protocol freeze.

4. **Training histories are not point-in-time correct.** `data/features.py:111` builds a final training-window history; `retrieval/train.py:75` reuses it for every example. Future events and the target itself can enter an earlier example's history. About 53.57% of click-positive targets occur in the supplied history, although that count includes legitimate earlier repeats. Construct histories strictly before each example's timestamp, with an explicit simultaneous-event rule. Use final train-window histories only for later evaluation. This is a training feature problem, not evidence of random-test gradient training.

5. **Retrieval comparisons are not controlled.** Baselines sample 2,000 users, two-tower 1,500, and cascade 1,000. ItemCF removes seen items while the other methods do not; ItemCF uses long-view positives while two-tower/popularity use clicks. Save one cohort and candidate eligibility policy, align objectives or explicitly ablate them, and use paired per-user comparisons. Same random seed does not make different-sized samples identical.

6. **Safety comparisons and labels need repair.** `pipeline_eval.py:156–172` filters only two-tower candidates, not the popularity baseline. `safety/judge.py:160–166` uses `is_hate` as a weak unsafe target despite the documented sanity-check-only intent. Cache keys at lines 144–146 omit policy/model/label-source provenance, so setting an API key can reuse heuristic labels. Separate dislike from policy violation; version label source, policy, model and cache. Apply identical filters to both candidate generators. The offline high-threshold/uncertain-band analysis is not the same policy as the served 0.5 classifier threshold.

7. **Run provenance is incomplete.** `eval/runlog.py` records HEAD but not dirty changes, data/checkpoint hashes or full resolved configuration, and overwrites fixed run filenames. The serving JSON bypasses the common logger. Use immutable run directories, data/split/cohort/transformer/model hashes, code state, resolved parameters, environment and per-user predictions. Check the discrepancy between logged best-fusion and evaluated-fusion AUC before attributing the old results to today's code.

## Why tiny NDCG alone does not settle project quality

Local random-log diagnostics: 1,186,059 impressions, 27,285 users, 7,583 videos; median user has observations for **22 videos**, approximately **0.29% of the candidate pool**. Median positive count across all users is one. There are **8,877 users without a positive long view**; the current positive-map evaluator excludes them. Its reported result therefore concerns users with at least one observed positive.

Full-pool recommendations are compared against sparse observed positives. Unobserved videos do not have known negative outcomes. Random exposure is useful, but the existing full-pool NDCG computation is not automatically an unbiased estimate of online feed value. Keep it as a clearly named observed-positive overlap diagnostic; report eligibility, missing feedback, cohort size and hit counts.

Add ranking metrics over observed randomized impressions (AUC/GAUC, log loss, calibration), and optionally within-user observed-exposure ranking with explicit candidate-set and zero-positive conventions. That smaller-candidate NDCG measures a different question: do not present its larger number as an improvement over full-pool NDCG. Add a random candidate baseline and paired confidence intervals. Off-policy estimators require an explicit estimand and justified logging propensities/support; do not invent propensities.

There is substantial distribution shift: training long-view rate is **33.66%**, versus **8.50%** in the full random log. Calibration should use separate development data and be reported separately from discrimination. Do not fit a constant baseline or calibration correction on final-test labels.

The [official KuaiRand documentation](https://kuairand.com/) explains that Pure retains candidate-pool logs and has incomplete sequences; it also distinguishes click/valid-play semantics across interfaces. Keep Pure for the current bounded project, document incomplete histories, stratify diagnostics by scenario, and consider 1K/27K only if rigorous long-history or off-policy work becomes the objective.

## Experiment sequence with decision criteria

Effort bands below are planning estimates, not promises. Preserve all stages; prioritize correctness before expansion.

| Order | Experiment/work | Evidence required to move on |
|---|---|---|
| 1 — first focused session | Freeze current artifacts; correct public claims; define splits, cohorts, repeat policy and metrics | No contradictory headline; documented exploratory status and reproducible manifest |
| 2 — correctness work | Shared feature transformer, true user inputs, strict histories, policy parity | Feature/API parity, chronology and personalization tests pass; regenerate affected artifacts |
| 3 — baseline cycle | Matched-input LR, simple personalized MLP, single-task DeepFM, shared-bottom and MMoE; train-prior baseline for log loss | Same rows/features/objectives/budget; validation curves; report all task metrics, not only the winning one |
| 4 — retrieval cycle | Click vs long-view training, history/no-history, logQ on/off, duplicate-positive handling; compare popularity and ItemCF | Same cohort, candidate rules and compute; Recall@50/200 and top-10 diagnostics with uncertainty |
| 5 — cascade isolation | Retrieval-only ordering; each retriever with the strongest validated ranker; candidate-count sweep | Identify whether retrieval ceiling, ranking or safety causes loss; paired per-user deltas |
| 6 — multitask cycle | Separate tasks vs shared-bottom vs MMoE, loss weights and fusion; validation-only calibration | Demonstrated transfer benefit or a documented simpler-model win; quantify task tradeoffs |
| 7 — safety cycle | Written caption policy; independent annotations; representative and positive-enriched sets reported separately | Precision/recall/confusion matrix and uncertainty; false-positive review; no claim that captions establish whole-video safety |
| 8 — serving cycle | Flat vs HNSW on user-query embeddings; K/threads/concurrency sweep, longer steady-state run | Quality/latency tradeoff, response validity, error rate, memory/CPU and workload recorded |
| 9 — final package | Freeze choices; evaluate once under revised protocol; repeat training seeds; reproducible demo | Paired bootstrap intervals plus training-seed variability; all public claims trace to artifacts |

Do not sweep every combination. Repair deterministic defects first, run a modest baseline comparison, then choose one hypothesis per experiment. A practical first model baseline is a personalized single-task long-view model; MMoE must show value beyond it. Full scope does not require every complex component to win.

Retrieval improvements worth testing after correctness: normalize numeric inputs using training statistics; treat categorical integer codes as categorical; use separate field embeddings or disjoint index offsets (the current shared categorical table mixes independently encoded field IDs); remove accidental duplicate-item negatives; align positive-label selection and logQ with the actual sampling distribution. Mixed uniform/in-batch negatives are a later controlled experiment, not a guaranteed gain. Do not mark every unobserved user-item pair a known dislike. With 7,583 candidates, a bounded full-softmax reference may also be practical; measure its compute cost and handle multiple positives explicitly.

Use an oracle ordering of *observed relevant candidates* only as a diagnostic upper bound under the chosen offline metric. If that bound is weak, focus on retrieval; if it is much better than actual ranking, focus on ranking. Label the bound clearly; it is not deployable performance.

For uncertainty, store per-user outputs and bootstrap paired model differences at the user level. Use multiple training seeds for learning variability; bootstrap intervals alone do not measure that. Report absolute differences alongside relative lift. A positive point estimate without adequate support is not a proven win.

## Evidence that makes this interview-worthy

- A coherent end-to-end implementation and a short working demo, with honest scale and limitations.
- One defensible investigation: observed failure → hypothesis → controlled change → measured outcome → deployment decision.
- A baseline table demonstrating why the chosen model is appropriate, even if LR or a simple MLP wins.
- Tests that would catch the actual failures found here: timestamp leakage, feature mismatch, missing personalization, unequal evaluation policy, and invalid API recommendations.
- A quality/latency decision: the present FAISS evidence favors Flat. Keep HNSW as a measured scaling experiment, not an automatic production choice.
- A safety evaluation with genuine annotation provenance and separate representative versus enriched sampling claims.
- Ability to explain sparse feedback, distribution shift, sampled-softmax correction, duplicate negatives, negative transfer, calibration, cold start and online experiment design.

The current 30-second Locust run includes client think time and accepts an empty items list; its ~49 QPS is the observed workload throughput, not saturation capacity. Strengthen it with response checks and sustained concurrency sweeps before claiming scalability. Synthetic candidate-pool scale tests must be labeled synthetic.

Practice these questions aloud: Why did Recall@200 rise while NDCG@10 fell? Why does random exposure not solve every evaluation problem? Why did LR beat MMoE? What does logQ correct? When is a duplicate candidate a false negative? Why choose exact search? How do you prove offline/API parity? What does 97% label agreement with zero unsafe human labels show? What would you monitor after deployment? What evidence would make you abandon your preferred architecture?

An honest project description today is: “Built a two-stage short-video recommendation prototype with separate retrieval, ranking, safety and serving evaluation; investigated why neural components did not yet improve end-to-end quality.” After repairs, replace the investigation clause with the actual reproducible outcome. Do not use the stale +59% claim, call weak labels human ground truth, or describe this offline study as an A/B test.

Continue coding/DSA and ML fundamentals preparation alongside this work. The goal is a defensible project you understand deeply, not a particular cosmetic metric or an ever-growing model list.

## Primary references

- [TikTok target job](https://lifeattiktok.com/search/7686439502455212293): role fit and qualification expectations.
- [KuaiRand authors' documentation](https://kuairand.com/): dataset construction, Pure limitations and label semantics.
- [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml/): training/serving consistency and disciplined ML engineering.
- [Meta: scaling Instagram Explore recommendations](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/): staged retrieval/ranking and engineering tradeoffs; not evidence of this project's scale.
- [Google: original MMoE paper](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/): task relationships motivate single-task and shared-bottom controls.
- [TensorFlow Recommenders retrieval API](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval): accidental-hit removal and sampling-probability correction are distinct mechanisms.
- [Google: mixed-negative sampling](https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/): motivation for a later sampling ablation, not a guaranteed local improvement.
- [Google: sampling-bias-corrected recommendations](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/): sampled-objective correction.
- [Microsoft: unbiased offline replay evaluation](https://www.microsoft.com/en-us/research/publication/unbiased-offline-evaluation-of-contextual-bandit-based-news-article-recommendation-algorithms-2/): estimator assumptions and partial-feedback evaluation; its guarantees do not automatically apply to this NDCG computation.
