# Retrieval seed sweep

Fixed cohort; fixed hyperparameters; no best-seed selection.

| Training seed | Recall@200 |
|---|---:|
| 42 | 0.069556 |
| 43 | 0.065714 |
| 44 | 0.064123 |
| 45 | 0.066361 |
| 46 | 0.067564 |

Mean ± sample SD: **0.066664 ± 0.002039**.

| Baseline | Recall@200 | Mean relative lift | Paired absolute-difference 95% CI* |
|---|---:|---:|---|
| popularity | 0.058418 | +14.11% | [0.000924, 0.015759] |
| itemcf | 0.053501 | +24.60% | [0.005487, 0.020763] |

*Bootstrap of seed-averaged per-user recall differences, 5,000 user resamples. Conditional on these fitted models; not an interval for a newly trained model or online utility. Per-seed intervals are in the JSON summary. An interval crossing zero is inconclusive, not evidence of equivalence.

Random-exposure evaluation remains exploratory; this does not repair prior holdout exposure. The original seed-42 serving model is retained. End-to-end ranking was not rerun and its prior NDCG interval still crosses zero.
