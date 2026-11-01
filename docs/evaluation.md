# Evaluation

| Purpose | Data | Use |
|---|---|---|
| Training | Standard log through 2022-04-21 | Fit model, feature vocabulary/bins and training aggregates |
| Selection | Standard log 2022-04-22 through 2022-04-28 | Select neural checkpoints and MMoE fusion weights |
| Additional available split | Standard log after 2022-04-28 | Prepared by the splitter; not the current results headline |
| Exploratory distribution-shift evaluation | Random-exposure log | Evaluate already selected models; previously inspected in older experiments |

The random log is **not a pristine final holdout**: earlier versions selected fusion weights on it. New code no longer does this, but cannot undo prior exposure. The current ranker comparison uses the same 400,000 training rows, 80,000 standard-validation rows and 80,000 random-evaluation rows, seed 42. Ranker family selection uses standard-validation long-view GAUC. Model checkpoints use validation log loss; MMoE fusion uses validation long-view AUC.

Retrieval methods use the same 2,000-user seed-42 cohort, long-view relevance and an **allow-repeat** candidate policy. Cascade comparisons use the same cohort and apply the same safety filter to both candidate generators. Training histories contain only events strictly before each training example; equal timestamps are excluded. KuaiRand-Pure histories are incomplete, and supplied static feature snapshots have their own temporal-availability limitations.

Full-pool NDCG measures overlap with **sparse observed positives**, not known relevance for all 7,583 videos or online user benefit. The median random-log user has observations for 22 videos. Positive-only retrieval/cascade metrics exclude users with no observed long-view positives; GAUC excludes users without both classes. We report user counts, positive hits and paired bootstrap intervals, not relative lift alone. Bootstrap uncertainty covers users, not repeated-training variation or model-selection uncertainty.

## Safety scope

This is an **unvalidated caption-only prototype**, not a video-safety system. Existing labels are 1,498 LLM-source rows and two heuristic fallbacks; the classifier's recorded F1 is about 0.11 against those labels. The 200-item review has all-safe human labels, so 97% agreement does not establish unsafe-content recall or validate annotation provenance.

New label runs version cache keys by policy/model/source, retry API fallbacks on subsequent runs, preserve existing human reviews, and do not turn `is_hate` into an unsafe label. The code repairs do not retroactively relabel the existing dataset. No paid relabeling was run during the correctness work.

The uncertain-band percentage is a **hypothetical offline routing count**; serving does not call an LLM. Missing safety scores pass through by explicit prototype policy. Dislike correlation is diagnostic only.

## Serving evidence

Load tests check nonempty unique recommendations, finite probabilities and valid timings. Report hardware, concurrency, duration and failures alongside QPS and p99. Locust includes client think time, so achieved throughput is not a maximum capacity claim. Earlier timings describe earlier artifacts and are superseded by a new run after retraining.
