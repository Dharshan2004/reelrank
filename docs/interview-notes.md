# Interview notes

The useful story is the investigation: a complex model initially lacked user inputs and used inconsistent features; tests and a shared contract fixed that, controlled comparisons selected a simpler ranker, and uncertainty limits performance claims. Explain why shortlist recall can improve while final ranking loses, why random exposure leaves missing feedback, why logQ does not fix duplicate negatives, and why exact search can beat ANN at small scale.

## Correctness work

- One serialized, training-fitted feature transformer is used by training, offline evaluation and the API; unseen IDs/categories map to zero.
- Rankers receive user/item information on matched fields; LR one-hot-encodes categories rather than treating IDs as continuous numbers.
- Retrieval training uses timestamp-correct histories and masks duplicate-item accidental negatives. Its logQ matches the selected positive sampling distribution.
- Tensor caching avoids per-example pandas lookups during retrieval training.
- Shared scoring and safety code prevent offline/API behavior drift. Tests cover LR/MMoE parity, unknown users, chronology, feature serialization, duplicate negatives and invalid responses.
- Each new result keeps an immutable `results/runs/<timestamp>-<run>/` record, source snapshot, configuration, source/data/model hashes and per-user cascade outputs. Top-level JSON files are latest-result aliases.

## Retrieval stability follow-up

Five predeclared training seeds produced Recall@200 0.06666 ± 0.00204 (sample SD), versus popularity 0.05842 and ItemCF 0.05350 on the same 2,000 users. The seed-averaged user-bootstrap differences exclude zero; three of five individual-seed intervals versus popularity cross zero. This supports an exploratory retrieval improvement across the measured runs, not a guaranteed gain for new training seeds or an end-to-end/online result. No best seed was selected.

Suggested résumé bullet: “Evaluated two-tower retrieval across five training seeds on a fixed 2,000-user cohort, achieving mean Recall@200 of 0.0667 ± 0.0020 versus 0.0584 for popularity.” Here ± is sample standard deviation, not a confidence interval.
