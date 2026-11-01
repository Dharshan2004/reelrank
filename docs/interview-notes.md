# Interview notes

The useful story is the investigation: a complex model initially lacked user inputs and used inconsistent features; tests and a shared contract fixed that, controlled comparisons selected a simpler ranker, and uncertainty limits performance claims. Explain why shortlist recall can improve while final ranking loses, why random exposure leaves missing feedback, why logQ does not fix duplicate negatives, and why exact search can beat ANN at small scale.

## Correctness work

- One serialized, training-fitted feature transformer is used by training, offline evaluation and the API; unseen IDs/categories map to zero.
- Rankers receive user/item information on matched fields; LR one-hot-encodes categories rather than treating IDs as continuous numbers.
- Retrieval training uses timestamp-correct histories and masks duplicate-item accidental negatives. Its logQ matches the selected positive sampling distribution.
- Tensor caching avoids per-example pandas lookups during retrieval training.
- Shared scoring and safety code prevent offline/API behavior drift. Tests cover LR/MMoE parity, unknown users, chronology, feature serialization, duplicate negatives and invalid responses.
- Each new result keeps an immutable `results/runs/<timestamp>-<run>/` record, source snapshot, configuration, source/data/model hashes and per-user cascade outputs. Top-level JSON files are latest-result aliases.
