# Retrieval seed sweep

This experiment checks training-seed stability of retrieval on the existing exploratory random-exposure evaluation. It does not establish online improvement, select a new serving model, or repair the earlier exposure of this evaluation log during model development.

## Fixed protocol

- Five training seeds: 42, 43, 44, 45, 46, in that order; no best-seed selection.
- Eight epochs, batch size 1,024, learning rate 0.001, embedding dimension 64, temperature 0.05, long-view positives.
- Same seed-42 cohort of 2,000 users with observed long-view positives, full 7,583-item candidate pool, repeat items allowed.
- Seed 42 must reproduce the recorded Recall@200 within 1e-8 before later seeds run. A failure stops execution for investigation.
- Popularity and ItemCF are evaluated once on the same cohort; their recommendation functions have no random sampling. The cohort sampler is fixed.
- Report all training seeds, arithmetic mean and sample standard deviation (ddof=1).
- Compute 5,000 paired user-bootstrap replicates with bootstrap seed 42. Report intervals for each fitted model versus each baseline, and for seed-averaged per-user differences.

The latter interval is conditional on the five fitted models. It is not a confidence interval for an arbitrary newly trained model. Seed SD describes training variation separately; it is not a confidence interval. Do not interpret an interval crossing zero as equivalence. There is no multiplicity-adjusted hypothesis test or online outcome claim.

## Reproduce

```bash
make install
make test
# Existing processed data and original retrieval result must be present.
.venv/bin/python -m reelrank.eval.seed_sweep --name retrieval_seed_sweep
```

The default seeds are predeclared above. The runner refuses existing output directories: inspect any partial run before starting a distinctly named experiment. It does not automatically restart failed runs. The five full eight-epoch runs each completed in 30-32 seconds in the existing local environment, plus about two seconds for baselines; this is not a guaranteed runtime on another machine.

Each child process sets `REELRANK_ARTIFACTS_DIR` and `REELRANK_RESULTS_DIR` before imports. Model bundles live under `artifacts/<name>/seed<seed>/`; evidence lives under `results/<name>/seed<seed>/`. Baselines have their own directories. This isolates checkpoints, embeddings, Flat/HNSW indexes and result aliases - not just result filenames. The top-level serving artifacts and pre-existing top-level result files are hashed before execution and checked afterward.

`protocol.json` is written before training. Each run records cohort, relevance and candidate-pool fingerprints plus source/input hashes and checksummed per-user scores in its immutable run archive. The summary rejects incompatible cohorts, input fingerprints, hyperparameters, duplicate users, altered score files and source changes across training seeds.

To regenerate the summary without training:

```bash
.venv/bin/python -m reelrank.eval.seed_summary results/retrieval_seed_sweep
.venv/bin/python -m reelrank.eval.make_readme_table
```

See [the measured summary](../results/retrieval_seed_sweep/SUMMARY.md) and [machine-readable comparison](../results/retrieval_seed_sweep.json). End-to-end NDCG results and serving performance remain measurements of the original bundle; they are not averaged over this retrieval sweep.
