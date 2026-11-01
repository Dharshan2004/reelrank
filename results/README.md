# Results index

Start with [RESULTS_TABLE.md](RESULTS_TABLE.md) and the [evaluation protocol](../docs/evaluation.md). Values are offline measurements, not online A/B results.

| Files | Interpretation |
|---|---|
| `retrieval_seed_sweep.json`, `retrieval_seed_sweep/SUMMARY.md` | Five-seed retrieval mean, sample SD, per-seed and seed-averaged paired intervals |
| `retrieval_two_tower.json`, `retrieval_popularity.json`, `retrieval_itemcf.json` | Matched 2,000-user retrieval comparison |
| `ranking_*_validation.json` | Standard-validation model comparison |
| `ranking_lr.json`, `ranking_deepfm.json`, `ranking_mmoe.json` | Exploratory random-exposure evaluation |
| `e2e_lr_*.json`, `e2e_mmoe_*.json` | Current cascade comparisons; read paired intervals alongside lift |
| `retrieval_faiss.json` | Exact/approximate search benchmark |
| `serving_loadtest.json` | Latest local load test, including hardware and concurrency |
| `safety_*.json` | Existing safety diagnostics; labels were not regenerated during correctness work |
| `e2e_no_safety.json`, `e2e_with_safety.json` | Historical cascade results, excluded from the current table |

Top-level JSON files are latest-result aliases, not immutable records. `runs/` preserves timestamped results, source snapshots and per-user outputs where available. It includes unsuccessful runs; consult their metrics before interpreting them. Historical source hashes describe the code used for that run, not today's checkout.

Keep the existing immutable records intact. Future large datasets, checkpoints, and bulk experiment archives belong in external artifact storage; retain compact result summaries and provenance here.
