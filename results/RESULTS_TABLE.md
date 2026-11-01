| Run | Stage / evaluation | Key metrics |
|---|---|---|
| `e2e_lr_no_safety` | e2e  | apply_safety=False, coverage_two_tower=0.0262, lift_vs_pop=0.3656, n_users=2000, ndcg@10_popularity=0.0026, ndcg@10_two_tower=0.0035, paired_bootstrap.confidence_level=0.95, paired_bootstrap.difference_ci95=[-0.00010947505929156876, 0.0020534712692531925], paired_bootstrap.mean_difference=0.0009457500385696628, paired_bootstrap.n_resamples=2000 |
| `e2e_lr_with_safety` | e2e  | apply_safety=True, coverage_two_tower=0.0270, lift_vs_pop=0.2854, n_users=2000, ndcg@10_popularity=0.0028, ndcg@10_two_tower=0.0036, paired_bootstrap.confidence_level=0.95, paired_bootstrap.difference_ci95=[-0.000264861529727631, 0.001891565315400187], paired_bootstrap.mean_difference=0.0007964381316195759, paired_bootstrap.n_resamples=2000 |
| `e2e_mmoe_no_safety` | e2e  | apply_safety=False, coverage_two_tower=0.0417, lift_vs_pop=0.1650, n_users=2000, ndcg@10_popularity=0.0025, ndcg@10_two_tower=0.0030, paired_bootstrap.confidence_level=0.95, paired_bootstrap.difference_ci95=[-0.0006973531353226777, 0.0014495805400423288], paired_bootstrap.mean_difference=0.0004205619474855734, paired_bootstrap.n_resamples=2000 |
| `e2e_mmoe_with_safety` | e2e  | apply_safety=True, coverage_two_tower=0.0415, lift_vs_pop=0.1378, n_users=2000, ndcg@10_popularity=0.0026, ndcg@10_two_tower=0.0030, paired_bootstrap.confidence_level=0.95, paired_bootstrap.difference_ci95=[-0.0007932175456436036, 0.0014402157716229106], paired_bootstrap.mean_difference=0.0003641577968301563, paired_bootstrap.n_resamples=2000 |
| `ranking_deepfm` | ranking random_exposure_exploratory | long_view.auc=0.6581877364373182, long_view.gauc=0.5989308736605529, long_view.logloss=0.4103180766105652, is_click.auc=0.641716577250757 |
| `ranking_deepfm_validation` | ranking standard_validation | long_view.auc=0.7282142134668667, long_view.gauc=0.6438954528273282, long_view.logloss=0.5493029356002808, is_click.auc=0.7230741406154252 |
| `ranking_lr` | ranking random_exposure_exploratory | long_view.auc=0.6558989906548438, long_view.gauc=0.5962679374706205, long_view.logloss=0.4459476261920577, is_click.auc=0.6364030981334416 |
| `ranking_lr_validation` | ranking standard_validation | long_view.auc=0.7296749794140058, long_view.gauc=0.6484461775374841, long_view.logloss=0.5525084797036336, is_click.auc=0.7241601565632529 |
| `ranking_mmoe` | ranking random_exposure_exploratory | long_view.auc=0.5923750717230547, long_view.gauc=0.5851750414495315, long_view.logloss=0.3721263110637665, is_click.auc=0.5636002352287536 |
| `ranking_mmoe_validation` | ranking standard_validation | long_view.auc=0.6738874269900168, long_view.gauc=0.6172564952031094, long_view.logloss=0.5798548460006714, is_click.auc=0.6666287543976456 |
| `retrieval_faiss` | ann  | hnsw_recall_vs_flat=0.9918, k=100, n_items=7583, n_queries=500, flat.batch_mean_ms=0.0779106940026395, flat.mean_ms=0.09999670373872505, flat.p50_ms=0.08233349944930524, flat.p99_ms=0.3940452150709455 |
| `retrieval_itemcf` | retrieval  | coverage=0.8820, hit_rate@10=0.0075, hit_rate@200=0.2220, hit_rate@50=0.0660, recall@10=0.0014, recall@200=0.0535 |
| `retrieval_popularity` | retrieval  | coverage=0.0264, hit_rate@10=0.0185, hit_rate@200=0.2395, hit_rate@50=0.0875, recall@10=0.0038, recall@200=0.0584 |
| `retrieval_seed_sweep` | retrieval_seed_sweep  | recall@200_mean=0.0667, recall@200_sample_sd=0.0020, popularity_relative_lift_pct=14.1148, popularity_paired_ci95=[0.0009239464983030143, 0.015758752437809307], itemcf_relative_lift_pct=24.6023, itemcf_paired_ci95=[0.005486882476418669, 0.020762915410663003] |
| `retrieval_two_tower` | retrieval  | coverage=0.3406, hit_rate@10=0.0240, hit_rate@200=0.2560, hit_rate@50=0.0885, recall@10=0.0042, recall@200=0.0696 |
| `safety_cascade` | safety  | blocked_classifier=112, blocked_rules=12, labeled_precision_proxy=0.7705, n_videos=7583, pct_to_llm=0.1718, uncertain_band_to_llm=1303, thresholds.high=0.7, thresholds.low=0.3 |
| `safety_classifier` | safety  | auc=0.6511, corr_safety_score_vs_hate_rate=-0.0223, f1=0.1081, n_test=375, n_train=1125, precision=0.0952 |
| `safety_human_kappa` | safety  | agreement=0.9700, cohen_kappa=0.0000, human_unsafe=0, judge_unsafe=6, n_labeled=200, n_spotcheck_rows=200, confusion_matrix.labels=['safe', 'unsafe'], confusion_matrix.matrix=[[194, 0], [6, 0]] |
| `serving_loadtest` | serving  | concurrency=8, errors=0, mean_ms=36.9154, n_requests=2890, p50_ms=31.0000, p95_ms=79.0000, hardware.platform=macOS-26.5.1-arm64-arm-64bit-Mach-O, hardware.processor=arm, hardware.python=3.13.15 |

Legacy e2e_no_safety/e2e_with_safety JSON files are retained for history and excluded from this current table. Safety records predate the correctness fixes; labels were not regenerated.

Random-exposure results are exploratory reuse. Positive lift requires its paired interval and scope; no online benefit is established.
