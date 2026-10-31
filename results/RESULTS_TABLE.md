| Run | Stage | Key metrics |
|---|---|---|
| `e2e_no_safety` | e2e | apply_safety=False, coverage_two_tower=0.0220, lift_vs_pop=0.4983, n_users=600, ndcg@10_popularity_mmoe=0.0029, ndcg@10_two_tower_mmoe=0.0043 |
| `e2e_with_safety` | e2e | apply_safety=True, coverage_two_tower=0.0216, lift_vs_pop=0.5876, n_users=600, ndcg@10_popularity_mmoe=0.0029, ndcg@10_two_tower_mmoe=0.0046 |
| `ranking_deepfm` | ranking | is_click.auc=0.510923080098223, is_click.gauc=0.514411544971561, is_click.logloss=1.5905382633209229, is_like.auc=0.5317245411181489 |
| `ranking_lr` | ranking | is_click.auc=0.5795071712192681, is_click.gauc=0.5857334900562214, is_click.logloss=0.4819810688495636, is_like.auc=0.613251701766724 |
| `ranking_mmoe` | ranking | fusion.auc_long_view=0.5808569300685058, fusion.gauc_long_view=0.5812802153328303, fusion_best.auc_long=0.5814683888917138, fusion_best.w_click=0.25 |
| `retrieval_faiss` | ann | hnsw_recall_vs_flat=0.9816, k=100, n_items=7583, n_queries=500, flat.batch_mean_ms=0.015989895999913035, flat.mean_ms=0.06821044400157916, flat.p50_ms=0.06712899994454347, flat.p99_ms=0.08465008984785524 |
| `retrieval_itemcf` | retrieval | coverage=0.8803, hit_rate@10=0.0250, hit_rate@200=0.2360, hit_rate@50=0.0885, recall@10=0.0047, recall@200=0.0592 |
| `retrieval_popularity` | retrieval | coverage=0.0264, hit_rate@10=0.0155, hit_rate@200=0.2285, hit_rate@50=0.0825, recall@10=0.0037, recall@200=0.0542 |
| `retrieval_two_tower` | retrieval | coverage=0.5362, hit_rate@10=0.0093, hit_rate@200=0.2293, hit_rate@50=0.0747, recall@10=0.0024, recall@200=0.0631 |
| `safety_cascade` | safety | blocked_classifier=67, blocked_rules=12, labeled_precision_proxy=1.0000, n_videos=7583, pct_to_llm=0.9243, uncertain_band_to_llm=7009, thresholds.high=0.7, thresholds.low=0.3 |
| `safety_classifier` | safety | auc=0.5322, corr_safety_score_vs_hate_rate=0.1253, f1=0.2414, n_test=375, n_train=1125, precision=0.2625 |
| `serving_loadtest` | serving | n_requests=100, concurrency=4, errors=0, qps=40.1118, p50_ms=98.0281, p95_ms=121.6057, hardware.platform=Linux-6.12.94+-x86_64-with-glibc2.39, hardware.processor=x86_64, hardware.python=3.12.3 |
