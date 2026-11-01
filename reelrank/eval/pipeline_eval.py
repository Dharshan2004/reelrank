"""End-to-end offline cascade: retrieve -> safety filter -> rank -> NDCG@10."""

from __future__ import annotations

import argparse
import json

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.eval.metrics import catalogue_coverage, mean_metric, ndcg_at_k
from reelrank.eval.runlog import config_hash, write_result
from reelrank.ranking.inference import SAFETY_THRESHOLD, filter_candidates, load_ranker_artifact, rank_candidates
from reelrank.retrieval.baselines import build_user_positives, popularity_recommend
from reelrank.retrieval.two_tower import TwoTower


def _load_two_tower():
    ckpt = torch.load(ARTIFACTS / "two_tower.pt", map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model = TwoTower(
        n_users=meta["n_users"],
        n_videos=meta["n_videos"],
        n_user_cats=meta["n_user_cats"],
        n_video_cats=meta["n_video_cats"],
        user_num_dim=len(meta["user_num_cols"]),
        emb_dim=meta["emb_dim"],
        temperature=meta["temperature"],
        history_n=meta["history_n"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


def _user_batch(uid, meta, users_df, histories, user_id_map):
    user_cat_cols = meta["user_cat_cols"]
    user_num_cols = meta["user_num_cols"]
    if uid in users_df.index:
        row = users_df.loc[uid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cat = np.array([int(row.get(c, 0)) for c in user_cat_cols] or [0], dtype=np.int64)
        num = np.array([float(row.get(c, 0.0)) for c in user_num_cols] or [0.0], dtype=np.float32)
    else:
        cat = np.zeros(max(len(user_cat_cols), 1), dtype=np.int64)
        num = np.zeros(max(len(user_num_cols), 1), dtype=np.float32)
    hist = histories.get(str(uid), [])
    hist_arr = np.zeros(meta["history_n"], dtype=np.int64)
    if hist:
        take = hist[-meta["history_n"] :]
        hist_arr[-len(take) :] = np.array(take, dtype=np.int64)
    return {
        "user_idx": torch.tensor([user_id_map.get(str(uid), user_id_map.get(uid, 0))], dtype=torch.long),
        "user_cat": torch.tensor(cat, dtype=torch.long).unsqueeze(0),
        "user_num": torch.tensor(num, dtype=torch.float32).unsqueeze(0),
        "hist_idx": torch.tensor(hist_arr, dtype=torch.long).unsqueeze(0),
    }


def paired_bootstrap(treatment: np.ndarray, baseline: np.ndarray, seed: int = 42, n_bootstrap: int = 2000) -> dict:
    """Resample paired users, retaining shared user difficulty across methods."""
    treatment = np.asarray(treatment, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if treatment.shape != baseline.shape or treatment.ndim != 1 or not len(treatment):
        raise ValueError("Bootstrap requires nonempty paired one-dimensional scores.")
    rng = np.random.default_rng(seed)
    differences, lifts = [], []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(treatment), size=len(treatment))
        a, b = treatment[sample].mean(), baseline[sample].mean()
        differences.append(float(a - b))
        if b > 0:
            lifts.append(float(a / b - 1))
    return {"n_resamples": n_bootstrap, "confidence_level": 0.95,
            "mean_difference": float((treatment - baseline).mean()),
            "difference_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "relative_lift_ci95": np.quantile(lifts, [0.025, 0.975]).tolist() if len(lifts) == n_bootstrap else None,
            "zero_baseline_resamples": n_bootstrap - len(lifts)}


def run_pipeline(
    max_users: int = 2000,
    retrieve_k: int = 200,
    top_k: int = 10,
    apply_safety: bool = True,
    seed: int = 42,
    ranker: str = "lr",
) -> dict:
    id_meta = json.loads((PROCESSED / "id_maps.json").read_text())
    histories = json.loads((PROCESSED / "histories.json").read_text())
    user_id_map = {**id_meta["user_id_map"], **{int(k): v for k, v in id_meta["user_id_map"].items()}}
    users_df = pd.read_parquet(PROCESSED / "users.parquet").set_index("user_id")
    videos_df = pd.read_parquet(PROCESSED / "videos.parquet").set_index("video_id")
    train = pd.read_parquet(PROCESSED / "interactions_train.parquet")
    test = pd.read_parquet(PROCESSED / "interactions_test_random.parquet")

    model, meta = _load_two_tower()
    item_ids = json.loads((ARTIFACTS / "item_embedding_ids.json").read_text())
    index = faiss.read_index(str(ARTIFACTS / "index_flat.faiss"))

    ranking = load_ranker_artifact(ARTIFACTS, ranker)

    safety = None
    if apply_safety:
        safety_path = ARTIFACTS / "video_safety_scores.parquet"
        if not safety_path.exists():
            raise FileNotFoundError("Safety-enabled evaluation requires video_safety_scores.parquet; run make train-safety or use --no-safety.")
        safety = pd.read_parquet(safety_path).set_index("video_id")

    user_rel = build_user_positives(test, label="long_view")
    rng = np.random.default_rng(seed)
    users = sorted(user_rel.keys())
    if len(users) > max_users:
        users = sorted(rng.choice(users, size=max_users, replace=False).tolist())
    user_rel = {u: user_rel[u] for u in users}

    # Popularity floor -> same ranker
    pop_recs_raw = popularity_recommend(train, users, k=retrieve_k)

    def retrieve(uid: int) -> list[int]:
        batch = _user_batch(uid, meta, users_df, histories, user_id_map)
        with torch.no_grad():
            uemb = model.encode_user(
                batch["user_idx"], batch["user_cat"], batch["user_num"], batch["hist_idx"]
            )
        _, idxs = index.search(uemb.cpu().numpy().astype(np.float32), retrieve_k)
        cands = [int(item_ids[j]) for j in idxs[0] if j >= 0]
        return cands

    tt_recs = {}
    pop_ranked = {}
    for uid in tqdm(users, desc="pipeline"):
        cands = retrieve(uid)
        tt_recs[uid] = rank_candidates(ranking, uid, filter_candidates(cands, safety), users_df, videos_df, top_k)
        pop_ranked[uid] = rank_candidates(ranking, uid, filter_candidates(pop_recs_raw[uid][:retrieve_k], safety), users_df, videos_df, top_k)

    metrics = {
        "ranker": ranking.name,
        "ndcg@10_two_tower": mean_metric(user_rel, tt_recs, top_k, ndcg_at_k),
        "ndcg@10_popularity": mean_metric(user_rel, pop_ranked, top_k, ndcg_at_k),
        "coverage_two_tower": catalogue_coverage(tt_recs.values(), catalogue_size=len(item_ids)),
        "apply_safety": apply_safety,
        "n_users": len(users),
        "retrieve_k": retrieve_k,
    }
    if metrics["ndcg@10_popularity"] > 0:
        metrics["lift_vs_pop"] = (
            metrics["ndcg@10_two_tower"] / metrics["ndcg@10_popularity"] - 1.0
        )
    per_user = pd.DataFrame([
        {"user_id": uid, "ndcg_two_tower": ndcg_at_k(user_rel[uid], tt_recs[uid], top_k),
         "ndcg_popularity": ndcg_at_k(user_rel[uid], pop_ranked[uid], top_k),
         "positive_hits_two_tower": len(set(user_rel[uid]) & set(tt_recs[uid])),
         "positive_hits_popularity": len(set(user_rel[uid]) & set(pop_ranked[uid])),
         "n_relevant": len(set(user_rel[uid])),
         "recommendations_two_tower": tt_recs[uid], "recommendations_popularity": pop_ranked[uid]}
        for uid in users
    ])
    if per_user.empty:
        raise ValueError("No users with observed long_view positives in the evaluation set.")
    metrics["paired_bootstrap"] = paired_bootstrap(per_user.ndcg_two_tower.to_numpy(), per_user.ndcg_popularity.to_numpy(), seed=seed)
    for method in ("two_tower", "popularity"):
        hits = per_user[f"positive_hits_{method}"]
        metrics[f"positive_hits_{method}"] = int(hits.sum())
        metrics[f"users_with_positive_hit_{method}"] = int((hits > 0).sum())
    run_id = f"e2e_{ranking.name}_with_safety" if apply_safety else f"e2e_{ranking.name}_no_safety"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    per_user_path = ARTIFACTS / f"{run_id}_per_user.parquet"
    per_user.to_parquet(per_user_path, index=False)
    write_result(run_id, "e2e", metrics, config={
        "ranker": ranking.name, "retrieve_k": retrieve_k, "top_k": top_k, "apply_safety": apply_safety,
        "safety_threshold": SAFETY_THRESHOLD, "fusion_weights": list(ranking.fusion_weights),
        "feature_transformer_hash": config_hash(ranking.transformer.to_dict()), "index": "flat",
        "max_users": max_users, "evaluation_file": "interactions_test_random.parquet",
        "cohort": "users with at least one observed long_view positive",
    }, seed=seed, extra={"per_user_scores": str(per_user_path),
        "interpretation": "Observed-positive offline ranking; missing user-item outcomes are not observed negatives. Bootstrap covers user sampling, not training-seed variation."})
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--ranker", choices=["lr", "mmoe"], default="lr")
    args = parser.parse_args()
    run_pipeline(max_users=args.max_users, apply_safety=not args.no_safety, seed=args.seed, ranker=args.ranker)
    if not args.no_safety:
        run_pipeline(max_users=args.max_users, apply_safety=False, seed=args.seed, ranker=args.ranker)


if __name__ == "__main__":
    main()
