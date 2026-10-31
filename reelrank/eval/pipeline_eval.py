"""End-to-end offline cascade: retrieve -> safety filter -> rank -> NDCG@10."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.eval.metrics import catalogue_coverage, mean_metric, ndcg_at_k
from reelrank.eval.runlog import write_result
from reelrank.ranking.fusion import fuse
from reelrank.ranking.models import MMoE
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


def run_pipeline(
    max_users: int = 1000,
    retrieve_k: int = 200,
    top_k: int = 10,
    apply_safety: bool = True,
    seed: int = 42,
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

    mmoe_ckpt = torch.load(ARTIFACTS / "mmoe.pt", map_location="cpu", weights_only=False)
    ranker = MMoE(mmoe_ckpt["field_dims"], n_experts=mmoe_ckpt["n_experts"])
    ranker.load_state_dict(mmoe_ckpt["state_dict"])
    ranker.eval()
    field_cols = mmoe_ckpt["field_cols"]
    fusion_w = tuple(mmoe_ckpt.get("fusion_weights", [0.3, 0.5, 0.2]))

    safety = None
    if apply_safety and (ARTIFACTS / "video_safety_scores.parquet").exists():
        safety = pd.read_parquet(ARTIFACTS / "video_safety_scores.parquet").set_index("video_id")

    user_rel = build_user_positives(test, label="long_view")
    rng = np.random.default_rng(seed)
    users = sorted(user_rel.keys())
    if len(users) > max_users:
        users = sorted(rng.choice(users, size=max_users, replace=False).tolist())
    user_rel = {u: user_rel[u] for u in users}

    # Popularity floor -> same ranker
    pop_recs_raw = popularity_recommend(train, users, k=retrieve_k)

    field_dims = mmoe_ckpt["field_dims"]

    def rank_candidates(uid: int, cand_ids: list[int]) -> list[int]:
        if not cand_ids:
            return []
        rows = []
        for vid in cand_ids:
            feats = {}
            if uid in users_df.index:
                urow = users_df.loc[uid]
                if isinstance(urow, pd.DataFrame):
                    urow = urow.iloc[0]
            else:
                urow = None
            vrow = videos_df.loc[vid] if vid in videos_df.index else None
            for c in field_cols:
                if c in ("log_dur_bin", "log_pop_bin"):
                    if vrow is None:
                        feats[c] = 0
                    elif c == "log_dur_bin":
                        feats[c] = int(min(10, np.log1p(float(vrow.get("video_duration", 0) or 0)))) + 1
                    else:
                        feats[c] = int(min(10, np.log1p(float(vrow.get("pop_clicks", 0) or 0)))) + 1
                elif urow is not None and c in urow.index:
                    feats[c] = int(urow.get(c, 0) or 0)
                elif vrow is not None and c in vrow.index:
                    feats[c] = int(vrow.get(c, 0) or 0)
                else:
                    feats[c] = 0
            # Clamp to embedding table sizes saved at train time
            row = []
            for i, c in enumerate(field_cols):
                row.append(int(np.clip(feats[c], 0, field_dims[i])))
            rows.append(row)
        x = torch.tensor(np.array(rows, dtype=np.int64), dtype=torch.long)
        with torch.no_grad():
            proba = torch.sigmoid(ranker(x)).numpy()
        scores = fuse(proba, fusion_w)
        order = np.argsort(-scores)
        return [cand_ids[i] for i in order[:top_k]]

    def retrieve(uid: int) -> list[int]:
        batch = _user_batch(uid, meta, users_df, histories, user_id_map)
        with torch.no_grad():
            uemb = model.encode_user(
                batch["user_idx"], batch["user_cat"], batch["user_num"], batch["hist_idx"]
            )
        _, idxs = index.search(uemb.cpu().numpy().astype(np.float32), retrieve_k)
        cands = [int(item_ids[j]) for j in idxs[0] if j >= 0]
        if apply_safety and safety is not None:
            filtered = []
            for vid in cands:
                if vid in safety.index:
                    row = safety.loc[vid]
                    if bool(row["rule_hit"]) or float(row["safety_score"]) >= 0.5:
                        continue
                filtered.append(vid)
            cands = filtered
        return cands

    tt_recs = {}
    pop_ranked = {}
    for uid in tqdm(users, desc="pipeline"):
        cands = retrieve(uid)
        tt_recs[uid] = rank_candidates(uid, cands)
        pop_ranked[uid] = rank_candidates(uid, pop_recs_raw[uid][:retrieve_k])

    metrics = {
        "ndcg@10_two_tower_mmoe": mean_metric(user_rel, tt_recs, top_k, ndcg_at_k),
        "ndcg@10_popularity_mmoe": mean_metric(user_rel, pop_ranked, top_k, ndcg_at_k),
        "coverage_two_tower": catalogue_coverage(tt_recs.values(), catalogue_size=len(item_ids)),
        "apply_safety": apply_safety,
        "n_users": len(users),
        "retrieve_k": retrieve_k,
    }
    if metrics["ndcg@10_popularity_mmoe"] > 0:
        metrics["lift_vs_pop"] = (
            metrics["ndcg@10_two_tower_mmoe"] / metrics["ndcg@10_popularity_mmoe"] - 1.0
        )
    run_id = "e2e_with_safety" if apply_safety else "e2e_no_safety"
    write_result(run_id, "e2e", metrics, config={"retrieve_k": retrieve_k, "top_k": top_k}, seed=seed)
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-safety", action="store_true")
    args = parser.parse_args()
    run_pipeline(max_users=args.max_users, apply_safety=not args.no_safety, seed=args.seed)
    if not args.no_safety:
        run_pipeline(max_users=args.max_users, apply_safety=False, seed=args.seed)


if __name__ == "__main__":
    main()
