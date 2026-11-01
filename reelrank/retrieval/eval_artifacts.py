"""Evaluate a saved two-tower checkpoint without retraining (FAISS + Recall@K)."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.eval.runlog import write_result
from reelrank.retrieval.baselines import build_user_positives, evaluate_retrieval, evaluation_cohort
from reelrank.retrieval.index import build_and_benchmark
from reelrank.retrieval.train import InteractionDataset, _fix_maps
from reelrank.retrieval.two_tower import TwoTower


def eval_saved(
    max_eval_users: int = 2000,
    seed: int = 42,
    skip_ann_bench: bool = False,
) -> None:
    faiss.omp_set_num_threads(1)
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

    item_embs = np.load(ARTIFACTS / "item_embeddings.npy").astype(np.float32)
    video_ids = json.loads((ARTIFACTS / "item_embedding_ids.json").read_text())

    if not skip_ann_bench:
        ann = build_and_benchmark(item_embs, seed=seed)
        write_result(
            "retrieval_faiss",
            "ann",
            ann,
            config={"index": "flat_vs_hnsw", "source": "eval_artifacts"},
            seed=seed,
        )
        print("ann", {k: ann[k] for k in ann if k in ("hnsw_recall_vs_flat", "hnsw_error", "n_items")})

    id_maps = json.loads((PROCESSED / "id_maps.json").read_text())
    user_id_map, _ = _fix_maps(id_maps)
    users_df = pd.read_parquet(PROCESSED / "users.parquet")
    videos_df = pd.read_parquet(PROCESSED / "videos.parquet")
    histories = json.loads((PROCESSED / "histories.json").read_text())
    ds = InteractionDataset(
        pd.read_parquet(PROCESSED / "interactions_train.parquet").head(1),
        user_id_map,
        {},
        users_df,
        videos_df,
        histories,
        meta["user_cat_cols"],
        meta["video_cat_cols"],
        meta["user_num_cols"],
        meta["history_n"],
        training=False,
    )

    test = pd.read_parquet(PROCESSED / "interactions_test_random.parquet")
    user_rel = build_user_positives(test, label="long_view")
    user_rel = evaluation_cohort(user_rel, max_eval_users, seed)
    users = list(user_rel)

    index = faiss.read_index(str(ARTIFACTS / "index_flat.faiss"))
    user_recs: dict[int, list[int]] = {}
    with torch.no_grad():
        for uid in tqdm(users, desc="eval users"):
            cat, num, hist = ds._user_feats(uid)
            uemb = model.encode_user(
                torch.tensor([user_id_map.get(str(uid), user_id_map.get(uid, 0))], dtype=torch.long),
                torch.tensor(cat, dtype=torch.long).unsqueeze(0),
                torch.tensor(num, dtype=torch.float32).unsqueeze(0),
                torch.tensor(hist, dtype=torch.long).unsqueeze(0),
            )
            _, idxs = index.search(uemb.cpu().numpy().astype(np.float32), 200)
            user_recs[uid] = [int(video_ids[j]) for j in idxs[0] if j >= 0]

    metrics = evaluate_retrieval(user_rel, user_recs, catalogue_size=len(video_ids))
    write_result(
        "retrieval_two_tower",
        "retrieval",
        metrics,
        config={
            "method": "two_tower",
            "epochs": "from_checkpoint",
            "emb_dim": meta["emb_dim"],
            "temperature": meta["temperature"],
            "max_eval_users": max_eval_users,
            "source": "eval_artifacts",
            "training_label": meta.get("training_label", "is_click"),
            "evaluation_label": "long_view",
            "seen_item_policy": "allow_repeats",
        },
        seed=seed,
    )
    print("two_tower", metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-eval-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ann-bench", action="store_true")
    args = parser.parse_args()
    eval_saved(args.max_eval_users, args.seed, args.skip_ann_bench)


if __name__ == "__main__":
    main()
