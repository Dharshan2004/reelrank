"""Retrieval baselines: global popularity and ItemCF."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.eval.metrics import (
    catalogue_coverage,
    hit_rate_at_k,
    mean_metric,
    recall_at_k,
)
from reelrank.eval.runlog import write_result


def build_user_positives(df: pd.DataFrame, label: str = "long_view") -> dict[int, list[int]]:
    pos = df[df[label] == 1]
    out: dict[int, list[int]] = defaultdict(list)
    for uid, vid in zip(pos["user_id"].tolist(), pos["video_id"].tolist()):
        out[int(uid)].append(int(vid))
    # unique preserve order
    return {u: list(dict.fromkeys(vs)) for u, vs in out.items()}


def popularity_recommend(
    train: pd.DataFrame,
    users: list[int],
    k: int = 200,
    label: str = "is_click",
) -> dict[int, list[int]]:
    scores = train.groupby("video_id")[label].sum().sort_values(ascending=False)
    top = scores.index.astype(int).tolist()[:k]
    return {int(u): top for u in users}


def itemcf_recommend(
    train: pd.DataFrame,
    users: list[int],
    k: int = 200,
    label: str = "long_view",
    top_neighbors: int = 50,
) -> dict[int, list[int]]:
    pos = train[train[label] == 1][["user_id", "video_id"]].drop_duplicates()
    users_u = sorted(pos["user_id"].unique())
    items_u = sorted(pos["video_id"].unique())
    u_index = {u: i for i, u in enumerate(users_u)}
    i_index = {v: i for i, v in enumerate(items_u)}
    rows = [u_index[u] for u in pos["user_id"]]
    cols = [i_index[v] for v in pos["video_id"]]
    mat = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(users_u), len(items_u)),
    )
    # item-item cosine via normalized columns
    item_mat = normalize(mat.T, norm="l2", axis=1)
    sims = item_mat @ item_mat.T  # items x items
    # For each user, score = sum sim(item, history)
    user_hist = defaultdict(list)
    for u, v in zip(pos["user_id"], pos["video_id"]):
        user_hist[int(u)].append(int(v))

    recs: dict[int, list[int]] = {}
    item_ids = np.array(items_u)
    for uid in users:
        hist = user_hist.get(int(uid), [])
        if not hist:
            # cold: fall back to popularity among items
            pop = np.asarray(mat.sum(axis=0)).ravel()
            order = np.argsort(-pop)[:k]
            recs[int(uid)] = item_ids[order].tolist()
            continue
        score = np.zeros(len(items_u), dtype=np.float32)
        seen = set()
        for v in hist:
            if v not in i_index:
                continue
            seen.add(v)
            row = sims.getrow(i_index[v])
            # top neighbors only for speed
            if row.nnz > top_neighbors:
                idx = row.indices
                data = row.data
                keep = np.argpartition(data, -top_neighbors)[-top_neighbors:]
                score[idx[keep]] += data[keep]
            else:
                score[row.indices] += row.data
        for v in seen:
            if v in i_index:
                score[i_index[v]] = -1e9
        order = np.argsort(-score)[:k]
        recs[int(uid)] = item_ids[order].tolist()
    return recs


def evaluate_retrieval(
    user_rel: dict[int, list[int]],
    user_recs: dict[int, list[int]],
    catalogue_size: int,
    ks: tuple[int, ...] = (10, 50, 200),
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = mean_metric(user_rel, user_recs, k, recall_at_k)
        metrics[f"hit_rate@{k}"] = mean_metric(user_rel, user_recs, k, hit_rate_at_k)
    metrics["coverage"] = catalogue_coverage(user_recs.values(), catalogue_size)
    return metrics


def run_baselines(max_eval_users: int = 2000, seed: int = 42) -> None:
    train = pd.read_parquet(PROCESSED / "interactions_train.parquet")
    test = pd.read_parquet(PROCESSED / "interactions_test_random.parquet")
    videos = pd.read_parquet(PROCESSED / "videos.parquet")
    catalogue_size = int(videos["video_id"].nunique())

    user_rel = build_user_positives(test, label="long_view")
    rng = np.random.default_rng(seed)
    users = sorted(user_rel.keys())
    if len(users) > max_eval_users:
        users = sorted(rng.choice(users, size=max_eval_users, replace=False).tolist())
    user_rel = {u: user_rel[u] for u in users}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pop_recs = popularity_recommend(train, users, k=200)
    pop_metrics = evaluate_retrieval(user_rel, pop_recs, catalogue_size)
    write_result(
        "retrieval_popularity",
        "retrieval",
        pop_metrics,
        config={"method": "popularity", "max_eval_users": max_eval_users},
        seed=seed,
    )
    print("popularity", pop_metrics)

    cf_recs = itemcf_recommend(train, users, k=200)
    cf_metrics = evaluate_retrieval(user_rel, cf_recs, catalogue_size)
    write_result(
        "retrieval_itemcf",
        "retrieval",
        cf_metrics,
        config={"method": "itemcf", "max_eval_users": max_eval_users},
        seed=seed,
    )
    print("itemcf", cf_metrics)

    # Persist popularity list for serving fallback
    pop = train.groupby("video_id")["is_click"].sum().sort_values(ascending=False)
    (ARTIFACTS / "popularity_top.json").write_text(
        json.dumps([int(x) for x in pop.index.tolist()[:500]])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-eval-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_baselines(args.max_eval_users, args.seed)


if __name__ == "__main__":
    main()
