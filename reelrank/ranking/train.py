"""Train ranking baselines (LR, DeepFM) and MMoE multi-task ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.eval.metrics import auc_score, binary_logloss, gauc
from reelrank.eval.runlog import write_result
from reelrank.ranking.fusion import fuse, fusion_grid
from reelrank.ranking.models import DeepFM, MMoE

TASKS = ["is_click", "long_view", "is_like"]


def _build_feature_frame(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    videos: pd.DataFrame,
    user_cat_cols: list[str],
    video_cat_cols: list[str],
    user_num_cols: list[str],
) -> pd.DataFrame:
    u = users.set_index("user_id")
    v = videos.set_index("video_id")
    df = interactions.copy()
    for c in user_cat_cols + user_num_cols:
        if c in u.columns:
            df[c] = df["user_id"].map(u[c]).fillna(0)
    for c in video_cat_cols:
        if c in v.columns:
            df[c] = df["video_id"].map(v[c]).fillna(0)
    df["log_dur"] = np.log1p(df["video_id"].map(v.get("video_duration", pd.Series(dtype=float))).fillna(0))
    df["log_pop"] = np.log1p(df["video_id"].map(v.get("pop_clicks", pd.Series(dtype=float))).fillna(0))
    return df


class RankDataset(Dataset):
    def __init__(self, df: pd.DataFrame, field_cols: list[str], tasks: list[str] = TASKS):
        self.x = df[field_cols].fillna(0).astype(np.int64).values
        self.y = df[tasks].astype(np.float32).values
        self.users = df["user_id"].astype(np.int64).values

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.users[idx], dtype=torch.long),
        )


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def train_lr(train_df, val_df, feature_cols, seed: int = 42) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    metrics = {}
    models = {}
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].fillna(0).astype(np.float32))
    x_val = scaler.transform(val_df[feature_cols].fillna(0).astype(np.float32))
    for task in TASKS:
        clf = LogisticRegression(max_iter=200, n_jobs=1, random_state=seed)
        clf.fit(x_train, train_df[task].astype(int))
        proba = clf.predict_proba(x_val)[:, 1]
        metrics[task] = {
            "auc": auc_score(val_df[task], proba),
            "gauc": gauc(val_df[task], proba, val_df["user_id"]),
            "logloss": binary_logloss(val_df[task], proba),
        }
        models[task] = clf
    joblib.dump({"models": models, "scaler": scaler, "features": feature_cols}, ARTIFACTS / "rank_lr.joblib")
    write_result("ranking_lr", "ranking", metrics, config={"model": "lr"}, seed=seed)
    return metrics


def train_deepfm(
    train_df,
    val_df,
    field_cols: list[str],
    field_dims: list[int],
    epochs: int = 2,
    batch_size: int = 2048,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for t_i, task in enumerate(TASKS):
        model = DeepFM(field_dims)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ds = RankDataset(train_df, field_cols, tasks=TASKS)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        model.train()
        for _ in range(epochs):
            for x, y, _ in loader:
                logit = model(x)
                loss = F.binary_cross_entropy_with_logits(logit, y[:, t_i])
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(val_df[field_cols].fillna(0).astype(np.int64).values, dtype=torch.long)
            proba = torch.sigmoid(model(xv)).numpy()
        all_metrics[task] = {
            "auc": auc_score(val_df[task], proba),
            "gauc": gauc(val_df[task], proba, val_df["user_id"]),
            "logloss": binary_logloss(val_df[task], proba),
        }
        torch.save(model.state_dict(), ARTIFACTS / f"deepfm_{task}.pt")
    write_result(
        "ranking_deepfm",
        "ranking",
        all_metrics,
        config={"model": "deepfm", "epochs": epochs},
        seed=seed,
    )
    (ARTIFACTS / "deepfm_fields.json").write_text(
        json.dumps({"field_cols": field_cols, "field_dims": field_dims})
    )
    return all_metrics


def train_mmoe(
    train_df,
    val_df,
    field_cols: list[str],
    field_dims: list[int],
    epochs: int = 3,
    batch_size: int = 2048,
    n_experts: int = 4,
    loss_weights: tuple[float, float, float] = (1.0, 1.5, 1.0),
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = MMoE(field_dims, n_tasks=3, n_experts=n_experts)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = RankDataset(train_df, field_cols)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    w = torch.tensor(loss_weights, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        losses = []
        for x, y, _ in tqdm(loader, desc=f"mmoe {epoch+1}/{epochs}"):
            logits = model(x)
            loss = sum(
                w[i] * F.binary_cross_entropy_with_logits(logits[:, i], y[:, i]) for i in range(3)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        xv = torch.tensor(val_df[field_cols].fillna(0).astype(np.int64).values, dtype=torch.long)
        proba = torch.sigmoid(model(xv)).numpy()

    metrics = {}
    for i, task in enumerate(TASKS):
        metrics[task] = {
            "auc": auc_score(val_df[task], proba[:, i]),
            "gauc": gauc(val_df[task], proba[:, i], val_df["user_id"]),
            "logloss": binary_logloss(val_df[task], proba[:, i]),
        }
    fused = fuse(proba, (0.3, 0.5, 0.2))
    metrics["fusion"] = {
        "auc_long_view": auc_score(val_df["long_view"], fused),
        "gauc_long_view": gauc(val_df["long_view"], fused, val_df["user_id"]),
    }
    trade = fusion_grid(proba, val_df["long_view"].values, n=5)
    metrics["fusion_best"] = trade[0] if trade else {}

    torch.save(
        {
            "state_dict": model.state_dict(),
            "field_cols": field_cols,
            "field_dims": field_dims,
            "n_experts": n_experts,
            "fusion_weights": [0.3, 0.5, 0.2],
        },
        ARTIFACTS / "mmoe.pt",
    )
    write_result(
        "ranking_mmoe",
        "ranking",
        metrics,
        config={
            "model": "mmoe",
            "epochs": epochs,
            "n_experts": n_experts,
            "loss_weights": list(loss_weights),
        },
        seed=seed,
        extra={"fusion_tradeoff_top5": trade[:5]},
    )
    print("mmoe", metrics)
    return metrics


def run(seed: int = 42, max_train: int = 400_000, max_val: int = 80_000) -> None:
    meta = json.loads((PROCESSED / "id_maps.json").read_text())
    users = pd.read_parquet(PROCESSED / "users.parquet")
    videos = pd.read_parquet(PROCESSED / "videos.parquet")
    train = pd.read_parquet(PROCESSED / "interactions_train.parquet")
    # Use random log as the unbiased ranking eval set (subsample for speed)
    val = pd.read_parquet(PROCESSED / "interactions_test_random.parquet")

    user_cat_cols = [c for c in meta["user_cat_idx_cols"] if c in users.columns][:4]
    video_cat_cols = [c for c in meta["video_cat_idx_cols"] if c in videos.columns][:4]
    user_num_cols = [c for c in meta["user_num_cols"] if c in users.columns][:6]

    train = _sample(train, max_train, seed)
    val = _sample(val, max_val, seed)
    train_f = _build_feature_frame(train, users, videos, user_cat_cols, video_cat_cols, user_num_cols)
    val_f = _build_feature_frame(val, users, videos, user_cat_cols, video_cat_cols, user_num_cols)

    # Discrete fields for DeepFM/MMoE: bucket continuous nums
    for col, src in [("log_dur_bin", "log_dur"), ("log_pop_bin", "log_pop")]:
        train_f[col] = pd.qcut(train_f[src], q=10, labels=False, duplicates="drop").fillna(0).astype(int) + 1
        # apply same edges roughly via rank bins on val
        val_f[col] = pd.qcut(val_f[src], q=10, labels=False, duplicates="drop").fillna(0).astype(int) + 1

    field_cols = user_cat_cols + video_cat_cols + ["log_dur_bin", "log_pop_bin"]
    # Ensure ints
    for c in field_cols:
        train_f[c] = train_f[c].fillna(0).astype(int)
        val_f[c] = val_f[c].fillna(0).astype(int)
    field_dims = [int(max(train_f[c].max(), val_f[c].max())) for c in field_cols]

    lr_features = user_num_cols + ["log_dur", "log_pop"] + field_cols
    print("training LR...")
    train_lr(train_f, val_f, lr_features, seed=seed)
    print("training DeepFM...")
    train_deepfm(train_f, val_f, field_cols, field_dims, epochs=2, seed=seed)
    print("training MMoE...")
    train_mmoe(train_f, val_f, field_cols, field_dims, epochs=3, n_experts=4, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=400000)
    parser.add_argument("--max-val", type=int, default=80000)
    args = parser.parse_args()
    run(seed=args.seed, max_train=args.max_train, max_val=args.max_val)


if __name__ == "__main__":
    main()
