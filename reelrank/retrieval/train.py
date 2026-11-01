"""Train two-tower retrieval with in-batch softmax + logQ."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Avoid Torch+FAISS OpenMP double-init crashes on macOS (seen as make Error 139).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.data.history import TimestampHistory
from reelrank.eval.runlog import write_result
from reelrank.retrieval.baselines import build_user_positives, evaluate_retrieval, evaluation_cohort
from reelrank.retrieval.index import build_and_benchmark
from reelrank.retrieval.two_tower import TwoTower, sampled_softmax_loss


class InteractionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_id_map: dict,
        video_id_map: dict,
        users_df: pd.DataFrame,
        videos_df: pd.DataFrame,
        histories: dict,
        user_cat_cols: list[str],
        video_cat_cols: list[str],
        user_num_cols: list[str],
        history_n: int,
        label: str = "long_view",
        training: bool = True,
    ) -> None:
        pos = df[df[label] == 1].reset_index(drop=True)
        self.df = pos
        self.user_id_map = user_id_map
        self.video_id_map = video_id_map
        self.users = users_df.set_index("user_id")
        self.videos = videos_df.set_index("video_id")
        self.histories = histories
        self.user_cat_cols = user_cat_cols
        self.video_cat_cols = video_cat_cols
        self.user_num_cols = user_num_cols
        self.history_n = history_n
        self.training_history = TimestampHistory(df, video_id_map, history_n) if training else None
        # Materialize entity features once instead of doing pandas .loc for
        # every training example in every epoch. Histories remain per-example.
        self._uids = self.df["user_id"].to_numpy(dtype=np.int64)
        self._vids = self.df["video_id"].to_numpy(dtype=np.int64)
        self._user_cache = {}
        for uid in np.unique(self._uids):
            cat, num, _ = self._user_feats(int(uid))
            self._user_cache[int(uid)] = (
                torch.tensor(user_id_map.get(str(uid), user_id_map.get(int(uid), 0)), dtype=torch.long),
                torch.from_numpy(cat), torch.from_numpy(num),
            )
        self._item_cache = {}
        for vid in np.unique(self._vids):
            cat, num = self._item_feats(int(vid))
            self._item_cache[int(vid)] = (
                torch.tensor(video_id_map.get(str(vid), video_id_map.get(int(vid), 0)), dtype=torch.long),
                torch.from_numpy(cat), torch.from_numpy(num),
            )
        history_rows = np.zeros((len(self.df), history_n), dtype=np.int64)
        for uid, rows in self.df.groupby("user_id", sort=False).indices.items():
            if self.training_history is not None:
                history_rows[rows] = self.training_history.before_many(
                    int(uid), self.df.iloc[rows]["time_ms"].to_numpy()
                )
            else:
                history_rows[rows] = self._user_feats(int(uid))[2]
        self._history_rows = torch.from_numpy(history_rows)

    def __len__(self) -> int:
        return len(self.df)

    def _user_feats(self, uid: int, timestamp: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if uid not in self.users.index:
            cat = np.zeros(max(len(self.user_cat_cols), 1), dtype=np.int64)
            num = np.zeros(max(len(self.user_num_cols), 1), dtype=np.float32)
        else:
            row = self.users.loc[uid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cat = (
                np.array([int(row.get(c, 0)) for c in self.user_cat_cols], dtype=np.int64)
                if self.user_cat_cols
                else np.zeros(1, dtype=np.int64)
            )
            num = (
                np.array([float(row.get(c, 0.0)) for c in self.user_num_cols], dtype=np.float32)
                if self.user_num_cols
                else np.zeros(1, dtype=np.float32)
            )
        if timestamp is not None:
            if self.training_history is None:
                raise ValueError("Point-in-time histories require training=True")
            return cat, num, self.training_history.before(uid, timestamp)
        # No timestamp is used only for inference after the training window.
        hist = self.histories.get(str(uid), self.histories.get(uid, []))
        hist_arr = np.zeros(self.history_n, dtype=np.int64)
        if hist:
            take = hist[-self.history_n :]
            hist_arr[-len(take) :] = np.array(take, dtype=np.int64)
        return cat, num, hist_arr

    def _item_feats(self, vid: int) -> tuple[np.ndarray, np.ndarray]:
        if vid not in self.videos.index:
            cat = np.zeros(max(len(self.video_cat_cols), 1), dtype=np.int64)
            num = np.zeros(2, dtype=np.float32)
        else:
            row = self.videos.loc[vid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cat = (
                np.array([int(row.get(c, 0)) for c in self.video_cat_cols], dtype=np.int64)
                if self.video_cat_cols
                else np.zeros(1, dtype=np.int64)
            )
            dur = float(row.get("video_duration", 0.0) or 0.0)
            pop = float(row.get("pop_clicks", 0.0) or 0.0)
            num = np.array([np.log1p(dur), np.log1p(pop)], dtype=np.float32)
        return cat, num

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_idx, user_cat, user_num = self._user_cache[int(self._uids[idx])]
        video_idx, video_cat, video_num = self._item_cache[int(self._vids[idx])]
        return {
            "user_idx": user_idx,
            "video_idx": video_idx,
            "user_cat": user_cat,
            "video_cat": video_cat,
            "user_num": user_num,
            "video_num": video_num,
            "hist_idx": self._history_rows[idx],
        }


def _fix_maps(meta: dict) -> tuple[dict, dict]:
    # JSON keys are strings; also keep int keys
    user_id_map = {**meta["user_id_map"], **{int(k): v for k, v in meta["user_id_map"].items()}}
    video_id_map = {**meta["video_id_map"], **{int(k): v for k, v in meta["video_id_map"].items()}}
    return user_id_map, video_id_map


def train_two_tower(
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 1e-3,
    emb_dim: int = 64,
    temperature: float = 0.05,
    seed: int = 42,
    max_eval_users: int = 2000,
    label: str = "long_view",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    meta = json.loads((PROCESSED / "id_maps.json").read_text())
    histories = json.loads((PROCESSED / "histories.json").read_text())
    user_id_map, video_id_map = _fix_maps(meta)
    users_df = pd.read_parquet(PROCESSED / "users.parquet")
    videos_df = pd.read_parquet(PROCESSED / "videos.parquet")
    train_df = pd.read_parquet(PROCESSED / "interactions_train.parquet")

    user_cat_cols = [c for c in meta["user_cat_idx_cols"] if c in users_df.columns][:4]
    video_cat_cols = [c for c in meta["video_cat_idx_cols"] if c in videos_df.columns][:4]
    user_num_cols = [c for c in meta["user_num_cols"] if c in users_df.columns][:8]
    if not user_num_cols:
        users_df["u_bias"] = 0.0
        user_num_cols = ["u_bias"]

    n_user_cats = int(max((users_df[c].max() for c in user_cat_cols), default=1)) + 1
    n_video_cats = int(max((videos_df[c].max() for c in video_cat_cols), default=1)) + 1

    ds = InteractionDataset(
        train_df,
        user_id_map,
        video_id_map,
        users_df,
        videos_df,
        histories,
        user_cat_cols,
        video_cat_cols,
        user_num_cols,
        meta["history_n"],
        label=label,
    )
    if len(ds) == 0:
        raise ValueError(f"No positive training events for {label}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    # item frequency for logQ
    freqs = ds.df["video_id"].map(
        lambda x: video_id_map.get(str(x), video_id_map.get(x, 0))
    )
    freq_counts = np.bincount(freqs.dropna().astype(int), minlength=meta["n_videos"] + 1).astype(
        np.float32
    )
    # Unseen items are never sampled; avoid smoothing their mass into q.
    sampling_q = freq_counts / freq_counts.sum()
    logq_table = torch.tensor(np.log(np.maximum(sampling_q, np.finfo(np.float32).tiny)), dtype=torch.float32)

    model = TwoTower(
        n_users=meta["n_users"],
        n_videos=meta["n_videos"],
        n_user_cats=n_user_cats,
        n_video_cats=n_video_cats,
        user_num_dim=len(user_num_cols),
        emb_dim=emb_dim,
        temperature=temperature,
        history_n=meta["history_n"],
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}"):
            u, v = model(batch)
            logq = logq_table[batch["video_idx"]]
            loss = sampled_softmax_loss(
                u, v, logq=logq, temperature=temperature, candidate_ids=batch["video_idx"]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    ckpt = {
        "state_dict": model.state_dict(),
        "meta": {
            **{k: meta[k] for k in ["n_users", "n_videos", "history_n", "user_num_cols"]},
            "user_cat_cols": user_cat_cols,
            "video_cat_cols": video_cat_cols,
            "user_num_cols": user_num_cols,
            "emb_dim": emb_dim,
            "temperature": temperature,
            "n_user_cats": n_user_cats,
            "n_video_cats": n_video_cats,
            "training_label": label,
            "training_history": "strictly_earlier_timestamp",
        },
    }
    torch.save(ckpt, ARTIFACTS / "two_tower.pt")

    # Encode all items
    model.eval()
    video_ids = videos_df["video_id"].tolist()
    item_embs = []
    with torch.no_grad():
        for i in range(0, len(videos_df), 2048):
            chunk = videos_df.iloc[i : i + 2048]
            vidx = torch.tensor(
                [video_id_map.get(str(v), video_id_map.get(int(v), 0)) for v in chunk["video_id"]],
                dtype=torch.long,
            )
            if video_cat_cols:
                vcat = torch.tensor(chunk[video_cat_cols].fillna(0).astype(int).values, dtype=torch.long)
            else:
                vcat = torch.zeros((len(chunk), 1), dtype=torch.long)
            dur = np.log1p(chunk.get("video_duration", pd.Series([0] * len(chunk))).fillna(0).astype(float).values)
            pop = np.log1p(chunk.get("pop_clicks", pd.Series([0] * len(chunk))).fillna(0).astype(float).values)
            vnum = torch.tensor(np.stack([dur, pop], axis=1), dtype=torch.float32)
            emb = model.encode_item(vidx, vcat, vnum)
            item_embs.append(emb.cpu().numpy())
    item_embs = np.vstack(item_embs).astype(np.float32)
    np.save(ARTIFACTS / "item_embeddings.npy", item_embs)
    (ARTIFACTS / "item_embedding_ids.json").write_text(json.dumps([int(x) for x in video_ids]))

    # Build FAISS + evaluate retrieval on random log
    ann = build_and_benchmark(item_embs)
    write_result("retrieval_faiss", "ann", ann, config={"index": "flat_vs_hnsw"}, seed=seed)

    test = pd.read_parquet(PROCESSED / "interactions_test_random.parquet")
    user_rel = build_user_positives(test, label="long_view")
    user_rel = evaluation_cohort(user_rel, max_eval_users, seed)
    users = list(user_rel)

    # Encode users and search
    import faiss

    index = faiss.read_index(str(ARTIFACTS / "index_flat.faiss"))
    user_recs: dict[int, list[int]] = {}
    id_list = video_ids
    with torch.no_grad():
        for uid in tqdm(users, desc="eval users"):
            cat, num, hist = ds._user_feats(uid)
            batch = {
                "user_idx": torch.tensor(
                    [user_id_map.get(str(uid), user_id_map.get(uid, 0))], dtype=torch.long
                ),
                "user_cat": torch.tensor(cat, dtype=torch.long).unsqueeze(0),
                "user_num": torch.tensor(num, dtype=torch.float32).unsqueeze(0),
                "hist_idx": torch.tensor(hist, dtype=torch.long).unsqueeze(0),
            }
            uemb = model.encode_user(
                batch["user_idx"], batch["user_cat"], batch["user_num"], batch["hist_idx"]
            )
            scores, idxs = index.search(uemb.cpu().numpy().astype(np.float32), 200)
            user_recs[uid] = [int(id_list[j]) for j in idxs[0] if j >= 0]

    metrics = evaluate_retrieval(user_rel, user_recs, catalogue_size=len(video_ids))
    write_result(
        "retrieval_two_tower",
        "retrieval",
        metrics,
        config={
            "method": "two_tower",
            "epochs": epochs,
            "batch_size": batch_size,
            "emb_dim": emb_dim,
            "temperature": temperature,
            "max_eval_users": max_eval_users,
            "training_label": label,
            "evaluation_label": "long_view",
            "seen_item_policy": "allow_repeats",
            "training_history": "strictly_earlier_timestamp",
        },
        seed=seed,
    )
    print("two_tower", metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval-users", type=int, default=2000)
    parser.add_argument("--label", choices=["long_view", "is_click"], default="long_view")
    args = parser.parse_args()
    train_two_tower(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        emb_dim=args.emb_dim,
        temperature=args.temperature,
        seed=args.seed,
        max_eval_users=args.max_eval_users,
        label=args.label,
    )


if __name__ == "__main__":
    main()
