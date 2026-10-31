"""Time-based train/val/test splits + random-exposure test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reelrank.data.paths import PROCESSED, PURE


LABEL_COLS = ["is_click", "long_view", "is_like", "is_hate"]
KEEP_COLS = [
    "user_id",
    "video_id",
    "date",
    "time_ms",
    "tab",
    "is_rand",
    "play_time_ms",
    "duration_ms",
    *LABEL_COLS,
]


def _load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df[KEEP_COLS].copy()


def build_splits(
    pure_dir: Path = PURE,
    out_dir: Path = PROCESSED,
    train_end: int = 20220421,
    val_end: int = 20220428,
) -> dict[str, Path]:
    """
    Standard log:
      train: date <= train_end (early window through 4/21)
      val:   train_end < date <= val_end
      test:  date > val_end
    Random log: full held-out unbiased test (never used for training).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    early = _load_log(pure_dir / "log_standard_4_08_to_4_21_pure.csv")
    late = _load_log(pure_dir / "log_standard_4_22_to_5_08_pure.csv")
    random = _load_log(pure_dir / "log_random_4_22_to_5_08_pure.csv")

    standard = pd.concat([early, late], ignore_index=True)
    standard = standard.sort_values("time_ms").reset_index(drop=True)

    train = standard[standard["date"] <= train_end].copy()
    val = standard[(standard["date"] > train_end) & (standard["date"] <= val_end)].copy()
    test_std = standard[standard["date"] > val_end].copy()

    paths = {
        "train": out_dir / "interactions_train.parquet",
        "val": out_dir / "interactions_val.parquet",
        "test_std": out_dir / "interactions_test_std.parquet",
        "test_random": out_dir / "interactions_test_random.parquet",
        "standard_all": out_dir / "interactions_standard_all.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)
    test_std.to_parquet(paths["test_std"], index=False)
    random.to_parquet(paths["test_random"], index=False)
    standard.to_parquet(paths["standard_all"], index=False)

    summary = {
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_std_rows": int(len(test_std)),
        "test_random_rows": int(len(random)),
        "train_users": int(train["user_id"].nunique()),
        "train_items": int(train["video_id"].nunique()),
        "train_end": train_end,
        "val_end": val_end,
        "date_min": int(standard["date"].min()),
        "date_max": int(standard["date"].max()),
    }
    pd.Series(summary).to_json(out_dir / "split_summary.json")
    print(summary)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pure-dir", type=Path, default=PURE)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED)
    parser.add_argument("--train-end", type=int, default=20220421)
    parser.add_argument("--val-end", type=int, default=20220428)
    args = parser.parse_args()
    build_splits(args.pure_dir, args.out_dir, args.train_end, args.val_end)


if __name__ == "__main__":
    main()
