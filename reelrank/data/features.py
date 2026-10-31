"""User/video feature tables and per-user long-view histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reelrank.data.paths import PROCESSED, PURE, RAW


def _encode_categoricals(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    maps: dict[str, dict] = {}
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        vals = out[c].fillna("__NA__").astype(str)
        uniq = sorted(vals.unique())
        mapping = {v: i + 1 for i, v in enumerate(uniq)}  # 0 reserved for unknown
        maps[c] = mapping
        out[c + "_idx"] = vals.map(mapping).astype(np.int32)
    return out, maps


def build_features(
    pure_dir: Path = PURE,
    raw_dir: Path = RAW,
    out_dir: Path = PROCESSED,
    history_n: int = 20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    users = pd.read_csv(pure_dir / "user_features_pure.csv")
    videos = pd.read_csv(pure_dir / "video_features_basic_pure.csv")
    train = pd.read_parquet(out_dir / "interactions_train.parquet")

    # Join categories / captions when available
    captions_path = raw_dir / "kuairand_video_captions.csv"
    cats_path = raw_dir / "kuairand_video_categories.csv"
    if captions_path.exists() and captions_path.stat().st_size > 0:
        captions = pd.read_csv(captions_path)
        captions = captions.rename(columns={"final_video_id": "video_id"})
        videos = videos.merge(
            captions[["video_id", "caption", "show_cover_text"]],
            on="video_id",
            how="left",
        )
    else:
        videos["caption"] = ""
        videos["show_cover_text"] = ""

    if cats_path.exists() and cats_path.stat().st_size > 0:
        cats = pd.read_csv(cats_path)
        cats = cats.rename(columns={"final_video_id": "video_id"})
        keep = [
            "video_id",
            "first_level_category_id",
            "first_level_category_name",
            "second_level_category_id",
            "second_level_category_name",
        ]
        videos = videos.merge(cats[[c for c in keep if c in cats.columns]], on="video_id", how="left")

    user_cat_cols = [c for c in users.columns if c != "user_id" and users[c].dtype == object]
    # Prefer a small stable set of numeric/user fields
    user_num_candidates = [
        c
        for c in users.columns
        if c != "user_id" and pd.api.types.is_numeric_dtype(users[c])
    ]
    user_num_cols = user_num_candidates[:12]

    users_enc, user_maps = _encode_categoricals(users, user_cat_cols[:8])
    for c in user_num_cols:
        users_enc[c] = users_enc[c].fillna(0).astype(np.float32)

    video_cat_cols = [
        c
        for c in [
            "author_id",
            "video_type",
            "upload_type",
            "visible_status",
            "first_level_category_name",
            "second_level_category_name",
        ]
        if c in videos.columns
    ]
    videos_enc, video_maps = _encode_categoricals(videos, video_cat_cols)
    if "video_duration" in videos_enc.columns:
        videos_enc["video_duration"] = videos_enc["video_duration"].fillna(0).astype(np.float32)
    elif "duration_ms" in videos_enc.columns:
        videos_enc["video_duration"] = videos_enc["duration_ms"].fillna(0).astype(np.float32)
    else:
        videos_enc["video_duration"] = 0.0

    # Contiguous id maps for embedding tables (0 = padding)
    user_ids = sorted(train["user_id"].unique().tolist())
    video_ids = sorted(pd.concat([train["video_id"], videos_enc["video_id"]]).unique().tolist())
    user_id_map = {u: i + 1 for i, u in enumerate(user_ids)}
    video_id_map = {v: i + 1 for i, v in enumerate(video_ids)}

    users_enc["user_idx"] = users_enc["user_id"].map(user_id_map).fillna(0).astype(np.int32)
    videos_enc["video_idx"] = videos_enc["video_id"].map(video_id_map).fillna(0).astype(np.int32)
    videos_enc["caption"] = videos_enc.get("caption", "").fillna("").astype(str)

    # Long-view histories from train (chronological)
    lv = train[train["long_view"] == 1][["user_id", "video_id", "time_ms"]].sort_values(
        ["user_id", "time_ms"]
    )
    histories: dict[int, list[int]] = {}
    for uid, g in lv.groupby("user_id", sort=False):
        vids = g["video_id"].tolist()[-history_n:]
        histories[int(uid)] = [int(video_id_map.get(v, 0)) for v in vids]

    # Popularity from train clicks
    pop = train.groupby("video_id")["is_click"].sum().astype(np.float32)
    videos_enc["pop_clicks"] = videos_enc["video_id"].map(pop).fillna(0).astype(np.float32)

    users_enc.to_parquet(out_dir / "users.parquet", index=False)
    videos_enc.to_parquet(out_dir / "videos.parquet", index=False)
    meta = {
        "user_id_map": {str(k): v for k, v in user_id_map.items()},
        "video_id_map": {str(k): v for k, v in video_id_map.items()},
        "n_users": len(user_id_map),
        "n_videos": len(video_id_map),
        "history_n": history_n,
        "user_num_cols": user_num_cols,
        "user_cat_idx_cols": [c + "_idx" for c in user_cat_cols[:8] if c in users.columns],
        "video_cat_idx_cols": [c + "_idx" for c in video_cat_cols],
        "user_maps": user_maps,
        "video_maps": video_maps,
    }
    (out_dir / "id_maps.json").write_text(json.dumps(meta))
    (out_dir / "histories.json").write_text(
        json.dumps({str(k): v for k, v in histories.items()})
    )
    print(
        {
            "users": len(users_enc),
            "videos": len(videos_enc),
            "n_users_mapped": len(user_id_map),
            "n_videos_mapped": len(video_id_map),
            "histories": len(histories),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-n", type=int, default=20)
    args = parser.parse_args()
    build_features(history_n=args.history_n)


if __name__ == "__main__":
    main()
