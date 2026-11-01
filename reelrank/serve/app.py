"""FastAPI serving demo for ReelRank cascade."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.ranking.inference import filter_candidates, load_ranker_artifact, score_candidates
from reelrank.retrieval.two_tower import TwoTower

STATE: dict[str, Any] = {}


def _load() -> None:
    id_meta = json.loads((PROCESSED / "id_maps.json").read_text())
    histories = json.loads((PROCESSED / "histories.json").read_text())
    user_id_map = {**id_meta["user_id_map"], **{int(k): v for k, v in id_meta["user_id_map"].items()}}
    users_df = pd.read_parquet(PROCESSED / "users.parquet").set_index("user_id")
    videos_df = pd.read_parquet(PROCESSED / "videos.parquet").set_index("video_id")

    ckpt = torch.load(ARTIFACTS / "two_tower.pt", map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    tt = TwoTower(
        n_users=meta["n_users"],
        n_videos=meta["n_videos"],
        n_user_cats=meta["n_user_cats"],
        n_video_cats=meta["n_video_cats"],
        user_num_dim=len(meta["user_num_cols"]),
        emb_dim=meta["emb_dim"],
        temperature=meta["temperature"],
        history_n=meta["history_n"],
    )
    tt.load_state_dict(ckpt["state_dict"])
    tt.eval()

    ranking = load_ranker_artifact(ARTIFACTS, os.environ.get("REELRANK_RANKER", "lr"))

    safety_path = ARTIFACTS / "video_safety_scores.parquet"
    safety = pd.read_parquet(safety_path).set_index("video_id") if safety_path.exists() else None

    STATE.update(
        {
            "tt": tt,
            "meta": meta,
            "ranking": ranking,
            "index": faiss.read_index(str(ARTIFACTS / "index_flat.faiss")),
            "item_ids": json.loads((ARTIFACTS / "item_embedding_ids.json").read_text()),
            "users_df": users_df,
            "videos_df": videos_df,
            "histories": histories,
            "user_id_map": user_id_map,
            "safety": safety,
            "pop": json.loads((ARTIFACTS / "popularity_top.json").read_text())
            if (ARTIFACTS / "popularity_top.json").exists()
            else [],
        }
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load()
    yield


app = FastAPI(title="ReelRank", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "models_loaded": bool(STATE)}


@app.get("/recommend")
def recommend(
    user_id: int = Query(...),
    k: int = Query(10, ge=1, le=50),
    retrieve_k: int = Query(200, ge=10, le=500),
) -> dict:
    if not STATE:
        raise HTTPException(503, "models not loaded")
    timings: dict[str, float] = {}

    # Retrieve
    t0 = time.perf_counter()
    meta = STATE["meta"]
    users_df = STATE["users_df"]
    histories = STATE["histories"]
    user_id_map = STATE["user_id_map"]
    uid = int(user_id)
    if uid in users_df.index:
        row = users_df.loc[uid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cat = np.array([int(row.get(c, 0)) for c in meta["user_cat_cols"]] or [0], dtype=np.int64)
        num = np.array([float(row.get(c, 0.0)) for c in meta["user_num_cols"]] or [0.0], dtype=np.float32)
    else:
        cat = np.zeros(max(len(meta["user_cat_cols"]), 1), dtype=np.int64)
        num = np.zeros(max(len(meta["user_num_cols"]), 1), dtype=np.float32)
    hist = histories.get(str(uid), [])
    hist_arr = np.zeros(meta["history_n"], dtype=np.int64)
    if hist:
        take = hist[-meta["history_n"] :]
        hist_arr[-len(take) :] = np.array(take, dtype=np.int64)
    with torch.no_grad():
        uemb = STATE["tt"].encode_user(
            torch.tensor([user_id_map.get(str(uid), user_id_map.get(uid, 0))], dtype=torch.long),
            torch.tensor(cat, dtype=torch.long).unsqueeze(0),
            torch.tensor(num, dtype=torch.float32).unsqueeze(0),
            torch.tensor(hist_arr, dtype=torch.long).unsqueeze(0),
        )
    _, idxs = STATE["index"].search(uemb.cpu().numpy().astype(np.float32), retrieve_k)
    cands = [int(STATE["item_ids"][j]) for j in idxs[0] if j >= 0]
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # Safety
    t1 = time.perf_counter()
    safety = STATE["safety"]
    filtered = filter_candidates(cands, safety)
    blocked = len(cands) - len(filtered)
    timings["safety_ms"] = (time.perf_counter() - t1) * 1000

    # Rank
    t2 = time.perf_counter()
    videos_df = STATE["videos_df"]
    keep_ids = filtered[:retrieve_k]
    items = []
    if keep_ids:
        proba, scores = score_candidates(STATE["ranking"], uid, keep_ids, users_df, videos_df)
        order = np.argsort(-scores, kind="stable")[:k]
        for i in order:
            vid = keep_ids[i]
            cap = ""
            if vid in videos_df.index:
                cap = str(videos_df.loc[vid].get("caption", "") or "")
            items.append(
                {
                    "video_id": vid,
                    "score": float(scores[i]),
                    "p_click": float(proba[i, 0]),
                    "p_long_view": float(proba[i, 1]),
                    "p_like": float(proba[i, 2]),
                    "caption": cap[:160],
                }
            )
    timings["ranking_ms"] = (time.perf_counter() - t2) * 1000
    timings["total_ms"] = sum(timings.values())

    return {
        "user_id": uid,
        "ranker": STATE["ranking"].name,
        "items": items,
        "timings_ms": timings,
        "diagnostics": {
            "retrieved": len(cands),
            "after_safety": len(filtered),
            "blocked": blocked,
        },
    }


def main() -> None:
    import uvicorn

    uvicorn.run("reelrank.serve.app:app", host="0.0.0.0", port=43123, reload=False)


if __name__ == "__main__":
    main()
