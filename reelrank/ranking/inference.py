"""Shared checkpoint-backed ranking and safety decisions for evaluation and serving."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

import numpy as np
import pandas as pd
import torch

from reelrank.ranking.features import RankingFeatureTransformer
from reelrank.ranking.fusion import fuse
from reelrank.ranking.models import MMoE

SAFETY_THRESHOLD = 0.5


@dataclass
class RankingRuntime:
    model: Any
    transformer: RankingFeatureTransformer
    fusion_weights: tuple[float, float, float]
    name: str = "mmoe"
    encoder: Any = None


def load_ranker(checkpoint: dict, name: str = "mmoe") -> RankingRuntime:
    if "feature_transformer" not in checkpoint:
        raise ValueError("Legacy ranking checkpoint lacks feature_transformer; rerun make train-ranker before evaluation or serving.")
    if name not in {"lr", "mmoe"}:
        raise ValueError(f"Unknown ranker: {name}; choose lr or mmoe.")
    transformer = RankingFeatureTransformer.from_dict(checkpoint["feature_transformer"])
    if name == "lr":
        if not all(key in checkpoint for key in ("models", "encoder", "features")):
            raise ValueError("Legacy LR checkpoint lacks encoded feature contract; rerun make train-ranker.")
        if list(checkpoint["features"]) != list(transformer.field_cols):
            raise ValueError("LR checkpoint feature schema disagrees with its saved transformer.")
        if not all(task in checkpoint["models"] for task in ("is_click", "long_view", "is_like")):
            raise ValueError("LR checkpoint is missing an engagement model; rerun make train-ranker.")
        # Fixed primary objective, not selected using the random-exposure outcomes.
        return RankingRuntime(checkpoint["models"], transformer, (0.0, 1.0, 0.0), "lr", checkpoint["encoder"])
    if list(checkpoint["field_cols"]) != list(transformer.field_cols) or list(checkpoint["field_dims"]) != list(transformer.field_dims):
        raise ValueError("Ranking checkpoint feature schema disagrees with its saved transformer.")
    if "fusion_weights" not in checkpoint:
        raise ValueError("Ranking checkpoint lacks validation-selected fusion_weights; rerun make train-ranker.")
    model = MMoE(checkpoint["field_dims"], n_experts=checkpoint["n_experts"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return RankingRuntime(model, transformer, tuple(checkpoint["fusion_weights"]))


def load_ranker_artifact(artifacts: Path, name: str = "lr") -> RankingRuntime:
    if name == "lr":
        checkpoint = joblib.load(artifacts / "rank_lr.joblib")
    elif name == "mmoe":
        checkpoint = torch.load(artifacts / "mmoe.pt", map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unknown ranker: {name}; choose lr or mmoe.")
    return load_ranker(checkpoint, name=name)


def _with_id_column(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame if name in frame.columns else frame.reset_index()


def score_candidates(runtime: RankingRuntime, uid: int, candidates: list[int], users: pd.DataFrame, videos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
    interactions = pd.DataFrame({"user_id": [uid] * len(candidates), "video_id": candidates})
    frame = runtime.transformer.transform(interactions, _with_id_column(users, "user_id"), _with_id_column(videos, "video_id"))
    if runtime.name == "lr":
        features = runtime.encoder.transform(frame[runtime.transformer.field_cols])
        probabilities = np.column_stack([runtime.model[task].predict_proba(features)[:, 1]
                                         for task in ("is_click", "long_view", "is_like")])
    else:
        features = torch.tensor(frame[runtime.transformer.field_cols].to_numpy(dtype=np.int64), dtype=torch.long)
        with torch.inference_mode():
            probabilities = torch.sigmoid(runtime.model(features)).cpu().numpy()
    return probabilities, fuse(probabilities, runtime.fusion_weights)


def rank_candidates(runtime: RankingRuntime, uid: int, candidates: list[int], users: pd.DataFrame, videos: pd.DataFrame, k: int) -> list[int]:
    _, scores = score_candidates(runtime, uid, candidates, users, videos)
    return [candidates[i] for i in np.argsort(-scores, kind="stable")[:k]]


def filter_candidates(candidates: list[int], safety: pd.DataFrame | None, threshold: float = SAFETY_THRESHOLD) -> list[int]:
    """Apply the identical caption-only policy to every candidate generator.

    Items without a safety score pass through; this is explicit prototype policy,
    not a claim that missing captions or scores establish content safety.
    """
    if safety is None:
        return list(candidates)
    if "video_id" in safety.columns:
        safety = safety.set_index("video_id")
    keep = []
    for vid in candidates:
        if vid in safety.index:
            row = safety.loc[vid]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Duplicate safety score for video {vid}")
            rule_hit = False if pd.isna(row["rule_hit"]) else bool(row["rule_hit"])
            if rule_hit or float(row["safety_score"]) >= threshold:
                continue
        keep.append(vid)
    return keep
