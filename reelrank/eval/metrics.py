"""Ranking and classification metrics used across ReelRank."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def recall_at_k(relevant: Sequence[int], recommended: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(recommended[:k])
    hits = sum(1 for r in relevant if r in top)
    return hits / len(set(relevant))


def hit_rate_at_k(relevant: Sequence[int], recommended: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(recommended[:k])
    return 1.0 if any(r in top for r in relevant) else 0.0


def ndcg_at_k(relevant: Sequence[int], recommended: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in rel_set:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_metric(
    user_relevant: dict[int, Sequence[int]],
    user_recs: dict[int, Sequence[int]],
    k: int,
    fn,
) -> float:
    scores = []
    for uid, rel in user_relevant.items():
        recs = user_recs.get(uid, [])
        scores.append(fn(rel, recs, k))
    return float(np.mean(scores)) if scores else 0.0


def auc_score(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def gauc(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    group_ids: Sequence[int] | np.ndarray,
) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    group_ids = np.asarray(group_ids)
    scores = []
    weights = []
    for gid in np.unique(group_ids):
        mask = group_ids == gid
        yt = y_true[mask]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(roc_auc_score(yt, y_score[mask]))
        weights.append(mask.sum())
    if not scores:
        return float("nan")
    return float(np.average(scores, weights=weights))


def binary_logloss(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.clip(np.asarray(y_score), 1e-7, 1 - 1e-7)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(log_loss(y_true, y_score))


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_score >= bins[i]) & (y_score < bins[i + 1] if i < n_bins - 1 else y_score <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_score[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def catalogue_coverage(recommended_lists: Iterable[Sequence[int]], catalogue_size: int) -> float:
    seen: set[int] = set()
    for recs in recommended_lists:
        seen.update(recs)
    if catalogue_size <= 0:
        return 0.0
    return float(len(seen / catalogue_size))
