"""Weighted fusion of multi-task ranking scores."""

from __future__ import annotations

import numpy as np


def fuse(scores: np.ndarray, weights: tuple[float, float, float] = (0.3, 0.5, 0.2)) -> np.ndarray:
    """scores: [N, 3] for click, long_view, like."""
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    return (scores * w).sum(axis=-1)


def fusion_grid(
    scores: np.ndarray,
    y_long: np.ndarray,
    n: int = 5,
) -> list[dict]:
    """Sweep fusion weights and report AUC on long_view labels for trade-off curve."""
    from sklearn.metrics import roc_auc_score

    results = []
    for a in np.linspace(0.1, 0.7, n):
        for b in np.linspace(0.1, 0.7, n):
            c = 1.0 - a - b
            if c < 0.05:
                continue
            s = fuse(scores, (a, b, c))
            if len(np.unique(y_long)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_long, s))
            results.append({"w_click": float(a), "w_long": float(b), "w_like": float(c), "auc_long": auc})
    results.sort(key=lambda r: (-(r["auc_long"] if r["auc_long"] == r["auc_long"] else -1)))
    return results
