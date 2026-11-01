"""Point-in-time long-view histories for training examples.

KuaiRand-Pure histories are partial: only candidate-pool events are available.
Events sharing a timestamp are all excluded from one another's histories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TimestampHistory:
    def __init__(self, train: pd.DataFrame, video_id_map: dict, history_n: int) -> None:
        if history_n < 1:
            raise ValueError("history_n must be positive")
        self.history_n = history_n
        self.events: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        positive = train.loc[train["long_view"] == 1].sort_values(
            ["user_id", "time_ms"], kind="stable"
        )
        for uid, group in positive.groupby("user_id", sort=False):
            ids = np.asarray(
                [video_id_map.get(str(v), video_id_map.get(int(v), 0)) for v in group["video_id"]],
                dtype=np.int64,
            )
            self.events[int(uid)] = (group["time_ms"].to_numpy(), ids)

    def before(self, uid: int, timestamp: int) -> np.ndarray:
        result = np.zeros(self.history_n, dtype=np.int64)
        if uid in self.events:
            times, ids = self.events[uid]
            end = int(np.searchsorted(times, timestamp, side="left"))
            recent = ids[max(0, end - self.history_n):end]
            if len(recent):
                result[-len(recent):] = recent
        return result

    def before_many(self, uid: int, timestamps: np.ndarray) -> np.ndarray:
        """Vectorized queries for one user; storage scales with rows × history_n."""
        result = np.zeros((len(timestamps), self.history_n), dtype=np.int64)
        if uid in self.events:
            times, ids = self.events[uid]
            ends = np.searchsorted(times, timestamps, side="left")
            positions = ends[:, None] - np.arange(self.history_n, 0, -1)[None, :]
            valid = positions >= 0
            result[valid] = ids[positions[valid]]
        return result
