"""Offline cascade: rules -> classifier -> (optional) LLM uncertain band."""

from __future__ import annotations

import json

import pandas as pd

from reelrank.data.paths import ARTIFACTS
from reelrank.eval.runlog import write_result


def analyze_cascade(low: float = 0.3, high: float = 0.7) -> dict:
    scores = pd.read_parquet(ARTIFACTS / "video_safety_scores.parquet")
    labels_path = ARTIFACTS / "safety_labels.parquet"
    labels = pd.read_parquet(labels_path) if labels_path.exists() else None

    blocked_rules = int(scores["rule_hit"].sum())
    blocked_clf = int(((~scores["rule_hit"]) & (scores["safety_score"] >= high)).sum())
    uncertain = int(
        ((~scores["rule_hit"]) & (scores["safety_score"] >= low) & (scores["safety_score"] < high)).sum()
    )
    metrics = {
        "n_videos": int(len(scores)),
        "blocked_rules": blocked_rules,
        "blocked_classifier": blocked_clf,
        "uncertain_band_to_llm": uncertain,
        "pct_to_llm": float(uncertain / max(len(scores), 1)),
        "thresholds": {"low": low, "high": high},
    }
    if labels is not None:
        merged = labels.merge(scores, on="video_id", how="left")
        # simulate cascade decision
        pred = (
            merged["rule_hit"].fillna(False)
            | (merged["safety_score"].fillna(0) >= high)
        ).astype(int)
        # uncertain would go to LLM — treat as using label source if available
        metrics["labeled_precision_proxy"] = float(
            ((pred == 1) & (merged["y"] == 1)).sum() / max((pred == 1).sum(), 1)
        )
    cost = {}
    cost_path = ARTIFACTS / "safety_judge_cost.json"
    if cost_path.exists():
        cost = json.loads(cost_path.read_text())
    write_result("safety_cascade", "safety", metrics, config={"low": low, "high": high}, extra={"judge": cost})
    print(metrics)
    return metrics


if __name__ == "__main__":
    analyze_cascade()
