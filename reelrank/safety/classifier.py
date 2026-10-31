"""Char n-gram TF-IDF + logistic regression safety classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.eval.runlog import write_result
from reelrank.safety.rules import rule_hit


def train_classifier(seed: int = 42) -> dict:
    labels = pd.read_parquet(ARTIFACTS / "safety_labels.parquet")
    x_train, x_test, y_train, y_test = train_test_split(
        labels["caption"].astype(str),
        labels["y"].astype(int),
        test_size=0.25,
        random_state=seed,
        stratify=labels["y"] if labels["y"].nunique() > 1 else None,
    )
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=50000)
    xt = vec.fit_transform(x_train)
    xv = vec.transform(x_test)
    clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed)
    clf.fit(xt, y_train)
    proba = clf.predict_proba(xv)[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    metrics = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "unsafe_rate_test": float(y_test.mean()),
    }

    joblib.dump({"vectorizer": vec, "model": clf, "threshold": 0.5}, ARTIFACTS / "safety_clf.joblib")

    # Score all videos + correlate with is_hate rate
    videos = pd.read_parquet(PROCESSED / "videos.parquet")
    train = pd.read_parquet(PROCESSED / "interactions_train.parquet")
    hate = train.groupby("video_id")["is_hate"].mean()
    caps = videos["caption"].fillna("").astype(str)
    scores = clf.predict_proba(vec.transform(caps))[:, 1]
    videos = videos.copy()
    videos["safety_score"] = scores
    videos["rule_hit"] = caps.map(rule_hit)
    videos["hate_rate"] = videos["video_id"].map(hate).fillna(0)
    corr = float(np.corrcoef(videos["safety_score"], videos["hate_rate"])[0, 1])
    metrics["corr_safety_score_vs_hate_rate"] = corr
    videos[["video_id", "safety_score", "rule_hit"]].to_parquet(
        ARTIFACTS / "video_safety_scores.parquet", index=False
    )
    write_result("safety_classifier", "safety", metrics, config={"model": "tfidf_char_lr"}, seed=seed)
    print(metrics)
    return metrics


def should_filter(caption: str, score: float, threshold: float = 0.5) -> bool:
    return rule_hit(caption) or score >= threshold


if __name__ == "__main__":
    train_classifier()
