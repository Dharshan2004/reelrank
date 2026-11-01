"""No-download example of the shared ranking and filtering contract.

Run from the checkout with `make example`. Candidates are supplied explicitly;
this illustrates ranking, not trained two-tower retrieval or model quality.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from reelrank.ranking.features import RankingFeatureTransformer
from reelrank.ranking.inference import filter_candidates, load_ranker, score_candidates


def main() -> None:
    users = pd.DataFrame({"user_id": [1, 2], "activity": [3.0, 8.0]})
    videos = pd.DataFrame({
        "video_id": [10, 20, 30, 40],
        "video_duration": [5.0, 15.0, 30.0, 60.0],
        "pop_clicks": [20, 10, 5, 1],
    })
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "video_id": [10, 20, 30, 40] * 2,
    })
    transformer = RankingFeatureTransformer().fit(
        interactions, users, videos, {"user_num_cols": ["activity"]}
    )
    features = transformer.transform(interactions, users, videos)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    x = encoder.fit_transform(features[transformer.field_cols])
    labels = {
        "is_click": [1, 1, 0, 0, 1, 0, 1, 0],
        "long_view": [1, 0, 0, 0, 1, 1, 0, 0],
        "is_like": [0, 1, 0, 0, 1, 0, 0, 0],
    }
    models = {task: LogisticRegression(random_state=42).fit(x, y) for task, y in labels.items()}
    runtime = load_ranker({
        "feature_transformer": transformer.to_dict(),
        "features": transformer.field_cols, "encoder": encoder, "models": models,
    }, name="lr")
    safety = pd.DataFrame({
        "video_id": [10, 20, 30, 40],
        "rule_hit": [False, False, True, False],
        "safety_score": [0.0, 0.0, 1.0, 0.0],
    })
    candidates = filter_candidates([10, 20, 30, 40], safety)
    _, scores = score_candidates(runtime, 1, candidates, users, videos)
    print("Synthetic example: supplied candidates -> filter -> shared LR scoring")
    print("Video 30 is filtered by a synthetic rule flag. Scores are not benchmark results.")
    for vid, score in sorted(zip(candidates, scores), key=lambda pair: -pair[1]):
        print(f"video_id={vid}, long_view_score={score:.4f}")


if __name__ == "__main__":
    main()
