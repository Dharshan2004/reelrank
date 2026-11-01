import math

import numpy as np
import pandas as pd
import pytest
import torch

from reelrank.data.history import TimestampHistory
from reelrank.retrieval.baselines import evaluation_cohort, itemcf_recommend, popularity_recommend
from reelrank.retrieval.train import InteractionDataset
from reelrank.retrieval.two_tower import sampled_softmax_loss


def events():
    # Deliberately unsorted, with same-time positives and a click-only event.
    return pd.DataFrame({
        "user_id": [1, 1, 1, 1, 1, 2],
        "video_id": [30, 10, 20, 40, 50, 60],
        "time_ms": [300, 100, 200, 200, 250, 100],
        "long_view": [1, 1, 1, 1, 0, 1],
        "is_click": [1, 1, 1, 1, 1, 1],
    })


def test_histories_exclude_current_future_and_same_timestamp_events():
    history = TimestampHistory(events(), {str(v): v for v in [10, 20, 30, 40, 50, 60]}, 2)
    assert history.before(1, 100).tolist() == [0, 0]
    assert history.before(1, 200).tolist() == [0, 10]
    assert history.before(1, 300).tolist() == [20, 40]
    assert history.before(1, 400).tolist() == [40, 30]
    assert history.before(2, 101).tolist() == [0, 60]
    assert history.before(999, 999).tolist() == [0, 0]
    queries = np.array([400, 100, 200, 300])
    assert np.array_equal(history.before_many(1, queries), np.stack([history.before(1, t) for t in queries]))


def test_training_dataset_uses_row_timestamp_but_inference_uses_final_history():
    df = events()
    ds = InteractionDataset(
        df, {1: 1, 2: 2}, {v: v for v in df.video_id},
        pd.DataFrame({"user_id": [1, 2]}), pd.DataFrame({"video_id": df.video_id}),
        {"1": [40, 30]}, [], [], [], 2,
    )
    assert len(ds) == 5  # Default target is long_view, not all six clicks.
    assert ds[0]["hist_idx"].tolist() == [20, 40]  # Event at 300 excludes itself.
    assert ds[1]["hist_idx"].tolist() == [0, 0]  # Event at 100 excludes future events.
    assert ds._user_feats(1)[2].tolist() == [40, 30]


def test_duplicate_items_are_not_contradictory_negatives():
    users = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    items = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    unmasked = sampled_softmax_loss(users, items, temperature=1.)
    assert unmasked.item() == pytest.approx(math.log(2))
    masked = sampled_softmax_loss(users, items, temperature=1., candidate_ids=torch.tensor([7, 7]))
    assert masked.item() == pytest.approx(0.)
    masked.backward()
    assert torch.isfinite(users.grad).all()
    assert torch.isfinite(items.grad).all()


def test_unique_item_loss_preserves_sampling_probability_correction():
    users = torch.eye(3)
    items = torch.eye(3)
    logq = torch.log(torch.tensor([.6, .3, .1]))
    expected = torch.nn.functional.cross_entropy(users @ items.T - logq[None, :], torch.arange(3))
    actual = sampled_softmax_loss(users, items, logq, temperature=1., candidate_ids=torch.tensor([4, 8, 9]))
    assert actual.item() == pytest.approx(expected.item())


def test_retrieval_cohort_is_stable_and_allows_repeat_targets():
    rel = {i: [i] for i in range(20)}
    assert evaluation_cohort(rel, 5, 42) == evaluation_cohort(dict(reversed(list(rel.items()))), 5, 42)
    train = pd.DataFrame({"user_id": [1, 2, 2], "video_id": [10, 10, 20], "long_view": [1, 1, 1]})
    assert popularity_recommend(train, [1], k=2)[1] == [10, 20]
    assert itemcf_recommend(train, [1], k=1)[1] == [10]
