import json

import numpy as np
import pandas as pd
import torch

from reelrank.ranking.features import RankingFeatureTransformer
from reelrank.ranking.models import DeepFM


def fixture():
    users = pd.DataFrame({'user_id': [1, 2, 3], 'age': [20, 40, 900], 'group': ['a', 'b', 'c']})
    videos = pd.DataFrame({'video_id': [10, 20, 30], 'video_duration': [10., 20., 999.], 'pop_clicks': [1., 2., 999.], 'genre': ['x', 'y', 'z']})
    train = pd.DataFrame({'user_id': [1, 2, 1, 2], 'video_id': [10, 20, 20, 10], 'long_view': [0, 1, 1, 0]})
    meta = {'user_cat_idx_cols': ['group'], 'video_cat_idx_cols': ['genre'], 'user_num_cols': ['age']}
    return train, users, videos, meta


def test_serialized_batch_and_single_request_features_match():
    train, users, videos, meta = fixture()
    transformer = RankingFeatureTransformer().fit(train, users, videos, meta)
    restored = RankingFeatureTransformer.from_dict(json.loads(json.dumps(transformer.to_dict())))
    batch = transformer.transform(train, users, videos)
    for i in range(len(train)):
        single = restored.transform(train.iloc[[i]][['user_id', 'video_id']], users.set_index('user_id'), videos.set_index('video_id'))
        np.testing.assert_array_equal(single[restored.field_cols].iloc[0], batch[transformer.field_cols].iloc[i])
    assert batch.loc[0, 'user_idx'] != batch.loc[1, 'user_idx']
    assert batch.loc[0, 'user_age_bin'] != batch.loc[1, 'user_age_bin']


def test_unseen_vocab_and_missing_features_are_unknown_without_refit():
    train, users, videos, meta = fixture()
    transformer = RankingFeatureTransformer().fit(train, users, videos, meta)
    before = transformer.to_dict()
    new = pd.DataFrame({'user_id': [3, 99], 'video_id': [30, 99]})
    result = transformer.transform(new, users, videos)
    assert result.user_idx.tolist() == [0, 0]
    assert result.video_idx.tolist() == [0, 0]
    assert result.user_group.tolist() == [0, 0]
    assert result.video_genre.tolist() == [0, 0]
    assert result.user_age_bin.iloc[1] == 0
    assert transformer.to_dict() == before
    for field, maximum in zip(transformer.field_cols, transformer.field_dims):
        assert result[field].between(0, maximum).all()


def test_transform_independent_of_eval_cohort_distribution():
    train, users, videos, meta = fixture()
    transformer = RankingFeatureTransformer().fit(train, users, videos, meta)
    row = pd.DataFrame({'user_id': [1], 'video_id': [10]})
    other = pd.DataFrame({'user_id': [3] * 100, 'video_id': [30] * 100})
    single = transformer.transform(row, users, videos)
    batch = transformer.transform(pd.concat([row, other], ignore_index=True), users, videos)
    np.testing.assert_array_equal(single[transformer.field_cols].iloc[0], batch[transformer.field_cols].iloc[0])


def test_deepfm_initial_logits_not_saturated_and_padding_zero():
    torch.manual_seed(42)
    model = DeepFM([100] * 20)
    logits = model(torch.randint(1, 101, (128, 20)))
    assert logits.abs().max() < 1
    for emb in model.embeddings:
        assert torch.count_nonzero(emb.weight[0]) == 0
