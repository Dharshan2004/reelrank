import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from reelrank.eval.pipeline_eval import paired_bootstrap
from reelrank.ranking.features import RankingFeatureTransformer
from reelrank.ranking.inference import filter_candidates, load_ranker, rank_candidates, score_candidates
from reelrank.ranking.models import MMoE
from reelrank.serve.app import app


@pytest.fixture(params=["mmoe", "lr"])
def ranking_fixture(request):
    users = pd.DataFrame({'user_id': [1], 'activity': [3.]})
    videos = pd.DataFrame({'video_id': [10, 20, 30], 'video_duration': [4., 20., 90.], 'pop_clicks': [2, 5, 10], 'caption': ['a', 'b', 'c']})
    train = pd.DataFrame({'user_id': [1, 1, 1], 'video_id': [10, 20, 30]})
    transformer = RankingFeatureTransformer().fit(train, users, videos, {'user_num_cols': ['activity']})
    torch.manual_seed(13)
    model = MMoE(transformer.field_dims, n_experts=2)
    checkpoint = {'feature_transformer': transformer.to_dict(), 'field_cols': transformer.field_cols,
                  'field_dims': transformer.field_dims, 'n_experts': 2,
                  'state_dict': model.state_dict(), 'fusion_weights': [0.2, 0.7, 0.1]}
    if request.param == "lr":
        frame = transformer.transform(train, users, videos)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        x = encoder.fit_transform(frame[transformer.field_cols])
        models = {task: LogisticRegression().fit(x, [0, 1, 0]) for task in ["is_click", "long_view", "is_like"]}
        checkpoint = {"feature_transformer": transformer.to_dict(), "features": transformer.field_cols,
                      "encoder": encoder, "models": models}
    return load_ranker(checkpoint, name=request.param), users, videos


class FakeTower:
    def encode_user(self, *args):
        return torch.zeros((1, 2))


class FakeIndex:
    def search(self, embeddings, k):
        return np.zeros((1, 3)), np.array([[0, 1, 2]])


@pytest.mark.parametrize('uid', [1, 999])
def test_api_and_offline_scores_match_frozen_training_transform(ranking_fixture, monkeypatch, uid):
    runtime, users, videos = ranking_fixture
    safety = pd.DataFrame({'video_id': [10, 20, 30], 'rule_hit': [False, True, False], 'safety_score': [0., 0., 0.2]}).set_index('video_id')
    state = {'ranking': runtime, 'users_df': users.set_index('user_id'), 'videos_df': videos.set_index('video_id'),
             'histories': {}, 'user_id_map': {'1': 1}, 'tt': FakeTower(), 'index': FakeIndex(),
             'item_ids': [10, 20, 30], 'safety': safety,
             'meta': {'user_cat_cols': [], 'user_num_cols': ['activity'], 'history_n': 2}}
    monkeypatch.setattr('reelrank.serve.app.STATE', state)
    # No lifespan context: avoid loading real production artifacts in this test.
    response = TestClient(app).get('/recommend', params={'user_id': uid, 'k': 2})
    assert response.status_code == 200
    body = response.json()
    kept = filter_candidates([10, 20, 30], safety)
    expected = rank_candidates(runtime, uid, kept, users, videos, 2)
    assert [item['video_id'] for item in body['items']] == expected
    proba, scores = score_candidates(runtime, uid, kept, users, videos)
    # Check actual training transform/model calculation, not only shared helper agreement.
    frame = runtime.transformer.transform(pd.DataFrame({'user_id': [uid] * 2, 'video_id': kept}), users, videos)
    if runtime.name == "lr":
        encoded = runtime.encoder.transform(frame[runtime.transformer.field_cols])
        train_proba = np.column_stack([runtime.model[t].predict_proba(encoded)[:, 1] for t in ["is_click", "long_view", "is_like"]])
        np.testing.assert_allclose(scores, train_proba[:, 1])
    else:
        with torch.inference_mode():
            train_proba = torch.sigmoid(runtime.model(torch.tensor(frame[runtime.transformer.field_cols].values))).numpy()
    assert body["ranker"] == runtime.name
    np.testing.assert_allclose(proba, train_proba)
    for item in body['items']:
        i = kept.index(item['video_id'])
        assert item['score'] == pytest.approx(float(scores[i]))
        assert item['p_long_view'] == pytest.approx(float(train_proba[i, 1]))
    assert body['diagnostics'] == {'retrieved': 3, 'after_safety': 2, 'blocked': 1}


def test_policy_is_order_independent_for_both_candidate_sources():
    safety = pd.DataFrame({'video_id': [10, 20, 30], 'rule_hit': [True, False, False], 'safety_score': [0., 0.5, 0.49]})
    neural, popular = [10, 20, 30, 40], [40, 30, 20, 10]
    assert filter_candidates(neural, safety) == [30, 40]
    assert filter_candidates(popular, safety) == [40, 30]
    assert filter_candidates(neural, None) == neural


def test_legacy_checkpoint_fails_explicitly():
    with pytest.raises(ValueError, match='rerun make train-ranker'):
        load_ranker({})
    with pytest.raises(ValueError, match="rerun make train-ranker"):
        load_ranker({"models": {}}, name="lr")


def test_paired_bootstrap_preserves_user_pairing():
    result = paired_bootstrap(np.array([0.2, 0.4, 0.8]), np.array([0.1, 0.3, 0.7]), n_bootstrap=100)
    np.testing.assert_allclose(result['difference_ci95'], [0.1, 0.1])
    zero = paired_bootstrap(np.zeros(3), np.zeros(3), n_bootstrap=100)
    assert zero['relative_lift_ci95'] is None
    assert zero['zero_baseline_resamples'] == 100


def test_api_reports_unavailable_models_and_rejects_invalid_request(monkeypatch):
    monkeypatch.setattr('reelrank.serve.app.STATE', {})
    client = TestClient(app)
    assert client.get('/recommend', params={'user_id': 1}).status_code == 503
    assert client.get('/recommend', params={'user_id': 1, 'k': 0}).status_code == 422
    assert client.get('/recommend', params={'user_id': 1, 'retrieve_k': 501}).status_code == 422


def test_empty_and_unseen_candidates_are_safe_to_score(ranking_fixture):
    runtime, users, videos = ranking_fixture
    probabilities, scores = score_candidates(runtime, 999, [], users, videos)
    assert probabilities.shape == (0, 3)
    assert scores.shape == (0,)
    probabilities, scores = score_candidates(runtime, 999, [9999], users, videos)
    assert np.isfinite(probabilities).all()
    assert np.isfinite(scores).all()
    assert rank_candidates(runtime, 999, [], users, videos, 10) == []
