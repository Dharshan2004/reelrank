import numpy as np

from reelrank.eval.metrics import (
    auc_score,
    catalogue_coverage,
    expected_calibration_error,
    gauc,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_and_hit():
    rel = [1, 2, 3]
    rec = [3, 9, 1, 8]
    assert recall_at_k(rel, rec, 2) == 1 / 3
    assert hit_rate_at_k(rel, rec, 1) == 1.0
    assert hit_rate_at_k(rel, [9, 8], 2) == 0.0


def test_ndcg_perfect():
    rel = [1, 2]
    rec = [1, 2, 3]
    assert abs(ndcg_at_k(rel, rec, 2) - 1.0) < 1e-9


def test_auc_gauc():
    y = [0, 1, 0, 1]
    s = [0.1, 0.9, 0.2, 0.8]
    assert auc_score(y, s) > 0.9
    g = gauc(y, s, [1, 1, 2, 2])
    assert g > 0.9


def test_ece_and_coverage():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    assert expected_calibration_error(y, s) >= 0
    assert catalogue_coverage([[1, 2], [2, 3]], 5) == 0.6
