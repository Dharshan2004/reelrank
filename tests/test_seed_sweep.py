import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from reelrank.eval.seed_summary import paired_interval, validate_pair, CONTRACT, load_record
from reelrank.eval.seed_sweep import child_environment


def test_paired_interval_preserves_pairing_and_is_reproducible():
    assert paired_interval(np.full(20, .2), 200) == pytest.approx([.2, .2])
    values = np.array([-.2, .1, .3, -.1])
    assert paired_interval(values, 200) == paired_interval(values, 200)
    with pytest.raises(ValueError):
        paired_interval([float('nan')])


def test_comparison_rejects_mismatched_users_or_contract():
    record = {"config": dict.fromkeys(CONTRACT, 42),
              "provenance": {"input_files": {"data/processed/events.parquet": "same"}}}
    a = pd.Series([.1, .2], index=[1, 2])
    validate_pair(record, record, a, a)
    with pytest.raises(ValueError, match="users"):
        validate_pair(record, record, a, pd.Series([.1, .2], index=[1, 3]))
    other = {**record, "config": {**record['config'], "relevance_hash": "different"}}
    with pytest.raises(ValueError, match="contracts"):
        validate_pair(record, other, a, a)


def test_training_seeds_share_cohort_and_keep_outputs_isolated(tmp_path):
    data = tmp_path / "data" / "processed"
    data.mkdir(parents=True)
    meta = {"user_id_map": {str(u): u for u in range(1, 6)},
            "video_id_map": {str(v): v for v in range(1, 5)},
            "n_users": 5, "n_videos": 4, "history_n": 2,
            "user_cat_idx_cols": [], "video_cat_idx_cols": [], "user_num_cols": ["activity"]}
    (data / 'id_maps.json').write_text(json.dumps(meta))
    (data / 'histories.json').write_text(json.dumps({str(u): [1, 3] for u in range(1, 6)}))
    pd.DataFrame({'user_id': range(1, 6), 'activity': [1.] * 5}).to_parquet(data / 'users.parquet')
    pd.DataFrame({'video_id': range(1, 5), 'video_duration': [1., 2., 3., 4.],
                  'pop_clicks': [3., 2., 1., 0.]}).to_parquet(data / 'videos.parquet')
    events = pd.DataFrame([{'user_id': u, 'video_id': v, 'time_ms': v*100,
                           'long_view': int(v % 2 == 1)} for u in range(1, 6) for v in range(1, 5)])
    events.to_parquet(data / 'interactions_train.parquet')
    events.to_parquet(data / 'interactions_test_random.parquet')
    records = []
    sentinel = tmp_path / 'serving.pt'
    sentinel.write_bytes(b'original')
    for seed in (42, 43):
        artifacts = tmp_path / f'artifacts{seed}'
        results = tmp_path / f'results{seed}'
        env = child_environment(artifacts, results)
        env['REELRANK_DATA_DIR'] = str(data.parent)
        process = subprocess.run([sys.executable, '-m', 'reelrank.retrieval.train',
                                  '--epochs', '1', '--batch-size', '8', '--emb-dim', '4',
                                  '--seed', str(seed), '--cohort-seed', '42', '--max-eval-users', '2'],
                                 env=env, capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        record, scores = load_record(results / 'retrieval_two_tower.json')
        assert record['seed'] == seed
        assert (artifacts / 'two_tower.pt').is_file()
        assert (artifacts / 'index_flat.faiss').is_file()
        assert len(scores) == 2
        records.append((record, scores))
    validate_pair(records[0][0], records[1][0], records[0][1], records[1][1])
    assert sentinel.read_bytes() == b'original'


def test_record_loader_rejects_changed_evidence(tmp_path):
    from reelrank.eval.runlog import file_hash
    scores = tmp_path / 'scores.parquet'
    pd.DataFrame({'user_id': [1, 2], 'recall@200': [.1, .3]}).to_parquet(scores)
    record = {'extra': {'archived_per_user_scores': str(scores), 'per_user_sha256': file_hash(scores)},
              'metrics': {'recall@200': .2}}
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(record))
    load_record(path)
    record['metrics']['recall@200'] = .5
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='reported mean'):
        load_record(path)
    pd.DataFrame({'user_id': [1, 2], 'recall@200': [.2, .4]}).to_parquet(scores)
    with pytest.raises(ValueError, match='checksum'):
        load_record(path)
