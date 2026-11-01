"""Comparable rankers with train-only features and standard-validation selection.

The random log has been inspected in earlier experiments: final random metrics
are exploratory reuse, not a newly untouched holdout.
"""
from __future__ import annotations

import argparse
import copy
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.eval.metrics import auc_score, binary_logloss, gauc
from reelrank.eval.runlog import write_result
from reelrank.ranking.features import RankingFeatureTransformer
from reelrank.ranking.fusion import fuse, fusion_grid
from reelrank.ranking.models import DeepFM, MMoE

TASKS = ['is_click', 'long_view', 'is_like']


def _sample(df, n, seed):
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def _metrics(frame, probabilities):
    return {task: {'auc': auc_score(frame[task], probabilities[:, i]),
                   'gauc': gauc(frame[task], probabilities[:, i], frame['user_id']),
                   'logloss': binary_logloss(frame[task], probabilities[:, i])}
            for i, task in enumerate(TASKS)}


def predict_neural(model, frame, fields, batch_size=8192):
    model.eval()
    x = frame[fields].to_numpy(dtype=np.int64)
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(x[i:i + batch_size]))).numpy()
                               for i in range(0, len(x), batch_size)])


def _fit_neural(model, train, val, fields, epochs, batch_size, task_index=None, weights=(1., 1.5, 1.)):
    dataset = TensorDataset(torch.tensor(train[fields].to_numpy(), dtype=torch.long),
                            torch.tensor(train[TASKS].to_numpy(), dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_loss, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in loader:
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y[:, task_index]) if task_index is not None else sum(
                weights[i] * F.binary_cross_entropy_with_logits(logits[:, i], y[:, i]) for i in range(3)) / sum(weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
        probabilities = predict_neural(model, val, fields)
        validation_loss = binary_logloss(val[TASKS[task_index]], probabilities) if task_index is not None else sum(
            weights[i] * binary_logloss(val[task], probabilities[:, i]) for i, task in enumerate(TASKS)) / sum(weights)
        history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation_logloss': validation_loss})
        print(history[-1], flush=True)
        if validation_loss < best_loss:
            best_loss, best_state, best_epoch = validation_loss, copy.deepcopy(model.state_dict()), epoch + 1
    if best_state is None:
        raise ValueError('No finite validation checkpoint; check label support')
    model.load_state_dict(best_state)
    return best_epoch, history


def run(seed=42, max_train=400_000, max_val=80_000, epochs=3, batch_size=2048):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PROCESSED / 'id_maps.json').read_text())
    users = pd.read_parquet(PROCESSED / 'users.parquet')
    videos = pd.read_parquet(PROCESSED / 'videos.parquet')
    train = _sample(pd.read_parquet(PROCESSED / 'interactions_train.parquet'), max_train, seed)
    val = _sample(pd.read_parquet(PROCESSED / 'interactions_val.parquet'), max_val, seed)
    transformer = RankingFeatureTransformer().fit(train, users, videos, meta)
    train, val = [transformer.transform(frame, users, videos) for frame in [train, val]]
    fields, dims = transformer.field_cols, transformer.field_dims
    transformer_payload = transformer.to_dict()
    common = dict(seed=seed, max_train=max_train, max_val=max_val, train_rows=len(train), validation_rows=len(val),
                  selection_split='standard_validation', feature_fields=fields,
                  random_evaluation_status='previously_inspected_exploratory_reuse')
    selected = {}

    print('training LR', flush=True)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
    x_train = encoder.fit_transform(train[fields])
    models = {}
    for task in TASKS:
        clf = LogisticRegression(max_iter=200, solver='liblinear', random_state=seed)
        clf.fit(x_train, train[task])
        models[task] = clf
    lr_payload = dict(models=models, encoder=encoder, features=fields, feature_transformer=transformer_payload)
    joblib.dump(lr_payload, ARTIFACTS / 'rank_lr.joblib')
    selected['lr'] = lr_payload

    print('training DeepFM', flush=True)
    deep_models, deep_epochs, deep_history = [], {}, {}
    for i, task in enumerate(TASKS):
        model = DeepFM(dims)
        best_epoch, history = _fit_neural(model, train, val, fields, epochs, batch_size, task_index=i)
        deep_models.append(model)
        deep_epochs[task], deep_history[task] = best_epoch, history
        torch.save(dict(state_dict=model.state_dict(), field_cols=fields, field_dims=dims,
                        feature_transformer=transformer_payload, selected_epoch=best_epoch), ARTIFACTS / f'deepfm_{task}.pt')
    selected['deepfm'] = deep_models
    (ARTIFACTS / 'deepfm_fields.json').write_text(json.dumps(dict(field_cols=fields, field_dims=dims, feature_transformer=transformer_payload)))

    print('training MMoE', flush=True)
    model = MMoE(dims, n_experts=4)
    best_epoch, history = _fit_neural(model, train, val, fields, epochs, batch_size)
    val_probs = predict_neural(model, val, fields)
    trade = fusion_grid(val_probs, val['long_view'].to_numpy())
    best = trade[0]
    weights = [best['w_click'], best['w_long'], best['w_like']]
    torch.save(dict(state_dict=model.state_dict(), field_cols=fields, field_dims=dims, n_experts=4,
                    feature_transformer=transformer_payload, fusion_weights=weights, selected_epoch=best_epoch,
                    selection_split='standard_validation'), ARTIFACTS / 'mmoe.pt')
    selected['mmoe'] = model

    # No random labels are loaded until every checkpoint and fusion choice is fixed.
    random = _sample(pd.read_parquet(PROCESSED / 'interactions_test_random.parquet'), max_val, seed)
    random = transformer.transform(random, users, videos)
    for split_name, frame in [('standard_validation', val), ('random_exposure_exploratory', random)]:
        for name, fitted in selected.items():
            if name == 'lr':
                x = fitted['encoder'].transform(frame[fields])
                proba = np.column_stack([fitted['models'][t].predict_proba(x)[:, 1] for t in TASKS])
            elif name == 'deepfm':
                proba = np.column_stack([predict_neural(m, frame, fields) for m in fitted])
            else:
                proba = predict_neural(fitted, frame, fields)
            metrics = _metrics(frame, proba)
            extra = {}
            if name == 'mmoe':
                scores = fuse(proba, tuple(weights))
                metrics['fusion'] = dict(auc_long_view=auc_score(frame.long_view, scores), gauc_long_view=gauc(frame.long_view, scores, frame.user_id))
                extra = dict(validation_fusion_selection=best, training_history=history, selected_epoch=best_epoch)
            elif name == 'deepfm':
                extra = dict(training_history=deep_history, selected_epochs=deep_epochs)
            run_id = f'ranking_{name}' if split_name == 'random_exposure_exploratory' else f'ranking_{name}_validation'
            write_result(run_id, 'ranking', metrics, config={**common, 'model': name, 'epochs': epochs if name != 'lr' else None,
                         'evaluation_split': split_name, 'evaluation_rows': len(frame),
                         **({'fusion_weights': weights, 'n_experts': 4, 'loss_weights': [1., 1.5, 1.]} if name == 'mmoe' else {})}, seed=seed, extra=extra)
            print(run_id, metrics, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train', type=int, default=400000)
    parser.add_argument('--max-val', type=int, default=80000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
