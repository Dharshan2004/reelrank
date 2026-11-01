"""Train-fitted feature contract shared by training, evaluation and serving."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd


class RankingFeatureTransformer:
    """Encode IDs/categories and numerical quantiles; zero always means unknown.

    Vocabulary and cut points are learned only from training impressions. Numeric
    values outside that range use the outer bins, without fitting on new data.
    """
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.specs: list[dict] = []
        self.vocabularies: dict[str, dict[str, int]] = {}
        self.edges: dict[str, list[float]] = {}

    @property
    def field_cols(self) -> list[str]:
        return [s['name'] for s in self.specs]

    @property
    def field_dims(self) -> list[int]:
        return [len(self.edges[s['name']]) + 1 if s['numeric'] else len(self.vocabularies[s['name']]) for s in self.specs]

    @staticmethod
    def _table(df, key):
        return df.set_index(key) if key in df.columns else df

    def _values(self, interactions, users, videos, spec):
        if spec['source'] == 'interaction':
            return interactions[spec['column']]
        table = self._table(users, 'user_id') if spec['source'] == 'user' else self._table(videos, 'video_id')
        key = 'user_id' if spec['source'] == 'user' else 'video_id'
        if spec['column'] not in table.columns:
            return pd.Series(np.nan, index=interactions.index)
        return interactions[key].map(table[spec['column']])

    @staticmethod
    def _keys(values):
        # Stable across integer/float serialization and one-row serving inputs.
        return values.map(lambda x: None if pd.isna(x) else str(int(x)) if isinstance(x, (int, float, np.number)) and float(x).is_integer() else str(x))

    def fit(self, interactions, users, videos, meta):
        if interactions.empty:
            raise ValueError('Cannot fit ranking features without training interactions')
        self.specs = [dict(name='user_idx', source='interaction', column='user_id', numeric=False), dict(name='video_idx', source='interaction', column='video_id', numeric=False)]
        for source, cols, table in [('user', meta.get('user_cat_idx_cols', []), users), ('video', meta.get('video_cat_idx_cols', []), videos)]:
            for c in cols:
                if c in table.columns:
                    self.specs.append(dict(name=f'{source}_{c}', source=source, column=c, numeric=False))
        for c in meta.get('user_num_cols', []):
            if c in users.columns:
                self.specs.append(dict(name=f'user_{c}_bin', source='user', column=c, numeric=True))
        for c in ['video_duration', 'pop_clicks']:
            self.specs.append(dict(name=f'video_{c}_bin', source='video', column=c, numeric=True))
        self.vocabularies, self.edges = {}, {}
        for spec in self.specs:
            values = self._values(interactions, users, videos, spec)
            if spec['numeric']:
                numbers = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                cuts = np.unique(np.quantile(numbers, np.linspace(0, 1, self.n_bins + 1)[1:-1])) if len(numbers) else np.array([])
                self.edges[spec['name']] = cuts.tolist()
            else:
                keys = self._keys(values).dropna().unique()
                self.vocabularies[spec['name']] = {key: i + 1 for i, key in enumerate(sorted(keys))}
        return self

    def transform(self, interactions, users, videos):
        if not self.specs:
            raise ValueError('RankingFeatureTransformer must be fitted')
        # Index each entity table once per batch, not once per feature field.
        users = self._table(users, 'user_id')
        videos = self._table(videos, 'video_id')
        result = interactions.copy()
        for spec in self.specs:
            values = self._values(interactions, users, videos, spec)
            if spec['numeric']:
                numbers = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
                encoded = np.searchsorted(self.edges[spec['name']], numbers, side='right') + 1
                encoded[~np.isfinite(numbers)] = 0
            else:
                encoded = self._keys(values).map(self.vocabularies[spec['name']]).fillna(0).to_numpy(dtype=np.int64)
            result[spec['name']] = np.asarray(encoded, dtype=np.int64)
        return result

    def to_dict(self):
        return copy.deepcopy(dict(version=1, n_bins=self.n_bins, specs=self.specs, vocabularies=self.vocabularies, edges=self.edges))

    @classmethod
    def from_dict(cls, payload):
        if payload.get('version') != 1:
            raise ValueError('Unsupported ranking feature transformer version')
        obj = cls(payload['n_bins'])
        obj.specs = copy.deepcopy(payload['specs'])
        obj.vocabularies = copy.deepcopy(payload['vocabularies'])
        obj.edges = copy.deepcopy(payload['edges'])
        return obj
