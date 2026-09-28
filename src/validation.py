"""Валидация: общая реализация для ноутбуков и CLI."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .text import normalize

CONTEXT = ['search_query', 'search_category', 'search_location_id', 'search_is_delivery_search', 'search_infm_params_text']


def in_validation(text):
    """Выделить около 15% нормализованных текстов запросов по исходному правилу SHA256."""
    digest = hashlib.sha256(('avito-validation-v1:42:' + text).encode()).digest()
    return int.from_bytes(digest[:8], 'big') < int(.15 * 2**64)


def validation_data(path):
    """Воспроизвести нормализованные контексты и отложенные тексты из 01_eda.ipynb."""
    frame = pd.read_parquet(path, columns=CONTEXT + ['item_id'], dtype_backend='pyarrow')
    for c in ['search_category', 'search_location_id', 'item_id']:
        frame[c] = frame[c].astype('string')
    for c in ['search_query', 'search_infm_params_text']:
        frame[c] = frame[c].map(normalize).astype('string')
    frame['context_code'] = pd.factorize(pd.MultiIndex.from_frame(frame[CONTEXT]), sort=False)[0]
    mask = frame.search_query.map({q: in_validation(q) for q in frame.search_query.unique()})
    assert not set(frame.loc[mask, 'search_query']) & set(frame.loc[~mask, 'search_query'])
    held = frame.loc[mask].drop_duplicates(['context_code', 'item_id'])
    queries = held.drop_duplicates('context_code').set_index('context_code')[CONTEXT]
    grouped = held.groupby('context_code', sort=False).item_id.agg(list).reindex(queries.index)
    truth = [set(values) for values in grouped.tolist()]
    return queries, truth


def quick_validation(queries, truth, manifest, n=2452):
    """Получить фиксированную выборку без учёта моделей; запретить незаметное изменение split."""
    keys = [hashlib.sha256(('quick-validation-v1:42:' + json.dumps(
        [str(v) for v in row], ensure_ascii=False)).encode()).hexdigest()
        for row in queries[CONTEXT].itertuples(index=False, name=None)]
    positions = np.argsort(keys)[:n]
    if len(positions) != n:
        raise ValueError('Not enough validation contexts')
    payload = {'version': 1, 'size': n, 'rule': 'lowest SHA256 of full normalized context, seed 42',
               'context_codes': queries.index[positions].tolist(),
               'keys_sha256': hashlib.sha256(''.join(keys[i] for i in positions).encode()).hexdigest()}
    manifest = Path(manifest)
    if manifest.exists():
        if json.loads(manifest.read_text()) != payload:
            raise ValueError('Quick-validation manifest no longer matches the data; do not resample')
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return queries.iloc[positions], [truth[i] for i in positions]


def load_validation(root, subset='quick'):
    """Загрузить исходную full, quick или отдельную по контекстам confirmation-выборку.

    Confirmation точно воспроизводит seed и порядок строк исходного ноутбука.
    Она использовалась для сравнения финалистов и не является нетронутым тестом."""
    root = Path(root)
    queries, truth = validation_data(root / 'data/train.parquet')
    if subset == 'full':
        return queries, truth
    quick, quick_truth = quick_validation(
        queries, truth, root / 'notebooks/quick_validation.json')
    if subset == 'quick':
        return quick, quick_truth
    if subset != 'confirmation':
        raise ValueError('subset must be full, quick or confirmation')
    remaining = np.flatnonzero(~queries.index.isin(quick.index))
    positions = np.random.default_rng(20260927).choice(remaining, size=2452, replace=False)
    positions.sort()
    return queries.iloc[positions].copy(), [truth[i] for i in positions]
