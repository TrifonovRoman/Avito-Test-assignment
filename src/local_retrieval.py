"""Дополнительный sparse-поиск в локации запроса без фильтрации глобальной выдачи."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sparse_dot_topn import sp_matmul_topn


def local_candidates(model, texts, query_locations, item_locations, cache_dir, k=100, *, cache_only=False):
    """Искать top-k внутри локации, сохраняя исходную ось item_id и TF-IDF-веса."""
    key = hashlib.sha256((model.cache_key + 'local-v1:' + str(k) + json.dumps(
        [texts, list(query_locations), list(item_locations)], ensure_ascii=False)).encode()).hexdigest()[:20]
    path = Path(cache_dir) / f'local-{key}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            return saved['positions'], saved['scores']
    if cache_only:
        raise FileNotFoundError(f'Local retrieval cache missing: {path}')
    positions = np.zeros((len(texts), k), dtype=np.int32)
    scores = np.zeros((len(texts), k), dtype=np.float32)
    query_matrix = model.vectorizer.transform(texts)
    column_index = model.index.tocsc()
    for location in np.unique(query_locations):
        query_rows = np.flatnonzero(query_locations == location)
        item_rows = np.flatnonzero(item_locations == location)
        if not len(item_rows):
            continue
        index = column_index[:, item_rows].tocsr()
        for start in range(0, len(query_rows), 128):
            batch = query_rows[start:start+128]
            found = sp_matmul_topn(query_matrix[batch], index, top_n=min(k, len(item_rows)),
                                  threshold=0., sort=True, n_threads=2)
            for offset, row in enumerate(batch):
                values = found.getrow(offset)
                ids = item_rows[values.indices]
                order = np.lexsort((ids, -values.data))
                count = len(order)
                positions[row, :count], scores[row, :count] = ids[order], values.data[order]
    np.savez_compressed(path, positions=positions, scores=scores)
    return positions, scores
