"""Общий интерфейс выполнения поиска для экспериментов и генерации ответа."""

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .cache import cached_fit, cached_search
from .fusion import reciprocal_rank_fusion
from .text import query_texts


@dataclass
class Candidates:
    """Позиции и оценки top-k на отсортированной оси item_id и сведения о кешировании."""

    item_ids: np.ndarray
    positions: np.ndarray
    scores: np.ndarray
    build_or_load_s: float
    search_or_load_s: float
    index_mib: float
    cache_hit: bool | None

    @property
    def source(self):
        """Пара позиций и оценок для объединения выдач и расчёта покрытия кандидатов."""
        return self.positions, self.scores


def retrieve(corpus, queries, spec, cache_dir, *, model_info=None, top_k=200,
             search_dir=None, cache_only=False):
    """Получить выдачу источника, используя кеши индексов, векторов и top-k запросов."""
    cache_dir = Path(cache_dir)
    started = time.perf_counter()
    if spec['kind'] == 'dense':
        from .dense_retrieval import DenseE5Retriever, download_model, load_encoder
        if cache_only:
            info = json.loads((cache_dir / 'e5/model.json').read_text())
            if info['revision'] != model_info['revision']:
                raise ValueError('Cached E5 revision differs from configuration')
        else:
            info = download_model(cache_dir / 'e5', revision=model_info['revision'])
        model = DenseE5Retriever(corpus, cache_dir / 'e5', info)
        if not model.embedding_path.exists():
            if cache_only:
                raise FileNotFoundError(f'Incomplete E5 embeddings: {model.directory}')
            encoder = load_encoder(info)
            model.encode(encoder)
            model.query_embeddings(query_texts(queries, spec['query']), encoder)
            del encoder
            gc.collect()
    else:
        model, _ = cached_fit(corpus, cache_dir / 'word_baseline', spec['document'],
                              spec['kind'], cache_only=cache_only)
    build_s = time.perf_counter() - started
    started = time.perf_counter()
    if spec.get('scope') == 'same_location':
        from .local_retrieval import local_candidates
        positions, scores = local_candidates(
            model, query_texts(queries, spec['query']),
            queries.search_location_id.astype(str).to_numpy(), item_locations(corpus, model.item_ids),
            cache_dir / 'final_pass', k=top_k, cache_only=cache_only)
        hit = None
    else:
        positions, scores, hit = cached_search(
            model, query_texts(queries, spec['query']), search_dir or cache_dir / 'hybrid',
            k=top_k, cache_only=cache_only)
    result = Candidates(model.item_ids, positions, scores, build_s,
                        time.perf_counter() - started, model.index_mib, hit)
    del model
    gc.collect()
    return result


def retrieve_sources(corpus, queries, config, cache_dir, **options):
    """Последовательно получить выдачи источников и проверить общую ось item_id."""
    results = {}
    for spec in config['sources']:
        name = spec.get('name', spec['kind'])
        if name in results:
            raise ValueError('Источникам одного типа нужны разные name')
        result = retrieve(corpus, queries, spec, cache_dir,
                          model_info=config.get('e5_model'),
                          top_k=spec.get('top_k', config['top_k_per_source']), **options)
        if results:
            np.testing.assert_array_equal(next(iter(results.values())).item_ids, result.item_ids)
        results[name] = result
    return results


def item_locations(corpus, item_ids):
    """Сопоставить локации позициям по item_id, а не по порядку строк в parquet."""
    frame = pd.read_parquet(corpus, columns=['item_id', 'item_location_id'], dtype_backend='pyarrow')
    values = frame.drop_duplicates('item_id').set_index('item_id').item_location_id.reindex(item_ids)
    if values.isna().any():
        raise ValueError('Missing item location metadata')
    return values.astype(str).to_numpy()


def fuse(results, queries, config, locations):
    """Применить фиксированные веса RRF и мягкий бонус локации к найденным кандидатам."""
    return reciprocal_rank_fusion(
        [results[spec.get('name', spec['kind'])].source for spec in config['sources']],
        weights=config['weights'], constant=config['rrf_constant'],
        item_locations=locations,
        query_locations=queries.search_location_id.astype(str).to_numpy(),
        location_weight=config['location_weight'])


def predict(corpus, queries, config, cache_dir, *, cache_only=False):
    """Вернуть позиции top-50 и item_id по фиксированной hybrid-конфигурации."""
    results = retrieve_sources(corpus, queries, config, cache_dir, cache_only=cache_only)
    ids = next(iter(results.values())).item_ids
    positions = fuse(results, queries, config, item_locations(corpus, ids))
    return positions, ids
