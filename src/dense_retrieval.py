"""Локальная E5-small: возобновляемое построение float32-векторов и точный поиск FAISS."""

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .cache import corpus_hash
from .text import normalize

MODEL_NAME = 'intfloat/multilingual-e5-small'


def download_model(cache_dir, revision=None):
    """Однократное скачивание модели с закреплённой revision."""
    from huggingface_hub import HfApi, snapshot_download
    cache_dir = Path(cache_dir)
    manifest = cache_dir / 'model.json'
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if Path(info['path']).exists() and (revision is None or info['revision'] == revision):
            return info
    remote = HfApi().model_info(MODEL_NAME, revision=revision)
    files = {s.rfilename for s in remote.siblings}
    weights = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    path = snapshot_download(MODEL_NAME, revision=remote.sha, cache_dir=str(cache_dir / 'model'),
                             allow_patterns=['*.json', '*.model', weights], max_workers=2)
    info = {'name': MODEL_NAME, 'revision': remote.sha, 'path': str(Path(path).resolve())}
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(info, indent=2))
    return info


def load_encoder(info, max_seq_length=128):
    """Загрузить локальные веса E5 и использовать два CPU-потока без внешнего inference."""
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    model = SentenceTransformer(info['path'], device='cpu', local_files_only=True)
    model.max_seq_length = max_seq_length
    return model


class DenseE5Retriever:
    """Возобновляемое построение векторов заголовков и точный cosine-поиск через FAISS."""
    def __init__(self, corpus, cache_dir, model_info, max_seq_length=128):
        self.model_info, self.max_seq_length = model_info, max_seq_length
        self.metadata = {'model': model_info['name'], 'revision': model_info['revision'],
                         'max_seq_length': max_seq_length, 'representation': 'passage: normalized_title',
                         'corpus_sha256': corpus_hash(corpus), 'dtype': 'float32', 'version': 1,
                         'packages': {p: importlib.metadata.version(p) for p in
                                      ['sentence-transformers', 'torch', 'transformers', 'faiss-cpu', 'numpy']}}
        self.cache_key = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        self.directory = Path(cache_dir) / self.cache_key[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'metadata.json').write_text(json.dumps(self.metadata, indent=2))
        self.embedding_path = self.directory / 'items.npy'
        self.progress_path = self.directory / 'progress.json'
        if self.embedding_path.exists():
            self.item_ids = np.load(self.directory / 'item_ids.npy')
            self.embeddings = np.load(self.embedding_path, mmap_mode='r')
            self.titles = None
        else:
            frame = pd.read_parquet(corpus, columns=['item_id', 'item_title_raw'], dtype_backend='pyarrow')
            frame = frame.drop_duplicates('item_id').sort_values('item_id')
            self.item_ids = np.asarray(frame.item_id.tolist())
            self.inverse, titles = pd.factorize(frame.item_title_raw.map(normalize), sort=False)
            self.titles = titles.tolist()
            np.save(self.directory / 'item_ids.npy', self.item_ids)

    def progress(self):
        """Прочитать последний сохранённый checkpoint или начальное состояние нового корпуса."""
        return json.loads(self.progress_path.read_text()) if self.progress_path.exists() else {
            'completed': 0, 'seconds': 0., 'unique_titles': len(self.titles)}

    def encode(self, encoder, max_new=None, batch_size=32, max_seconds=2400):
        """Продолжить с последнего записанного блока"""
        if self.embedding_path.exists():
            return self.progress()
        state = self.progress()
        start = state['completed']
        stop = len(self.titles) if max_new is None else min(len(self.titles), start + max_new)
        unique_path = self.directory / 'unique.partial.npy'
        dim = encoder.get_embedding_dimension()
        if unique_path.exists():
            vectors = np.load(unique_path, mmap_mode='r+')
        else:
            vectors = np.lib.format.open_memmap(unique_path, mode='w+', dtype=np.float32,
                                              shape=(len(self.titles), dim))
        began = time.perf_counter()
        previous_seconds = state['seconds']
        for offset in range(start, stop, 512):
            end = min(offset + 512, stop)
            texts = ['passage: ' + t for t in self.titles[offset:end]]
            vectors[offset:end] = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                                 show_progress_bar=False, convert_to_numpy=True)
            vectors.flush()
            elapsed = time.perf_counter() - began
            state.update(completed=end, seconds=previous_seconds + elapsed, dimension=dim)
            pending = self.progress_path.with_suffix('.tmp')
            pending.write_text(json.dumps(state, indent=2))
            pending.replace(self.progress_path)
            rate = (end - start) / elapsed
            if max_new is not None or (end-start)//8192 > (offset-start)//8192 or end == stop:
                print(f'E5 {end}/{len(self.titles)} unique titles · {rate:.1f}/s · '
                      f'elapsed {elapsed:.0f}s · ETA {(len(self.titles)-end)/rate/60:.1f} min', flush=True)
            if elapsed > max_seconds and end < len(self.titles):
                raise TimeoutError('E5 encoding budget exhausted; partial embeddings saved')
        if state['completed'] == len(self.titles):
            destination = self.directory / 'items.partial.npy'
            expanded = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32,
                                                shape=(len(self.item_ids), dim))
            for offset in range(0, len(self.item_ids), 4096):
                expanded[offset:offset + 4096] = vectors[self.inverse[offset:offset + 4096]]
            expanded.flush()
            del expanded
            destination.replace(self.embedding_path)
            self.embeddings = np.load(self.embedding_path, mmap_mode='r')
        return state

    def query_embeddings(self, texts, encoder=None):
        """Кешировать нормированные векторы запросов в каталоге соответствующих корпуса и модели."""
        key = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()[:20]
        path = self.directory / f'queries-{key}.npy'
        if path.exists():
            return np.load(path)
        owns_encoder = encoder is None
        if owns_encoder:
            encoder = load_encoder(self.model_info, self.max_seq_length)
        vectors = encoder.encode(['query: ' + normalize(t) for t in texts], batch_size=32,
                                 normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True).astype(np.float32)
        np.save(path, vectors)
        if owns_encoder:
            del encoder
            gc.collect()
        return vectors

    def search(self, texts, k=200):
        import faiss
        if not self.embedding_path.exists():
            raise ValueError('Corpus embeddings are incomplete')
        q = self.query_embeddings(texts)
        faiss.omp_set_num_threads(2)
        index = faiss.IndexFlatIP(self.embeddings.shape[1])
        index.add(self.embeddings)
        scores, positions = index.search(q, k)
        for row in range(len(q)):
            order = np.lexsort((positions[row], -scores[row]))
            positions[row], scores[row] = positions[row, order], scores[row, order]
        return positions.astype(np.int32), scores.astype(np.float32)

    @property
    def index_mib(self):
        return self.embeddings.nbytes / 2**20


def check_cached_embeddings(corpus, texts, cache_dir, model_info):
    """Проверить небольшой готовый набор векторов без загрузки и запуска трансформера."""
    import faiss
    model = DenseE5Retriever(corpus, cache_dir, model_info)
    if not model.embedding_path.exists():
        raise FileNotFoundError('Corpus embeddings are incomplete')
    key = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()[:20]
    queries = np.load(model.directory / f'queries-{key}.npy')[:3]
    items = np.asarray(model.embeddings[:1000])
    assert items.dtype == queries.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(items, axis=1), 1., atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(queries, axis=1), 1., atol=1e-5)
    faiss.omp_set_num_threads(2)
    index = faiss.IndexFlatIP(items.shape[1])
    index.add(items)
    scores, _ = index.search(queries, 5)
    direct = np.sort(queries @ items.T, axis=1)[:, -5:][:, ::-1]
    np.testing.assert_allclose(scores, direct, atol=1e-5)
    return {'sample_items': len(items), 'sample_queries': len(queries),
            'dimension': items.shape[1], 'dtype': str(items.dtype), 'checks': 'passed'}
