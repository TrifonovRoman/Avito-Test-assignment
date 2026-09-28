"""Кеширование: общая реализация для ноутбуков и CLI."""

import hashlib
import json
from pathlib import Path
import sys
import types
import joblib
import numpy as np
from .sparse_retrieval import WordRetriever, CharRetriever


def corpus_hash(path):
    """Хешировать содержимое файла, чтобы перенос корпуса не делал индекс неактуальным."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cached_fit(path, cache_dir, representation, kind='word', *, cache_only=False):
    import importlib.metadata
    path, cache_dir = Path(path), Path(cache_dir)
    versions = tuple(importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'sparse-dot-topn', 'pandas', 'pyarrow', 'joblib'])
    digest = hashlib.sha256(repr(('sparse-v1', kind, representation, corpus_hash(path), versions)).encode())
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'{kind}-{representation}-{digest.hexdigest()[:16]}.joblib'
    if cache.exists():
        try:
            model = joblib.load(cache)
        except ModuleNotFoundError as error:
            if error.name != 'retrieval':
                raise
            legacy = types.ModuleType('retrieval')
            legacy.WordRetriever = WordRetriever
            legacy.CharRetriever = CharRetriever
            sys.modules['retrieval'] = legacy
            try:
                model = joblib.load(cache)
            finally:
                del sys.modules['retrieval']
        model.cache_key = digest.hexdigest()
        return model, True
    if cache_only:
        raise FileNotFoundError(f'Sparse index cache missing: {cache}')
    if kind not in ['word', 'char'] or kind == 'char' and representation != 'title':
        raise ValueError('Supported indexes: word title/title_params, char title')
    model = (WordRetriever(representation) if kind == 'word' else CharRetriever()).fit(path)
    model.cache_key = digest.hexdigest()
    joblib.dump(model, cache, compress=0)
    return model, False


def cached_search(model, texts, cache_dir, k=200, *, cache_only=False):
    """Кешировать top-k для точных текстов запросов, индекса, кода поиска и значения k."""
    import inspect
    import ast
    import textwrap
    source = inspect.getsource(type(model).search)
    compatibility = Path(cache_dir) / 'search_source_compat.json'
    if compatibility.exists():
        tree = ast.parse(textwrap.dedent(source))
        if ast.get_docstring(tree.body[0]) is not None:
            tree.body[0].body.pop(0)
        code_key = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
        source = json.loads(compatibility.read_text()).get(code_key, source)
    key = hashlib.sha256((model.cache_key + source
                          + json.dumps(texts, ensure_ascii=False) + str(k)).encode()).hexdigest()[:20]
    path = Path(cache_dir) / f'search-{key}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as data:
            return data['positions'], data['scores'], True
    if cache_only:
        raise FileNotFoundError(f'Retrieval cache missing: {path}')
    positions, scores = model.search(texts, k=k)
    np.savez_compressed(path, positions=positions, scores=scores)
    return positions, scores, False
