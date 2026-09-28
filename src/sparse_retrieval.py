"""Разреженный поиск: общая реализация для ноутбуков и CLI."""

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from .text import normalize


class WordRetriever:
    """Word TF-IDF с отсортированной осью item_id и пакетным разреженным поиском top-k."""
    def __init__(self, representation='title_params', max_features=150_000):
        if representation not in ['title', 'title_params']:
            raise ValueError('representation must be title or title_params')
        self.representation = representation
        self.vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=(1, 2), min_df=2, max_df=.98,
            sublinear_tf=True, norm='l2', dtype=np.float32, max_features=max_features,
        )

    def fit(self, path):
        """Убрать дубли ID перед векторизацией, не сохраняя все исходные тексты в памяти."""
        ids, seen = [], set()
        columns = ['item_id', 'item_title_raw']
        if self.representation == 'title_params':
            columns.append('item_infm_params_text')

        def documents():
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
                for row in batch.to_pylist():
                    item_id = row['item_id']
                    if not isinstance(item_id, str):
                        raise ValueError('item_id must be a string')
                    if item_id in seen:
                        continue
                    seen.add(item_id)
                    ids.append(item_id)
                    title = normalize(row['item_title_raw'])
                    yield title if self.representation == 'title' else (
                        title + ' ' + title + ' ' + normalize(row['item_infm_params_text']))

        matrix = self.vectorizer.fit_transform(documents())
        order = np.argsort(ids)
        self.item_ids = np.asarray(ids)[order]
        self.index = matrix[order].T.tocsr()
        self.index.sort_indices()
        return self

    def search(self, texts, k=50, batch_size=128, n_threads=2):
        """Вернуть позиции объявлений и cosine без плотной матрицы запросы * каталог.

        При нехватке совпадений выдача дополняется первыми ID в лексикографическом порядке.
        Равные положительные оценки разрешаются детерминированным отбором, затем порядком ID."""
        if not 1 <= k <= len(self.item_ids):
            raise ValueError('k must be between 1 and corpus size')
        positions = np.empty((len(texts), k), dtype=np.int32)
        scores = np.zeros((len(texts), k), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            q = self.vectorizer.transform(texts[start:start + batch_size])
            result = sp_matmul_topn(q, self.index, top_n=k, threshold=0., sort=True, n_threads=n_threads)
            for offset in range(result.shape[0]):
                row = result.getrow(offset)
                order = np.lexsort((row.indices, -row.data))
                idx, values = row.indices[order], row.data[order]
                if len(idx) < k:
                    used = set(idx)
                    padding = [i for i in range(k) if i not in used][:k - len(idx)]
                    idx = np.concatenate([idx, padding])
                positions[start + offset] = idx
                scores[start + offset, :len(values)] = values
        return positions, scores

    @property
    def index_mib(self):
        return sum(a.nbytes for a in [self.index.data, self.index.indices, self.index.indptr]) / 2**20


class CharRetriever(WordRetriever):
    """Символьные n-граммы заголовков для вариантов написания и частичных совпадений слов."""
    def __init__(self):
        self.representation = 'title'
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=150_000,
            sublinear_tf=True, norm='l2', dtype=np.float32,
        )
