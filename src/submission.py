"""Подготовка ответа: общая реализация для ноутбуков и CLI."""

from pathlib import Path
import pandas as pd


def validate_submission(answer, query_ids, item_ids):
    """Проверить сформированную таблицу или CSV, прочитанный с dtype=str."""
    if list(answer.columns) != ['query_id', 'answer']:
        raise ValueError('Expected exactly query_id, answer columns')
    if not answer.query_id.is_unique or answer.query_id.tolist() != list(query_ids):
        raise ValueError('Query IDs must match the input, once each and in original order')
    known = set(item_ids)
    for value in answer.answer:
        if not isinstance(value, str):
            raise ValueError('Answers must be strings')
        ids = value.split()
        if not 1 <= len(ids) <= 50 or len(set(ids)) != len(ids) or not set(ids) <= known:
            raise ValueError('Each answer must contain 1–50 unique known item IDs')


def build_submission(queries, positions, item_ids):
    """Преобразовать позиции выдачи в исходные строковые ID с сохранением регистра."""
    if (queries.query_id.isna().any() or not queries.query_id.is_unique
            or not all(isinstance(value, str) for value in queries.query_id)):
        raise ValueError('Input query IDs must be unique non-null strings')
    answer = pd.DataFrame({'query_id': queries.query_id.tolist(),
                          'answer': [' '.join(item_ids[row]) for row in positions]})
    validate_submission(answer, queries.query_id, item_ids)
    return answer


def save_submission(answer, path, query_ids, item_ids):
    """Проверить данные до и после записи UTF-8 CSV; не преобразовывать ID в числа."""
    validate_submission(answer, query_ids, item_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(path, index=False, encoding='utf-8')
    reloaded = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_submission(reloaded, query_ids, item_ids)
    pd.testing.assert_frame_equal(answer, reloaded)


def check_submission(path, queries_path, items_path):
    """Проверить готовый CSV, прочитав из benchmark только столбцы идентификаторов."""
    answer = pd.read_csv(path, dtype=str, keep_default_na=False)
    queries = pd.read_parquet(queries_path, columns=['query_id'])
    items = pd.read_parquet(items_path, columns=['item_id'])
    validate_submission(answer, queries.query_id, items.item_id.unique())
    return {'queries': len(answer), 'candidates_per_query': answer.answer.str.split().map(len).unique().tolist()}
