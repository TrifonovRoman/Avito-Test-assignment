"""Подготовка текста: общая реализация для ноутбуков и CLI."""

import re
import pandas as pd


def normalize(text):
    """Одинаково нормализовать регистр, ё и пунктуацию в запросах и заголовках."""
    text = '' if pd.isna(text) else str(text).lower().replace('ё', 'е')
    return ' '.join(re.sub(r'[^\w\s]|_', ' ', text).split())


def query_texts(frame, representation='query_params'):
    """Подготовить запросы; добавление параметров явно задаётся вариантом эксперимента."""
    if representation == 'query_only':
        return frame.search_query.map(normalize).tolist()
    if representation != 'query_params':
        raise ValueError('Unknown query representation')
    return (frame.search_query.fillna('') + ' ' + frame.search_infm_params_text.fillna('')).map(normalize).tolist()
