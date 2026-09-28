"""Метрики: общая реализация для ноутбуков и CLI."""

import numpy as np


def recall_per_query(positions, truth, item_ids, k=50):
    """Доля найденных уникальных релевантных объявлений для каждого контекста запроса."""
    if len(positions) != len(truth):
        raise ValueError('Predictions and ground truth must have the same length')
    if any(not values for values in truth):
        raise ValueError('Ground truth must be non-empty for each query')
    return np.asarray([len(set(item_ids[row[:k]]) & relevant) / len(relevant)
                       for row, relevant in zip(positions, truth)], dtype=np.float64)


def candidate_recall(sources, truth, item_ids):
    """Средний recall объединения источников без технического дополнения с нулевыми оценками."""
    values = []
    for row, relevant in enumerate(truth):
        union = set()
        for indices, scores in sources:
            union.update(item_ids[indices[row][scores[row] > 0]])
        values.append(len(union & relevant) / len(relevant))
    return float(np.mean(values))


def evaluate(positions, truth, item_ids, sources):
    """Посчитать средние Recall@10/20/50 и покрытие кандидатов на одних контекстах."""
    return {**{f'R@{k}': float(recall_per_query(positions, truth, item_ids, k).mean())
               for k in (10, 20, 50)},
            'candidate_recall': candidate_recall(sources, truth, item_ids)}
