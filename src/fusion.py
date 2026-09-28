"""Объединение выдач: общая реализация для ноутбуков и CLI."""

import numpy as np


def reciprocal_rank_fusion(sources, weights=None, k=50, constant=60,
                           item_locations=None, query_locations=None, location_weight=0.):
    """Объединить позиции и оценки источников на общей отсортированной оси item_id.

    Дополнения sparse-выдачи с нулевой оценкой не считаются найденными кандидатами.
    Бонус локации получают только кандидаты, уже присутствующие в объединении.
    """
    weights = [1.] * len(sources) if weights is None else weights
    if len(weights) != len(sources) or not sources:
        raise ValueError('One weight required per source')
    n = len(sources[0][0])
    output = np.empty((n, k), dtype=np.int32)
    for row in range(n):
        totals = {}
        for (indices, scores), weight in zip(sources, weights):
            if weight <= 0:
                continue
            for rank, (idx, score) in enumerate(zip(indices[row], scores[row]), 1):
                if score > 0:
                    totals[int(idx)] = totals.get(int(idx), 0.) + weight / (constant + rank)
        if location_weight:
            for idx in totals:
                totals[idx] += location_weight * (item_locations[idx] == query_locations[row])
        ranked = sorted(totals, key=lambda idx: (-totals[idx], idx))[:k]
        used = set(ranked)
        ranked.extend(i for i in range(k) if i not in used)
        output[row] = ranked[:k]
    return output
