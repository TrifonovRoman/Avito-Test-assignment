"""Генерация ответа по финальной hybrid-конфигурации."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.pipeline import predict
from src.submission import build_submission, save_submission


def main():
    """Разобрать аргументы, выполнить поиск и проверить сохранение CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=Path('data/benchmark_queries.parquet'))
    parser.add_argument('--items', type=Path, default=Path('data/benchmark_items.parquet'))
    parser.add_argument('--output', type=Path, default=Path('answer.csv'))
    parser.add_argument('--config', type=Path, default=Path('hybrid_config.json'))
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--cache-only', action='store_true',
                        help='Fail on missing hybrid cache rather than compute')
    args = parser.parse_args()
    queries = pd.read_parquet(args.queries, dtype_backend='pyarrow')
    config = json.loads(args.config.read_text())
    positions, ids = predict(args.items, queries, config,
                             args.cache_dir or Path('.cache'), cache_only=args.cache_only)
    answer = build_submission(queries, positions, ids)
    save_submission(answer, args.output, queries.query_id, ids)
    print(f'Saved and verified {len(answer)} queries: {args.output}')


if __name__ == '__main__':
    main()
