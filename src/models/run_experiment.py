import argparse
from pathlib import Path
import json
from typing import Literal

from src.models.experiment import run_single_experiment, TaskType


def main():
    parser = argparse.ArgumentParser(description="Run single ML experiment")
    parser.add_argument("csv", type=Path, help="Путь к CSV файлу с фичами для тикера")
    parser.add_argument("target", type=str, help="Название таргета, например target_1d")
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        required=True,
        help="Тип задачи",
    )
    parser.add_argument(
        "--media",
        action="store_true",
        help="Добавлять ли медиа-признаки",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgb", "lgbm"],
        default="xgb",
    )
    parser.add_argument("--output", type=Path, default=None, help="Файл JSON для записи метрик")

    args = parser.parse_args()
    metrics, fi = run_single_experiment(
        csv_path=args.csv,
        target_col=args.target,
        task=TaskType(args.task),
        use_media=args.media,
        model_family=args.model,
    )

    print("Metrics:\n", json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main() 