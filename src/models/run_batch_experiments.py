import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.batch_runner import iter_experiments, save_results


def main():
    parser = argparse.ArgumentParser(description="Выполнить серию ML-экспериментов по всем тикерам")
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Папка с *_final.csv (по умолчанию data/features_final)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/batch_results"),
        help="Путь (без расширения) для сохранения CSV и JSON c результатами",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=None,
        help="Список таргетов (если не указано – полный набор)",
    )
    args = parser.parse_args()

    df = iter_experiments(
        features_dir=args.features_dir or Path("data/features_final"),
        targets=args.targets or None,
    )

    save_results(df, args.output)
    print(f"Готово! Результаты сохранены в {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main() 