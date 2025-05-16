import argparse
import logging
import math
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Определяем корневую директорию проекта (3 уровня вверх от файла)
project_root = Path(__file__).resolve().parent.parent.parent

# Добавляем корень проекта в sys.path для корректных импортов (если понадобятся внутренние пакеты)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


def split_jsonl_file(
    input_path: Path,
    first_part_size: int = 50,
    other_part_size: int = 3831,
    output_dir: Path | None = None,
) -> list[Path]:
    """Разбивает JSONL-файл по схеме 525 / 300 / остаток.

    Первая часть содержит *first_part_size* строк (по умолчанию 525), каждая следующая — *other_part_size*
    строк (по умолчанию 300), а последняя часть — остаток (если он меньше *other_part_size*).

    Args:
        input_path: Путь к исходному JSONL-файлу.
        first_part_size: Сколько строк должно быть в первой части (>0).
        other_part_size: Сколько строк должно быть в остальных частях (>0).
        output_dir: Каталог, куда сохранять части. По умолчанию — рядом с исходным файлом.

    Returns:
        Список путей к созданным файлам.
    """

    if first_part_size <= 0 or other_part_size <= 0:
        raise ValueError("Размеры частей должны быть положительными.")

    if not input_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    output_dir = output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    if total == 0:
        _logger.warning("Исходный файл пуст — нечего разбивать.")
        return []

    created: list[Path] = []

    _logger.info(
        f"Первая часть: {first_part_size} строк, следующие по {other_part_size}, всего строк: {total}."
    )

    index = 0
    part_num = 1

    # Первая часть
    first_end = min(first_part_size, total)
    part_lines = lines[index:first_end]
    part_path = output_dir / f"{input_path.stem}_part{part_num}{input_path.suffix}"
    with part_path.open("w", encoding="utf-8") as pf:
        pf.writelines(part_lines)
    created.append(part_path)
    _logger.info(f"Создан файл: {part_path} (строки {index + 1}–{first_end})")

    index = first_end
    part_num += 1

    # Следующие части по other_part_size
    while index < total:
        end = min(index + other_part_size, total)
        part_lines = lines[index:end]
        part_path = output_dir / f"{input_path.stem}_part{part_num}{input_path.suffix}"
        with part_path.open("w", encoding="utf-8") as pf:
            pf.writelines(part_lines)
        created.append(part_path)
        _logger.info(f"Создан файл: {part_path} (строки {index + 1}–{end})")

        index = end
        part_num += 1

    return created


def main():
    parser = argparse.ArgumentParser(description="Разбивает готовый .jsonl файл Batch API на несколько частей.")

    parser.add_argument(
        "--input-jsonl",
        default="data/external/text/batch/batch_input_blogs_history.jsonl",
        help="Путь к существующему .jsonl файлу, который нужно разбить.",
    )
    parser.add_argument(
        "--first-size",
        type=int,
        default=1940,
        help="Сколько строк должно быть в первой части. По умолчанию 525.",
    )
    parser.add_argument(
        "--other-size",
        type=int,
        default=1941,
        help="Сколько строк должно быть в остальных частях. По умолчанию 300.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Опциональный путь к директории, куда сохранять части. По умолчанию — рядом с исходным файлом.",
    )

    args = parser.parse_args()

    # Загружаем .env на случай, если скрипт запускается в той же среде, где нужны переменные (не обязательно)
    load_dotenv(dotenv_path=project_root / ".env")

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        parts = split_jsonl_file(
            input_path=input_path,
            first_part_size=args.first_size,
            other_part_size=args.other_size,
            output_dir=output_dir,
        )
        if parts:
            print("\nФайл успешно разбит на части:")
            for p in parts:
                print(f" - {p}")
            print("Готово.")
        else:
            print("\nРазбиение не выполнено (см. предупреждения выше).")
    except Exception as e:
        _logger.error(f"Ошибка разбиения файла: {e}", exc_info=True)
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 