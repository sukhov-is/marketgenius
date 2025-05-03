# src/etl/process_batch_results.py
import argparse
import logging
import os
import sys
from pathlib import Path
import json
import pandas as pd # Для сохранения в CSV

# Добавляем корневую папку проекта в sys.path
project_root = Path(__file__).resolve().parent.parent.parent # Поднимаемся на 2 уровня из src/etl
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

try:
    # Импортируем только нужную функцию
    from src.utils.gpt_batch_analyzer import process_batch_output
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что скрипт запускается из папки src/etl или корневой папки проекта, и структура проекта верна.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)

# Путь по умолчанию для файла с ID последнего задания
DEFAULT_BATCH_INFO_PATH = "data/external/text/batch/latest_batch_info.json"

def main():
    parser = argparse.ArgumentParser(description="Обработка скачанных файлов результатов OpenAI Batch API и сохранение в CSV.")

    # Способ 1: Указать файлы напрямую
    parser.add_argument("--results-file", default=None, help="Путь к скачанному файлу результатов (*_results_*.jsonl).")
    parser.add_argument("--errors-file", default=None, help="Путь к скачанному файлу ошибок (*_errors_*.jsonl) (опционально).")

    # Способ 2: Указать ID и директорию
    parser.add_argument("--batch-id", default=None, help="ID пакетного задания, чьи результаты нужно обработать (используется, если файлы не указаны явно).")
    parser.add_argument("--read-id-from", default=DEFAULT_BATCH_INFO_PATH, help=f"Путь к JSON файлу для чтения Batch ID, если --batch-id не указан (по умолчанию: {DEFAULT_BATCH_INFO_PATH}).")
    parser.add_argument("--results-dir", default=None, help="Директория, где лежат скачанные файлы results_{id}.jsonl и errors_{id}.jsonl (используется с --batch-id).")

    # Обязательный аргумент для вывода
    parser.add_argument("--output-csv", required=True, help="Путь для сохранения итогового CSV файла.")

    args = parser.parse_args()

    # Загрузка .env (хотя API ключ тут не нужен, но для консистентности)
    load_dotenv(dotenv_path=project_root / '.env')

    results_file_path = Path(args.results_file) if args.results_file else None
    errors_file_path = Path(args.errors_file) if args.errors_file else None
    batch_id_to_process = args.batch_id
    read_id_path = Path(args.read_id_from)
    results_dir_path = Path(args.results_dir) if args.results_dir else None
    output_csv_path = Path(args.output_csv)

    # --- Определяем пути к файлам для обработки ---
    if not results_file_path: # Если файл результатов не указан явно
        _logger.info("Путь к файлу результатов не указан явно.")
        # Пытаемся определить ID
        if not batch_id_to_process:
            _logger.info(f"Batch ID не указан, попытка чтения из файла: {read_id_path}")
            if read_id_path.is_file():
                try:
                    with read_id_path.open("r", encoding="utf-8") as f:
                        batch_info_data = json.load(f)
                        batch_id_to_process = batch_info_data.get("batch_id")
                        if batch_id_to_process:
                            _logger.info(f"Найден Batch ID в файле: {batch_id_to_process}")
                        else:
                             _logger.error(f"Файл {read_id_path} найден, но не содержит ключ 'batch_id'.")
                except Exception as e:
                     _logger.error(f"Ошибка чтения или парсинга файла {read_id_path}: {e}")
            else:
                _logger.warning(f"Файл для чтения Batch ID ({read_id_path}) не найден.")

        # Если ID определен и указана директория с результатами
        if batch_id_to_process and results_dir_path:
             _logger.info(f"Поиск файлов для Batch ID {batch_id_to_process} в директории {results_dir_path}")
             results_file_path = results_dir_path / f"results_{batch_id_to_process}.jsonl"
             errors_file_path = results_dir_path / f"errors_{batch_id_to_process}.jsonl"

             # Проверяем, существует ли хотя бы файл результатов
             if not results_file_path.is_file():
                  _logger.warning(f"Ожидаемый файл результатов {results_file_path} не найден.")
                  results_file_path = None # Сбрасываем, если не найден

             if not errors_file_path.is_file():
                  _logger.info(f"Ожидаемый файл ошибок {errors_file_path} не найден (это может быть нормально).")
                  errors_file_path = None # Сбрасываем, если не найден

        elif batch_id_to_process and not results_dir_path:
             _logger.error("Указан Batch ID, но не указана директория с результатами (--results-dir).")
             print("\nОшибка: Указан Batch ID, но не указана директория --results-dir.")
             sys.exit(1)

    # Проверяем, что у нас есть хотя бы какой-то файл для обработки
    if not results_file_path and not errors_file_path:
         _logger.error("Не удалось определить файлы для обработки. Укажите --results-file или --batch-id вместе с --results-dir.")
         print("\nОшибка: Не найдены файлы для обработки. Проверьте аргументы.")
         sys.exit(1)

    _logger.info(f"--- Начало обработки результатов ---")
    _logger.info(f"Файл результатов: {results_file_path or 'Не указан'}")
    _logger.info(f"Файл ошибок:     {errors_file_path or 'Не указан/Не найден'}")
    _logger.info(f"Итоговый CSV:     {output_csv_path}")

    try:
        # Вызов основной функции обработки
        final_df = process_batch_output(results_file_path, errors_file_path)

        if not final_df.empty:
            # Сохраняем итоговый CSV
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig') # utf-8-sig для Excel
            print(f"\nУспешно! Итоговый DataFrame ({len(final_df)} строк) сохранен в: {output_csv_path}")
            _logger.info(f"--- Обработка результатов завершена успешно. Сохранено в {output_csv_path} ---")
        else:
            print("\nНе удалось создать итоговый DataFrame из результатов (возможно, все запросы завершились с ошибкой, файлы пусты или произошла ошибка обработки).")
            _logger.warning("--- Обработка результатов завершена, но итоговый DataFrame пуст ---")

    except FileNotFoundError as e:
         # Эта ошибка не должна возникать здесь, так как мы проверяем файлы раньше,
         # но на всякий случай оставим обработку.
        _logger.error(f"Ошибка: Входной файл не найден во время обработки - {e}.")
        print(f"\nОшибка: Входной файл не найден - {e}.")
        sys.exit(1)
    except Exception as e:
        _logger.error(f"Произошла непредвиденная ошибка во время обработки: {e}", exc_info=True)
        print(f"\nПроизошла непредвиденная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()