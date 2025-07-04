# src/etl/process_batch_results.py
import argparse
import logging
import os
import sys
from pathlib import Path
import json
import pandas as pd # Для сохранения в CSV
from datetime import datetime  # Для генерации уникального имени файла

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
    parser.add_argument("--results-file", default="data/processed/gpt/gpt_news_history.jsonl", help="Путь к скачанному файлу результатов (*_results_*.jsonl).")
    parser.add_argument("--errors-file", default=None, help="Путь к скачанному файлу ошибок (*_errors_*.jsonl) (опционально).")

    # Способ 2: Указать ID и директорию
    parser.add_argument("--batch-id", default=None, help="ID пакетного задания, чьи результаты нужно обработать (используется, если файлы не указаны явно).")
    parser.add_argument("--read-id-from", default=DEFAULT_BATCH_INFO_PATH, help=f"Путь к JSON файлу для чтения Batch ID, если --batch-id не указан (по умолчанию: {DEFAULT_BATCH_INFO_PATH}).")
    parser.add_argument("--results-dir", default='data/processed/gpt', help="Директория, где лежат скачанные файлы results_{id}.jsonl и errors_{id}.jsonl (используется с --batch-id).")

    # Обязательный аргумент для вывода
    parser.add_argument("--output-csv", default='data/processed/gpt/results_gpt_news.csv', help="Путь для сохранения итогового CSV файла.")
    parser.add_argument("--append", action="store_true", help="Если указан этот флаг и файл CSV уже существует, новые данные будут дозаписаны (append) к существующим вместо перезаписи.")

    args = parser.parse_args()

    # Загрузка .env (хотя API ключ тут не нужен, но для консистентности)
    load_dotenv(dotenv_path=project_root / '.env')

    results_file_path = Path(args.results_file) if args.results_file else None
    errors_file_path = Path(args.errors_file) if args.errors_file else None
    batch_id_to_process = args.batch_id
    read_id_path = Path(args.read_id_from)
    results_dir_path = Path(args.results_dir) if args.results_dir else None

    # --- Генерируем уникальное имя для выходного CSV, если пользователь не указал своё ---
    default_output_csv = parser.get_default('output_csv')
    output_csv_path = Path(args.output_csv)

    if args.output_csv == default_output_csv:  # Пользователь не передал --output-csv
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Например: results_gpt_20240513_123456.csv
        output_csv_path = output_csv_path.with_name(f"{output_csv_path.stem}_{timestamp_str}{output_csv_path.suffix}")

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
        final_df, errored_custom_ids = process_batch_output(results_file_path, errors_file_path)

        if errored_custom_ids:
            _logger.warning("Обнаружены ошибки при обработке следующих custom_id:")
            for error_id in errored_custom_ids:
                _logger.warning(f"- {error_id}")
            print("\nВнимание: При обработке некоторых Batch API запросов возникли ошибки.")
            print("Список custom_id с ошибками (см. лог для подробностей):")
            for error_id in errored_custom_ids:
                print(f"- {error_id}")

        if not final_df.empty:
            # Сохраняем итоговый CSV
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)

            if args.append and output_csv_path.is_file():
                try:
                    existing_df = pd.read_csv(output_csv_path, encoding="utf-8-sig")
                    combined_df = pd.concat([existing_df, final_df], ignore_index=True)
                    combined_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
                    _logger.info(
                        f"К существующему CSV ({len(existing_df)} строк) добавлено {len(final_df)} строк. Итог: {len(combined_df)}."
                    )
                except Exception as e:
                    _logger.error(f"Не удалось дозаписать данные в {output_csv_path}: {e}")
                    print(f"\nОшибка при дозаписи данных: {e}. Попробуйте запустить без флага --append.")
                    sys.exit(1)
            else:
                final_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")  # utf-8-sig для Excel

            print(
                f"\nУспешно! Итоговый DataFrame ({len(final_df)} строк) сохранен в: {output_csv_path}"
            )
            _logger.info(
                f"--- Обработка результатов завершена успешно. Сохранено в {output_csv_path} ---"
            )
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