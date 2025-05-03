# src/etl/monitor_and_download.py
import argparse
import logging
import os
import sys
from pathlib import Path
import json
import time

# Добавляем корневую папку проекта в sys.path
project_root = Path(__file__).resolve().parent.parent.parent # Поднимаемся на 2 уровня из src/etl
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

try:
    # Импортируем нужные функции
    from src.utils.gpt_batch_analyzer import check_batch_status, download_batch_results
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что скрипт запускается из папки src/etl или корневой папки проекта, и структура проекта верна.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)

# Путь по умолчанию для файла с ID последнего задания
DEFAULT_BATCH_INFO_PATH = "data/external/text/batch/latest_batch_info.json"

def main():
    parser = argparse.ArgumentParser(description="Мониторинг статуса пакетного задания OpenAI и скачивание результатов по завершении.")

    parser.add_argument("--batch-id", default=None, help="ID пакетного задания для мониторинга (приоритет над --read-id-from).")
    parser.add_argument("--read-id-from", default=DEFAULT_BATCH_INFO_PATH, help=f"Путь к JSON файлу для чтения Batch ID, если --batch-id не указан (по умолчанию: {DEFAULT_BATCH_INFO_PATH}).")
    parser.add_argument("--output-dir", required=True, help="Директория для сохранения скачанных файлов результатов/ошибок.")
    parser.add_argument("--interval", type=int, default=60, help="Интервал проверки статуса в секундах (по умолчанию: 60).")

    args = parser.parse_args()

    # Загрузка API ключа
    load_dotenv(dotenv_path=project_root / '.env')
    if not os.getenv("OPENAI_API_KEY"):
        _logger.error("Ошибка: OPENAI_API_KEY не найден.")
        print("\nОшибка: OPENAI_API_KEY не найден.")
        sys.exit(1)

    batch_id_to_monitor = args.batch_id
    read_path = Path(args.read_id_from)
    output_dir = Path(args.output_dir)

    # Определяем ID для мониторинга
    if not batch_id_to_monitor:
        _logger.info(f"Batch ID не указан, попытка чтения из файла: {read_path}")
        if read_path.is_file():
            try:
                with read_path.open("r", encoding="utf-8") as f:
                    batch_info_data = json.load(f)
                    batch_id_to_monitor = batch_info_data.get("batch_id")
                    if batch_id_to_monitor:
                        _logger.info(f"Найден Batch ID в файле: {batch_id_to_monitor}")
                    else:
                         _logger.error(f"Файл {read_path} найден, но не содержит ключ 'batch_id'.")
            except Exception as e:
                 _logger.error(f"Ошибка чтения или парсинга файла {read_path}: {e}")
        else:
            _logger.warning(f"Файл для чтения Batch ID ({read_path}) не найден.")

    if not batch_id_to_monitor:
        _logger.error("Не удалось определить Batch ID для мониторинга.")
        print(f"\nОшибка: Не удалось определить Batch ID. Укажите его через --batch-id или проверьте файл {read_path}.")
        sys.exit(1)

    _logger.info(f"--- Начало мониторинга Batch ID: {batch_id_to_monitor} ---")
    print(f"Наблюдение за Batch ID: {batch_id_to_monitor}")
    print(f"Интервал проверки: {args.interval} секунд")
    print(f"Результаты будут сохранены в: {output_dir}")
    print("Нажмите Ctrl+C для прерывания...")

    final_status = None
    status_info = None

    try:
        while True:
            status_info = check_batch_status(batch_id_to_monitor)

            if not status_info:
                print("Не удалось получить статус. Повторная попытка через {args.interval} сек...")
                _logger.warning(f"Не удалось получить статус для {batch_id_to_monitor}. Повтор через {args.interval} сек.")
                time.sleep(args.interval)
                continue

            current_status = status_info.get('status')
            request_counts = status_info.get('request_counts', {})
            completed_req = request_counts.get('completed', 'N/A')
            failed_req = request_counts.get('failed', 'N/A')
            total_req = request_counts.get('total', 'N/A')

            print(f"[{pd.Timestamp.now()}] Статус: {current_status} (Выполнено: {completed_req}/{total_req}, Ошибки: {failed_req})")

            if current_status == 'completed':
                print("Задание завершено! Скачивание результатов...")
                _logger.info(f"Задание {batch_id_to_monitor} завершено (completed). Скачивание результатов.")
                final_status = current_status
                break # Выходим из цикла для скачивания

            elif current_status in ['failed', 'expired', 'cancelled']:
                print(f"Задание завершилось со статусом {current_status}. Попытка скачать файл ошибок...")
                _logger.warning(f"Задание {batch_id_to_monitor} завершилось со статусом {current_status}.")
                final_status = current_status
                break # Выходим из цикла для скачивания (ошибок)
            else:
                # Статусы: validating, in_progress, finalizing, cancelling
                _logger.info(f"Задание {batch_id_to_monitor} в статусе {current_status}. Ожидание {args.interval} секунд.")
                time.sleep(args.interval)

        # --- Скачивание после выхода из цикла ---
        if final_status and status_info:
            print(f"Скачивание файлов для задания {batch_id_to_monitor} (статус: {final_status})...")
            results_path, errors_path = download_batch_results(status_info, output_dir)

            print("\n--- Результаты скачивания ---")
            if results_path:
                print(f"Файл результатов: {results_path}")
                _logger.info(f"Файл результатов сохранен: {results_path}")
            else:
                 print("Файл результатов отсутствует или не удалось скачать.")
                 _logger.warning(f"Файл результатов не скачан для {batch_id_to_monitor}.")

            if errors_path:
                print(f"Файл ошибок:     {errors_path}")
                _logger.info(f"Файл ошибок сохранен: {errors_path}")
            else:
                # Это нормально, если ошибок не было
                print("Файл ошибок отсутствует (вероятно, ошибок не было).")
                _logger.info(f"Файл ошибок не скачан для {batch_id_to_monitor} (вероятно, отсутствовал).")
            _logger.info(f"--- Мониторинг и скачивание для {batch_id_to_monitor} завершены ---")
        else:
             print("\nНе удалось определить финальный статус или получить информацию о задании. Скачивание не выполнено.")
             _logger.error(f"Не удалось выполнить скачивание для {batch_id_to_monitor}, так как финальный статус не определен.")
             sys.exit(1)


    except KeyboardInterrupt:
         print("\nМониторинг прерван пользователем.")
         _logger.info(f"Мониторинг для {batch_id_to_monitor} прерван пользователем.")
         sys.exit(0)
    except Exception as e:
        _logger.error(f"Произошла непредвиденная ошибка во время мониторинга/скачивания: {e}", exc_info=True)
        print(f"\nПроизошла непредвиденная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Добавим импорт pandas для красивого вывода времени
    import pandas as pd
    main()