# src/etl/07_process_batch_sequence.py
import argparse
import logging
import os
import sys
import time
from pathlib import Path
import json
import pandas as pd

# Добавляем корневую папку проекта в sys.path
project_root = Path(__file__).resolve().parent.parent.parent  # Поднимаемся на 2 уровня из src/etl
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Импортируем функции работы с Batch API
try:
    from src.utils.gpt_batch_analyzer import (
        submit_batch_job,
        check_batch_status,
        download_batch_results,
    )
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print(
        "Убедитесь, что скрипт запускается из папки src/etl или корневой, и структура проекта верна."
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
_logger = logging.getLogger(__name__)


STATUS_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def wait_for_completion(
    batch_id: str,
    interval: int,
    max_none_checks: int = 15,
    max_minutes: int | None = 60,
) -> dict | None:
    """Ожидает завершения пакетного задания.

    Если подряд `max_none_checks` раз не удаётся получить статус ИЛИ общее время ожидания
    превышает `max_minutes`, функция возвращает None, чтобы вызывающий код мог перейти к
    повторной попытке или завершить работу.
    """

    _logger.info(f"Начало ожидания завершения Batch ID: {batch_id}")
    start_ts = time.time()
    none_counter = 0

    while True:
        # Проверка таймаута
        if max_minutes is not None and (time.time() - start_ts) > max_minutes * 60:
            _logger.error(
                f"Превышено максимальное время ожидания ({max_minutes} мин) для {batch_id}."
            )
            return None

        status_info = check_batch_status(batch_id)
        if not status_info:
            none_counter += 1
            _logger.warning(
                f"Не удалось получить статус для {batch_id} (попытка {none_counter}/{max_none_checks})."
            )
            if none_counter >= max_none_checks:
                _logger.error(
                    f"Количество неудачных запросов статуса превысило лимит ({max_none_checks})."
                )
                return None
            time.sleep(interval)
            continue

        # Сброс счётчика, если статус получен
        none_counter = 0

        status = status_info.get("status")
        req_counts = status_info.get("request_counts", {})
        completed_req = req_counts.get("completed", "N/A")
        failed_req = req_counts.get("failed", "N/A")
        total_req = req_counts.get("total", "N/A")

        print(
            f"[{pd.Timestamp.now()}] Статус: {status} (Выполнено: {completed_req}/{total_req}, Ошибки: {failed_req})"
        )

        if status in STATUS_TERMINAL:
            _logger.info(f"Задание {batch_id} достигло конечного статуса: {status}")
            return status_info

        time.sleep(interval)


def process_single_file(
    input_file: Path,
    output_dir: Path,
    interval: int,
    save_info_dir: Path | None = None,
    max_retries: int = 2,
    retry_wait: int = 30,
    success_wait: int = 30,
) -> None:
    """Отправляет один файл в Batch API, ждёт окончания и скачивает результаты."""
    if not input_file.is_file():
        _logger.error(f"Файл не найден: {input_file}")
        return

    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        # Метаданные можно расширить при необходимости
        metadata = {
            "description": f"Batch processing for {input_file.name}",
            "attempt": str(attempt),
        }

        print(f"\n=== Отправка файла: {input_file} (попытка {attempt}/{max_retries + 1}) ===")
        batch_id = submit_batch_job(str(input_file), metadata=metadata)
        if not batch_id:
            _logger.error("Не удалось отправить пакет. \n")
            if attempt > max_retries:
                print("Достигнут лимит попыток. Переходим к следующему файлу.")
                return
            time.sleep(retry_wait)
            continue

        # Сохраняем информацию о запуске (если указана директория)
        if save_info_dir:
            save_info_dir.mkdir(parents=True, exist_ok=True)
            info_path = save_info_dir / f"batch_info_{batch_id}.json"
            with info_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "batch_id": batch_id,
                        "input_file": str(input_file.resolve()),
                        "submitted_at": pd.Timestamp.now().isoformat(),
                        "attempt": attempt,
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            _logger.info(f"Информация о задании сохранена в {info_path}")

        # Ожидаем завершения задания
        status_info = wait_for_completion(batch_id, interval)
        if not status_info:
            print("Не удалось получить финальный статус задания.")
            if attempt > max_retries:
                print("Достигнут лимит попыток. Переходим к следующему файлу.")
                return
            time.sleep(retry_wait)
            continue

        final_status = status_info.get("status")
        print(f"Финальный статус: {final_status}. Начинаем скачивание файлов...")

        results_path, errors_path = download_batch_results(status_info, output_dir)

        print("--- Итоги скачивания ---")
        if results_path:
            print(f"Файл результатов: {results_path}")
        else:
            print("Файл результатов отсутствует или не удалось скачать.")

        if errors_path:
            print(f"Файл ошибок:     {errors_path}")
        else:
            print("Файл ошибок отсутствует (вероятно, ошибок не было).")

        # Проверяем статус выполнения
        if final_status == "completed" and results_path:
            _logger.info("Пакет успешно обработан и результаты скачаны.")
            _logger.info(f"Пауза {success_wait} сек. перед следующим файлом.")
            time.sleep(success_wait)
            return

        # Если статус финальный и неуспешный — нет смысла ретраить
        if final_status in {"failed", "cancelled", "expired"}:
            _logger.error(f"Задание {batch_id} завершилось со статусом {final_status}. Повторные попытки прерываются.")
            return

        # Если дошли сюда — попытка не удалась
        if attempt > max_retries:
            _logger.error("Все попытки отправки пакета исчерпаны. Переходим к следующему файлу.")
            return

        _logger.warning(f"Попытка {attempt} не удалась. Повтор через {retry_wait} сек.")
        time.sleep(retry_wait)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Последовательная отправка нескольких .jsonl файлов в OpenAI Batch API, "
            "ожидание завершения каждого задания и скачивание результатов."
        )
    )

    parser.add_argument(
        "--input-dir",
        default="data/external/text/batch",
        help="Директория, содержащая .jsonl файлы для отправки.",
    )
    parser.add_argument(
        "--file-prefix",
        default="batch_input_blogs_history_part",
        help="Префикс имён файлов (до номера части).",
    )
    parser.add_argument("--start", type=int, default=1, help="Номер первой части.")
    parser.add_argument(
        "--end", type=int, default=20, help="Номер последней части (включительно)."
    )
    parser.add_argument(
        "--output-dir",
        default="data/external/gpt",
        help="Директория для сохранения результатов/ошибок.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Интервал проверки статуса в секундах.",
    )
    parser.add_argument(
        "--save-info-dir",
        default="data/external/text/batch",
        help="Директория для сохранения JSON с информацией о каждом Batch ID.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Максимальное количество повторных попыток при неудаче обработки пакета.",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=30,
        help="Время ожидания между повторными попытками (сек).",
    )
    parser.add_argument(
        "--success-wait",
        type=int,
        default=5,
        help="Пауза после успешной обработки пакета (сек).",
    )

    args = parser.parse_args()

    # Загружаем ключ API
    load_dotenv(dotenv_path=project_root / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        _logger.error("OPENAI_API_KEY не найден. Завершение.")
        print("\nОшибка: OPENAI_API_KEY не найден.")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    save_info_dir = Path(args.save_info_dir) if args.save_info_dir else None

    # Перебор файлов
    for part_num in range(args.start, args.end + 1):
        file_name = f"{args.file_prefix}{part_num}.jsonl"
        input_file = input_dir / file_name

        process_single_file(
            input_file=input_file,
            output_dir=output_dir,
            interval=args.interval,
            save_info_dir=save_info_dir,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
            success_wait=args.success_wait,
        )

    print("\n=== Обработка всех файлов завершена ===")


if __name__ == "__main__":
    main() 