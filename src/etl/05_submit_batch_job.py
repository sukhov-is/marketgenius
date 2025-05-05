# src/etl/submit_batch_job.py
import argparse
import logging
import os
import sys
from pathlib import Path
import json # Для парсинга метаданных
import pandas as pd

# Добавляем корневую папку проекта в sys.path
project_root = Path(__file__).resolve().parent.parent.parent # Поднимаемся на 2 уровня из src/etl
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

try:
    from src.utils.gpt_batch_analyzer import submit_batch_job
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что скрипт запускается из папки src/etl или корневой папки проекта, и структура проекта верна.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)

def parse_metadata(metadata_json):
    """Пытается распарсить строку JSON с метаданными."""
    if not metadata_json:
        return None
    try:
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("Метаданные должны быть словарем (JSON object).")
        return metadata
    except json.JSONDecodeError:
        _logger.error("Ошибка: Некорректный формат JSON для метаданных.")
        raise ValueError("Некорректный формат JSON для --metadata.")
    except ValueError as e:
        _logger.error(f"Ошибка валидации метаданных: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Отправка .jsonl файла в OpenAI Batch API и запуск задания.")

    parser.add_argument("--input-jsonl", default="data/external/text/batch/batch_input_example_1.jsonl", help="Путь к подготовленному .jsonl файлу.")
    parser.add_argument("--metadata", default=None, help='Метаданные для пакетного задания в формате JSON строки (например, \'{"description": "My batch", "run_id": 123}\').')
    parser.add_argument("--save-id-to", default="data/external/text/batch/latest_batch_info.json", help="Опциональный путь к JSON файлу для сохранения Batch ID и метаданных.")

    args = parser.parse_args()

    # Загрузка API ключа (обязателен для этого шага)
    load_dotenv(dotenv_path=project_root / '.env')
    if not os.getenv("OPENAI_API_KEY"):
        _logger.error("Ошибка: OPENAI_API_KEY не найден. Создайте .env файл или установите переменную окружения.")
        print("\nОшибка: OPENAI_API_KEY не найден.")
        sys.exit(1)

    _logger.info("--- Начало отправки Batch Job ---")

    try:
        # Парсинг метаданных из строки JSON
        parsed_metadata = parse_metadata(args.metadata)

        # Отправка задания
        _logger.info(f"Отправка файла {args.input_jsonl}...")
        batch_id = submit_batch_job(
            input_file_path=args.input_jsonl,
            metadata=parsed_metadata
        )

        if batch_id:
            print(f"\nУспешно! Пакетное задание запущено.")
            print(f"Batch ID: {batch_id}")
            print("Используйте этот ID для проверки статуса и получения результатов.")
            _logger.info(f"--- Отправка Batch Job завершена успешно. Batch ID: {batch_id} ---")

            # ===> Сохранение Batch ID в файл <===
            if args.save_id_to:
                save_path = Path(args.save_id_to)
                save_path.parent.mkdir(parents=True, exist_ok=True) # Создаем директорию, если нужно
                batch_info_to_save = {
                    "batch_id": batch_id,
                    "input_file": str(Path(args.input_jsonl).resolve()), # Сохраняем абсолютный путь
                    "submitted_at": pd.Timestamp.now().isoformat(), # Добавляем время отправки
                    "metadata": parsed_metadata
                }
                try:
                    with save_path.open("w", encoding="utf-8") as f:
                        json.dump(batch_info_to_save, f, indent=4, ensure_ascii=False)
                    _logger.info(f"Информация о задании (Batch ID: {batch_id}) сохранена в {save_path}")
                    print(f"Информация о задании сохранена в: {save_path}")
                except Exception as e:
                    _logger.error(f"Не удалось сохранить информацию о задании в {save_path}: {e}")
                    print(f"\nПредупреждение: Не удалось сохранить информацию о задании в {save_path}")
            # =================================
        else:
            print("\nНе удалось запустить пакетное задание. Проверьте логи для деталей.")
            _logger.error("--- Отправка Batch Job не удалась ---")
            sys.exit(1)

    except FileNotFoundError as e:
        _logger.error(f"Ошибка: Входной файл .jsonl не найден - {e}.")
        print(f"\nОшибка: Входной файл .jsonl не найден - {e}.")
        sys.exit(1)
    except ValueError as e: # Ловим ошибки парсинга метаданных
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        _logger.error(f"Произошла непредвиденная ошибка: {e}", exc_info=True)
        print(f"\nПроизошла непредвиденная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()