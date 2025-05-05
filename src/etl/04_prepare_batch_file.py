import argparse
import logging
from pathlib import Path
import os # Добавляем os
from dotenv import load_dotenv # Добавляем dotenv
import sys # <-- Добавляем sys

# Определяем корневую директорию проекта
project_root = Path(__file__).resolve().parent.parent.parent

# Добавляем корневую директорию в sys.path для корректных импортов
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.utils.gpt_analyzer import GPTNewsAnalyzer
    from src.utils.gpt_batch_analyzer import prepare_batch_input_file
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что скрипт запускается из папки scripts или корневой папки проекта.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Подготовка .jsonl файла для OpenAI Batch API.")

    # Обязательные аргументы
    parser.add_argument("--input-csv", default="data/external/text/representative_news.csv", help="Путь к входному CSV/TSV файлу с новостями.")
    parser.add_argument("--output-jsonl", default="data/external/text/batch/batch_input_example.jsonl", help="Путь для сохранения выходного .jsonl файла.")

    # Аргументы для конфигурации
    parser.add_argument("--config-path", default="configs/companies_config.json", help="Путь к JSON конфигу компаний.")
    parser.add_argument("--prompt-path", default="src/prompts", help="Путь к директории с файлами промптов (*_promt.txt).")
    parser.add_argument("--prompt-type", default="news", help="Тип промпта для использования (имя файла без _promt.txt).")
    parser.add_argument("--model", default="gpt-4o-mini", help="Модель OpenAI для указания в запросах.")

    # Аргументы для данных
    parser.add_argument("--date-col", default="datetime", help="Название колонки с датой.")
    parser.add_argument("--text-col", default="news", help="Название колонки с текстом новости.")
    parser.add_argument("--title-col", default="channel_name", help="Название колонки с источником/заголовком.")
    parser.add_argument("--start-date", default=None, help="Начальная дата для фильтрации (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Конечная дата для фильтрации (YYYY-MM-DD).")

    args = parser.parse_args()

    # Загрузка API ключа (хотя он не нужен для подготовки, хорошо иметь его проверку)
    load_dotenv(dotenv_path=project_root / '.env')
    if not os.getenv("OPENAI_API_KEY"):
        print("Предупреждение: OPENAI_API_KEY не найден. Он понадобится для отправки задания.")
        # Не прерываем выполнение, так как для подготовки ключ не нужен

    _logger.info("--- Начало подготовки Batch Input файла ---")

    try:
        # Инициализация анализатора (нужен для настроек и метода _split_into_chunks)
        _logger.info("Инициализация GPTNewsAnalyzer...")
        analyzer_instance = GPTNewsAnalyzer(
            model=args.model,
            config_path=args.config_path,
            prompt_path=args.prompt_path,
        )

        # Загрузка и фильтрация данных
        _logger.info(f"Загрузка данных из {args.input_csv}")
        df = analyzer_instance._load_dataframe(args.input_csv, args.date_col, args.text_col, args.title_col)
        _logger.info(f"Фильтрация дат: с {args.start_date or 'начала'} по {args.end_date or 'конец'}")
        df_filtered = analyzer_instance._filter_dates(df, args.date_col, args.start_date, args.end_date)

        if df_filtered.empty:
            _logger.warning("Нет данных для обработки после фильтрации по дате.")
            print("Нет данных для обработки после фильтрации. Файл не будет создан.")
            sys.exit(0) # Успешный выход, так как нет данных для ошибки

        # Создание папки для выходного файла, если её нет
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Подготовка файла
        _logger.info(f"Подготовка файла {args.output_jsonl}...")
        num_requests = prepare_batch_input_file(
            analyzer=analyzer_instance,
            df=df_filtered,
            date_col=args.date_col,
            text_col=args.text_col,
            title_col=args.title_col,
            prompt_type=args.prompt_type,
            output_file_path=output_path,
        )

        if num_requests > 0:
            print(f"\nУспешно! Создан файл: {output_path}")
            print(f"Количество запросов в файле: {num_requests}")
            _logger.info(f"--- Подготовка Batch Input файла завершена успешно ({num_requests} запросов) ---")
        else:
            print("\nНе удалось создать файл с запросами (возможно, не было валидных данных или произошла ошибка при записи).")
            _logger.warning("--- Подготовка Batch Input файла завершена, но запросы не были записаны ---")

    except FileNotFoundError as e:
        _logger.error(f"Ошибка: Файл не найден - {e}. Проверьте пути.")
        print(f"\nОшибка: Файл не найден - {e}. Убедитесь, что пути верны.")
        sys.exit(1)
    except KeyError as e:
        _logger.error(f"Ошибка: Колонка не найдена - {e}. Проверьте аргументы --date-col, --text-col, --title-col.")
        print(f"\nОшибка: Колонка не найдена - {e}. Проверьте названия колонок.")
        sys.exit(1)
    except ValueError as e:
         _logger.error(f"Ошибка значения: {e}")
         print(f"\nОшибка значения: {e}")
         sys.exit(1)
    except Exception as e:
        _logger.error(f"Произошла непредвиденная ошибка: {e}", exc_info=True)
        print(f"\nПроизошла непредвиденная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()