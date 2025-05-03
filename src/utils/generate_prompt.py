import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


def _load_tickers(config_path: str | Path) -> str:
    """Загружает и форматирует блок тикеров/индексов из JSON конфига."""
    try:
        cfg_path = Path(config_path)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        companies = cfg.get("companies", {})
        indices = cfg.get("indices", {})

        ticker_lines = []

        # Обрабатываем компании
        for ticker, data in companies.items():
            if isinstance(data, dict) and 'names' in data:
                # Новый формат: объединяем имена через запятую
                joined_names = ", ".join(data['names'])
                ticker_lines.append(f"{ticker} : {joined_names}")
            elif isinstance(data, str):
                # Старый формат для обратной совместимости
                ticker_lines.append(f"{ticker} : {data}")

        # Индексы
        for ticker, description in indices.items():
            ticker_lines.append(f"{ticker} : {description}")

        if not ticker_lines:
            _logger.warning(f"Не найдено тикеров или индексов в {config_path}")
            return ""

        return "\n".join(ticker_lines)
    except FileNotFoundError:
        _logger.error(f"Файл конфигурации не найден: {config_path}")
        raise
    except json.JSONDecodeError:
        _logger.error(f"Ошибка декодирования JSON в файле: {config_path}")
        raise
    except Exception as e:
        _logger.error(f"Неожиданная ошибка при загрузке тикеров из {config_path}: {e}")
        raise


def _load_and_filter_data(
    data_path: str | Path,
    target_date_str: str,
    date_col: str,
    text_col: str,
    title_col: str,
) -> str:
    """Загружает данные из CSV/TSV, фильтрует по дате и форматирует сообщения."""
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        dp = Path(data_path)
        sep = "\t" if dp.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(dp, sep=sep)

        required_cols = {date_col, text_col, title_col}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"В файле данных отсутствуют колонки: {missing}")

        # Преобразуем колонку с датой/временем в дату
        try:
            # Пытаемся автоматически определить формат, обрабатываем ошибки как NaT (Not a Time)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True).dt.date
            # Удаляем строки, где дата не распозналась
            original_rows = len(df)
            df.dropna(subset=[date_col], inplace=True)
            if len(df) < original_rows:
                _logger.warning(f"Удалено {original_rows - len(df)} строк с нераспознанным форматом даты в колонке '{date_col}'")
        except Exception as e:
            _logger.error(f"Критическая ошибка при преобразовании колонки даты '{date_col}': {e}. Убедитесь, что формат даты распознается pandas.")
            raise

        _logger.info(f"Найдено {len(df)} строк с корректной датой до фильтрации.")

        # Фильтруем по дате
        df_filtered = df[df[date_col] == target_date].copy()
        _logger.info(f"Найдено {len(df_filtered)} строк для даты {target_date_str}.")

        if df_filtered.empty:
            # Это предупреждение уже есть, но оставляем для ясности
            _logger.warning(f"Нет данных для даты {target_date_str} в файле {data_path} после фильтрации.")
            return ""

        # Очистка и форматирование текстовых колонок
        df_filtered[text_col] = df_filtered[text_col].astype(str).fillna("").str.strip().str.replace(r'\s+', ' ', regex=True)
        df_filtered[title_col] = df_filtered[title_col].astype(str).fillna("Unknown").str.strip().str.replace(r'\s+', ' ', regex=True)

        rows_before_empty_filter = len(df_filtered)
        # Убираем строки с пустым текстом ПОСЛЕ очистки
        df_filtered = df_filtered[df_filtered[text_col] != '']
        rows_after_empty_filter = len(df_filtered)

        if rows_after_empty_filter < rows_before_empty_filter:
            _logger.info(f"Отфильтровано {rows_before_empty_filter - rows_after_empty_filter} строк с пустым текстом новостей после очистки.")

        if df_filtered.empty:
            _logger.warning(f"Нет непустых сообщений для даты {target_date_str} после очистки.")
            return ""

        # Форматируем строки новостей
        news_lines = [
            f"{row[title_col]} : {row[text_col]}"
            for _, row in df_filtered.iterrows()
        ]

        return "\n".join(news_lines)

    except FileNotFoundError:
        _logger.error(f"Файл данных не найден: {data_path}")
        raise
    except ValueError as ve:
        _logger.error(f"Ошибка значения: {ve}")
        raise
    except Exception as e:
        _logger.error(f"Неожиданная ошибка при загрузке и фильтрации данных из {data_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Генерирует файл промпта для указанной даты.")
    parser.add_argument(
        "--date",
        type=str,
        default="2025-04-01",
        help="Дата в формате YYYY-MM-DD, для которой генерируется промпт.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/external/text/representative_news.csv",
        help="Путь к CSV/TSV файлу с данными (новости/сообщения).",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="src/prompts/promt.txt",
        help="Путь к файлу шаблона промпта.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/companies_config.json",
        help="Путь к JSON файлу конфигурации с тикерами/индексами.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Путь к выходному .txt файлу. Если не указан, генерируется автоматически в 'output/prompt_YYYY-MM-DD.txt'.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="datetime",
        help="Название колонки с датой/временем в файле данных.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="news",
        help="Название колонки с текстом сообщения в файле данных.",
    )
    parser.add_argument(
        "--title-col",
        type=str,
        default="channel_name",
        help="Название колонки с заголовком/источником сообщения.",
    )

    args = parser.parse_args()

    try:
        # 1. Загрузка шаблона
        prompt_template_path = Path(args.prompt_template)
        try:
            prompt_template = prompt_template_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            _logger.error(f"Файл шаблона не найден: {prompt_template_path}")
            return
        except Exception as e:
            _logger.error(f"Ошибка чтения файла шаблона {prompt_template_path}: {e}")
            return

        # 2. Загрузка тикеров
        tickers_block = _load_tickers(args.config_file)

        # 3. Загрузка и форматирование данных
        news_lines_str = _load_and_filter_data(
            args.data_file, args.date, args.date_col, args.text_col, args.title_col
        )

        # 4. Форматирование промпта
        try:
            final_prompt = prompt_template.format(
                TICKERS_AND_INDICES=tickers_block,
                DATE=args.date,
                NEWS_LINES=news_lines_str,
            )
        except KeyError as e:
            _logger.error(f"Ошибка форматирования шаблона: отсутствует ключ {e}. Убедитесь, что шаблон содержит {{TICKERS_AND_INDICES}}, {{DATE}}, {{NEWS_LINES}}.")
            return
        except Exception as e:
             _logger.error(f"Неожиданная ошибка при форматировании промпта: {e}")
             return

        # 5. Определение пути и сохранение файла
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            output_dir = Path("output")
            output_path = output_dir / f"prompt_{args.date}.txt"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            output_path.write_text(final_prompt, encoding="utf-8")
            _logger.info(f"Промпт успешно сохранен в: {output_path}")
        except Exception as e:
            _logger.error(f"Ошибка записи в файл {output_path}: {e}")

    except Exception as e:
        _logger.error(f"Критическая ошибка выполнения скрипта: {e}")


if __name__ == "__main__":
    main() 