import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv

# Импортируем все наши классы-загрузчики
from src.data_ingestion.moex_parser import MoexLoader
# FinancialReportLoader не импортируем, т.к. отчеты обновляются редко
from src.data_ingestion.indexes_moex import MoexIndexLoader
from src.data_ingestion.oil_prices_loader import AlphaVantageBrentLoader
from src.data_ingestion.usd_rub_loader import UsdRubLoader
from src.data_ingestion.key_rate_loader import KeyRateLoader

# Загружаем переменные окружения из .env файла
# Ищем .env в текущей директории или выше
project_root = Path(__file__).resolve().parent.parent # Поднимаемся на уровень выше (в корень проекта)
dotenv_path = project_root / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Если .env не найден относительно скрипта, пробуем стандартный поиск
    load_dotenv()

def setup_logging(log_level_str: str):
    """Настраивает базовое логирование."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Установлен уровень логирования: {log_level_str.upper()}")


def append_to_csv(new_data: Optional[pd.DataFrame], filepath: Path, date_col: str = 'DATE'):
    """Добавляет новые данные в CSV, избегая дубликатов по дате.

    Args:
        new_data (Optional[pd.DataFrame]): DataFrame с новыми данными.
        filepath (Path): Путь к CSV файлу.
        date_col (str): Название колонки с датой.
    """
    logger = logging.getLogger(__name__) # Получаем логгер
    if new_data is None or new_data.empty:
        logger.warning(f"Нет новых данных для добавления в {filepath}")
        return

    # Убедимся, что колонка даты есть и имеет правильный тип
    if date_col not in new_data.columns:
        logger.error(f"Отсутствует колонка даты '{date_col}' в новых данных для {filepath}")
        return
    try:
        # Убедимся, что данные в колонке - datetime объекты
        new_data[date_col] = pd.to_datetime(new_data[date_col])
    except Exception as e:
        logger.error(f"Не удалось конвертировать колонку '{date_col}' в дату для {filepath}: {e}")
        return

    # Получаем уникальные даты из новых данных
    new_dates = set(new_data[date_col].dt.date)
    if not new_dates:
        logger.warning(f"Нет корректных дат в новых данных для {filepath}")
        return

    try:
        if filepath.exists():
            existing_df = pd.read_csv(filepath)
            # Проверяем наличие колонки даты в существующем файле
            if date_col not in existing_df.columns:
                logger.error(f"Отсутствует колонка даты '{date_col}' в существующем файле {filepath}. Перезапись файла новыми данными.")
                # В случае ошибки структуры существующего файла, можно его перезаписать
                # new_data.sort_values(by=date_col).to_csv(filepath, index=False)
                # Или пропустить обновление - пропустим для безопасности
                return

            existing_df[date_col] = pd.to_datetime(existing_df[date_col])
            existing_dates = set(existing_df[date_col].dt.date)

            # Отфильтровываем даты, которые уже есть
            dates_to_add = new_dates - existing_dates

            if not dates_to_add:
                logger.info(f"Данные за даты {sorted(list(new_dates))} уже существуют в {filepath}. Пропуск добавления.")
                return

            # Фильтруем новые данные, оставляя только строки с уникальными датами для добавления
            filtered_new_data = new_data[new_data[date_col].dt.date.isin(dates_to_add)].copy()

            # Проверяем консистентность колонок
            if set(filtered_new_data.columns) != set(existing_df.columns):
                logger.warning(f"Колонки в новых данных ({filtered_new_data.columns.tolist()}) отличаются от существующих в {filepath} ({existing_df.columns.tolist()}). Обновление может быть неполным.")
                # Приводим колонки к общему знаменателю (колонки существующего файла)
                filtered_new_data = filtered_new_data.reindex(columns=existing_df.columns)

            # Добавляем только отфильтрованные данные
            updated_df = pd.concat([existing_df, filtered_new_data], ignore_index=True)
            # Сортируем и сохраняем весь файл
            updated_df.sort_values(by=date_col).to_csv(filepath, index=False)
            logger.info(f"Добавлено {len(filtered_new_data)} строк с датами {sorted(list(dates_to_add))} в {filepath}")
        else:
            # Если файла нет, просто сохраняем новые данные
            new_data.sort_values(by=date_col).to_csv(filepath, index=False)
            logger.info(f"Создан файл {filepath} с {len(new_data)} строками.")

    except pd.errors.EmptyDataError:
        logger.warning(f"Существующий файл {filepath} пуст. Перезаписываем новыми данными.")
        new_data.sort_values(by=date_col).to_csv(filepath, index=False)
        logger.info(f"Создан файл {filepath} с {len(new_data)} строками.")
    except Exception as e:
        logger.exception(f"Ошибка при чтении/добавлении данных в файл {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Запуск ежедневного обновления данных.")
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help="Дата для загрузки в формате YYYY-MM-DD (по умолчанию: вчерашний день)."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help="Корневая директория, где хранятся сырые данные (по умолчанию: data/raw)."
    )
    parser.add_argument(
        '--companies-config',
        type=str,
        default='configs/companies_config.json',
        help="Путь к JSON файлу конфигурации компаний (по умолчанию: configs/companies_config.json)."
    )
    parser.add_argument(
        '--indices-config',
        type=str,
        default=None,
        help="(Опционально) Путь к JSON файлу конфигурации индексов MOEX."
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Уровень логирования (по умолчанию: INFO)."
    )

    args = parser.parse_args()

    # Настройка логирования
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__) # Получаем логгер

    # Определение даты для загрузки
    if args.date:
        try:
            date_to_fetch = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"Неверный формат даты: {args.date}. Используйте YYYY-MM-DD.")
            return
    else:
        date_to_fetch = date.today() - timedelta(days=1)

    logger.info(f"Загрузка данных за дату: {date_to_fetch}")

    # Определение путей
    base_output_path = Path(args.output_dir)
    if not base_output_path.exists():
         logger.warning(f"Директория {base_output_path} не существует. Создание невозможно из скрипта обновления. Запустите сначала run_initial_load.py.")
         # Можно либо создать её здесь, либо прервать выполнение.
         # Прервем, чтобы избежать случайного создания в неверном месте.
         return

    moex_output_dir = base_output_path / 'moex_shares'
    macro_output_dir = base_output_path / 'macro'
    indices_output_file = macro_output_dir / 'moex_indices.csv'
    brent_output_file = macro_output_dir / "brent_prices.csv"
    usd_rub_output_file = macro_output_dir / 'usd_rub_rate.csv'
    key_rate_output_file = macro_output_dir / 'key_rate.csv'

    # Получаем ключ Alpha Vantage из окружения
    alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not alpha_vantage_api_key:
        logger.warning("Ключ ALPHA_VANTAGE_API_KEY не найден в переменных окружения. Обновление цен Brent (Alpha Vantage) не будет выполнено.")
    else:
        logger.info("Ключ ALPHA_VANTAGE_API_KEY успешно загружен.")

    # --- Этап 1: Обновление данных по акциям MOEX --- #
    logger.info("--- Начало обновления данных акций MOEX ---")
    try:
        moex_loader = MoexLoader(config_path=args.companies_config)
        tickers_to_update = list(moex_loader.companies.keys())
        logger.info(f"Будут обновлены данные для {len(tickers_to_update)} тикеров.")
        for ticker in tickers_to_update:
            logger.debug(f"Загрузка MOEX для {ticker} за {date_to_fetch}")
            df_share = moex_loader.load_security_data(ticker, date_to_fetch, date_to_fetch)
            if df_share is not None:
                 ticker_file = moex_output_dir / f"{ticker}_moex_data.csv"
                 append_to_csv(df_share, ticker_file, date_col='TRADEDATE')
            else:
                logger.warning(f"Нет данных MOEX для {ticker} за {date_to_fetch}")
        logger.info("--- Обновление данных акций MOEX завершено ---")
    except FileNotFoundError:
        logger.error(f"Ошибка: Не найден файл конфигурации компаний: {args.companies_config}")
    except Exception as e:
        logger.exception(f"Ошибка при обновлении данных акций MOEX: {e}")

    # --- Этап 2: Обновление индексов MOEX --- #
    logger.info("--- Начало обновления индексов MOEX ---")
    try:
        index_loader = MoexIndexLoader(config_path=args.indices_config)
        indices_to_update = index_loader.indices # Получаем список индексов из загрузчика
        all_indices_df_today = []
        logger.info(f"Обновление для {len(indices_to_update)} индексов.")
        for index_name in indices_to_update:
             logger.debug(f"Загрузка индекса {index_name} за {date_to_fetch}")
             df_index = index_loader._fetch_single_index(index_name, date_to_fetch, date_to_fetch)
             if df_index is not None:
                  all_indices_df_today.append(df_index)

        if all_indices_df_today:
            # Объединяем данные только за сегодняшний день перед добавлением
            try:
                merged_df_today = all_indices_df_today[0]
                for i in range(1, len(all_indices_df_today)):
                    merged_df_today = pd.merge(merged_df_today, all_indices_df_today[i], on='DATE', how='outer')
                # Добавляем объединенные данные дня в общий файл
                append_to_csv(merged_df_today, indices_output_file, date_col='DATE')
            except Exception as merge_err:
                logger.exception(f"Ошибка при объединении или добавлении данных индексов за {date_to_fetch}: {merge_err}")
        else:
            logger.warning(f"Нет данных для обновления индексов MOEX за {date_to_fetch}.")
        logger.info("--- Обновление индексов MOEX завершено ---")
    except Exception as e:
        logger.exception(f"Ошибка при обновлении индексов MOEX: {e}")

    # --- Этап 3: Обновление цен на нефть (Urals, Brent) --- #
    logger.info("--- Начало обновления цен на нефть (Brent / Alpha Vantage) ---")
    if alpha_vantage_api_key:
        try:
            oil_loader = AlphaVantageBrentLoader(alpha_vantage_api_key=alpha_vantage_api_key)
            logger.debug(f"Загрузка Brent (Alpha Vantage) за {date_to_fetch}")
            df_brent = oil_loader._fetch_brent_prices(date_to_fetch, date_to_fetch)
            append_to_csv(df_brent, brent_output_file, date_col='DATE')
            logger.info("--- Обновление цен на нефть (Brent / Alpha Vantage) завершено ---")
        except ValueError as ve:
            logger.error(f"Ошибка инициализации AlphaVantageBrentLoader: {ve}")
        except Exception as e:
            logger.exception(f"Ошибка при обновлении цен на нефть (Brent / Alpha Vantage): {e}")
    else:
        logger.warning("Пропуск обновления цен на нефть (Brent) из-за отсутствия ALPHA_VANTAGE_API_KEY.")

    # --- Этап 4: Обновление курса USD/RUB --- #
    logger.info("--- Начало обновления курса USD/RUB ---")
    try:
        usd_rub_loader = UsdRubLoader()
        logger.debug(f"Загрузка USD/RUB за {date_to_fetch}")
        df_usd_rub = usd_rub_loader.fetch_rates(date_to_fetch, date_to_fetch)
        append_to_csv(df_usd_rub, usd_rub_output_file, date_col='DATE')
        logger.info("--- Обновление курса USD/RUB завершено ---")
    except Exception as e:
        logger.exception(f"Ошибка при обновлении курса USD/RUB: {e}")

    # --- Этап 5: Обновление ключевой ставки --- #
    logger.info("--- Начало обновления истории ключевой ставки ---")
    try:
        key_rate_loader = KeyRateLoader()
        # Ключевая ставка не меняется каждый день, но запрос безопасен
        logger.debug(f"Загрузка Key Rate за {date_to_fetch}")
        df_key_rate = key_rate_loader.fetch_key_rate(date_to_fetch, date_to_fetch)
        # Добавляем, только если были изменения (хотя append_to_csv сам справится с дубликатами)
        append_to_csv(df_key_rate, key_rate_output_file, date_col='DATE')
        logger.info("--- Обновление ключевой ставки завершено ---")
    except Exception as e:
        logger.exception(f"Ошибка при обновлении ключевой ставки: {e}")

    logger.info("=== Ежедневное обновление данных завершено ===")

if __name__ == "__main__":
    main() 