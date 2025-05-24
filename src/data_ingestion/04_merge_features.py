import pandas as pd
import os
import glob
import logging
import argparse
from typing import List, Optional, Tuple, Dict, Any

# --- Константы ---
# Имена колонок и суффиксы можно вынести сюда для легкого изменения
DATE_COL_MACRO = 'DATE'
DATE_COL_TECH = 'TRADEDATA'
INDEX_COL_FIN = 0 # Финансовые данные - дата в первой колонке (индексе)

# Закомментированы, т.к. суффиксы не добавляются при загрузке
# TECH_SUFFIX = ''
# FIN_SUFFIX = '_fin'
# BRENT_SUFFIX = '_brent'
# RATE_SUFFIX = '_rate'
# USD_RUB_SUFFIX = '_usd_rub'
# MOEX_SUFFIX = '_moex'

# Ставим DATE_COL_MACRO первым в списке для MOEX
POTENTIAL_DATE_COLS_MOEX = [DATE_COL_MACRO, 'date', 'Date', 'begin', 'tradedate', 'TRADEDATE']

# --- Настройка логирования ---
def setup_logging(level=logging.INFO):
    """Настраивает базовую конфигурацию логирования."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Функции загрузки данных ---

def load_csv_with_date_index(filepath: str, expected_date_col: Optional[str] = None, index_col: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Загружает CSV, устанавливает индекс даты (из колонки или существующего индекса).

    Args:
        filepath: Путь к CSV файлу.
        expected_date_col: Ожидаемое имя колонки с датой (если index_col не задан).
        index_col: Номер колонки, которая является индексом (если дата уже в индексе).

    Returns:
        DataFrame с индексом даты или None в случае ошибки.
    """
    try:
        df = None
        if index_col is not None:
            # 1а. Загружаем CSV, используя указанную колонку как индекс и сразу парсим даты
            logging.debug(f"Загрузка {filepath} с index_col={index_col}, parse_dates=True")
            try:
                 # Добавляем keep_default_na=False для предотвращения интерпретации строк типа "NA" как NaN
                df = pd.read_csv(filepath, index_col=index_col, parse_dates=True, keep_default_na=False)
                 # Проверяем, является ли индекс типом даты
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                     logging.warning(f"Индекс в файле {filepath} (колонка {index_col}) не был успешно распознан как дата. Тип индекса: {df.index.dtype}. Попытка преобразования...")
                     df.index = pd.to_datetime(df.index, errors='coerce')
                     if df.index.isnull().all():
                          logging.error(f"Не удалось преобразовать индекс файла {filepath} в валидные даты. Пропуск файла.")
                          return None
                     logging.info(f"Индекс файла {filepath} успешно преобразован в дату.")

            except ValueError as e:
                logging.error(f"Ошибка при парсинге дат во время загрузки индекса файла {filepath}: {e}. Возможно, колонка {index_col} не содержит дат. Пропуск файла.")
                return None
            except Exception as e:
                logging.error(f"Ошибка при загрузке файла {filepath} с index_col={index_col}: {e}. Пропуск файла.")
                return None

        elif expected_date_col is not None:
            # 1б. Загружаем CSV без парсинга дат
            logging.debug(f"Загрузка {filepath} для поиска колонки даты '{expected_date_col}'")
             # Добавляем keep_default_na=False
            df = pd.read_csv(filepath, keep_default_na=False)
            actual_date_col = None

            # 2. Ищем колонку с датой
            if expected_date_col in df.columns:
                actual_date_col = expected_date_col
                logging.debug(f"Найдена основная колонка даты '{actual_date_col}' в {filepath}.")
            else:
                logging.warning(f"Колонка '{expected_date_col}' не найдена в {filepath}. Попытка найти альтернативную...")
                potential_cols = [c for c in df.columns if 'date' in c.lower() or 'day' in c.lower() or 'time' in c.lower() or 'period' in c.lower()]
                if potential_cols:
                    actual_date_col = potential_cols[0]
                    logging.info(f"Используется альтернативная колонка '{actual_date_col}' как дата в {filepath}.")
                else:
                     logging.error(f"Не найдена колонка с датой в {filepath} (ожидалась '{expected_date_col}'). Реальные колонки: {list(df.columns)}. Пропуск файла.")
                     return None

            # 3. Преобразуем найденную колонку в datetime
            try:
                df[actual_date_col] = pd.to_datetime(df[actual_date_col], errors='coerce')
            except Exception as e:
                logging.error(f"Не удалось преобразовать колонку '{actual_date_col}' в дату в файле {filepath}: {e}. Пропуск файла.")
                return None

            if df[actual_date_col].isnull().all():
                logging.error(f"Колонка '{actual_date_col}' в файле {filepath} не содержит валидных дат после преобразования. Пропуск файла.")
                return None

            # 4. Устанавливаем индекс
            df = df.set_index(actual_date_col)
        else:
             logging.error(f"Не указан ни expected_date_col, ни index_col для файла {filepath}. Невозможно определить дату.")
             return None


        # 5. Удаляем колонки, которые полностью состоят из NaN
        df.dropna(axis=1, how='all', inplace=True)

        # 6. Добавляем суффикс - УБРАНО
        # if suffix:
        #     df = df.add_suffix(suffix)

        logging.info(f"Загружен и обработан файл: {filepath}, shape: {df.shape}")
        return df.sort_index()

    except FileNotFoundError:
        logging.warning(f"Файл не найден: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Непредвиденная ошибка при загрузке/обработке файла {filepath}: {e}")
        return None

def load_moex_indices(filepath: str, potential_date_cols: List[str]) -> Optional[pd.DataFrame]:
    """Загружает CSV с индексами MOEX, определяя колонку даты."""
    try:
        df = pd.read_csv(filepath)
        logging.debug(f"Колонки в файле {filepath}: {list(df.columns)}") # Отладочный вывод
        found_date_col = None
        for col in potential_date_cols:
            if col in df.columns:
                try:
                    # Попытка преобразовать колонку в дату без изменения DataFrame
                    pd.to_datetime(df[col], errors='raise')
                    found_date_col = col
                    logging.info(f"Найдена колонка даты '{found_date_col}' в {filepath}.")
                    break
                except Exception:
                    logging.debug(f"Колонка '{col}' в {filepath} не является валидной датой.")
                    continue # Попробовать следующую колонку

        if not found_date_col:
            logging.error(f"Не найдена подходящая колонка с датой в {filepath} из списка {potential_date_cols}. Реальные колонки: {list(df.columns)}. Пропуск файла.")
            return None

        # Преобразуем найденную колонку в datetime
        try:
            df[found_date_col] = pd.to_datetime(df[found_date_col], errors='coerce')
        except Exception as e:
             logging.error(f"Не удалось преобразовать колонку '{found_date_col}' в дату в файле {filepath}: {e}. Пропуск файла.")
             return None

        if df[found_date_col].isnull().all():
             logging.error(f"Колонка '{found_date_col}' в файле {filepath} не содержит валидных дат после преобразования. Пропуск файла.")
             return None

        df = df.set_index(found_date_col)
        df.dropna(axis=1, how='all', inplace=True)
        # УБРАНО добавление суффикса
        # if suffix:
        #     df = df.add_suffix(suffix)
        logging.info(f"Загружены индексы MOEX: {filepath}, shape: {df.shape}")
        return df.sort_index()

    except FileNotFoundError:
        logging.warning(f"Файл не найден: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Ошибка при загрузке/обработке файла {filepath}: {e}")
        return None


def load_and_prepare_macro_data(macro_dir: str) -> pd.DataFrame:
    """Загружает, подготавливает и объединяет макроэкономические данные."""
    logging.info("Загрузка макроэкономических данных...")
    brent_path = os.path.join(macro_dir, 'brent_prices.csv')
    key_rate_path = os.path.join(macro_dir, 'key_rate.csv')
    usd_rub_path = os.path.join(macro_dir, 'usd_rub_rate.csv')
    moex_indices_path = os.path.join(macro_dir, 'moex_indices.csv')

    macro_dfs = []

    # Вызываем без суффиксов
    brent_df = load_csv_with_date_index(brent_path, expected_date_col=DATE_COL_MACRO)
    if brent_df is not None: macro_dfs.append(brent_df)

    key_rate_df = load_csv_with_date_index(key_rate_path, expected_date_col=DATE_COL_MACRO)
    if key_rate_df is not None: macro_dfs.append(key_rate_df)

    usd_rub_df = load_csv_with_date_index(usd_rub_path, expected_date_col=DATE_COL_MACRO)
    if usd_rub_df is not None: macro_dfs.append(usd_rub_df)

    # MOEX обрабатывается отдельно, без суффикса
    moex_df = load_moex_indices(moex_indices_path, POTENTIAL_DATE_COLS_MOEX)
    if moex_df is not None: macro_dfs.append(moex_df)

    if not macro_dfs:
        logging.warning("Не удалось загрузить ни один файл с макроданными. Макропризнаки не будут добавлены.")
        return pd.DataFrame()

    # Объединение макро-данных
    # Используем outer join, чтобы сохранить все даты, затем ffill/bfill
    macro_final_df = pd.concat(macro_dfs, axis=1, join='outer')
    macro_final_df = macro_final_df.sort_index()

    # Удаляем дубликаты колонок (если вдруг суффиксы совпали или были исходные дубли)
    macro_final_df = macro_final_df.loc[:, ~macro_final_df.columns.duplicated()]

    logging.info(f"Макроданые объединены и отсортированы. Shape: {macro_final_df.shape}")
    # Замечание: ffill для макро данных будет применен позже, после мержа с основными данными

    return macro_final_df

# --- Функции обработки тикеров ---

def process_ticker(ticker: str, tech_indicators_dir: str, financial_features_dir: str, output_dir: str, macro_data: pd.DataFrame):
    """Обрабатывает один тикер: загружает данные, объединяет и сохраняет."""
    logging.info(f"Обработка тикера: {ticker}")

    tech_path = os.path.join(tech_indicators_dir, f"{ticker}_processed.csv")
    fin_path = os.path.join(financial_features_dir, f"{ticker}_features.csv")
    out_path = os.path.join(output_dir, f"{ticker}_merged.csv")

    # 1. Загрузка технических индикаторов (обязательно)
    # Вызываем без суффикса
    tech_df = load_csv_with_date_index(tech_path, expected_date_col=DATE_COL_TECH)
    if tech_df is None:
        logging.error(f"Не удалось загрузить технические индикаторы для {ticker}. Пропуск тикера.")
        return False # Сигнализируем об ошибке

    # 2. Загрузка финансовых признаков (опционально)
    # Вызываем без суффикса
    fin_df = load_csv_with_date_index(fin_path, index_col=INDEX_COL_FIN)

    # 3. Объединение тех. индикаторов и фин. признаков (если фин. данные есть)
    # Используем LEFT JOIN, чтобы сохранить все даты из tech_df
    if fin_df is not None:
        merged_df = pd.merge(tech_df, fin_df, left_index=True, right_index=True, how='left') # ВОЗВРАЩЕНО how='left'
        logging.debug(f"[{ticker}] Shape после LEFT MERGE с фин. признаками: {merged_df.shape}")
    else:
        logging.info(f"Финансовые признаки для тикера {ticker} не найдены или не загружены. Используются только тех. индикаторы.")
        merged_df = tech_df # Используем только технические данные

    # 4. Объединение с макроэкономическими данными (если макро данные есть)
    # Используем LEFT JOIN, чтобы сохранить все даты из merged_df
    if not macro_data.empty:
        final_df = pd.merge(merged_df, macro_data, left_index=True, right_index=True, how='left') # ВОЗВРАЩЕНО how='left'
        logging.debug(f"[{ticker}] Shape после LEFT MERGE с макро данными: {final_df.shape}")
    else:
         logging.info(f"[{ticker}] Макро данные не добавлены.")

    # 5. Очистка (удаление строк, где ВСЕ значения NaN) - оставляем, т.к. могут быть NaN из-за inner merge
    initial_rows = len(final_df)
    final_df.dropna(how='all', inplace=True)
    if initial_rows > len(final_df):
         logging.debug(f"[{ticker}] Удалено {initial_rows - len(final_df)} строк, состоящих полностью из NaN.")

    # 6. Сохранение результата
    try:
        # Сохраняем с индексом-датой для возможного удобства последующей загрузки
        final_df.to_csv(out_path, index=True, index_label=DATE_COL_MACRO)
        logging.info(f"Результат для тикера {ticker} сохранен в: {out_path}")
        return True # Успешная обработка
    except Exception as e:
        logging.error(f"Не удалось сохранить результат для тикера {ticker} в {out_path}: {e}")
        return False # Сигнализируем об ошибке

# --- Основная функция ---
def main(tech_dir: str, fin_dir: str, macro_dir: str, out_dir: str):
    """Главная функция скрипта: загрузка макро, обработка тикеров."""
    setup_logging()
    logging.info("Запуск скрипта объединения признаков...")

    # Создание выходной директории
    try:
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f"Выходная директория: {out_dir}")
    except OSError as e:
        logging.critical(f"Не удалось создать выходную директорию {out_dir}: {e}")
        return # Критическая ошибка, дальше нет смысла идти

    # Загрузка макро-данных
    macro_final_df = load_and_prepare_macro_data(macro_dir)

    # Получение списка файлов с техническими индикаторами
    tech_files_pattern = os.path.join(tech_dir, '*_processed.csv')
    tech_files = glob.glob(tech_files_pattern)
    if not tech_files:
        logging.warning(f"Не найдено файлов с техническими индикаторами по паттерну: {tech_files_pattern}")
        logging.info("Скрипт завершает работу, так как нет данных для обработки.")
        return

    logging.info(f"Найдено {len(tech_files)} файлов с техническими индикаторами.")

    processed_count = 0
    failed_tickers = []

    # Обработка каждого тикера
    for tech_path in tech_files:
        filename = os.path.basename(tech_path)
        # Извлекаем тикер (ожидаем формат TCKR_processed.csv)
        if filename.endswith('_processed.csv'):
             ticker = filename[:-len('_processed.csv')] # Убираем суффикс
             if ticker: # Убедимся, что тикер не пустой
                 if process_ticker(ticker, tech_dir, fin_dir, out_dir, macro_final_df):
                     processed_count += 1
                 else:
                     failed_tickers.append(ticker)
             else:
                logging.warning(f"Не удалось извлечь тикер из имени файла: {filename}")
                failed_tickers.append(filename) # Добавляем имя файла, если тикер не извлечен
        else:
             logging.warning(f"Файл {filename} не соответствует ожидаемому формату '*_processed.csv'. Пропуск.")
             # Можно добавить имя файла в failed_tickers, если это считается ошибкой
             # failed_tickers.append(filename)


    # Итоговая статистика
    logging.info("--- Завершено ---")
    logging.info(f"Успешно обработано тикеров: {processed_count}")
    if failed_tickers:
        logging.warning(f"Не удалось обработать тикеры/файлы ({len(failed_tickers)}): {', '.join(failed_tickers)}")
    else:
        logging.info("Все тикеры обработаны без ошибок.")
    logging.info(f"Результаты сохранены в: {out_dir}")

# --- Точка входа ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для объединения технических индикаторов, финансовых и макроэкономических признаков.")

    parser.add_argument(
        '--tech-dir',
        type=str,
        default='./data/processed/technical_indicators',
        help='Директория с файлами технических индикаторов (*_processed.csv).'
    )
    parser.add_argument(
        '--fin-dir',
        type=str,
        default='./data/processed/financial_features',
        help='Директория с файлами финансовых признаков (*_features.csv).'
    )
    parser.add_argument(
        '--macro-dir',
        type=str,
        default='./data/raw/macro',
        help='Директория с файлами макроэкономических данных.'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/processed/merged_features',
        help='Директория для сохранения объединенных данных.'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Уровень логирования.'
    )

    args = parser.parse_args()

    # Устанавливаем уровень логирования из аргументов
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=log_level_numeric)

    main(
        tech_dir=args.tech_dir,
        fin_dir=args.fin_dir,
        macro_dir=args.macro_dir,
        out_dir=args.out_dir
    ) 