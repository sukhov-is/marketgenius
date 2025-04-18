import pandas as pd
import os
import logging
from typing import List

# -- Конфигурация Логирования --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -- Константы --
INPUT_DIR = 'data/processed/final_features'
OUTPUT_DIR = 'data/processed/ready_for_training'
START_DATE = '2019-01-01'
END_DATE = '2025-04-11' # Новая константа для конечной даты
DATE_COL = 'DATE'
CLOSE_COL = 'CLOSE'
TARGET_PREFIX = 'target_'
PREDICTION_HORIZONS = [1, 3, 7, 30, 180] # Горизонты прогнозирования в днях

# -- Функция обработки одного файла --
def process_ticker_data(input_path: str, output_path: str, start_date_str: str, end_date_str: str, horizons: List[int]) -> bool:
    """
    Загружает, обрабатывает и сохраняет данные для одного тикера.

    Args:
        input_path: Путь к исходному CSV файлу.
        output_path: Путь для сохранения обработанного CSV файла.
        start_date_str: Начальная дата для фильтрации данных (YYYY-MM-DD).
        end_date_str: Конечная дата для фильтрации данных (YYYY-MM-DD).
        horizons: Список горизонтов прогнозирования в днях.

    Returns:
        True, если обработка и сохранение прошли успешно, False в противном случае.
    """
    try:
        # Загрузка данных
        logging.debug(f"Загрузка данных из {input_path}")
        df = pd.read_csv(input_path)

        # Проверка наличия DATE и CLOSE
        if DATE_COL not in df.columns:
            logging.warning(f"Столбец '{DATE_COL}' не найден в {input_path}. Пропуск файла.")
            return False
        if CLOSE_COL not in df.columns:
            logging.warning(f"Столбец '{CLOSE_COL}' не найден в {input_path}. Пропуск файла.")
            return False

        # Обработка даты
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df.set_index(DATE_COL, inplace=True)
        df.sort_index(inplace=True)
        logging.debug(f"Установлен и отсортирован индекс по дате.")

        # 1. Обрезка по дате
        df = df[df.index >= start_date_str]
        if df.empty:
            logging.warning(f"Нет данных после {start_date_str} в {input_path}. Пропуск файла.")
            return False
        # Добавляем обрезку по конечной дате
        df = df[df.index <= end_date_str]
        if df.empty:
            logging.warning(f"Нет данных в диапазоне {start_date_str} - {end_date_str} в {input_path}. Пропуск файла.")
            return False
        logging.debug(f"Данные обрезаны по диапазону дат [{start_date_str}, {end_date_str}]. Строк осталось: {len(df)}")

        # 2. Обрезка начальных пропусков в CLOSE
        first_valid_close_index = df[CLOSE_COL].first_valid_index()
        if first_valid_close_index is None:
            logging.warning(f"Нет валидных значений '{CLOSE_COL}' в {input_path} в диапазоне [{start_date_str}, {end_date_str}]. Пропуск файла.")
            return False

        df = df[df.index >= first_valid_close_index]
        if df.empty:
             logging.warning(f"DataFrame стал пустым после обрезки начальных NaN в '{CLOSE_COL}' для {input_path}. Файл не будет обработан.")
             return False
        logging.debug(f"Начальные NaN в '{CLOSE_COL}' обрезаны. Строк осталось: {len(df)}")

        # 3. Создание целевых переменных
        target_cols = []
        for h in horizons:
            target_col_name = f"{TARGET_PREFIX}{h}d"
            # Расчет процентного изменения к цене через h дней
            df.loc[:, target_col_name] = (df[CLOSE_COL].shift(-h) / df[CLOSE_COL]) - 1
            target_cols.append(target_col_name)
        logging.debug(f"Созданы целевые столбцы: {target_cols}")

        # 4. Обработка пропусков признаков (до удаления строк с NaN таргетами)
        # Удаляем столбцы, которые могли стать полностью NaN из-за обрезки
        initial_cols = df.shape[1]
        df.dropna(axis=1, how='all', inplace=True)
        if df.shape[1] < initial_cols:
             logging.debug(f"Удалено {initial_cols - df.shape[1]} полностью пустых столбцов перед ffill/bfill.")

        # Заполнение пропусков в признаках
        df.ffill(inplace=True)
        df.bfill(inplace=True) # Заполняем NaN в начале, если остались
        logging.debug("Пропуски в признаках заполнены с помощью ffill и bfill.")

        # 5. Удаление строк с NaN в *любом* из таргетов
        initial_rows = len(df)
        df.dropna(subset=target_cols, inplace=True, how='any')
        if len(df) < initial_rows:
             logging.debug(f"Удалено {initial_rows - len(df)} строк с NaN в целевых переменных.")

        # 6. Удаление полностью пустых столбцов (если вдруг появились снова)
        initial_cols = df.shape[1]
        df.dropna(axis=1, how='all', inplace=True)
        if df.shape[1] < initial_cols:
             logging.debug(f"Удалено {initial_cols - df.shape[1]} полностью пустых столбцов после обработки NaN.")

        # Финальная проверка на пустоту
        if df.empty:
             logging.warning(f"DataFrame стал пустым после всех операций для {input_path}. Файл не будет сохранен.")
             return False

        # Сохранение результата
        logging.debug(f"Сохранение обработанных данных в {output_path}")
        df.to_csv(output_path)
        return True

    except pd.errors.EmptyDataError:
        logging.warning(f"Файл {input_path} пуст. Пропуск файла.")
        return False
    except KeyError as e:
        logging.error(f"Отсутствует ожидаемый столбец {e} в файле {input_path}. Пропуск файла.")
        return False
    except Exception as e:
        logging.exception(f"Неожиданная ошибка при обработке файла {input_path}: {e}") # Используем logging.exception для вывода traceback
        return False

# -- Основной блок выполнения --
if __name__ == "__main__":
    # Создать выходную директорию, если не существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Получить список файлов
    try:
        all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_final.csv')]
        logging.info(f"Найдено {len(all_files)} файлов для обработки в директории {INPUT_DIR}.")
    except FileNotFoundError:
        logging.error(f"Директория {INPUT_DIR} не найдена.")
        exit()
    except Exception as e:
        logging.error(f"Ошибка при чтении директории {INPUT_DIR}: {e}")
        exit()

    processed_count = 0
    skipped_count = 0

    # Обработка каждого файла
    for filename in all_files:
        input_file_path = os.path.join(INPUT_DIR, filename)
        # Используем тикер (часть имени файла до '_final.csv') для имени выходного файла
        ticker = filename.replace('_final.csv', '')
        output_file_path = os.path.join(OUTPUT_DIR, f'{ticker}_processed.csv')

        logging.info(f"Обработка файла: {filename}...")
        success = process_ticker_data(
            input_path=input_file_path,
            output_path=output_file_path,
            start_date_str=START_DATE,
            end_date_str=END_DATE,
            horizons=PREDICTION_HORIZONS
        )

        if success:
            processed_count += 1
            logging.info(f"Файл {filename} успешно обработан и сохранен в {output_file_path}")
        else:
            skipped_count += 1
            logging.warning(f"Файл {filename} был пропущен или не сохранен из-за ошибок/пустых данных.")

    logging.info("="*30)
    logging.info("Обработка завершена.")
    logging.info(f"Успешно обработано и сохранено файлов: {processed_count}")
    logging.info(f"Пропущено файлов: {skipped_count}")
    logging.info("="*30)

# Директории
# input_dir = 'data/processed/final_features'
# output_dir = 'data/processed/ready_for_training'
# start_date = '2019-01-01'
#
# # Создать выходную директорию, если не существует
# os.makedirs(output_dir, exist_ok=True)
#
# # Получить список файлов
# try:
#     all_files = [f for f in os.listdir(input_dir) if f.endswith('_final.csv')]
#     print(f"Найдено {len(all_files)} файлов для обработки.")
# except FileNotFoundError:
#     print(f"Ошибка: Директория {input_dir} не найдена.")
#     exit()
# except Exception as e:
#     print(f"Ошибка при чтении директории {input_dir}: {e}")
#     exit()
#
# processed_count = 0
# skipped_empty_count = 0
#
# # Обработка каждого файла
# for filename in all_files:
#     input_path = os.path.join(input_dir, filename)
#     output_path = os.path.join(output_dir, filename.replace('_final.csv', '_processed.csv')) # Новое имя файла
#
#     print(f"Обработка файла: {filename}...")
#
#     try:
#         # Загрузка данных
#         df = pd.read_csv(input_path)
#
#         # Обработка даты
#         if 'DATE' not in df.columns:
#             print(f"  Предупреждение: Столбец 'DATE' не найден в {filename}. Пропуск файла.")
#             continue
#         df['DATE'] = pd.to_datetime(df['DATE'])
#         df.set_index('DATE', inplace=True)
#         df.sort_index(inplace=True)
#
#         # 1. Обрезка по дате
#         df = df[df.index >= start_date]
#         if df.empty:
#             print(f"  Предупреждение: Нет данных после {start_date} в {filename}. Пропуск файла.")
#             skipped_empty_count += 1
#             continue
#
#         # 2. Обрезка начальных пропусков в CLOSE
#         if 'CLOSE' not in df.columns:
#             print(f"  Предупреждение: Столбец 'CLOSE' не найден в {filename}. Пропуск файла.")
#             continue
#
#         first_valid_close_index = df['CLOSE'].first_valid_index()
#         if first_valid_close_index is None:
#             print(f"  Предупреждение: Нет валидных значений 'CLOSE' в {filename} после {start_date}. Пропуск файла.")
#             skipped_empty_count += 1
#             continue
#
#         df = df[df.index >= first_valid_close_index]
#         if df.empty:
#              print(f"  Предупреждение: DataFrame стал пустым после обрезки начальных NaN в 'CLOSE' для {filename}. Пропуск файла.")
#              skipped_empty_count += 1
#              continue
#
#         # 3. Создание целевой переменной (процентное изменение к следующему дню)
#         # Используем .loc для избежания SettingWithCopyWarning
#         df.loc[:, 'target'] = df['CLOSE'].pct_change().shift(-1)
#
#         # 4. Обработка пропусков
#         # Удаляем столбцы, которые могут быть полностью NaN *перед* ffill/bfill,
#         # особенно если они появились из-за обрезки дат/строк.
#         df.dropna(axis=1, how='all', inplace=True)
#
#         df.ffill(inplace=True)
#         df.bfill(inplace=True) # Заполняем NaN в начале, если остались
#
#         # Удаляем последнюю строку, где target == NaN
#         df.dropna(subset=['target'], inplace=True)
#
#         # 5. Удаление полностью пустых столбцов (если вдруг появились после ffill/bfill/dropna target)
#         df.dropna(axis=1, how='all', inplace=True)
#
#         # Проверка на пустоту после всех операций
#         if df.empty:
#              print(f"  Предупреждение: DataFrame стал пустым после всех операций для {filename}. Пропуск файла.")
#              skipped_empty_count += 1
#              continue
#
#         # Сохранение результата
#         df.to_csv(output_path)
#         processed_count += 1
#         print(f"  Файл {filename} обработан и сохранен в {output_path}")
#
#     except pd.errors.EmptyDataError:
#         print(f"  Предупреждение: Файл {filename} пуст. Пропуск файла.")
#         skipped_empty_count += 1
#     except KeyError as e:
#         print(f"  Ошибка: Отсутствует ожидаемый столбец {e} в файле {filename}. Пропуск файла.")
#     except Exception as e:
#         print(f"  Неожиданная ошибка при обработке файла {filename}: {e}")
#
# print(f"Обработка завершена.")
# print(f"Успешно обработано и сохранено файлов: {processed_count}")
# print(f"Пропущено пустых или невалидных файлов: {skipped_empty_count}") 