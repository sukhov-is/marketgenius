import pandas as pd
import os
import logging

# -- Конфигурация Логирования --
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -- Константы --
INPUT_DIR = 'data/processed/ready_for_training'
OUTPUT_FILE = 'data/processed/combined_data.csv'
DATE_COL = 'DATE' # Убедитесь, что это имя столбца с датой после reset_index
SECID_COL = 'SECID' # Убедитесь, что это имя столбца с тикером

# -- Функция объединения данных --
def combine_processed_data(input_dir: str, output_file: str):
    """
    Объединяет все обработанные CSV файлы из input_dir в один CSV файл.

    Args:
        input_dir: Директория с обработанными файлами (*_processed.csv).
        output_file: Путь для сохранения объединенного CSV файла.
    """
    try:
        all_files = [f for f in os.listdir(input_dir) if f.endswith('_processed.csv')]
        if not all_files:
            logging.warning(f"В директории {input_dir} не найдено файлов _processed.csv\".")
            return
        logging.info(f"Найдено {len(all_files)} файлов для объединения.")

        dfs_to_concat = []
        for filename in all_files:
            file_path = os.path.join(input_dir, filename)
            logging.debug(f"Загрузка файла: {filename}")
            try:
                # Загружаем CSV, предполагая, что DATE - первый столбец (индекс при сохранении)
                # и SECID - один из столбцов.
                df = pd.read_csv(file_path)
                if DATE_COL not in df.columns:
                     # Если DATE стал индексом при сохранении, читаем его как столбец
                     df = pd.read_csv(file_path, index_col=0) # Пробуем прочитать индекс как DATE
                     df.reset_index(inplace=True)

                # Проверяем наличие ключевых столбцов
                if DATE_COL not in df.columns:
                    logging.warning(f"Столбец '{DATE_COL}' не найден в {filename} после загрузки. Пропуск файла.")
                    continue
                if SECID_COL not in df.columns:
                    logging.warning(f"Столбец '{SECID_COL}' не найден в {filename}. Пропуск файла.")
                    continue

                # Преобразуем DATE в datetime на всякий случай
                df[DATE_COL] = pd.to_datetime(df[DATE_COL])

                dfs_to_concat.append(df)
            except Exception as e:
                logging.exception(f"Ошибка при загрузке файла {filename}: {e}")
                continue # Пропускаем файл, если ошибка

        if not dfs_to_concat:
            logging.error("Нет данных для объединения после попытки загрузки файлов.")
            return

        logging.info("Объединение загруженных DataFrame...")
        combined_df = pd.concat(dfs_to_concat, ignore_index=True) # ignore_index=True создаст новый числовой индекс
        logging.info(f"Объединенный DataFrame содержит {len(combined_df)} строк и {combined_df.shape[1]} столбцов.")

        # Сортировка для консистентности
        logging.info(f"Сортировка данных по '{SECID_COL}' и '{DATE_COL}'...")
        combined_df.sort_values(by=[SECID_COL, DATE_COL], inplace=True)

        # Сохранение результата
        logging.info(f"Сохранение объединенного DataFrame в {output_file}...")
        combined_df.to_csv(output_file, index=False) # Сохраняем без индекса
        logging.info("Объединенный DataFrame успешно сохранен.")

    except FileNotFoundError:
        logging.error(f"Директория {input_dir} не найдена.")
    except Exception as e:
        logging.exception(f"Неожиданная ошибка при объединении данных: {e}")

# -- Основной блок выполнения --
if __name__ == "__main__":
    combine_processed_data(INPUT_DIR, OUTPUT_FILE) 