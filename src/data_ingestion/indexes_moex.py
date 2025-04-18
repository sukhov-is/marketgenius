import pandas as pd
import requests
from datetime import date, datetime, timedelta
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict
import json
from tqdm import tqdm

class MoexIndexLoader:
    """Класс для загрузки данных по индексам Московской биржи.

    Поддерживает загрузку списка индексов за указанный период и сохранение
    в единый CSV-файл.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация загрузчика индексов.

        Parameters:
        -----------
        config_path : Optional[str], optional
            Путь к JSON файлу конфигурации. Ожидается ключ 'indices' со списком
            тикеров индексов. Если None, будет использован список по умолчанию.
            По умолчанию None.
        """
        self.base_url = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities"
        # Сначала настраиваем логгер
        self._setup_logging()
        # Затем загружаем индексы, т.к. _load_indices использует логгер
        self.indices = self._load_indices(config_path)

    def _load_indices(self, config_path: Optional[str]) -> List[str]:
        """Загрузка списка индексов из файла или использование списка по умолчанию."""
        default_indices = [
            "MRBC", "RTSI", "MCXSM", "IMOEX", "MOEXBC", "MOEXBMI",
            "MOEXCN", "MOEXIT", "MOEXRE", "MOEXEU", "MOEXFN", "MOEXINN",
            "MOEXMM", "MOEXOG", "MOEXTL", "MOEXTN", "MOEXCH"
        ]
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    indices_from_config = config.get('indices')
                    if isinstance(indices_from_config, list):
                        self.logger.info(f"Загружен список индексов из {config_path}")
                        return indices_from_config
                    else:
                        self.logger.warning(f"Ключ 'indices' в {config_path} не является списком. Используется список по умолчанию.")
            except FileNotFoundError:
                self.logger.warning(f"Файл конфигурации {config_path} не найден. Используется список по умолчанию.")
            except json.JSONDecodeError:
                self.logger.error(f"Ошибка декодирования JSON в файле: {config_path}. Используется список по умолчанию.")
            except Exception as e:
                self.logger.exception(f"Непредвиденная ошибка при загрузке конфига {config_path}: {e}. Используется список по умолчанию.")
        else:
            self.logger.info("Конфигурационный файл не указан. Используется список индексов по умолчанию.")

        return default_indices

    def _setup_logging(self) -> None:
        """Настройка логирования."""
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)

    def _fetch_single_index(self, index_name: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Получает данные по одному индексу с MOEX ISS API.

        Returns:
        --------
        Optional[pd.DataFrame]: DataFrame с колонками ['DATE', index_name] или None при ошибке.
        """
        url = f"{self.base_url}/{index_name}.json"
        params = {
            "from": start_date.strftime('%Y-%m-%d'),
            "till": end_date.strftime('%Y-%m-%d'),
            "history.columns": "TRADEDATE,CLOSE", # Запрашиваем только нужные колонки
            "iss.meta": "off",
            "start": 0,
            "limit": 100 # Ограничение API
            # "iss.json": "extended" # Не используем extended, т.к. он менее стабилен для истории
        }

        all_data = []
        self.logger.debug(f"Запрос данных для индекса {index_name} с {start_date} по {end_date}")

        while True:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # API истории возвращает данные в ключе 'history'
                history_block = data.get('history', {})
                index_data = history_block.get('data', [])
                columns = history_block.get('columns', [])

                if not index_data:
                    self.logger.debug(f"Нет данных 'history' для {index_name} в диапазоне с {params['start']}.")
                    break # Данных нет на этой странице

                all_data.extend(index_data)

                # Обработка пагинации для 'history'
                cursor_data = data.get('history.cursor', {})
                cursor_info = cursor_data.get('data', [])

                if not cursor_info:
                    # Нет информации о курсоре, предполагаем, что это последняя страница
                    break

                # Структура курсора: [INDEX, TOTAL, PAGESIZE]
                current_index, total_rows, page_size = cursor_info[0]
                start = current_index + page_size

                if start >= total_rows:
                    break # Загружены все строки

                params['start'] = start
                self.logger.debug(f"Запрос следующей страницы для {index_name}, start={start}")

            except requests.exceptions.Timeout:
                self.logger.error(f"Таймаут при получении данных для {index_name}.")
                return None
            except requests.exceptions.RequestException as e:
                status_code = e.response.status_code if e.response is not None else "N/A"
                self.logger.error(f"Ошибка сети (статус: {status_code}) при получении данных для {index_name}: {e}")
                return None
            except json.JSONDecodeError:
                self.logger.error(f"Ошибка декодирования JSON ответа для {index_name}.")
                return None
            except Exception as e:
                self.logger.exception(f"Непредвиденная ошибка при получении данных для {index_name}: {e}")
                return None

        if not all_data:
            self.logger.warning(f"Данные для индекса {index_name} за период {start_date} - {end_date} не найдены.")
            return None

        if not columns:
            self.logger.error(f"Не удалось получить названия колонок для индекса {index_name}.")
            return None

        try:
            df = pd.DataFrame(all_data, columns=columns)
            # Ищем нужные колонки (могут немного отличаться)
            date_col = 'TRADEDATE'
            close_col = 'CLOSE'
            if date_col not in df.columns or close_col not in df.columns:
                self.logger.error(f"Ожидаемые колонки '{date_col}' или '{close_col}' не найдены в данных для {index_name}. Доступные: {df.columns.tolist()}")
                return None

            # Выбираем, переименовываем и конвертируем
            df = df[[date_col, close_col]].copy()
            df.columns = ["DATE", index_name] # Переименовываем для удобства объединения
            df["DATE"] = pd.to_datetime(df["DATE"])
            df[index_name] = pd.to_numeric(df[index_name], errors='coerce') # Конвертируем Close в число
            df = df.dropna().sort_values("DATE").reset_index(drop=True)

            self.logger.info(f"Успешно загружено {len(df)} записей для индекса {index_name}.")
            return df
        except Exception as e:
            self.logger.exception(f"Ошибка при обработке DataFrame для индекса {index_name}: {e}")
            return None

    def download_indices(
        self,
        output_file: str,
        start_date: date,
        end_date: date,
        indices_list: Optional[List[str]] = None,
        delay: float = 1.0
        ) -> pd.DataFrame:
        """
        Загружает данные для списка индексов и сохраняет их в один CSV файл.

        Parameters:
        -----------
        output_file : str
            Путь к итоговому CSV файлу.
        start_date : date
            Дата начала периода.
        end_date : date
            Дата окончания периода.
        indices_list : Optional[List[str]], optional
            Список тикеров индексов для загрузки. Если None, используются все из конфига/по умолчанию.
            По умолчанию None.
        delay : float, optional
            Задержка между запросами к API в секундах. По умолчанию 1.0.

        Returns:
        --------
        pd.DataFrame:
             DataFrame с отчетом о результатах загрузки для каждого индекса.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Создаем директорию, если нужно

        if indices_list is None:
            indices_to_process = self.indices
        else:
            # Проверяем, что переданные индексы существуют в общем списке (если он был из конфига)
            # или просто используем переданный список, если конфига не было
            known_indices = set(self.indices)
            indices_to_process = [idx for idx in indices_list if idx in known_indices]
            missing = set(indices_list) - known_indices
            if missing:
                 self.logger.warning(f"Следующие индексы из списка не найдены в известном списке и будут пропущены: {missing}")
            if not indices_to_process:
                self.logger.error("Список индексов для обработки пуст после фильтрации.")
                return pd.DataFrame(columns=['index', 'status', 'rows']) # Возвращаем пустой отчет

        self.logger.info(f"Начало загрузки данных для {len(indices_to_process)} индексов...")
        all_indices_data = []
        results = []

        for index_name in tqdm(indices_to_process, desc="Загрузка индексов MOEX"):
            df_index = self._fetch_single_index(index_name, start_date, end_date)

            if df_index is not None and not df_index.empty:
                all_indices_data.append(df_index)
                status = "Успешно"
                rows = len(df_index)
            elif df_index is None:
                 status = "Ошибка загрузки"
                 rows = 0
            else: # df_index is empty
                 status = "Нет данных"
                 rows = 0

            results.append({
                'index': index_name,
                'status': status,
                'rows': rows,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })

            if len(indices_to_process) > 1:
                 time.sleep(delay) # Задержка только если больше одного индекса

        if not all_indices_data:
            self.logger.error("Не удалось загрузить данные ни для одного индекса.")
            return pd.DataFrame(results) # Возвращаем отчет об ошибках

        # Объединение данных всех индексов
        try:
            # Начинаем с первого DataFrame
            merged_df = all_indices_data[0]
            # Последовательно объединяем остальные по дате
            for i in range(1, len(all_indices_data)):
                merged_df = pd.merge(merged_df, all_indices_data[i], on='DATE', how='outer')

            # Сортировка по дате
            merged_df = merged_df.sort_values('DATE').reset_index(drop=True)

            # Сохранение в CSV
            merged_df.to_csv(output_path, index=False)
            self.logger.info(f"Данные по {len(merged_df.columns)-1} индексам успешно объединены и сохранены в {output_path}")

        except Exception as e:
            self.logger.exception(f"Ошибка при объединении или сохранении данных индексов: {e}")
            # В этом случае возвращаем только отчет results, т.к. файл не был сохранен

        return pd.DataFrame(results)