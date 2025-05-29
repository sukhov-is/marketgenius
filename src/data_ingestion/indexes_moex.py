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

    Поддерживает загрузку списка индексов и цен на драгоценные металлы
    за указанный период и сохранение в единый CSV-файл.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация загрузчика.

        Parameters:
        -----------
        config_path : Optional[str], optional
            Путь к JSON файлу конфигурации. Ожидается ключ 'indices' со списком
            тикеров индексов. Если None, будет использован список по умолчанию.
            По умолчанию None.
        """
        self.base_url_index = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities"
        self.base_url_metal = "https://iss.moex.com/iss/history/engines/currency/markets/selt/securities"
        self.metals = {
            'GLDRUB_TOM': 'GOLD',
            'SLVRUB_TOM': 'SILVER',
        }
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
        url = f"{self.base_url_index}/{index_name}.json"
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

    def _fetch_single_metal_price(self, metal_ticker: str, metal_name: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Получает данные по цене одного драгоценного металла с MOEX ISS API.

        Returns:
        --------
        Optional[pd.DataFrame]: DataFrame с колонками ['DATE', metal_name] или None при ошибке.
        """
        url = f"{self.base_url_metal}/{metal_ticker}.json"
        params = {
            "from": start_date.strftime('%Y-%m-%d'),
            "till": end_date.strftime('%Y-%m-%d'),
            "iss.meta": "off",
            "history.columns": "TRADEDATE,CLOSE,BOARDID", # Запрашиваем BOARDID для фильтрации
            "start": 0,
            "limit": 100
        }
        all_data = []
        self.logger.debug(f"Запрос данных для металла {metal_name} ({metal_ticker}) с {start_date} по {end_date}")

        while True:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                history_block = data.get('history', {})
                metal_data = history_block.get('data', [])
                columns = history_block.get('columns', [])

                if not metal_data:
                    self.logger.debug(f"Нет данных 'history' для {metal_name} ({metal_ticker}) в диапазоне с {params['start']}.")
                    break

                all_data.extend(metal_data)

                cursor_data = data.get('history.cursor', {})
                cursor_info = cursor_data.get('data', [])
                if not cursor_info:
                    break
                
                current_index, total_rows, page_size = cursor_info[0]
                start_pos = current_index + page_size
                if start_pos >= total_rows:
                    break
                params['start'] = start_pos
                self.logger.debug(f"Запрос следующей страницы для {metal_name} ({metal_ticker}), start={start_pos}")

            except requests.exceptions.Timeout:
                self.logger.error(f"Таймаут при получении данных для {metal_name} ({metal_ticker}).")
                return None
            except requests.exceptions.RequestException as e:
                status_code = e.response.status_code if e.response is not None else "N/A"
                self.logger.error(f"Ошибка сети (статус: {status_code}) при получении данных для {metal_name} ({metal_ticker}): {e}")
                return None
            except json.JSONDecodeError:
                self.logger.error(f"Ошибка декодирования JSON ответа для {metal_name} ({metal_ticker}).")
                return None
            except Exception as e:
                self.logger.exception(f"Непредвиденная ошибка при получении данных для {metal_name} ({metal_ticker}): {e}")
                return None
        
        if not all_data:
            self.logger.warning(f"Данные для металла {metal_name} ({metal_ticker}) за период {start_date} - {end_date} не найдены.")
            return None

        if not columns:
            self.logger.error(f"Не удалось получить названия колонок для металла {metal_name} ({metal_ticker}).")
            return None
        
        try:
            df = pd.DataFrame(all_data, columns=columns)
            
            # Фильтруем только записи с BOARDID 'CETS' и ценой > 0 (как в precious_metals_parser.py)
            df = df[(df['BOARDID'] == 'CETS') & (df['CLOSE'] > 0)]

            if df.empty:
                self.logger.warning(f"После фильтрации не осталось данных для {metal_name} ({metal_ticker}) за период {start_date} - {end_date}.")
                return None

            date_col = 'TRADEDATE'
            close_col = 'CLOSE'
            if date_col not in df.columns or close_col not in df.columns:
                self.logger.error(f"Ожидаемые колонки '{date_col}' или '{close_col}' не найдены в данных для {metal_name}. Доступные: {df.columns.tolist()}")
                return None

            df = df[[date_col, close_col]].copy()
            df.columns = ["DATE", metal_name]
            df["DATE"] = pd.to_datetime(df["DATE"])
            df[metal_name] = pd.to_numeric(df[metal_name], errors='coerce')
            df = df.dropna().sort_values("DATE").drop_duplicates(subset=['DATE'], keep='last').reset_index(drop=True) # Убираем дубликаты по дате, берем последнюю цену

            self.logger.info(f"Успешно загружено {len(df)} записей для металла {metal_name} ({metal_ticker}).")
            return df
        except Exception as e:
            self.logger.exception(f"Ошибка при обработке DataFrame для металла {metal_name} ({metal_ticker}): {e}")
            return None

    def download_indices(
        self,
        output_file: str,
        start_date: date,
        end_date: date,
        indices_list: Optional[List[str]] = None,
        include_metals: bool = True, # Новый параметр
        delay: float = 1.0
        ) -> pd.DataFrame:
        """
        Загружает данные для списка индексов (и опционально металлов) и сохраняет их в один CSV файл.

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
        include_metals : bool, optional
            Флаг, указывающий, следует ли включать данные о металлах. По умолчанию True.
        delay : float, optional
            Задержка между запросами к API в секундах. По умолчанию 1.0.

        Returns:
        --------
        pd.DataFrame:
             DataFrame с отчетом о результатах загрузки для каждого элемента.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Создаем директорию, если нужно

        items_to_process = []
        item_sources = {} # Для хранения источника (индекс/металл) и оригинального имени/тикера

        if indices_list is None:
            for index_name in self.indices:
                items_to_process.append(index_name)
                item_sources[index_name] = {'type': 'index', 'original_name': index_name}
            if include_metals:
                for metal_ticker, metal_name in self.metals.items():
                    # Используем metal_name как ключ для item_sources, если metal_ticker уже есть (маловероятно, но для уникальности)
                    # или сам metal_ticker, если это обеспечит уникальность с индексами
                    processing_key = metal_ticker # Обычно тикеры металлов не пересекаются с именами индексов
                    items_to_process.append(processing_key)
                    item_sources[processing_key] = {'type': 'metal', 'original_name': metal_ticker, 'display_name': metal_name}
        else:
            # Обрабатываем предоставленный список: могут быть и индексы, и тикеры металлов
            for item_name_or_ticker in indices_list:
                if item_name_or_ticker in self.indices:
                    items_to_process.append(item_name_or_ticker)
                    item_sources[item_name_or_ticker] = {'type': 'index', 'original_name': item_name_or_ticker}
                elif include_metals and item_name_or_ticker in self.metals:
                    items_to_process.append(item_name_or_ticker)
                    item_sources[item_name_or_ticker] = {'type': 'metal', 'original_name': item_name_or_ticker, 'display_name': self.metals[item_name_or_ticker]}
                else:
                    self.logger.warning(f"Элемент '{item_name_or_ticker}' из списка не найден ни среди известных индексов, ни среди металлов (или include_metals=False). Он будет пропущен.")
        
        if not items_to_process:
            self.logger.error("Список элементов для обработки пуст.")
            return pd.DataFrame(columns=['item', 'type', 'status', 'rows', 'start_date', 'end_date'])

        self.logger.info(f"Начало загрузки данных для {len(items_to_process)} элементов (индексы/металлы)...")
        all_data_list = [] # Переименовано из all_indices_data
        results = []

        for item_key in tqdm(items_to_process, desc="Загрузка данных MOEX"):
            item_info = item_sources[item_key]
            item_type = item_info['type']
            df_item = None
            item_display_name = item_key # По умолчанию

            if item_type == 'index':
                original_index_name = item_info['original_name']
                item_display_name = original_index_name
                df_item = self._fetch_single_index(original_index_name, start_date, end_date)
            elif item_type == 'metal':
                metal_ticker = item_info['original_name']
                metal_name = item_info['display_name']
                item_display_name = metal_name # Для отчета используем человекочитаемое имя
                df_item = self._fetch_single_metal_price(metal_ticker, metal_name, start_date, end_date)

            status = "Ошибка загрузки"
            rows = 0
            if df_item is not None and not df_item.empty:
                all_data_list.append(df_item)
                status = "Успешно"
                rows = len(df_item)
            elif df_item is None: # Явная ошибка при загрузке
                 status = "Ошибка загрузки"
            else: # df_item пустой, но не None
                 status = "Нет данных"
            
            results.append({
                'item': item_display_name, # Используем имя индекса или название металла
                'type': item_type,
                'status': status,
                'rows': rows,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })

            if len(items_to_process) > 1: # Задержка если обрабатывается больше одного элемента
                 time.sleep(delay)

        if not all_data_list:
            self.logger.error("Не удалось загрузить данные ни для одного элемента.")
            return pd.DataFrame(results)

        try:
            merged_df = all_data_list[0]
            for i in range(1, len(all_data_list)):
                merged_df = pd.merge(merged_df, all_data_list[i], on='DATE', how='outer')
            
            merged_df = merged_df.sort_values('DATE').reset_index(drop=True)
            
            # Заполнение пропусков (NaN) после 'outer' merge
            # Можно выбрать метод ffill или bfill, или оставить как есть
            # merged_df = merged_df.fillna(method='ffill') # Пример: прямое заполнение вперед
            
            merged_df.to_csv(output_path, index=False)
            num_series = len(merged_df.columns) -1 # Минус колонка DATE
            self.logger.info(f"Данные по {num_series} активам (индексы/металлы) успешно объединены и сохранены в {output_path}")

        except Exception as e:
            self.logger.exception(f"Ошибка при объединении или сохранении данных: {e}")

        return pd.DataFrame(results)