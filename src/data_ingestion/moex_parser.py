import pandas as pd
import requests
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import time


class MoexLoader:
    """Класс для загрузки исторических данных с Московской биржи"""
    
    def __init__(self, config_path: str):
        """
        Инициализация загрузчика данных MOEX
        
        Parameters:
        -----------
        config_path : str
            Путь к конфигурационному файлу с списком компаний
        """
        self.base_url = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities"
        self.companies = self._load_companies(config_path)
        self._setup_logging()
    
    def _load_companies(self, config_path: str) -> Dict[str, str]:
        """Загрузка списка компаний из конфигурационного файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                companies_data = config.get('companies', {})
                # Преобразуем новый формат к старому для обратной совместимости
                companies = {}
                for ticker, data in companies_data.items():
                    if isinstance(data, dict) and 'names' in data and data['names']:
                        # Берем первое название из списка как основное
                        companies[ticker] = data['names'][0]
                    elif isinstance(data, str):
                        # Поддержка старого формата
                        companies[ticker] = data
                return companies
        except Exception as e:
            logging.error(f"Ошибка при загрузке конфигурационного файла {config_path}: {str(e)}")
            raise Exception(f"Ошибка при загрузке конфигурационного файла: {str(e)}")
    
    def _setup_logging(self) -> None:
        """Настройка логирования"""
        if not logging.getLogger().hasHandlers():
             logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
             )
        self.logger = logging.getLogger(__name__)
    
    def load_security_data(self, security: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Загрузка данных для конкретной ценной бумаги за указанный период.
        
        Parameters:
        -----------
        security : str
            Тикер ценной бумаги.
        start_date : date
            Дата начала периода.
        end_date : date
            Дата окончания периода.
        
        Returns:
        --------
        Optional[pd.DataFrame]
            DataFrame с историческими данными или None в случае ошибки.
        """
        self.logger.info(f"Загрузка данных для {security} с {start_date} по {end_date}")
        max_retries = 3
        retry_delay = 5 # seconds
        request_timeout = 30 # seconds

        try:
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'till': end_date.strftime('%Y-%m-%d'),
                'start': 0,
                'limit': 100
            }

            all_data = []

            while True:
                url = f"{self.base_url}/{security}.json"
                for attempt in range(max_retries):
                    try:
                        r = requests.get(url, params=params, timeout=request_timeout)
                        r.raise_for_status()
                        data = r.json()
                        break  # Успешный запрос, выходим из цикла попыток
                    except requests.exceptions.Timeout as timeout_e:
                        self.logger.warning(f"Таймаут при запросе к MOEX API для {security} (попытка {attempt + 1}/{max_retries}): {timeout_e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            self.logger.error(f"Превышено количество попыток запроса к MOEX API для {security} после таймаутов.")
                            return None
                    except requests.exceptions.RequestException as req_e:
                        self.logger.error(f"Ошибка сети при запросе к MOEX API для {security} (попытка {attempt + 1}/{max_retries}): {req_e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay) # Добавляем задержку перед следующей попыткой
                        else:
                            self.logger.error(f"Превышено количество попыток запроса к MOEX API для {security} после сетевых ошибок.")
                            return None # Возвращаем None после всех неудачных попыток
                    except json.JSONDecodeError as json_e:
                        self.logger.error(f"Ошибка декодирования JSON от MOEX API для {security}: {json_e}")
                        return None # Ошибка декодирования JSON обычно не требует повторных попыток
                else: # Этот блок выполнится, если цикл for завершился без break (все попытки неудачны)
                    return None

                history_data = data.get('history', {}).get('data', [])
                if not history_data:
                    self.logger.info(f"Нет данных 'history' для {security} в диапазоне с {params['start']}.")
                    break

                all_data.extend(history_data)

                cursor_data = data.get('history.cursor', {}).get('data', [])
                if not cursor_data:
                    break

                current_index, total_rows, page_size = cursor_data[0]
                start = current_index + page_size

                if start >= total_rows:
                    break

                params['start'] = start
                self.logger.debug(f"Запрос следующей страницы для {security}, start={start}")


            if not all_data:
                self.logger.warning(f"Данные для {security} за период {start_date} - {end_date} не найдены.")
                return None

            columns = data.get('history', {}).get('columns', [])
            if not columns:
                 self.logger.error(f"Не найдены колонки 'history.columns' в ответе API для {security}")
                 return None

            df = pd.DataFrame(all_data, columns=columns)

            df = df[df['BOARDID'] == 'TQBR']
            if df.empty:
                self.logger.warning(f"Нет данных в режиме TQBR для {security} за период {start_date} - {end_date}.")
                return None

            needed_columns = ['TRADEDATE', 'SECID', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'WAPRICE']
            available_columns = [col for col in needed_columns if col in df.columns]
            if len(available_columns) != len(needed_columns):
                missing = set(needed_columns) - set(available_columns)
                self.logger.warning(f"В данных для {security} отсутствуют колонки: {missing}. Используются только доступные.")

            if not available_columns:
                 self.logger.error(f"Ни одна из необходимых колонок ({needed_columns}) не найдена для {security}.")
                 return None

            df = df[available_columns]

            numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'WAPRICE']
            for col in numeric_cols:
                if col in df.columns:
                     df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.sort_values('TRADEDATE').reset_index(drop=True)
            self.logger.info(f"Успешно загружено {len(df)} строк для {security}.")
            return df

        except Exception as e:
            self.logger.exception(f"Непредвиденная ошибка при загрузке данных для {security}: {str(e)}")
            return None
    
    def download_historical_range(
        self,
        output_dir: str,
        start_date: date,
        end_date: date,
        tickers_list: Optional[List[str]] = None,
        delay: float = 1.0
        ) -> None:
        """
        Загрузка и сохранение данных для списка тикеров за указанный диапазон дат.
        
        Parameters:
        -----------
        output_dir : str
            Директория для сохранения файлов CSV.
        start_date : date
            Дата начала периода.
        end_date : date
            Дата окончания периода.
        tickers_list : Optional[List[str]], optional
            Список тикеров для загрузки. Если None, используются все тикеры из конфига.
            По умолчанию None.
        delay : float, optional
            Задержка между запросами к API в секундах. По умолчанию 1.0.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []

        if tickers_list is None:
            tickers_to_load = list(self.companies.keys())
        else:
            tickers_to_load = tickers_list

        self.logger.info(f"Начало загрузки данных для {len(tickers_to_load)} тикеров...")

        for ticker in tqdm(tickers_to_load, desc="Загрузка данных MOEX"):
            company_name = self.companies.get(ticker, "N/A")
            self.logger.info(f"Обработка тикера: {ticker} ({company_name})")

            df = self.load_security_data(ticker, start_date, end_date)

            if df is not None and not df.empty:
                file_path = output_path / f"{ticker}_moex_data.csv"
                try:
                    df.to_csv(file_path, index=False)
                    status = "Успешно"
                    rows = len(df)
                    self.logger.info(f"Данные для {ticker} сохранены в {file_path}")
                except IOError as io_e:
                    status = "Ошибка сохранения"
                    rows = 0
                    self.logger.error(f"Не удалось сохранить файл для {ticker} в {file_path}: {io_e}")
            elif df is None:
                status = "Ошибка загрузки"
                rows = 0
            else:
                status = "Нет данных"
                rows = 0
                self.logger.warning(f"Нет данных для сохранения для тикера {ticker}")

            results.append({
                'ticker': ticker,
                'name': company_name,
                'status': status,
                'rows': rows,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })

            time.sleep(delay)

        report_path = output_path / 'download_report.csv'
        try:
            pd.DataFrame(results).to_csv(report_path, index=False)
            self.logger.info(f"Отчет о загрузке сохранен в {report_path}")
        except IOError as io_e:
            self.logger.error(f"Не удалось сохранить отчет о загрузке в {report_path}: {io_e}")