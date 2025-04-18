import requests
import pandas as pd
from datetime import date
import logging
from pathlib import Path
from typing import Optional

class UsdRubLoader:
    """Класс для загрузки исторических данных курса USD/RUB с сайта ЦБ РФ."""

    def __init__(self):
        """Инициализация загрузчика."""
        self.base_url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
        self.currency_code = 'R01235' # Код валюты USD в ЦБ РФ
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Настройка логирования."""
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)

    def fetch_rates(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Загружает данные о курсе USD/RUB за указанный период.

        Parameters:
        -----------
        start_date : date
            Дата начала периода.
        end_date : date
            Дата окончания периода.

        Returns:
        --------
        Optional[pd.DataFrame]:
             DataFrame с колонками ['DATE', 'USD_RUB'] или None в случае ошибки.
        """
        start_date_str = start_date.strftime('%d/%m/%Y')
        end_date_str = end_date.strftime('%d/%m/%Y')
        self.logger.info(f"Запрос курса USD/RUB с {start_date_str} по {end_date_str}")

        params = {
            'date_req1': start_date_str,
            'date_req2': end_date_str,
            'VAL_NM_RQ': self.currency_code
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status() # Проверка на HTTP ошибки (4xx, 5xx)

            # Попытка парсинга XML
            # Добавляем обработку пустого ответа или не-XML
            if not response.content or 'text/html' in response.headers.get('Content-Type', ''):
                self.logger.warning(f"Получен пустой или HTML ответ от ЦБ РФ для диапазона {start_date_str} - {end_date_str}. Возможно, нет данных.")
                return None

            df = pd.read_xml(response.content, xpath="//Record")

            if df.empty:
                 self.logger.warning(f"Нет данных (пустой Record) в ответе ЦБ РФ для диапазона {start_date_str} - {end_date_str}.")
                 return None

            # Проверка наличия ожидаемых колонок
            if 'Date' not in df.columns or 'Value' not in df.columns:
                self.logger.error(f"Ожидаемые колонки 'Date' или 'Value' не найдены в XML. Доступные: {df.columns.tolist()}")
                return None

            # Преобразование данных
            df['DATE'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
            # Заменяем запятую на точку и конвертируем в float, обрабатывая ошибки
            df['USD_RUB'] = pd.to_numeric(df['Value'].str.replace(',', '.', regex=False), errors='coerce')

            # Удаляем строки с ошибками конвертации
            original_len = len(df)
            df = df.dropna(subset=['USD_RUB'])
            if len(df) < original_len:
                self.logger.warning(f"Удалено {original_len - len(df)} строк с некорректными значениями курса.")

            # Выбираем нужные колонки и сортируем
            df = df[['DATE', 'USD_RUB']].sort_values('DATE').reset_index(drop=True)

            self.logger.info(f"Успешно загружено {len(df)} записей курса USD/RUB.")
            return df

        except pd.errors.XMLSyntaxError as xml_err:
             self.logger.error(f"Ошибка парсинга XML ответа от ЦБ РФ: {xml_err}")
             return None
        except requests.exceptions.Timeout:
            self.logger.error(f"Таймаут при запросе курса USD/RUB к ЦБ РФ.")
            return None
        except requests.exceptions.RequestException as req_err:
            status_code = req_err.response.status_code if req_err.response is not None else "N/A"
            self.logger.error(f"Ошибка сети (статус: {status_code}) при запросе курса USD/RUB: {req_err}")
            return None
        except Exception as e:
            self.logger.exception(f"Непредвиденная ошибка при загрузке курса USD/RUB: {e}")
            return None

    def download_rates(self, output_file: str, start_date: date, end_date: date) -> bool:
        """Загружает курс USD/RUB за период и сохраняет в CSV.

        Parameters:
        -----------
        output_file : str
            Путь к выходному CSV файлу.
        start_date : date
            Дата начала периода.
        end_date : date
            Дата окончания периода.

        Returns:
        --------
        bool: True в случае успеха, False в случае ошибки.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_rates = self.fetch_rates(start_date, end_date)

        if df_rates is not None and not df_rates.empty:
            try:
                df_rates.to_csv(output_path, index=False)
                self.logger.info(f"Данные курса USD/RUB сохранены в {output_path}")
                return True
            except IOError as e:
                self.logger.error(f"Ошибка записи файла {output_path}: {e}")
                return False
        elif df_rates is None:
             self.logger.error("Ошибка при загрузке данных курса USD/RUB, файл не будет сохранен.")
             return False
        else: # df_rates is empty
             self.logger.warning("Нет данных курса USD/RUB для сохранения.")
             # Считаем это успехом, так как ошибки загрузки не было, просто нет данных
             # Можно создать пустой файл или не создавать вовсе.
             # Давайте не будем создавать пустой файл, чтобы не путать.
             return True 