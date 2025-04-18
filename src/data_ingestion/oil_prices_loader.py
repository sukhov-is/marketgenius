import requests
import pandas as pd
from datetime import date, datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

class AlphaVantageBrentLoader:
    """Класс для загрузки исторических цен на нефть Brent с Alpha Vantage API."""

    def __init__(self, alpha_vantage_api_key: str):
        """Инициализация загрузчика.

        Args:
            alpha_vantage_api_key (str): Ваш API ключ для Alpha Vantage.
        """
        if not alpha_vantage_api_key:
            raise ValueError("API ключ Alpha Vantage не может быть пустым.")

        self.alpha_vantage_base_url = "https://www.alphavantage.co/query"
        self.api_key = alpha_vantage_api_key
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Настройка логирования."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
             self.logger.setLevel(logging.INFO)
        self.logger.setLevel(logging.INFO)

    def _fetch_brent_prices(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Загружает цены на нефть Brent с Alpha Vantage API.

        Args:
            start_date (date): Дата начала периода (включительно).
            end_date (date): Дата окончания периода (включительно).

        Returns:
            Optional[pd.DataFrame]: DataFrame с колонками ['DATE', 'BRENT_CLOSE']
                                     или None в случае ошибки.
                                     Примечание: Alpha Vantage API (BRENT function)
                                     обычно возвращает всю доступную историю, фильтрация
                                     по датам происходит после загрузки.
        """
        self.logger.info(f"Запрос цен на Brent (Alpha Vantage) с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")

        params = {
            "function": "BRENT",
            "interval": "daily",
            "apikey": self.api_key
        }

        try:
            response = requests.get(self.alpha_vantage_base_url, params=params, timeout=60)
            response.raise_for_status() # Проверка на HTTP ошибки (4xx, 5xx)
            data = response.json()

            # Проверка на сообщение об ошибке от API Alpha Vantage
            if 'Error Message' in data:
                self.logger.error(f"Ошибка API Alpha Vantage: {data['Error Message']}")
                return None
            if 'Note' in data:
                 self.logger.warning(f"Примечание API Alpha Vantage: {data['Note']}")
                 # Продолжаем выполнение, но логируем возможное ограничение частоты запросов

            # Проверка основной структуры ответа
            if 'data' not in data or not isinstance(data['data'], list) or 'name' not in data:
                 self.logger.warning(f"Неожиданный формат ответа от Alpha Vantage API: {str(data)[:200]}...")
                 return None

            if not data['data']:
                 self.logger.warning("Alpha Vantage API вернул пустой список данных для Brent.")
                 return None

            # Преобразование данных в DataFrame
            df = pd.DataFrame(data['data'])

            if 'date' not in df.columns or 'value' not in df.columns:
                self.logger.error(f"Ожидаемые колонки 'date' или 'value' не найдены в ответе Alpha Vantage. Доступные: {df.columns.tolist()}")
                return None

            # Переименование и преобразование типов
            df.rename(columns={'date': 'DATE', 'value': 'BRENT_CLOSE'}, inplace=True)
            df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d', errors='coerce')
            # Значение 'value' может быть строкой '.' для отсутствующих данных
            df['BRENT_CLOSE'] = pd.to_numeric(df['BRENT_CLOSE'], errors='coerce')

            # Удаление строк с некорректными датами или ценами
            df.dropna(subset=['DATE', 'BRENT_CLOSE'], inplace=True)

            # Фильтрация по дате уже после загрузки
            # Преобразуем start_date и end_date в datetime для сравнения
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())

            df = df[(df['DATE'] >= start_dt) & (df['DATE'] <= end_dt)].copy()

            if df.empty:
                self.logger.warning(f"Нет данных Brent от Alpha Vantage в указанном диапазоне дат: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
                # Возвращаем пустой DataFrame, а не None, т.к. загрузка прошла, но данных нет
                return df

            # Сортировка по дате
            df = df.sort_values('DATE').reset_index(drop=True)

            self.logger.info(f"Успешно загружено и отфильтровано {len(df)} записей цен Brent (Alpha Vantage).")
            return df

        except requests.exceptions.Timeout:
            self.logger.error("Таймаут при запросе цен Brent к Alpha Vantage API.")
            return None
        except requests.exceptions.RequestException as req_err:
            status_code = req_err.response.status_code if req_err.response is not None else "N/A"
            self.logger.error(f"Ошибка сети (статус: {status_code}) при запросе цен Brent к Alpha Vantage API: {req_err}")
            return None
        except json.JSONDecodeError:
             self.logger.error("Ошибка декодирования JSON ответа от Alpha Vantage API.")
             return None
        except Exception as e:
            self.logger.exception(f"Непредвиденная ошибка при загрузке цен Brent с Alpha Vantage: {e}")
            return None

    def download_prices(self, output_dir: str, start_date: date, end_date: date) -> bool:
        """Загружает цены на Brent (Alpha Vantage) за период и сохраняет в CSV файл.

        Args:
            output_dir (str): Директория для сохранения CSV файла (brent_prices.csv).
            start_date (date): Дата начала периода.
            end_date (date): Дата окончания периода.

        Returns:
            bool: True, если загрузка и сохранение прошли успешно (или данных нет в диапазоне),
                  False в случае ошибки загрузки/записи.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        success = False # Статус по умолчанию - неудача

        # Загрузка и сохранение Brent (Alpha Vantage)
        df_brent = self._fetch_brent_prices(start_date, end_date)

        # Важно: _fetch_brent_prices может вернуть пустой DataFrame, если данные
        # за диапазон дат отсутствуют, но сама загрузка прошла успешно.
        # Ошибкой считаем только если вернулся None.
        if df_brent is not None:
            if not df_brent.empty:
                brent_file = output_path / "brent_prices.csv"
                try:
                    df_brent.to_csv(brent_file, index=False)
                    self.logger.info(f"Данные Brent (Alpha Vantage) сохранены в {brent_file}")
                    success = True # Успех, если сохранили
                except IOError as e:
                    self.logger.error(f"Ошибка записи файла Brent {brent_file}: {e}")
                    # success остается False
            else:
                 # Загрузка прошла, но данных в диапазоне нет или API вернул пустой список
                 self.logger.warning(f"Нет данных Brent (Alpha Vantage) для сохранения в диапазоне {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}.")
                 success = True # Считаем успехом, т.к. ошибки загрузки не было
        else:
             # Если df_brent is None, значит была ошибка при загрузке
             self.logger.error("Ошибка при загрузке данных Brent (Alpha Vantage), файл не будет сохранен.")
             # success остается False

        return success 