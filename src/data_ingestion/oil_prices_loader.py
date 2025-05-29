import requests
import pandas as pd
from datetime import date, datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

class AlphaVantageBrentLoader:
    """Класс для загрузки исторических цен на нефть Brent и природный газ с Alpha Vantage API."""

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

    def _calculate_percentage_changes(self, df: pd.DataFrame, column_name: str = 'BRENT_CLOSE') -> pd.DataFrame:
        """Рассчитывает процентное изменение для указанной колонки за разные периоды."""
        df_copy = df.copy()
        df_copy = df_copy.sort_values('DATE') # Убедимся, что данные отсортированы по дате

        for period in [1, 3, 7, 30]:
            df_copy[f'{column_name}_change_{period}D'] = df_copy[column_name].pct_change(periods=period) * 100
        return df_copy

    def _fetch_commodity_prices(self, commodity_name: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Загружает цены на указанный товар (Brent или Natural Gas) с Alpha Vantage API.

        Args:
            commodity_name (str): Название товара ("BRENT" или "NATURAL_GAS").
            start_date (date): Дата начала периода (включительно).
            end_date (date): Дата окончания периода (включительно).

        Returns:
            Optional[pd.DataFrame]: DataFrame с колонками ['DATE', '{COMMODITY_NAME}_CLOSE']
                                     или None в случае ошибки.
        """
        self.logger.info(f"Запрос цен на {commodity_name} (Alpha Vantage) с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")

        params = {
            "function": commodity_name.upper(), # Используем commodity_name для параметра function
            "interval": "daily",
            "apikey": self.api_key
        }
        column_name_suffix = "CLOSE"
        log_commodity_name = commodity_name

        try:
            response = requests.get(self.alpha_vantage_base_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if 'Error Message' in data:
                self.logger.error(f"Ошибка API Alpha Vantage для {log_commodity_name}: {data['Error Message']}")
                return None
            if 'Note' in data:
                self.logger.warning(f"Примечание API Alpha Vantage для {log_commodity_name}: {data['Note']}")

            if 'data' not in data or not isinstance(data['data'], list) or 'name' not in data:
                self.logger.warning(f"Неожиданный формат ответа от Alpha Vantage API для {log_commodity_name}: {str(data)[:200]}...")
                return None

            if not data['data']:
                self.logger.warning(f"Alpha Vantage API вернул пустой список данных для {log_commodity_name}.")
                return None

            df = pd.DataFrame(data['data'])

            if 'date' not in df.columns or 'value' not in df.columns:
                self.logger.error(f"Ожидаемые колонки 'date' или 'value' не найдены в ответе Alpha Vantage для {log_commodity_name}. Доступные: {df.columns.tolist()}")
                return None

            df.rename(columns={'date': 'DATE', 'value': f'{commodity_name.upper()}_{column_name_suffix}'}, inplace=True)
            df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d', errors='coerce')
            df[f'{commodity_name.upper()}_{column_name_suffix}'] = pd.to_numeric(df[f'{commodity_name.upper()}_{column_name_suffix}'], errors='coerce')

            df.dropna(subset=['DATE', f'{commodity_name.upper()}_{column_name_suffix}'], inplace=True)

            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[(df['DATE'] >= start_dt) & (df['DATE'] <= end_dt)].copy()

            if df.empty:
                self.logger.warning(f"Нет данных {log_commodity_name} от Alpha Vantage в указанном диапазоне дат: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
                return df

            df = df.sort_values('DATE').reset_index(drop=True)
            self.logger.info(f"Успешно загружено и отфильтровано {len(df)} записей цен {log_commodity_name} (Alpha Vantage).")
            return df

        except requests.exceptions.Timeout:
            self.logger.error(f"Таймаут при запросе цен {log_commodity_name} к Alpha Vantage API.")
            return None
        except requests.exceptions.RequestException as req_err:
            status_code = req_err.response.status_code if req_err.response is not None else "N/A"
            self.logger.error(f"Ошибка сети (статус: {status_code}) при запросе цен {log_commodity_name} к Alpha Vantage API: {req_err}")
            return None
        except json.JSONDecodeError:
            self.logger.error(f"Ошибка декодирования JSON ответа от Alpha Vantage API для {log_commodity_name}.")
            return None
        except Exception as e:
            self.logger.exception(f"Непредвиденная ошибка при загрузке цен {log_commodity_name} с Alpha Vantage: {e}")
            return None

    def _fetch_brent_prices(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Загружает цены на нефть Brent с Alpha Vantage API."""
        return self._fetch_commodity_prices("BRENT", start_date, end_date)

    def _fetch_natural_gas_prices(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Загружает цены на природный газ с Alpha Vantage API."""
        return self._fetch_commodity_prices("NATURAL_GAS", start_date, end_date)

    def download_prices(self, output_dir: str, start_date: date, end_date: date) -> bool:
        """
        Загружает цены на Brent и природный газ (Alpha Vantage) за период,
        объединяет их и сохраняет в один CSV файл (commodity_prices.csv).

        Args:
            output_dir (str): Директория для сохранения CSV файла.
            start_date (date): Дата начала периода.
            end_date (date): Дата окончания периода.

        Returns:
            bool: True, если загрузка, объединение и сохранение прошли успешно
                  (или данных для объединения нет, но ошибки не было),
                  False в случае критической ошибки загрузки/записи.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Загрузка данных
        df_brent = self._fetch_brent_prices(start_date, end_date)
        df_natural_gas = self._fetch_natural_gas_prices(start_date, end_date)

        brent_fetched_successfully = df_brent is not None
        natural_gas_fetched_successfully = df_natural_gas is not None

        brent_has_data = brent_fetched_successfully and not df_brent.empty
        natural_gas_has_data = natural_gas_fetched_successfully and not df_natural_gas.empty

        final_df = None

        if brent_has_data and natural_gas_has_data:
            final_df = pd.merge(df_brent, df_natural_gas, on='DATE', how='outer')
            final_df = final_df.sort_values('DATE').reset_index(drop=True)
            self.logger.info("Данные по нефти Brent и природному газу успешно загружены и объединены.")
        elif brent_has_data:
            final_df = df_brent
            self.logger.info("Используются только данные по нефти Brent.")
            if not natural_gas_fetched_successfully:
                self.logger.error("Ошибка при загрузке данных по природному газу (Alpha Vantage).")
            elif natural_gas_fetched_successfully and df_natural_gas.empty:
                 self.logger.warning(f"Данные по природному газу (Alpha Vantage) пусты для диапазона {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}.")
        elif natural_gas_has_data:
            final_df = df_natural_gas
            self.logger.info("Используются только данные по природному газу.")
            if not brent_fetched_successfully:
                self.logger.error("Ошибка при загрузке данных по нефти Brent (Alpha Vantage).")
            elif brent_fetched_successfully and df_brent.empty:
                self.logger.warning(f"Данные по нефти Brent (Alpha Vantage) пусты для диапазона {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}.")
        else:
            # Логируем, если ни один из источников не дал данных или были ошибки
            if not brent_fetched_successfully:
                 self.logger.error("Ошибка при загрузке данных Brent (Alpha Vantage).")
            elif df_brent is not None and df_brent.empty: # Успешно, но пусто
                 self.logger.warning(f"Нет данных Brent (Alpha Vantage) в диапазоне {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}.")
            
            if not natural_gas_fetched_successfully:
                 self.logger.error("Ошибка при загрузке данных по природному газу (Alpha Vantage).")
            elif df_natural_gas is not None and df_natural_gas.empty: # Успешно, но пусто
                 self.logger.warning(f"Нет данных по природному газу (Alpha Vantage) в диапазоне {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}.")

            if not brent_fetched_successfully and not natural_gas_fetched_successfully:
                 self.logger.error("Не удалось загрузить данные ни по нефти, ни по газу. Файл не будет создан.")
            elif (brent_fetched_successfully and df_brent.empty) and \
                 (natural_gas_fetched_successfully and df_natural_gas.empty):
                 self.logger.warning("Данные по нефти Brent и природному газу пусты. Файл не будет создан.")


        if final_df is not None and not final_df.empty:
            output_file = output_path / "brent_prices.csv"
            try:
                final_df.to_csv(output_file, index=False)
                self.logger.info(f"Объединенные данные (нефть Brent и природный газ) сохранены в {output_file}")
                return True # Успех, если файл сохранен
            except IOError as e:
                self.logger.error(f"Ошибка записи объединенного файла {output_file}: {e}")
                return False # Ошибка записи файла
        else:
            # Нет данных для сохранения
            # Возвращаем True, если обе загрузки были "успешными" (не None), даже если вернули пустые DataFrame
            # Возвращаем False, если хотя бы одна загрузка вернула None (критическая ошибка)
            self.logger.warning(f"Нет итоговых данных для сохранения в файл в диапазоне {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}.")
            return brent_fetched_successfully and natural_gas_fetched_successfully 