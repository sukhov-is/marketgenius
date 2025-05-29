import requests
import pandas as pd
from datetime import date
import logging
from pathlib import Path
from typing import Optional, Dict, List
import xml.etree.ElementTree as ET

class UsdRubLoader:
    """Класс для загрузки исторических данных курсов нескольких валют (USD/RUB, EUR/RUB) с сайта ЦБ РФ.
    
    Важно: ЦБ РФ изменяет номинал валюты когда курс становится слишком большим (обычно > 100 руб).
    Например:
    - Если 1 CNY стоит больше 100 руб, ЦБ может публиковать курс за 10 CNY
    - Если 1 KZT стоит меньше 1 руб, ЦБ публикует курс за 100 KZT
    
    Этот класс автоматически нормализует все курсы к 1 единице валюты.
    """

    # Словарь с информацией о валютах: {Имя_колонки: Код_ЦБ_РФ}
    CURRENCIES_INFO: Dict[str, str] = {
        'USD_RUB': 'R01235', # Код валюты USD в ЦБ РФ
        'EUR_RUB': 'R01239', # Код валюты EUR в ЦБ РФ
        'CNY_RUB': 'R01375', # Код валюты CNY (Китайский юань) в ЦБ РФ
        'KZT_RUB': 'R01335', # Код валюты KZT (Казахстанский тенге) в ЦБ РФ
        'TRY_RUB': 'R01700J' # Код валюты TRY (Турецкая лира) в ЦБ РФ
    }

    def __init__(self):
        """Инициализация загрузчика."""
        self.base_url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Настройка логирования."""
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)

    def _parse_xml_with_nominal(self, xml_content: bytes) -> pd.DataFrame:
        """Парсит XML ответ ЦБ РФ, извлекая данные с учетом номинала.
        
        Returns:
        --------
        DataFrame с колонками: Date, Value, Nominal
        """
        try:
            root = ET.fromstring(xml_content)
            records = []
            
            for record in root.findall('.//Record'):
                date_str = record.get('Date')
                nominal = int(record.get('Nominal', '1'))
                value = record.find('.//Value')
                
                if value is not None and value.text:
                    records.append({
                        'Date': date_str,
                        'Value': value.text,
                        'Nominal': nominal
                    })
            
            if records:
                df = pd.DataFrame(records)
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
                df['Value'] = pd.to_numeric(df['Value'].str.replace(',', '.'), errors='coerce')
                return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге XML: {e}")
        
        return pd.DataFrame()

    def _calculate_features(self, df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
        """Рассчитывает процентное изменение и скользящие средние для указанных колонок за разные периоды."""
        df_copy = df.copy()
        df_copy = df_copy.sort_values('DATE')
        
        periods = [1, 3, 7, 30, 180]

        for column_name in column_names:
            if column_name not in df_copy.columns:
                self.logger.warning(f"Колонка {column_name} отсутствует в DataFrame, расчет признаков для нее пропущен.")
                continue
            for period in periods:
                # Относительное изменение (в процентах)
                df_copy[f'{column_name}_pct_change_{period}d'] = df_copy[column_name].pct_change(periods=period) * 100
                # Скользящее среднее
                df_copy[f'{column_name}_sma_{period}d'] = df_copy[column_name].rolling(window=period, min_periods=1).mean()
        return df_copy

    def fetch_rates(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Загружает данные о курсах валют за указанный период.
        
        Все курсы автоматически нормализуются к 1 единице валюты.

        Parameters:
        -----------
        start_date : date
            Дата начала периода.
        end_date : date
            Дата окончания периода.

        Returns:
        --------
        Optional[pd.DataFrame]:
             DataFrame с колонкой 'DATE' и колонками для каждой валюты (курс за 1 единицу)
             или None в случае ошибки.
        """
        start_date_str = start_date.strftime('%d/%m/%Y')
        end_date_str = end_date.strftime('%d/%m/%Y')
        self.logger.info(f"Запрос курсов валют с {start_date_str} по {end_date_str}")

        all_currencies_df: Optional[pd.DataFrame] = None

        for currency_name, cbr_currency_code in self.CURRENCIES_INFO.items():
            self.logger.info(f"Запрос курса {currency_name} (код: {cbr_currency_code}).")
            params = {
                'date_req1': start_date_str,
                'date_req2': end_date_str,
                'VAL_NM_RQ': cbr_currency_code
            }

            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()

                if not response.content or 'text/html' in response.headers.get('Content-Type', ''):
                    self.logger.warning(f"Получен пустой или HTML ответ от ЦБ РФ для {currency_name}")
                    current_currency_df = pd.DataFrame({'DATE': pd.to_datetime([])})
                else:
                    # Парсим XML с учетом номинала
                    df_parsed = self._parse_xml_with_nominal(response.content)
                    
                    if df_parsed.empty:
                        self.logger.warning(f"Нет данных для {currency_name}")
                        current_currency_df = pd.DataFrame({'DATE': pd.to_datetime([])})
                    else:
                        current_currency_df = pd.DataFrame()
                        current_currency_df['DATE'] = df_parsed['Date']
                        
                        # ВСЕГДА нормализуем к 1 единице валюты
                        current_currency_df[currency_name] = df_parsed['Value'] / df_parsed['Nominal']
                        
                        # Логируем изменения номинала
                        nominal_changes = df_parsed.groupby('Nominal').size()
                        if len(nominal_changes) > 1:
                            self.logger.warning(f"Обнаружены изменения номинала для {currency_name}:")
                            for nominal, count in nominal_changes.items():
                                self.logger.warning(f"  Номинал {nominal}: {count} записей")
                        
                        # Удаляем некорректные значения
                        original_len = len(current_currency_df)
                        current_currency_df = current_currency_df.dropna(subset=[currency_name])
                        if len(current_currency_df) < original_len:
                            self.logger.warning(f"Удалено {original_len - len(current_currency_df)} некорректных значений для {currency_name}")
                
                if all_currencies_df is None:
                    all_currencies_df = current_currency_df
                else:
                    if not current_currency_df.empty or 'DATE' in current_currency_df.columns:
                        all_currencies_df = pd.merge(all_currencies_df, current_currency_df, on='DATE', how='outer')

            except pd.errors.XMLSyntaxError as xml_err:
                self.logger.error(f"Ошибка парсинга XML для {currency_name}: {xml_err}")
                return None
            except requests.exceptions.Timeout:
                self.logger.error(f"Таймаут при запросе курса {currency_name}")
                return None
            except requests.exceptions.RequestException as req_err:
                status_code = req_err.response.status_code if hasattr(req_err.response, 'status_code') else "N/A"
                self.logger.error(f"Ошибка сети (статус: {status_code}) для {currency_name}: {req_err}")
                return None
            except Exception as e:
                self.logger.exception(f"Непредвиденная ошибка для {currency_name}: {e}")
                return None
        
        if all_currencies_df is not None and not all_currencies_df.empty:
            all_currencies_df = all_currencies_df.sort_values('DATE').reset_index(drop=True)
            self.logger.info(f"Успешно загружены данные для валют: {', '.join(self.CURRENCIES_INFO.keys())}")
            return all_currencies_df
        elif all_currencies_df is not None:
            self.logger.warning("Не удалось загрузить данные ни для одной валюты")
            return pd.DataFrame()
        else:
            self.logger.error("Ошибка при загрузке данных курсов валют")
            return None

    def download_rates(self, output_file: str, start_date: date, end_date: date) -> bool:
        """Загружает курсы валют за период и сохраняет в CSV.
        
        Все курсы автоматически нормализуются к 1 единице валюты.

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
                self.logger.info(f"Данные курсов валют сохранены в {output_path}")
                self.logger.info(f"Все курсы нормализованы к 1 единице валюты")
                return True
            except IOError as e:
                self.logger.error(f"Ошибка записи файла {output_path}: {e}")
                return False
            except Exception as e:
                self.logger.error(f"Непредвиденная ошибка при сохранении в {output_path}: {e}")
                return False
        elif df_rates is None:
            self.logger.error("Ошибка при загрузке данных, файл не сохранен")
            return False
        else:
            self.logger.warning("Нет данных для сохранения")
            return True 