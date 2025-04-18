import requests
import pandas as pd
from datetime import date
import logging
from pathlib import Path
from typing import Optional

class KeyRateLoader:
    """Класс для загрузки истории ключевой ставки ЦБ РФ.

    Загружает данные с HTML-страницы ЦБ РФ.
    """

    def __init__(self):
        """Инициализация загрузчика."""
        # URL страницы с историей ключевой ставки
        self.base_url = "https://cbr.ru/hd_base/KeyRate/"
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

    def fetch_key_rate(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Загружает историю ключевой ставки за указанный период.

        Parameters:
        -----------
        start_date : date
            Дата начала периода.
        end_date : date
            Дата окончания периода.

        Returns:
        --------
        Optional[pd.DataFrame]:
             DataFrame с колонками ['DATE', 'KEY_RATE'] или None в случае ошибки.
        """
        start_date_str = start_date.strftime('%d.%m.%Y')
        end_date_str = end_date.strftime('%d.%m.%Y')
        url = f'{self.base_url}?UniDbQuery.Posted=True&UniDbQuery.From={start_date_str}&UniDbQuery.To={end_date_str}'
        self.logger.info(f"Запрос истории ключевой ставки с {start_date_str} по {end_date_str}")
        self.logger.debug(f"URL запроса: {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Проверка на случай, если страница не вернула данные или вернула ошибку в HTML
            if not response.text:
                 self.logger.error("Получен пустой ответ от страницы ключевой ставки ЦБ РФ.")
                 return None

            # Парсинг HTML таблиц
            tables = pd.read_html(response.text, decimal=',', thousands='\xa0')

            if not tables:
                self.logger.error("Не найдено таблиц на странице ключевой ставки ЦБ РФ.")
                return None

            # Обычно нужная таблица первая
            df = tables[0]
            self.logger.debug(f"Найдено {len(tables)} таблиц, используется первая. Колонки: {df.columns.tolist()}")

            # Проверяем и переименовываем колонки
            # Ожидаемые названия могут меняться, делаем более гибко
            if len(df.columns) < 2:
                self.logger.error(f"В найденной таблице меньше 2 колонок: {df.columns.tolist()}")
                return None

            # Предполагаем, что первая колонка - дата, вторая - ставка
            date_col_name = df.columns[0]
            rate_col_name = df.columns[1]
            self.logger.info(f"Используются колонки: '{date_col_name}' (дата), '{rate_col_name}' (ставка)")

            # Преобразование данных
            df['DATE'] = pd.to_datetime(df[date_col_name], format='%d.%m.%Y')
            df['KEY_RATE'] = pd.to_numeric(df[rate_col_name], errors='coerce')

            # Удаляем строки с ошибками конвертации
            original_len = len(df)
            df = df.dropna(subset=['DATE', 'KEY_RATE'])
            if len(df) < original_len:
                self.logger.warning(f"Удалено {original_len - len(df)} строк с некорректными значениями даты или ставки.")

            if df.empty:
                 self.logger.warning(f"Нет корректных данных после обработки таблицы ключевой ставки.")
                 # Возвращаем пустой DataFrame, а не None, т.к. запрос прошел успешно
                 return pd.DataFrame(columns=['DATE', 'KEY_RATE'])

            # Выбираем нужные колонки и сортируем
            df = df[['DATE', 'KEY_RATE']].sort_values('DATE').reset_index(drop=True)

            self.logger.info(f"Успешно загружено и обработано {len(df)} записей ключевой ставки.")
            return df

        except ImportError as imp_err:
            # Ошибка, если не установлен lxml или html5lib
            self.logger.error(f"Ошибка импорта для парсинга HTML (требуется lxml или html5lib): {imp_err}")
            return None
        except ValueError as val_err:
             # Ошибки при конвертации типов или парсинге дат
             self.logger.error(f"Ошибка значения при обработке данных ключевой ставки: {val_err}")
             return None
        except requests.exceptions.Timeout:
            self.logger.error(f"Таймаут при запросе страницы ключевой ставки ЦБ РФ.")
            return None
        except requests.exceptions.RequestException as req_err:
            status_code = req_err.response.status_code if req_err.response is not None else "N/A"
            self.logger.error(f"Ошибка сети (статус: {status_code}) при запросе ключевой ставки: {req_err}")
            return None
        except IndexError:
             self.logger.error("Ошибка индекса при доступе к таблицам или колонкам. Возможно, структура страницы изменилась.")
             return None
        except Exception as e:
            self.logger.exception(f"Непредвиденная ошибка при загрузке ключевой ставки: {e}")
            return None

    def download_key_rate(self, output_file: str, start_date: date, end_date: date) -> bool:
        """Загружает историю ключевой ставки за период и сохраняет в CSV.

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

        df_rate = self.fetch_key_rate(start_date, end_date)

        if df_rate is not None and not df_rate.empty:
            try:
                df_rate.to_csv(output_path, index=False)
                self.logger.info(f"Данные ключевой ставки сохранены в {output_path}")
                return True
            except IOError as e:
                self.logger.error(f"Ошибка записи файла {output_path}: {e}")
                return False
        elif df_rate is None:
             self.logger.error("Ошибка при загрузке данных ключевой ставки, файл не будет сохранен.")
             return False
        else: # df_rate is empty
             self.logger.warning("Нет данных ключевой ставки для сохранения (получен пустой DataFrame).")
             # Можно создать пустой файл, но лучше этого не делать.
             # Возвращаем True, т.к. запрос был успешным, просто данных нет.
             return True 