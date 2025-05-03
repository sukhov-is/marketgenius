import json
import requests
import os
from time import sleep
from pathlib import Path
import logging
from typing import Dict, Optional, List
from tqdm import tqdm # Добавим tqdm для прогресс-бара
import pandas as pd # Импорт pandas нужен для возвращаемого типа и создания отчета

class FinancialReportLoader:
    """Класс для загрузки финансовых отчетов (квартальных и годовых) со smart-lab.ru"""

    def __init__(self, config_path: str):
        """
        Инициализация загрузчика.

        Parameters:
        -----------
        config_path : str
            Путь к конфигурационному файлу JSON со списком компаний.
            Ожидается словарь в ключе 'companies'.
        """
        self.companies = self._load_companies(config_path)
        self.base_url_template = "https://smart-lab.ru/q/{ticker}/f/{report_type}/MSFO/download/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            # Можно добавить другие заголовки при необходимости
        }
        self._setup_logging()

    def _load_companies(self, config_path: str) -> Dict[str, str]:
        """Загрузка списка компаний из конфигурационного файла.
           Возвращает пустой словарь при ошибке или отсутствии ключа.
        """
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
        except FileNotFoundError:
            logging.error(f"Конфигурационный файл не найден: {config_path}")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Ошибка декодирования JSON в файле: {config_path}")
            return {}
        except Exception as e:
            logging.exception(f"Непредвиденная ошибка при загрузке конфига {config_path}: {e}")
            return {}

    def _setup_logging(self) -> None:
        """Настройка логирования.
           Использует стандартный logging, уровень INFO.
        """
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)

    def _download_single_report(self, ticker: str, report_type: str, filename: Path) -> bool:
        """Скачивает один отчет (квартальный 'q' или годовой 'y').

        Returns:
        --------
        bool: True в случае успеха, False в случае ошибки.
        """
        url = self.base_url_template.format(ticker=ticker, report_type=report_type)
        report_type_name = "квартальный" if report_type == 'q' else "годовой"

        try:
            self.logger.debug(f"Запрос {report_type_name} отчета для {ticker} по URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=30) # Добавим таймаут
            response.raise_for_status() # Проверка на HTTP ошибки

            # Проверка, что скачался не HTML (признак ошибки или отсутствия файла на smart-lab)
            if 'text/html' in response.headers.get('Content-Type', ''):
                self.logger.warning(f"Получен HTML вместо CSV для {ticker} ({report_type_name}). Вероятно, отчет отсутствует.")
                return False

            # Создаем родительские директории перед записью файла
            filename.parent.mkdir(parents=True, exist_ok=True)

            with open(filename, 'wb') as f:
                f.write(response.content)
            self.logger.info(f"Успешно сохранен {report_type_name} отчет для {ticker} в {filename}")
            return True
        except requests.exceptions.Timeout:
             self.logger.error(f"Таймаут при скачивании {report_type_name} отчета для {ticker} с {url}")
             return False
        except requests.exceptions.RequestException as e:
            # Логируем код статуса, если он есть
            status_code = e.response.status_code if e.response is not None else "N/A"
            self.logger.error(f"Ошибка сети (статус: {status_code}) при скачивании {report_type_name} отчета для {ticker}: {e}")
            return False
        except IOError as e:
            self.logger.error(f"Ошибка записи файла {filename}: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Непредвиденная ошибка при скачивании {report_type_name} отчета для {ticker}: {e}")
            return False

    def download_reports(
        self,
        output_dir: str,
        tickers_list: Optional[List[str]] = None,
        delay_q: float = 1.0, # Отдельные задержки
        delay_y: float = 2.0
        ) -> pd.DataFrame:
        """
        Скачивает квартальные и годовые отчеты для указанных тикеров.

        Parameters:
        -----------
        output_dir : str
            Корневая директория для сохранения отчетов (внутри будут созданы quarterly/ и yearly/).
        tickers_list : Optional[List[str]], optional
            Список тикеров для загрузки. Если None, используются все из конфига.
            По умолчанию None.
        delay_q : float, optional
            Задержка после скачивания квартального отчета (сек). По умолчанию 1.0.
        delay_y : float, optional
            Задержка после скачивания годового отчета (сек). По умолчанию 2.0.

        Returns:
        --------
        pd.DataFrame:
             DataFrame с отчетом о результатах скачивания.
        """
        base_output_path = Path(output_dir)
        quarterly_dir = base_output_path / 'quarterly'
        yearly_dir = base_output_path / 'yearly'

        if not self.companies:
            self.logger.error("Список компаний пуст. Загрузка отменена.")
            return pd.DataFrame() # Возвращаем пустой DataFrame

        if tickers_list is None:
            tickers_to_process = list(self.companies.keys())
        else:
            # Фильтруем tickers_list, оставляя только те, что есть в конфиге
            tickers_to_process = [t for t in tickers_list if t in self.companies]
            if len(tickers_to_process) != len(tickers_list):
                missing = set(tickers_list) - set(self.companies.keys())
                self.logger.warning(f"Следующие тикеры из списка не найдены в конфиге и будут пропущены: {missing}")

        self.logger.info(f"Начало загрузки фин. отчетов для {len(tickers_to_process)} тикеров...")
        results = []

        for ticker in tqdm(tickers_to_process, desc="Скачивание фин. отчетов"):
            company_name = self.companies.get(ticker, "N/A")
            self.logger.info(f"Обработка тикера: {ticker} ({company_name})")

            # Скачиваем квартальный отчет
            q_filename = quarterly_dir / f"{ticker}_quarterly.csv"
            q_success = self._download_single_report(ticker, 'q', q_filename)
            results.append({
                'ticker': ticker,
                'name': company_name,
                'report_type': 'quarterly',
                'status': 'Успешно' if q_success else 'Ошибка',
                'filename': str(q_filename) if q_success else None
            })
            sleep(delay_q) # Пауза

            # Скачиваем годовой отчет
            y_filename = yearly_dir / f"{ticker}_yearly.csv"
            y_success = self._download_single_report(ticker, 'y', y_filename)
            results.append({
                'ticker': ticker,
                'name': company_name,
                'report_type': 'yearly',
                'status': 'Успешно' if y_success else 'Ошибка',
                'filename': str(y_filename) if y_success else None
            })
            sleep(delay_y) # Пауза

        self.logger.info("Загрузка фин. отчетов завершена!")
        report_df = pd.DataFrame(results)

        # Сохраняем отчет
        report_path = base_output_path / 'fin_reports_download_report.csv'
        try:
            report_df.to_csv(report_path, index=False)
            self.logger.info(f"Отчет о загрузке фин. отчетов сохранен в {report_path}")
        except IOError as io_e:
            self.logger.error(f"Не удалось сохранить отчет о загрузке в {report_path}: {io_e}")

        return report_df