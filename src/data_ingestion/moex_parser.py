import pandas as pd
import requests
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import Dict, Optional
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
                return config.get('companies', {})
        except Exception as e:
            raise Exception(f"Ошибка при загрузке конфигурационного файла: {str(e)}")
    
    def _setup_logging(self) -> None:
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_security_data(self, security: str, years: int) -> Optional[pd.DataFrame]:
        """
        Загрузка данных для конкретной ценной бумаги
        
        Parameters:
        -----------
        security : str
            Тикер ценной бумаги
        years : int
            Количество лет истории для загрузки
        
        Returns:
        --------
        Optional[pd.DataFrame]
            DataFrame с историческими данными или None в случае ошибки
        """
        try:
            # Формируем даты начала и конца периода
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=365 * years)
            
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'till': end_date.strftime('%Y-%m-%d'),
                'start': 0
            }
            
            all_data = []
            
            while True:
                url = f"{self.base_url}/{security}.json"
                r = requests.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                
                history_data = data.get('history', {}).get('data', [])
                if not history_data:
                    break
                
                all_data.extend(history_data)
                
                cursor = data.get('history.cursor', {}).get('data', [])
                if not cursor:
                    break
                
                total_rows = cursor[0][1]
                start = cursor[0][0] + cursor[0][2]
                
                if start >= total_rows:
                    break
                    
                params['start'] = start
            
            if not all_data:
                return None
                
            columns = data.get('history', {}).get('columns', [])
            df = pd.DataFrame(all_data, columns=columns)
            
            # Фильтруем только по основному режиму торгов TQBR
            df = df[df['BOARDID'] == 'TQBR']
            
            # Оставляем только нужные колонки
            needed_columns = ['TRADEDATE', 'SECID', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'WAPRICE']
            df = df[needed_columns]
            
            # Преобразуем даты и сортируем
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
            df = df.sort_values('TRADEDATE').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных для {security}: {str(e)}")
            return None
    
    def download_all(self, output_dir: str, years: int, delay: float = 1.0) -> None:
        """
        Загрузка данных для всех компаний
        
        Parameters:
        -----------
        output_dir : str
            Директория для сохранения файлов
        years : int
            Количество лет истории
        delay : float
            Задержка между запросами в секундах
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        
        for ticker, name in tqdm(self.companies.items(), desc="Загрузка данных"):
            logging.info(f"Загрузка данных для {ticker} ({name})")
            
            df = self.load_security_data(ticker, years)
            if df is not None:
                file_path = output_path / f"{ticker}_data.csv"
                df.to_csv(file_path, index=False)
                status = "Успешно"
                rows = len(df)
            else:
                status = "Ошибка"
                rows = 0
                
            results.append({
                'ticker': ticker,
                'name': name,
                'status': status,
                'rows': rows
            })
            
            time.sleep(delay)
        
        # Сохраняем отчет о загрузке
        pd.DataFrame(results).to_csv(output_path / 'download_report.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Загрузка исторических данных с MOEX')
    parser.add_argument('--config', type=str, default='configs/companies_config.json',
                      help='Путь к конфигурационному файлу')
    parser.add_argument('--output', type=str, default='data/moex',
                      help='Директория для сохранения данных')
    parser.add_argument('--years', type=int, default=5,
                      help='Количество лет истории')
    parser.add_argument('--delay', type=float, default=1.0,
                      help='Задержка между запросами (сек)')
    
    args = parser.parse_args()
    
    loader = MoexLoader(args.config)
    loader.download_all(args.output, args.years, args.delay)


if __name__ == "__main__":
    main()