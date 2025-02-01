import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Optional, Union
from pathlib import Path
import time
from tqdm import tqdm  # для отображения прогресса
import logging
from datetime import datetime, timedelta

def load_moex_data(security='GAZP', years=7):
    """
    Загружает исторические данные с MOEX для указанной ценной бумаги.
    
    Parameters:
    -----------
    security : str
        Тикер ценной бумаги (например, 'GAZP' для Газпрома)
    years : int
        Количество лет истории для загрузки
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame с историческими данными, содержащий следующие колонки:
        - TRADEDATE: дата торгов
        - SECID: идентификатор ценной бумаги
        - OPEN: цена открытия
        - HIGH: максимальная цена
        - LOW: минимальная цена
        - CLOSE: цена закрытия
        - VOLUME: объем торгов в лотах
        - WAPRICE: средневзвешенная цена
    """
    # Формируем даты начала и конца периода
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365 * years)
    
    # Базовый URL ISS API для исторических данных
    base_url = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities"
    
    # Параметры запроса
    params = {
        'from': start_date.strftime('%Y-%m-%d'),
        'till': end_date.strftime('%Y-%m-%d'),
        'start': 0
    }
    
    # Список для накопления данных
    all_data = []
    
    while True:
        # Формируем URL для запроса
        url = f"{base_url}/{security}.json"
        
        # Делаем запрос к API
        r = requests.get(url, params=params)
        r.raise_for_status()  # Проверяем на ошибки
        data = r.json()
        
        # Получаем данные из секции 'history'
        history_data = data.get('history', {}).get('data', [])
        if not history_data:
            break  # Нет данных или достигнут конец
        
        all_data.extend(history_data)
        
        # Проверяем, есть ли еще данные
        cursor = data.get('history.cursor', {}).get('data', [])
        if not cursor:
            break
        
        # Получаем информацию о пагинации
        total_rows = cursor[0][1]
        start = cursor[0][0] + cursor[0][2]  # current_start + page_size
        
        if start >= total_rows:
            break
            
        params['start'] = start
    
    # Создаем DataFrame
    columns = data.get('history', {}).get('columns', [])
    df = pd.DataFrame(all_data, columns=columns)
    
    # Фильтруем только по основному режиму торгов TQBR
    df = df[df['BOARDID'] == 'TQBR']
    
    # Оставляем только нужные колонки
    needed_columns = [
        'TRADEDATE',
        'SECID',
        'OPEN',
        'HIGH',
        'LOW',
        'CLOSE',
        'VOLUME',
        'WAPRICE'
    ]
    df = df[needed_columns]
    
    # Преобразуем даты и сортируем
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    df = df.sort_values('TRADEDATE').reset_index(drop=True)
    
    # Проверяем на пропущенные значения
    if df.isnull().any().any():
        print(f"Warning: Found {df.isnull().sum().sum()} missing values in the data")
    
    return df

def add_technical_indicators(df: pd.DataFrame, 
                           rsi_period: int = 14,
                           macd_periods: tuple = (12, 26, 9),
                           bb_period: int = 20,
                           bb_std: int = 2,
                           ma_periods: List[int] = [5, 10, 20, 50, 200],
                           vwap_period: int = 20,
                           atr_period: int = 14,
                           stoch_period: int = 14,
                           stoch_smooth: int = 3,
                           obv_ma_period: int = 20) -> pd.DataFrame:
    """
    Рассчитывает технические индикаторы для данных MOEX.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с данными MOEX (должен содержать колонки: TRADEDATE, OPEN, HIGH, LOW, CLOSE, VOLUME, WAPRICE)
    rsi_period : int
        Период для расчета RSI
    macd_periods : tuple
        Периоды для расчета MACD (fast, slow, signal)
    bb_period : int
        Период для расчета полос Боллинджера
    bb_std : int
        Количество стандартных отклонений для полос Боллинджера
    ma_periods : List[int]
        Список периодов для расчета скользящих средних
    vwap_period : int
        Период для расчета VWAP
    atr_period : int
        Период для расчета ATR
    stoch_period : int
        Период для расчета Stochastic Oscillator
    stoch_smooth : int
        Период сглаживания для Stochastic Oscillator
    obv_ma_period : int
        Период для скользящей средней OBV
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame с добавленными техническими индикаторами
    """
    # Создаем копию DataFrame
    df = df.copy()
    
    # -- Трендовые индикаторы --
    
    # Скользящие средние
    for period in ma_periods:
        df[f'SMA_{period}'] = df['CLOSE'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['CLOSE'].ewm(span=period, adjust=False).mean()
    
    # MACD
    fast, slow, signal = macd_periods
    df['EMA_fast'] = df['CLOSE'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['CLOSE'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df = df.drop(['EMA_fast', 'EMA_slow'], axis=1)
    
    # Полосы Боллинджера
    df['BB_middle'] = df['CLOSE'].rolling(window=bb_period).mean()
    bb_std_val = df['CLOSE'].rolling(window=bb_period).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * bb_std_val)
    df['BB_lower'] = df['BB_middle'] - (bb_std * bb_std_val)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # -- Моментум индикаторы --
    
    # RSI
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    df['LOW_MIN'] = df['LOW'].rolling(window=stoch_period).min()
    df['HIGH_MAX'] = df['HIGH'].rolling(window=stoch_period).max()
    df['STOCH_K'] = 100 * ((df['CLOSE'] - df['LOW_MIN']) / (df['HIGH_MAX'] - df['LOW_MIN']))
    df['STOCH_D'] = df['STOCH_K'].rolling(window=stoch_smooth).mean()
    df = df.drop(['LOW_MIN', 'HIGH_MAX'], axis=1)
    
    # -- Волатильность --
    
    # ATR (Average True Range)
    df['TR'] = np.maximum(
        np.maximum(
            df['HIGH'] - df['LOW'],
            abs(df['HIGH'] - df['CLOSE'].shift())
        ),
        abs(df['LOW'] - df['CLOSE'].shift())
    )
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()
    df = df.drop(['TR'], axis=1)
    
    # -- Объемные индикаторы --
    
    # OBV (On Balance Volume)
    df['OBV'] = (np.sign(df['CLOSE'].diff()) * df['VOLUME']).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=obv_ma_period).mean()
    
    # VWAP (Volume Weighted Average Price)
    df['VWAP'] = (df['WAPRICE'] * df['VOLUME']).rolling(window=vwap_period).sum() / \
                 df['VOLUME'].rolling(window=vwap_period).sum()
                 
    # -- Дополнительные метрики --
    
    # Волатильность (стандартное отклонение доходности)
    df['Returns'] = df['CLOSE'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)  # Годовая волатильность
    
    # Момент цены
    df['Price_Momentum'] = df['CLOSE'] / df['CLOSE'].shift(20) - 1
    
    # Удаляем промежуточные колонки
    df = df.drop(['Returns'], axis=1)
    
    return df

def get_market_data(security: str = 'GAZP',
                   years: int = 5,
                   add_indicators: bool = True,
                   **indicator_params) -> pd.DataFrame:
    """
    Загружает данные с MOEX и добавляет технические индикаторы.
    
    Parameters:
    -----------
    security : str
        Тикер ценной бумаги
    years : int
        Количество лет истории
    add_indicators : bool
        Добавлять ли технические индикаторы
    indicator_params : dict
        Параметры для расчета индикаторов
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame с данными и индикаторами
    """
    # Импортируем функцию загрузки данных
    # from moex_parser import load_moex_data
    
    # Загружаем базовые данные
    df = load_moex_data(security=security, years=years)
    
    # Добавляем индикаторы при необходимости
    if add_indicators:
        df = add_technical_indicators(df, **indicator_params)
        
    return df

def get_moex_blue_chips() -> List[str]:
    """
    Получает список тикеров голубых фишек с MOEX.
    
    Returns:
    --------
    List[str]
        Список тикеров голубых фишек
    """
    # Список наиболее ликвидных акций MOEX
    blue_chips = [
        'SBER' #'GAZP', 'LKOH', 'YNDX', 'GMKN', 'ROSN', 'NVTK', 
        # 'PLZL', 'POLY', 'MGNT', 'MTSS', 'SNGS', 'TATN', 'TCSG', 
        # 'VTBR', 'ALRS', 'CHMF', 'FIVE', 'NLMK', 'PHOR'
    ]
    return blue_chips

def setup_logging(log_dir: str = 'logs') -> None:
    """
    Настраивает логирование процесса загрузки и обработки данных.
    
    Parameters:
    -----------
    log_dir : str
        Директория для сохранения лог-файлов
    """
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f'moex_data_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def process_blue_chips(output_dir: str = 'data/external/moex_data',
                      years: int = 5,
                      add_indicators: bool = True,
                      retry_attempts: int = 3,
                      delay_between_requests: float = 1.0,
                      **indicator_params) -> Dict[str, str]:
    """
    Загружает, обрабатывает и сохраняет данные по всем голубым фишкам.
    
    Parameters:
    -----------
    output_dir : str
        Директория для сохранения файлов
    years : int
        Количество лет истории для загрузки
    add_indicators : bool
        Добавлять ли технические индикаторы
    retry_attempts : int
        Количество попыток повторной загрузки при ошибке
    delay_between_requests : float
        Задержка между запросами к API в секундах
    indicator_params : dict
        Параметры для расчета индикаторов
        
    Returns:
    --------
    Dict[str, str]
        Словарь с результатами обработки {тикер: статус}
    """
    # Настраиваем логирование
    setup_logging()
    logging.info(f"Starting blue chips data processing for {years} years of history")
    
    # Создаем директорию для данных если её нет
    data_dir = Path(output_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Получаем список тикеров
    tickers = get_moex_blue_chips()
    logging.info(f"Found {len(tickers)} blue chip tickers")
    
    # Словарь для хранения результатов
    results = {}
    
    # Обрабатываем каждый тикер
    for ticker in tqdm(tickers, desc="Processing tickers"):
        for attempt in range(retry_attempts):
            try:
                logging.info(f"Processing {ticker} (attempt {attempt + 1}/{retry_attempts})")
                
                # Загружаем и обрабатываем данные
                df = get_market_data(
                    security=ticker,
                    years=years,
                    add_indicators=add_indicators,
                    **indicator_params
                )
                
                # Формируем имя файла
                file_path = data_dir / f"{ticker}_data.csv"
                
                # Сохраняем данные
                df.to_csv(file_path, index=False)
                
                # Добавляем информацию о размере данных
                results[ticker] = f"Success ({len(df)} rows)"
                logging.info(f"Successfully processed {ticker}: {len(df)} rows saved to {file_path}")
                
                # Делаем паузу между запросами
                time.sleep(delay_between_requests)
                break
                
            except Exception as e:
                logging.error(f"Error processing {ticker} (attempt {attempt + 1}): {str(e)}")
                if attempt == retry_attempts - 1:
                    results[ticker] = f"Failed: {str(e)}"
                time.sleep(delay_between_requests)
    
    # Сохраняем сводную информацию
    summary_df = pd.DataFrame(
        [(ticker, status) for ticker, status in results.items()],
        columns=['Ticker', 'Status']
    )
    summary_df.to_csv(data_dir / 'processing_summary.csv', index=False)
    
    logging.info("Processing completed")
    return results

if __name__ == "__main__":
    # Параметры для расчета индикаторов
    indicator_params = {
        'rsi_period': 14,
        'macd_periods': (12, 26, 9),
        'bb_period': 20,
        'ma_periods': [5, 10, 20, 50, 200],
        'vwap_period': 20
    }
    
    # Запускаем обработку
    results = process_blue_chips(
        output_dir='data/external/moex_data',
        years=10,
        add_indicators=True,
        retry_attempts=3,
        delay_between_requests=1.0,
        **indicator_params
    )
    
    # Выводим результаты
    print("\nProcessing Results:")
    for ticker, status in results.items():
        print(f"{ticker}: {status}")