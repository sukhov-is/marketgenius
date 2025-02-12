import pandas as pd
import numpy as np
from typing import List, Optional, Union
import logging
from pathlib import Path


class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""
    
    def __init__(self):
        """Инициализация калькулятора технических индикаторов"""
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def calculate_moving_averages(self, df: pd.DataFrame, 
                                periods: List[int] = [5, 10, 20, 50, 200],
                                price_column: str = 'CLOSE') -> pd.DataFrame:
        """
        Расчет простых (SMA) и экспоненциальных (EMA) скользящих средних
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными
        periods : List[int]
            Список периодов для расчета
        price_column : str
            Название колонки с ценами
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленными индикаторами
        """
        df = df.copy()
        
        for period in periods:
            df[f'SMA_{period}'] = df[price_column].rolling(window=period).mean()
            df[f'EMA_{period}'] = df[price_column].ewm(span=period, adjust=False).mean()
            
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, 
                     period: int = 14,
                     price_column: str = 'CLOSE') -> pd.DataFrame:
        """
        Расчет индикатора относительной силы (RSI)
        """
        df = df.copy()
        
        delta = df[price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_macd(self, df: pd.DataFrame,
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9,
                      price_column: str = 'CLOSE') -> pd.DataFrame:
        """
        Расчет индикатора MACD
        """
        df = df.copy()
        
        # Расчет быстрой и медленной EMA
        fast_ema = df[price_column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[price_column].ewm(span=slow_period, adjust=False).mean()
        
        # Расчет MACD и сигнальной линии
        df['MACD'] = fast_ema - slow_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame,
                                period: int = 20,
                                std_dev: float = 2.0,
                                price_column: str = 'CLOSE') -> pd.DataFrame:
        """
        Расчет полос Боллинджера
        """
        df = df.copy()
        
        # Расчет средней линии и стандартного отклонения
        df['BB_Middle'] = df[price_column].rolling(window=period).mean()
        bb_std = df[price_column].rolling(window=period).std()
        
        # Расчет верхней и нижней полос
        df['BB_Upper'] = df['BB_Middle'] + (std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (std_dev * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        return df
    
    def calculate_stochastic(self, df: pd.DataFrame,
                           period: int = 14,
                           smooth_k: int = 3,
                           smooth_d: int = 3) -> pd.DataFrame:
        """
        Расчет стохастического осциллятора
        """
        df = df.copy()
        
        # Расчет минимумов и максимумов за период
        low_min = df['LOW'].rolling(window=period).min()
        high_max = df['HIGH'].rolling(window=period).max()
        
        # Расчет %K
        df['STOCH_K'] = 100 * ((df['CLOSE'] - low_min) / (high_max - low_min))
        
        # Сглаживание %K и расчет %D
        df['STOCH_K'] = df['STOCH_K'].rolling(window=smooth_k).mean()
        df['STOCH_D'] = df['STOCH_K'].rolling(window=smooth_d).mean()
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Расчет индикатора ATR (Average True Range)
        """
        df = df.copy()
        
        # Расчет True Range
        df['TR'] = np.maximum(
            np.maximum(
                df['HIGH'] - df['LOW'],
                abs(df['HIGH'] - df['CLOSE'].shift())
            ),
            abs(df['LOW'] - df['CLOSE'].shift())
        )
        
        # Расчет ATR
        df['ATR'] = df['TR'].rolling(window=period).mean()
        df.drop('TR', axis=1, inplace=True)
        
        return df
    
    def calculate_volume_indicators(self, df: pd.DataFrame,
                                 vwap_period: int = 20,
                                 obv_ma_period: int = 20) -> pd.DataFrame:
        """
        Расчет индикаторов объема (VWAP и OBV)
        """
        df = df.copy()
        
        # Расчет VWAP
        df['VWAP'] = (df['WAPRICE'] * df['VOLUME']).rolling(window=vwap_period).sum() / \
                     df['VOLUME'].rolling(window=vwap_period).sum()
        
        # Расчет OBV и его скользящей средней
        df['OBV'] = (np.sign(df['CLOSE'].diff()) * df['VOLUME']).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(window=obv_ma_period).mean()
        
        return df
    
    def calculate_all_indicators(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Расчет всех технических индикаторов
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными
        **kwargs : dict
            Параметры для расчета индикаторов
            
        Returns:
        --------
        pd.DataFrame
            DataFrame со всеми рассчитанными индикаторами
        """
        try:
            df = df.copy()
            
            # Получаем параметры или используем значения по умолчанию
            ma_periods = kwargs.get('ma_periods', [5, 10, 20, 50, 200])
            rsi_period = kwargs.get('rsi_period', 14)
            macd_periods = kwargs.get('macd_periods', (12, 26, 9))
            bb_period = kwargs.get('bb_period', 20)
            bb_std = kwargs.get('bb_std', 2.0)
            stoch_period = kwargs.get('stoch_period', 14)
            stoch_smooth = kwargs.get('stoch_smooth', 3)
            atr_period = kwargs.get('atr_period', 14)
            vwap_period = kwargs.get('vwap_period', 20)
            obv_ma_period = kwargs.get('obv_ma_period', 20)
            
            # Расчет всех индикаторов
            df = self.calculate_moving_averages(df, periods=ma_periods)
            df = self.calculate_rsi(df, period=rsi_period)
            df = self.calculate_macd(df, fast_period=macd_periods[0],
                                   slow_period=macd_periods[1],
                                   signal_period=macd_periods[2])
            df = self.calculate_bollinger_bands(df, period=bb_period, std_dev=bb_std)
            df = self.calculate_stochastic(df, period=stoch_period,
                                         smooth_k=stoch_smooth,
                                         smooth_d=stoch_smooth)
            df = self.calculate_atr(df, period=atr_period)
            df = self.calculate_volume_indicators(df, vwap_period=vwap_period,
                                               obv_ma_period=obv_ma_period)
            
            return df
            
        except Exception as e:
            logging.error(f"Ошибка при расчете индикаторов: {str(e)}")
            raise
    
    def process_files(self, input_dir: str, output_dir: str, **kwargs) -> None:
        """
        Обработка всех файлов в директории
        
        Parameters:
        -----------
        input_dir : str
            Директория с исходными файлами
        output_dir : str
            Директория для сохранения обработанных файлов
        **kwargs : dict
            Параметры для расчета индикаторов
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        
        # Обрабатываем все CSV файлы в директории
        for file_path in tqdm(list(input_path.glob('*_data.csv')), desc="Обработка файлов"):
            try:
                # Загружаем данные
                df = pd.read_csv(file_path)
                ticker = file_path.stem.replace('_data', '')
                
                # Преобразуем даты
                df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
                
                # Рассчитываем индикаторы
                df_with_indicators = self.calculate_all_indicators(df, **kwargs)
                
                # Сохраняем результат
                output_file = output_path / f"{ticker}_processed.csv"
                df_with_indicators.to_csv(output_file, index=False)
                
                results.append({
                    'ticker': ticker,
                    'status': 'Успешно',
                    'rows': len(df_with_indicators)
                })
                
            except Exception as e:
                logging.error(f"Ошибка при обработке {file_path}: {str(e)}")
                results.append({
                    'ticker': ticker,
                    'status': f'Ошибка: {str(e)}',
                    'rows': 0
                })
        
        # Сохраняем отчет об обработке
        pd.DataFrame(results).to_csv(output_path / 'processing_report.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Расчет технических индикаторов')
    parser.add_argument('--input', type=str, default='data/moex',
                      help='Директория с исходными данными')
    parser.add_argument('--output', type=str, default='data/processed',
                      help='Директория для сохранения обработанных данных')
    
    args = parser.parse_args()
    
    # Создаем экземпляр класса и обрабатываем файлы
    calculator = TechnicalIndicators()
    calculator.process_files(args.input, args.output)


if __name__ == "__main__":
    main()