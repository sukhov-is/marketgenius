import pandas as pd
import numpy as np
from typing import List, Optional, Union
import logging
from pathlib import Path
import argparse
from tqdm import tqdm


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
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Расчет индикатора Вильямса %R
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными
        period : int
            Период для расчета
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        highest_high = df['HIGH'].rolling(window=period).max()
        lowest_low = df['LOW'].rolling(window=period).min()
        
        df['Williams_%R'] = -100 * (highest_high - df['CLOSE']) / (highest_high - lowest_low)
        
        return df
    
    def calculate_momentum(self, df: pd.DataFrame, period: int = 14, 
                         price_column: str = 'CLOSE') -> pd.DataFrame:
        """
        Расчет индикатора моментума
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными
        period : int
            Период для расчета
        price_column : str
            Название колонки с ценами
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        df['Momentum'] = df[price_column] - df[price_column].shift(period)
        
        return df
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Расчет индекса среднего направленного движения (ADX)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными
        period : int
            Период для расчета
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленными индикаторами ADX, +DI, -DI
        """
        df = df.copy()
        
        # Расчет истинного диапазона (TR)
        df['TR'] = np.maximum(
            np.maximum(
                df['HIGH'] - df['LOW'],
                abs(df['HIGH'] - df['CLOSE'].shift())
            ),
            abs(df['LOW'] - df['CLOSE'].shift())
        )
        
        # Направленные движения (DM)
        df['Plus_DM'] = np.where(
            (df['HIGH'] - df['HIGH'].shift() > df['LOW'].shift() - df['LOW']) & 
            (df['HIGH'] - df['HIGH'].shift() > 0),
            df['HIGH'] - df['HIGH'].shift(),
            0
        )
        
        df['Minus_DM'] = np.where(
            (df['LOW'].shift() - df['LOW'] > df['HIGH'] - df['HIGH'].shift()) & 
            (df['LOW'].shift() - df['LOW'] > 0),
            df['LOW'].shift() - df['LOW'],
            0
        )
        
        # Расчет экспоненциальных средних
        for col in ['TR', 'Plus_DM', 'Minus_DM']:
            df[f'{col}{period}'] = df[col].ewm(alpha=1/period, adjust=False).mean()
        
        # Расчет индексов направленного движения (DI)
        df['Plus_DI'] = 100 * df[f'Plus_DM{period}'] / df[f'TR{period}']
        df['Minus_DI'] = 100 * df[f'Minus_DM{period}'] / df[f'TR{period}']
        
        # Расчет направленного индекса (DX)
        df['DX'] = 100 * abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'])
        
        # Расчет ADX
        df['ADX'] = df['DX'].ewm(alpha=1/period, adjust=False).mean()
        
        # Удаление промежуточных колонок
        df.drop(['TR', 'Plus_DM', 'Minus_DM', f'TR{period}', 
                f'Plus_DM{period}', f'Minus_DM{period}', 'DX'], axis=1, inplace=True)
        
        return df
    
    def calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Расчет индекса потока денег (MFI)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными и объемами
        period : int
            Период для расчета
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        # Расчет типичной цены
        df['Typical_Price'] = (df['HIGH'] + df['LOW'] + df['CLOSE']) / 3
        
        # Расчет денежного потока
        df['Money_Flow'] = df['Typical_Price'] * df['VOLUME']
        
        # Определение положительного и отрицательного потоков
        df['Positive_Flow'] = np.where(df['Typical_Price'] > df['Typical_Price'].shift(), 
                                     df['Money_Flow'], 0)
        df['Negative_Flow'] = np.where(df['Typical_Price'] < df['Typical_Price'].shift(), 
                                     df['Money_Flow'], 0)
        
        # Расчет суммы положительных и отрицательных потоков за период
        df['Positive_Flow_Sum'] = df['Positive_Flow'].rolling(window=period).sum()
        df['Negative_Flow_Sum'] = df['Negative_Flow'].rolling(window=period).sum()
        
        # Расчет коэффициента денежного потока
        df['Money_Flow_Ratio'] = df['Positive_Flow_Sum'] / df['Negative_Flow_Sum']
        
        # Расчет MFI
        df['MFI'] = 100 - (100 / (1 + df['Money_Flow_Ratio']))
        
        # Удаление промежуточных колонок
        df.drop(['Typical_Price', 'Money_Flow', 'Positive_Flow', 'Negative_Flow', 
               'Positive_Flow_Sum', 'Negative_Flow_Sum', 'Money_Flow_Ratio'], 
               axis=1, inplace=True)
        
        return df
    
    def calculate_pvo(self, df: pd.DataFrame, fast_period: int = 12, 
                    slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Расчет процентного объёмного осциллятора (PVO)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с данными объема
        fast_period : int
            Период для быстрой EMA
        slow_period : int
            Период для медленной EMA
        signal_period : int
            Период для сигнальной линии
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        # Расчет EMA объема
        df['Volume_EMA_Fast'] = df['VOLUME'].ewm(span=fast_period, adjust=False).mean()
        df['Volume_EMA_Slow'] = df['VOLUME'].ewm(span=slow_period, adjust=False).mean()
        
        # Расчет PVO
        df['PVO'] = ((df['Volume_EMA_Fast'] - df['Volume_EMA_Slow']) / 
                    df['Volume_EMA_Slow']) * 100
        
        # Расчет сигнальной линии
        df['PVO_Signal'] = df['PVO'].ewm(span=signal_period, adjust=False).mean()
        
        # Расчет гистограммы
        df['PVO_Hist'] = df['PVO'] - df['PVO_Signal']
        
        # Удаление промежуточных колонок
        df.drop(['Volume_EMA_Fast', 'Volume_EMA_Slow'], axis=1, inplace=True)
        
        return df
    
    def calculate_chaikin_ad(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет индикатора Чайкина накопления/распределения (Chaikin A/D line)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными и объемами
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        # Расчет множителя денежного потока (Money Flow Multiplier)
        df['MFM'] = ((df['CLOSE'] - df['LOW']) - (df['HIGH'] - df['CLOSE'])) / (df['HIGH'] - df['LOW'])
        
        # Расчет объема денежного потока (Money Flow Volume)
        df['MFV'] = df['MFM'] * df['VOLUME']
        
        # Расчет линии A/D
        df['Chaikin_AD'] = df['MFV'].cumsum()
        
        # Удаление промежуточных колонок
        df.drop(['MFM', 'MFV'], axis=1, inplace=True)
        
        return df
    
    def calculate_chaikin_oscillator(self, df: pd.DataFrame, 
                                   fast_period: int = 3, 
                                   slow_period: int = 10) -> pd.DataFrame:
        """
        Расчет осциллятора Чайкина
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными и объемами
        fast_period : int
            Период для быстрой EMA
        slow_period : int
            Период для медленной EMA
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        # Сначала нужно рассчитать линию Чайкина A/D
        df = self.calculate_chaikin_ad(df)
        
        # Расчет быстрой и медленной EMA от линии A/D
        df['Chaikin_AD_Fast_EMA'] = df['Chaikin_AD'].ewm(span=fast_period, adjust=False).mean()
        df['Chaikin_AD_Slow_EMA'] = df['Chaikin_AD'].ewm(span=slow_period, adjust=False).mean()
        
        # Расчет осциллятора Чайкина
        df['Chaikin_Oscillator'] = df['Chaikin_AD_Fast_EMA'] - df['Chaikin_AD_Slow_EMA']
        
        # Удаление промежуточных колонок
        df.drop(['Chaikin_AD_Fast_EMA', 'Chaikin_AD_Slow_EMA'], axis=1, inplace=True)
        
        return df
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Расчет индекса товарного канала (CCI)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными
        period : int
            Период для расчета
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        # Расчет типичной цены
        df['Typical_Price'] = (df['HIGH'] + df['LOW'] + df['CLOSE']) / 3
        
        # Расчет скользящей средней типичной цены
        df['TP_SMA'] = df['Typical_Price'].rolling(window=period).mean()
        
        # Расчет среднего абсолютного отклонения
        df['TP_Deviation'] = abs(df['Typical_Price'] - df['TP_SMA'])
        df['TP_MAD'] = df['TP_Deviation'].rolling(window=period).mean()
        
        # Расчет CCI
        df['CCI'] = (df['Typical_Price'] - df['TP_SMA']) / (0.015 * df['TP_MAD'])
        
        # Удаление промежуточных колонок
        df.drop(['Typical_Price', 'TP_SMA', 'TP_Deviation', 'TP_MAD'], axis=1, inplace=True)
        
        return df
    
    def calculate_emv(self, df: pd.DataFrame, volume_divisor: int = 10000) -> pd.DataFrame:
        """
        Расчет индекса лёгкости движения (EMV)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными и объемами
        volume_divisor : int
            Делитель объема для нормализации
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        # Расчет средней цены для текущего и предыдущего периодов
        df['Midpoint'] = (df['HIGH'] + df['LOW']) / 2
        df['Midpoint_Prev'] = df['Midpoint'].shift(1)
        
        # Расчет изменения
        df['Change'] = df['Midpoint'] - df['Midpoint_Prev']
        
        # Расчет бокса-диапазона
        df['Box_Ratio'] = (df['VOLUME'] / volume_divisor) / (df['HIGH'] - df['LOW'])
        
        # Расчет однодневного EMV
        df['EMV_1'] = df['Change'] / df['Box_Ratio']
        
        # Расчет 14-дневного EMV
        df['EMV'] = df['EMV_1'].rolling(window=14).mean()
        
        # Удаление промежуточных колонок
        df.drop(['Midpoint', 'Midpoint_Prev', 'Change', 'Box_Ratio', 'EMV_1'], 
               axis=1, inplace=True)
        
        return df
    
    def calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет индикатора накопления/распределения (A/D)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными и объемами
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        # Расчет коэффициента распределения объема
        df['CLV'] = ((df['CLOSE'] - df['LOW']) - (df['HIGH'] - df['CLOSE'])) / (df['HIGH'] - df['LOW'])
        df['CLV'] = df['CLV'].replace([np.inf, -np.inf], 0)
        
        # Расчет потока объема
        df['Flow_Volume'] = df['CLV'] * df['VOLUME']
        
        # Кумулятивная сумма потока объема
        df['A/D_Line'] = df['Flow_Volume'].cumsum()
        
        # Удаление промежуточных колонок
        df.drop(['CLV', 'Flow_Volume'], axis=1, inplace=True)
        
        return df
    
    def calculate_bull_bear_power(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """
        Расчет индикаторов Силы быков и медведей
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными
        period : int
            Период для расчета EMA
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленными индикаторами
        """
        df = df.copy()
        
        # Расчет EMA
        df['EMA'] = df['CLOSE'].ewm(span=period, adjust=False).mean()
        
        # Расчет Силы быков и медведей
        df['Bull_Power'] = df['HIGH'] - df['EMA']
        df['Bear_Power'] = df['LOW'] - df['EMA']
        
        # Удаление промежуточной колонки
        df.drop(['EMA'], axis=1, inplace=True)
        
        return df
    
    def calculate_tema(self, df: pd.DataFrame, period: int = 20, 
                     price_column: str = 'CLOSE') -> pd.DataFrame:
        """
        Расчет тройной экспоненциальной скользящей средней (TEMA)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame с ценовыми данными
        period : int
            Период для расчета
        price_column : str
            Название колонки с ценами
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленным индикатором
        """
        df = df.copy()
        
        # Расчет первой EMA
        df['EMA1'] = df[price_column].ewm(span=period, adjust=False).mean()
        
        # Расчет EMA от EMA1
        df['EMA2'] = df['EMA1'].ewm(span=period, adjust=False).mean()
        
        # Расчет EMA от EMA2
        df['EMA3'] = df['EMA2'].ewm(span=period, adjust=False).mean()
        
        # Расчет TEMA
        df['TEMA'] = 3 * df['EMA1'] - 3 * df['EMA2'] + df['EMA3']
        
        # Удаление промежуточных колонок
        df.drop(['EMA1', 'EMA2', 'EMA3'], axis=1, inplace=True)
        
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
            williams_period = kwargs.get('williams_period', 14)
            momentum_period = kwargs.get('momentum_period', 14)
            adx_period = kwargs.get('adx_period', 14)
            mfi_period = kwargs.get('mfi_period', 14)
            pvo_periods = kwargs.get('pvo_periods', (12, 26, 9))
            cci_period = kwargs.get('cci_period', 20)
            tema_period = kwargs.get('tema_period', 20)
            bull_bear_period = kwargs.get('bull_bear_period', 13)
            
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
            df = self.calculate_williams_r(df, period=williams_period)
            df = self.calculate_momentum(df, period=momentum_period)
            df = self.calculate_adx(df, period=adx_period)
            df = self.calculate_mfi(df, period=mfi_period)
            df = self.calculate_pvo(df, fast_period=pvo_periods[0],
                                  slow_period=pvo_periods[1],
                                  signal_period=pvo_periods[2])
            df = self.calculate_chaikin_ad(df)
            df = self.calculate_chaikin_oscillator(df)
            df = self.calculate_cci(df, period=cci_period)
            df = self.calculate_emv(df)
            df = self.calculate_accumulation_distribution(df)
            df = self.calculate_bull_bear_power(df, period=bull_bear_period)
            df = self.calculate_tema(df, period=tema_period)
            
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
        for file_path in tqdm(list(input_path.glob('*_moex_data.csv')), desc="Обработка файлов"):
            try:
                # Загружаем данные
                df = pd.read_csv(file_path)
                ticker = file_path.stem.replace('_moex_data', '')
                
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
    parser.add_argument('--input', type=str, default='data/raw/moex_shares',
                      help='Директория с исходными данными')
    parser.add_argument('--output', type=str, default='data/processed/technical_indicators',
                      help='Директория для сохранения обработанных данных')
    
    args = parser.parse_args()
    
    # Создаем экземпляр класса и обрабатываем файлы
    calculator = TechnicalIndicators()
    calculator.process_files(args.input, args.output)


if __name__ == "__main__":
    main()