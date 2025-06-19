# scripts/10_generate_predictions.py

import pandas as pd
import numpy as np
import os
import glob
from lightgbm import LGBMClassifier, LGBMRegressor
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Определение констант
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'features_final')

# Используем расширенный набор признаков, как в ноутбуке
# --- Компоненты признаков ---
price_volume_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'WAPRICE']
ma_cols = ['SMA_5', 'EMA_5', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_200', 'EMA_200']
oscillator_cols = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
    'STOCH_K', 'STOCH_D', 'ATR', 'VWAP', 'OBV', 'OBV_MA', 'Williams_%R', 'Momentum',
    'Plus_DI', 'Minus_DI', 'ADX', 'MFI', 'PVO', 'PVO_Signal', 'PVO_Hist', 'Chaikin_AD',
    'Chaikin_Oscillator', 'CCI', 'EMV', 'A/D_Line', 'Bull_Power', 'Bear_Power', 'TEMA'
]
fundamental_cols = [
    'Assets_q', 'Assets_y', 'CAPEX_q', 'CAPEX_y', 'Cash_q', 'Cash_y', 'Debt_q', 'Debt_y',
    'DividendsPaid_q', 'DividendsPaid_y', 'EBITDA_q', 'EBITDA_y', 'Equity_q', 'Equity_y',
    'NetDebt_q', 'NetDebt_y', 'NetProfit_q', 'NetProfit_y', 'OperatingCashFlow_q', 'OperatingCashFlow_y',
    'OperatingExpenses_q', 'OperatingExpenses_y', 'OperatingProfit_q', 'OperatingProfit_y',
    'Revenue_q', 'Revenue_y'
]
macro_index_cols = [
    'BRENT_CLOSE', 'NATURAL_GAS_CLOSE', 'KEY_RATE', 'CPI', 'USD_RUB', 'EUR_RUB', 'CNY_RUB', 'KZT_RUB', 'TRY_RUB',
    'MRBC', 'RTSI', 'MCXSM', 'IMOEX', 'MOEXBC', 'MOEXBMI', 'MOEXCN', 'MOEXIT',
    'MOEXRE', 'MOEXEU', 'MOEXFN', 'MOEXINN', 'MOEXMM',
    'MOEXOG', 'MOEXTL', 'MOEXTN', 'MOEXCH', 'GOLD', 'SILVER'
]
relative_coeff_cols = [
    'ROE_y', 'ROA_y', 'EBITDA_Margin_y', 'NetProfit_Margin_y', 'Debt_Equity_q', 'Debt_Equity_y',
    'NetDebt_EBITDA_y_q', 'NetDebt_EBITDA_y_y', 'EPS_y', 'BVPS_q', 'BVPS_y', 'SPS_y',
    'PE_y', 'PB_q', 'PB_y', 'PS_y', 'EV_EBITDA_y'
]
X_media_features = [
    'score_blog', 'score_blog_roll_avg_15', 'score_blog_roll_avg_50',
    'Index_MOEX_blog_score', 'Avg_Other_Indices_blog_score',
    'Avg_Other_Indices_blog_score_roll_avg_15', 'Avg_Other_Indices_blog_score_roll_avg_50',
    'score_news', 'score_news_roll_avg_15', 'score_news_roll_avg_50',
    'Index_MOEX_news_score', 'Avg_Other_Indices_news_score',
    'Avg_Other_Indices_news_score_roll_avg_15', 'Avg_Other_Indices_news_score_roll_avg_50'
]

# --- Основные наборы признаков ---
X_base_features = price_volume_cols + ma_cols + oscillator_cols + fundamental_cols + macro_index_cols + relative_coeff_cols
X_extended_features = X_base_features + X_media_features

# --- Целевые переменные ---
Y_CLF_BINARY_TARGETS = ['target_1d_binary', 'target_3d_binary', 'target_7d_binary', 'target_30d_binary', 'target_180d_binary', 'target_365d_binary']
Y_REG_TARGETS = ['target_1d', 'target_3d', 'target_7d', 'target_30d', 'target_180d', 'target_365d']

# Лучшие параметры из Optuna в ноутбуке для LGBMClassifier
BEST_LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 900,
    'learning_rate': 0.03634446643805034,
    'num_leaves': 60,
    'max_depth': 3,
    'min_child_samples': 20,
    'subsample': 0.6,
    'colsample_bytree': 0.9,
    'reg_alpha': 1.9152584521132006e-06,
    'reg_lambda': 1.5666922059058115e-06
}

# Параметры для регрессора (используем базовые, как в ноутбуке)
BEST_LGBM_REG_PARAMS = {
    'objective': 'regression_l1', # MAE
    'metric': 'mae',
    'verbosity': -1,
    'random_state': 42,
    'n_jobs': -1
}

def generate_predictions():
    """
    Обучает модели LightGBMClassifier и LGBMRegressor на объединенных данных 
    и делает прогнозы для каждого тикера, обновляя файлы в папке data/features_final.
    """
    try:
        print("1. Загрузка и объединение данных...")
        all_files = glob.glob(os.path.join(DATA_PATH, "*_final.csv"))
        if not all_files:
            print(f"В папке {DATA_PATH} не найдено файлов. Завершение работы.")
            return

        df_list = [pd.read_csv(filename, low_memory=False) for filename in all_files]
        all_stocks_df = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"Объединено {len(df_list)} файлов. Размер датасета: {all_stocks_df.shape}")

        # Убедимся, что все признаки существуют в датафрейме
        available_features = [f for f in X_extended_features if f in all_stocks_df.columns]
        print(f"Используется {len(available_features)} из {len(X_extended_features)} признаков.")
        
        # 2. Обучение моделей
        print("\n2. Обучение моделей...")
        
        trained_clf_models = {}
        print("--- Обучение моделей LightGBMClassifier ---")
        for target_col in Y_CLF_BINARY_TARGETS:
            print(f"--- Обучение для {target_col} ---")
            if target_col not in all_stocks_df.columns:
                print(f"Целевая переменная {target_col} отсутствует. Пропуск.")
                continue

            # Подготовка данных для обучения
            df_train_clf = all_stocks_df.dropna(subset=[target_col])
            X_train_clf = df_train_clf[available_features]
            y_train_clf = df_train_clf[target_col]

            if len(y_train_clf.unique()) < 2:
                print(f"В таргете {target_col} только один класс ({y_train_clf.unique()}) после удаления NaN. Модель не будет обучена.")
                continue
            
            # Обучение модели
            model = LGBMClassifier(**BEST_LGBM_PARAMS)
            model.fit(X_train_clf, y_train_clf)
            trained_clf_models[target_col] = model
            print(f"Модель для {target_col} обучена. Количество строк для обучения: {len(X_train_clf)}")

        trained_reg_models = {}
        print("\n--- Обучение моделей LGBMRegressor ---")
        for target_col in Y_REG_TARGETS:
            print(f"--- Обучение для {target_col} ---")
            if target_col not in all_stocks_df.columns:
                print(f"Целевая переменная {target_col} отсутствует. Пропуск.")
                continue

            # Подготовка данных для обучения
            df_train_reg = all_stocks_df.dropna(subset=[target_col])
            X_train_reg = df_train_reg[available_features]
            y_train_reg = df_train_reg[target_col]
            
            # Обучение модели
            model = LGBMRegressor(**BEST_LGBM_REG_PARAMS)
            model.fit(X_train_reg, y_train_reg)
            trained_reg_models[target_col] = model
            print(f"Модель для {target_col} обучена. Количество строк для обучения: {len(X_train_reg)}")


        if not trained_clf_models and not trained_reg_models:
            print("Ни одной модели не было обучено. Завершение работы.")
            return
            
        # 3. Создание и сохранение прогнозов
        print("\n3. Создание и сохранение прогнозов для каждого тикера...")
        for file_path in tqdm(all_files, desc="Обновление файлов"):
            try:
                ticker_df = pd.read_csv(file_path, low_memory=False)
                
                # X_predict должен иметь те же колонки, что и X_train (available_features)
                X_predict_base = ticker_df.copy()
                # Добавляем недостающие колонки (если есть) и заполняем их NaN
                for col in available_features:
                    if col not in X_predict_base.columns:
                        X_predict_base[col] = np.nan
                
                # Выбираем признаки в правильном порядке
                X_predict = X_predict_base[available_features]

                # Делаем прогнозы для классификаторов
                for target_col, model in trained_clf_models.items():
                    pred_col = f'{target_col}_pred'
                    predictions = model.predict(X_predict)
                    ticker_df[pred_col] = predictions
                
                # Делаем прогнозы для регрессоров
                for target_col, model in trained_reg_models.items():
                    pred_col = f'{target_col}_pred'
                    predictions = model.predict(X_predict)
                    ticker_df[pred_col] = predictions

                # Сохраняем обновленный файл
                ticker_df.to_csv(file_path, index=False)
            except Exception as e:
                print(f"Ошибка при обработке файла {file_path}: {e}")

        print("\nПроцесс создания прогнозов завершен.")
    
    except Exception as e:
        print(f"Произошла глобальная ошибка: {e}")


if __name__ == "__main__":
    generate_predictions()