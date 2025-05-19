import pandas as pd
import numpy as np
import os
import glob
import re
import logging

# --- Конфигурация Логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Константы ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MERGED_FEATURES_DIR = os.path.join(PROJECT_ROOT, "data/processed/merged_features")
REFERENCE_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw/reference")
FINAL_FEATURES_DIR = os.path.join(PROJECT_ROOT, "data/processed/final_features")
ISSUE_SIZE_FILE = os.path.join(REFERENCE_DATA_DIR, "issue_sizes.csv")

# Список коэффициентов, которые мы пытаемся рассчитать
# (LTM аппроксимируется годовыми данными _y)
TARGET_RATIOS = [
    'ROE_y', 'ROA_y', 'EBITDA_Margin_y', 'NetProfit_Margin_y', # Profitability
    'Debt_Equity_q', 'Debt_Equity_y', 'NetDebt_EBITDA_y_q', 'NetDebt_EBITDA_y_y', # Leverage
    'EPS_y', 'BVPS_q', 'BVPS_y', 'SPS_y', # Per Share
    'PE_y', 'PB_q', 'PB_y', 'PS_y', 'EV_EBITDA_y' # Multiples
]

# --- Функции ---

def load_issue_sizes(filepath):
    """Загружает данные IssueSize из CSV в словарь {SECID: IssueSize}."""
    issue_sizes = {}
    try:
        df = pd.read_csv(filepath)
        # Используем последнее доступное значение для каждого тикера
        # Предполагаем, что файл содержит DATE, SECID, IssueSize
        # и отсортирован так, что последняя запись актуальна, или содержит только одну запись
        df = df.drop_duplicates(subset=['SECID'], keep='last')
        issue_sizes = pd.Series(df.IssueSize.values, index=df.SECID).to_dict()
        logging.info(f"Загружено {len(issue_sizes)} записей IssueSize из {filepath}")
    except FileNotFoundError:
        logging.error(f"Файл IssueSize не найден: {filepath}")
    except Exception as e:
        logging.error(f"Ошибка загрузки файла IssueSize {filepath}: {e}")
    return issue_sizes

def safe_divide(numerator, denominator):
    """Безопасное деление, обрабатывает 0 и NaN в знаменателе."""
    # Копируем, чтобы не изменять исходные ряды
    num = numerator.copy()
    den = denominator.copy()
    # Заменяем 0 и отрицательные значения в знаменателе на NaN, чтобы избежать деления на 0/отрицательное
    den[den <= 0] = np.nan 
    # Выполняем деление, результат будет NaN там, где den был NaN или num был NaN
    result = num / den
    return result

def calculate_ratios(df, issue_size):
    """Рассчитывает финансовые коэффициенты для DataFrame одного тикера."""
    calculated_ratios_set = set()
    
    # Копируем DataFrame, чтобы не изменять оригинал
    df_res = df.copy()

    # --- LTM Аппроксимация (используем годовые _y) ---
    # Используем _y напрямую как LTM прокси
    net_profit_ltm = df_res.get('NetProfit_y')
    revenue_ltm = df_res.get('Revenue_y')
    ebitda_ltm = df_res.get('EBITDA_y')

    # --- Рентабельность ---
    if net_profit_ltm is not None and 'Equity_y' in df_res:
        df_res['ROE_y'] = safe_divide(net_profit_ltm, df_res['Equity_y']) * 100 # В процентах
        if df_res['ROE_y'].notna().any(): calculated_ratios_set.add('ROE_y')
        
    if net_profit_ltm is not None and 'Assets_y' in df_res:
        df_res['ROA_y'] = safe_divide(net_profit_ltm, df_res['Assets_y']) * 100 # В процентах
        if df_res['ROA_y'].notna().any(): calculated_ratios_set.add('ROA_y')
        
    if ebitda_ltm is not None and revenue_ltm is not None:
        df_res['EBITDA_Margin_y'] = safe_divide(ebitda_ltm, revenue_ltm) * 100 # В процентах
        if df_res['EBITDA_Margin_y'].notna().any(): calculated_ratios_set.add('EBITDA_Margin_y')
        
    if net_profit_ltm is not None and revenue_ltm is not None:
        df_res['NetProfit_Margin_y'] = safe_divide(net_profit_ltm, revenue_ltm) * 100 # В процентах
        if df_res['NetProfit_Margin_y'].notna().any(): calculated_ratios_set.add('NetProfit_Margin_y')

    # --- Долговая нагрузка ---
    if 'Debt_q' in df_res and 'Equity_q' in df_res:
        df_res['Debt_Equity_q'] = safe_divide(df_res['Debt_q'], df_res['Equity_q'])
        if df_res['Debt_Equity_q'].notna().any(): calculated_ratios_set.add('Debt_Equity_q')

    if 'Debt_y' in df_res and 'Equity_y' in df_res:
        df_res['Debt_Equity_y'] = safe_divide(df_res['Debt_y'], df_res['Equity_y'])
        if df_res['Debt_Equity_y'].notna().any(): calculated_ratios_set.add('Debt_Equity_y')

    if 'NetDebt_q' in df_res and ebitda_ltm is not None:
        df_res['NetDebt_EBITDA_y_q'] = safe_divide(df_res['NetDebt_q'], ebitda_ltm)
        if df_res['NetDebt_EBITDA_y_q'].notna().any(): calculated_ratios_set.add('NetDebt_EBITDA_y_q')
        
    if 'NetDebt_y' in df_res and ebitda_ltm is not None:
        df_res['NetDebt_EBITDA_y_y'] = safe_divide(df_res['NetDebt_y'], ebitda_ltm)
        if df_res['NetDebt_EBITDA_y_y'].notna().any(): calculated_ratios_set.add('NetDebt_EBITDA_y_y')
        
    # --- Показатели на акцию (только если есть issue_size) ---
    if issue_size and issue_size > 0:
        if net_profit_ltm is not None:
            df_res['EPS_y'] = net_profit_ltm / issue_size * 1e6 # Приводим к рублям на акцию (т.к. показатели в млн)
            if df_res['EPS_y'].notna().any(): calculated_ratios_set.add('EPS_y')
            
        if 'Equity_q' in df_res:
            df_res['BVPS_q'] = df_res['Equity_q'] / issue_size * 1e6
            if df_res['BVPS_q'].notna().any(): calculated_ratios_set.add('BVPS_q')

        if 'Equity_y' in df_res:
            df_res['BVPS_y'] = df_res['Equity_y'] / issue_size * 1e6
            if df_res['BVPS_y'].notna().any(): calculated_ratios_set.add('BVPS_y')

        if revenue_ltm is not None:
            df_res['SPS_y'] = revenue_ltm / issue_size * 1e6
            if df_res['SPS_y'].notna().any(): calculated_ratios_set.add('SPS_y')
            
        # --- Мультипликаторы (только если есть цена и показатели на акцию) ---
        if 'CLOSE' in df_res:
            price = df_res['CLOSE']
            
            if 'EPS_y' in df_res:
                df_res['PE_y'] = safe_divide(price, df_res['EPS_y'])
                if df_res['PE_y'].notna().any(): calculated_ratios_set.add('PE_y')
                
            if 'BVPS_q' in df_res:
                df_res['PB_q'] = safe_divide(price, df_res['BVPS_q'])
                if df_res['PB_q'].notna().any(): calculated_ratios_set.add('PB_q')
                
            if 'BVPS_y' in df_res:
                df_res['PB_y'] = safe_divide(price, df_res['BVPS_y'])
                if df_res['PB_y'].notna().any(): calculated_ratios_set.add('PB_y')
                
            if 'SPS_y' in df_res:
                df_res['PS_y'] = safe_divide(price, df_res['SPS_y'])
                if df_res['PS_y'].notna().any(): calculated_ratios_set.add('PS_y')
                
            # EV/EBITDA
            if 'NetDebt_y' in df_res and ebitda_ltm is not None:
                market_cap = price * issue_size
                enterprise_value = market_cap + df_res['NetDebt_y'] * 1e6 # NetDebt уже в млн, приводим EV к млн
                df_res['EV_EBITDA_y'] = safe_divide(enterprise_value / 1e6, ebitda_ltm) # ebitda_ltm тоже в млн
                if df_res['EV_EBITDA_y'].notna().any(): calculated_ratios_set.add('EV_EBITDA_y')
                
    return df_res, calculated_ratios_set

def main():
    logging.info("--- Расчет финансовых коэффициентов ---")
    
    # Загрузка IssueSize
    issue_sizes = load_issue_sizes(ISSUE_SIZE_FILE)
    if not issue_sizes:
        logging.warning("Данные IssueSize не загружены. Показатели на акцию и мультипликаторы не будут рассчитаны.")
        # Продолжаем без них, но некоторые target ratios не будут достигнуты

    # Создание выходной директории
    os.makedirs(FINAL_FEATURES_DIR, exist_ok=True)
    logging.info(f"Результаты будут сохранены в: {FINAL_FEATURES_DIR}")

    # Получение списка файлов для обработки
    merged_files = glob.glob(os.path.join(MERGED_FEATURES_DIR, "*_merged.csv"))
    if not merged_files:
        logging.error(f"Не найдены файлы в {MERGED_FEATURES_DIR}. Прерывание.")
        return
        
    logging.info(f"Найдено {len(merged_files)} файлов для обработки.")
    
    fully_calculated_tickers = []
    target_ratios_set = set(TARGET_RATIOS)
    processed_count = 0

    # Обработка каждого тикера
    for filepath in merged_files:
        filename = os.path.basename(filepath)
        match = re.match(r"([A-Z0-9]+)_merged.csv", filename)
        if not match:
            logging.warning(f"Не удалось извлечь тикер из имени файла: {filename}. Пропуск.")
            continue
            
        ticker = match.group(1)
        logging.info(f"--- Обработка тикера: {ticker} ---")
        
        try:
            # Загрузка данных
            df = pd.read_csv(filepath, parse_dates=['DATE'], index_col='DATE')
            
            # Получение IssueSize для текущего тикера
            current_issue_size = issue_sizes.get(ticker)
            if current_issue_size is None:
                 logging.warning(f"IssueSize для {ticker} не найден. Показатели на акцию/мультипликаторы не будут рассчитаны.")
            
            # Расчет коэффициентов
            df_final, calculated_ratios = calculate_ratios(df, current_issue_size)
            
            # Проверка полноты расчета
            missing_ratios = target_ratios_set - calculated_ratios
            if not missing_ratios:
                fully_calculated_tickers.append(ticker)
                logging.info(f"Все {len(target_ratios_set)} целевых коэффициентов рассчитаны для {ticker}.")
            else:
                # Логируем только если issue_size был доступен, иначе это ожидаемо
                if current_issue_size is not None:
                    logging.warning(f"Не все коэффициенты рассчитаны для {ticker}. Отсутствуют: {sorted(list(missing_ratios))}")
                # Если issue_size не было, не логируем как warning, т.к. это ожидаемый пропуск
                    
            # Сохранение результата
            output_filename = f"{ticker}_final.csv"
            output_path = os.path.join(FINAL_FEATURES_DIR, output_filename)
            df_final.to_csv(output_path)
            processed_count += 1
            logging.info(f"Результат для {ticker} сохранен в {output_path}")

        except pd.errors.EmptyDataError:
            logging.warning(f"Файл {filename} пуст. Пропуск.")
        except Exception as e:
            logging.error(f"Ошибка при обработке файла {filename}: {e}", exc_info=True)

    logging.info("--- Обработка завершена ---")
    logging.info(f"Успешно обработано файлов: {processed_count}")
    logging.info(f"Тикеры, для которых рассчитаны ВСЕ целевые коэффициенты ({len(fully_calculated_tickers)}):")
    if fully_calculated_tickers:
        # Выводим порциями для читаемости
        for i in range(0, len(fully_calculated_tickers), 10):
             logging.info(f"  {fully_calculated_tickers[i:i+10]}")
    else:
        logging.info("  (Нет таких тикеров)")

if __name__ == "__main__":
    main() 