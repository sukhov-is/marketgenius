import pandas as pd
import os
import glob
import re
from datetime import datetime, date
import numpy as np

# --- Константы ---
# Получаем директорию, где находится сам скрипт
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Предполагаем, что корень проекта на один уровень выше папки scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_REPORTS_DIR = os.path.join(PROJECT_ROOT, "data/raw/financial_reports")
PROCESSED_FEATURES_DIR = os.path.join(PROJECT_ROOT, "data/processed/financial_features")
QUARTERLY_DIR = os.path.join(RAW_REPORTS_DIR, "quarterly")
YEARLY_DIR = os.path.join(RAW_REPORTS_DIR, "yearly")

# Словарь для поиска метрик в отчетах
METRIC_NAMES_MAP = {
    "Revenue": ["Выручка"],
    "EBITDA": ["EBITDA"],
    "OperatingProfit": ["Операционная прибыль"],
    "NetProfit": ["Чистая прибыль", "Чистая прибыль н/с"], # Искать второе, если нет первого
    "Assets": ["Активы", "Активы банка"],
    "Equity": ["Капитал", "Чистые активы", "Баланс стоимость"],
    "Debt": ["Долг"],
    "Cash": ["Наличность"],
    "OperatingCashFlow": ["Операционный денежный поток"],
    "CAPEX": ["CAPEX"],
    "OperatingExpenses": ["Операционные расходы", "Опер. расходы"],
    "DividendsPaid": ["Див.выплата"]
}
# Название строки с датой публикации отчета
DATE_REPORT_NAME = "Дата отчета"

# --- Функции ---

def clean_numeric_value(value):
    """Очищает строковое значение, пытаясь преобразовать его в число."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Удаляем кавычки, пробелы, заменяем запятую на точку
        cleaned_value = value.replace('"', '').replace(' ', '').replace(',', '.')
        # Обрабатываем случай с точкой с запятой вместо NaN
        if cleaned_value == ';':
             return np.nan
        # Обрабатываем случай с процентами
        if cleaned_value.endswith('%'):
            cleaned_value = cleaned_value.replace('%', '')
            try:
                # Преобразуем процент в долю
                return float(cleaned_value) / 100.0
            except ValueError:
                return np.nan # Если после удаления % осталось не число
        # Пробуем преобразовать в число
        try:
            return float(cleaned_value)
        except ValueError:
            # Если не получилось, возможно это просто пустая строка или мусор
            return np.nan
    return np.nan # Возвращаем NaN для других типов или если очистка не помогла

def parse_period_string(period_str):
    """Преобразует строку периода ('ГГГГQК' или 'ГГГГ') в дату конца периода."""
    if pd.isna(period_str) or not isinstance(period_str, str):
        return pd.NaT

    cleaned_period_str = period_str.strip()

    # Попробуем re.search и проверим длину совпадения для квартала
    quarter_match = re.search(r"(\d{4})Q(\d)", cleaned_period_str)
    year_match = re.fullmatch(r"(\d{4})", cleaned_period_str)

    # --- УДАЛЕНА Отладочная печать внутри parse_period_string ---
    # print(f"    DEBUG parse_period_string: input='{period_str}', cleaned='{cleaned_period_str}'")
    # print(f"      DEBUG: quarter_match result: {quarter_match}")
    # print(f"      DEBUG: year_match result: {year_match}")
    # --- Конец УДАЛЕНИЯ отладки ---

    # Проверяем, что квартальный матч покрывает всю строку (имитация fullmatch)
    if quarter_match and quarter_match.group(0) == cleaned_period_str:
        # print("      DEBUG: Matched quarter (full string)") # Убираем отладку
        year = int(quarter_match.group(1))
        quarter = int(quarter_match.group(2))
        try:
            if not 1 <= quarter <= 4: raise ValueError("Invalid quarter")
            # --- ЗАМЕНА pd.Period --- 
            # return pd.Period(year=year, quarter=quarter).end_time.date()
            last_month_of_quarter = quarter * 3
            # Находим первый день следующего месяца, затем вычитаем один день
            next_month = last_month_of_quarter + 1
            next_year = year
            if next_month > 12:
                 next_month = 1
                 next_year += 1
            first_day_next_month = date(next_year, next_month, 1)
            last_day_of_quarter = first_day_next_month - pd.Timedelta(days=1)
            return last_day_of_quarter # Уже объект date
            # --- КОНЕЦ ЗАМЕНЫ ---
        except ValueError:
             # print("      DEBUG: Invalid quarter/year for Period or date calculation") # Убираем отладку
             return pd.NaT
    elif year_match:
        # print("      DEBUG: Matched year (fullmatch)") # Убираем отладку
        year = int(year_match.group(1))
        try:
            return datetime(year, 12, 31).date()
        except ValueError:
             # print("      DEBUG: Invalid year for datetime") # Убираем отладку
             return pd.NaT
    else:
        # print("      DEBUG: No valid match found") # Убираем отладку
        return pd.NaT

def parse_publication_date(date_str):
    """Преобразует строку с датой публикации в datetime.date."""
    if pd.isna(date_str) or not isinstance(date_str, str) or date_str.strip() == '':
        return pd.NaT
    try:
        # Попробуем формат ДД.ММ.ГГГГ
        return datetime.strptime(date_str.strip(), "%d.%m.%Y").date()
    except ValueError:
        # Можно добавить другие форматы при необходимости
        return pd.NaT

def find_metric_row(df, possible_names):
    """Находит индекс строки, полное имя и масштаб (млрд=1e9, млн=1e6) метрики."""
    for name_to_find in possible_names:
        for idx, actual_index_name in enumerate(df.index):
            if isinstance(actual_index_name, str):
                 cleaned_actual_name = actual_index_name.strip()
                 if cleaned_actual_name.startswith(name_to_find):
                    scale = 1.0
                    if "млрд" in cleaned_actual_name:
                         scale = 1e9
                    elif "млн" in cleaned_actual_name:
                         scale = 1e6
                    # print(f"DEBUG: Found '{name_to_find}' in '{cleaned_actual_name}', scale={scale}") # Отладка
                    return idx, actual_index_name, scale # Возвращаем индекс, полное имя, масштаб
    return -1, None, 1.0 # Возвращаем -1 и масштаб 1.0 если не найдено

def parse_report_file(file_path):
    """Парсит один файл отчета (годовой или квартальный)."""
    print(f"Обработка файла: {os.path.basename(file_path)}")
    try:
        # Читаем без index_col, чтобы первая колонка была как данные
        # df = pd.read_csv(file_path, sep=';', header=0, encoding='utf-8')
        # Устанавливаем первую колонку как индекс
        # if df.columns.empty:
        #      print(f"Ошибка: Файл {file_path} пуст или не содержит заголовков.")
        #      return None
        # df.set_index(df.columns[0], inplace=True)
        # df.index.name = None # Убираем имя индекса

        # Читаем CSV, используя первую колонку как индекс напрямую
        df = pd.read_csv(file_path, sep=';', header=0, index_col=0, encoding='utf-8')
        df.index.name = None # Убираем имя индекса
        if df.columns.empty:
             print(f"Ошибка: Файл {file_path} не содержит колонок с данными (периодами).")
             return None
        # --- УДАЛЕНА Печать колонок для отладки ---
        # print(f"  Колонки, прочитанные pandas: {list(df.columns)}")
        # --- Конец УДАЛЕНИЯ печати колонок ---

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден {file_path}")
        return None
    except Exception as e:
        print(f"Ошибка чтения файла {file_path}: {e}")
        return None

    filename = os.path.basename(file_path)
    ticker_match = re.match(r"([A-Z]+)_", filename)
    if not ticker_match:
        print(f"Не удалось извлечь тикер из имени файла: {filename}")
        return None
    ticker = ticker_match.group(1)
    report_type = "quarterly" if "quarterly" in filename else "yearly"

    # Найти строку с датами публикации
    date_row_idx, _, _ = find_metric_row(df, [DATE_REPORT_NAME]) # Принимаем 3 значения
    if date_row_idx == -1:
        print(f"Предупреждение: Строка '{DATE_REPORT_NAME}' не найдена в файле {filename}")
        publication_dates_raw = pd.Series([pd.NaT] * len(df.columns), index=df.columns) # Заглушка
    else:
        publication_dates_raw = df.iloc[date_row_idx]

    report_data = []

    # Извлекаем периоды и даты публикации
    periods = []
    publication_dates = []
    valid_columns = []
    for col in df.columns:
        period_end = parse_period_string(col)

        # --- УДАЛЕНА Детальная отладка парсинга периода и проверки года ---
        # print(f"  DEBUG: Processing column '{col}'")
        # print(f"    DEBUG: parse_period_string returned: {period_end} (type: {type(period_end)})")
        # if pd.isna(period_end):
        #      print(f"    DEBUG: Skipping column '{col}' because period_end is NaT.")
        #      continue
        # --- Конец УДАЛЕНИЯ отладки ---

        if pd.notna(period_end):
            # --- Добавляем проверку года --- 
            year_check_passed = False
            try:
                year_val = period_end.year
                if year_val >= 2018:
                    year_check_passed = True
                    # print(f"    DEBUG: Year check PASSED for '{col}' (Year: {year_val})") # Убираем отладку года
                # else:
                    # print(f"    DEBUG: Year check FAILED for '{col}' (Year: {year_val} < 2018)") # Убираем отладку года
            except AttributeError:
                 # print(f"    DEBUG: Could not get year from period_end ({type(period_end)}). Skipping '{col}'.") # Убираем отладку года
                 continue

            if not year_check_passed:
                 continue

            pub_date_raw = publication_dates_raw.get(col)
            pub_date = parse_publication_date(pub_date_raw)

            # Эвристика для отсутствующей даты публикации
            if pd.isna(pub_date):
                days_offset = 90 if report_type == "yearly" else 45
                # Используем pd.Timedelta для корректного добавления дней к дате
                # Преобразуем period_end в Timestamp, если это date, для сложения с Timedelta
                # Используем импортированный класс date
                if isinstance(period_end, date) and not isinstance(period_end, datetime):
                    period_end_ts = pd.Timestamp(period_end)
                else:
                    period_end_ts = period_end # Уже Timestamp или совместимый тип

                try:
                    pub_date_estimated = period_end_ts + pd.Timedelta(days=days_offset)
                    # print(f"  Предупреждение: Дата публикации для {ticker} {col} не найдена ('{pub_date_raw}'). Используется эвристика: {pub_date_estimated.date()}")
                    pub_date = pub_date_estimated.date() # Используем .date() для согласованности типов
                except Exception as e:
                    print(f" Ошибка при расчете эвристической даты для {ticker} {col}: {e}")
                    continue # Пропускаем колонку, если расчет даты не удался

            # Дата публикации (или расчетная) должна быть ПОСЛЕ или РАВНА концу периода
            # Добавил проверки на NaT перед сравнением + исправил сравнение дат
            comparison_passed = False
            if pd.notna(period_end) and pd.notna(pub_date):
                try:
                    # Сравниваем только объекты date (или datetime)
                    if isinstance(period_end, date) and isinstance(pub_date, date):
                         comparison_passed = pub_date >= period_end
                    else:
                         print(f"  Предупреждение: Не удалось сравнить даты для {ticker} {col}. period_end: {type(period_end)}, pub_date: {type(pub_date)}")
                except TypeError as e:
                     print(f"  Ошибка сравнения дат для {ticker} {col}: {e}")

            if comparison_passed:
                periods.append(period_end)
                publication_dates.append(pub_date)
                valid_columns.append(col)
            # else: # Условие для отладки пропущенных колонок
            #     if pd.notna(period_end) and pd.notna(pub_date) and period_end.year >= 2018:
            #          print(f"  Колонка {col} пропущена: Сравнение дат не прошло (Pub: {pub_date}, PeriodEnd: {period_end})")
        # else:
            # print(f"  Пропуск колонки '{col}': не удалось распознать период.")

    if not valid_columns:
        print(f"Предупреждение: Не найдено валидных колонок с периодами и датами публикации в {filename}")
        return None

    # Извлекаем значения метрик
    for metric_key, possible_names in METRIC_NAMES_MAP.items():
        row_idx, found_name, scale_factor = find_metric_row(df, possible_names) # Получаем масштаб
        if row_idx != -1:
            metric_values_raw = df.iloc[row_idx][valid_columns] # Берем значения только для валидных колонок
            metric_values = [clean_numeric_value(v) for v in metric_values_raw]

            for i in range(len(periods)):
                # Добавляем только если значение метрики не NaN
                if pd.notna(metric_values[i]):
                    scaled_value = metric_values[i] * scale_factor # Применяем масштаб
                    report_data.append({
                        "Ticker": ticker,
                        "ReportType": report_type,
                        "PeriodEnd": periods[i],
                        "PublicationDate": publication_dates[i],
                        "Metric": metric_key,
                        "Value": scaled_value # Сохраняем масштабированное значение
                    })

    if not report_data:
        # print(f"Предупреждение: Не удалось извлечь никаких метрик из файла {filename}")
        return None

    return pd.DataFrame(report_data)

# --- Основная логика ---
def main():
    # Создаем папку для обработанных данных, если она не существует
    os.makedirs(PROCESSED_FEATURES_DIR, exist_ok=True)

    # Находим все файлы отчетов
    quarterly_files = glob.glob(os.path.join(QUARTERLY_DIR, "*_quarterly.csv"))
    yearly_files = glob.glob(os.path.join(YEARLY_DIR, "*_yearly.csv"))
    all_files = quarterly_files + yearly_files

    if not all_files:
        print("Ошибка: Не найдены файлы отчетов в папках data/raw/financial_reports/quarterly и data/raw/financial_reports/yearly")
        return

    # Группируем файлы по тикерам
    files_by_ticker = {}
    for f in all_files:
        ticker_match = re.match(r"([A-Z]+)_", os.path.basename(f))
        if ticker_match:
            ticker = ticker_match.group(1)
            if ticker not in files_by_ticker:
                files_by_ticker[ticker] = []
            files_by_ticker[ticker].append(f)

    all_parsed_data = []
    # Парсим все файлы
    for ticker, files in files_by_ticker.items():
         print(f"--- Обработка тикера: {ticker} ---")
         for file_path in files:
            parsed_df = parse_report_file(file_path)
            if parsed_df is not None:
                all_parsed_data.append(parsed_df)

    if not all_parsed_data:
        print("Ошибка: Не удалось обработать ни одного файла отчетов.")
        return

    # Объединяем все данные
    combined_data = pd.concat(all_parsed_data, ignore_index=True)

    # Преобразуем типы данных для дат
    combined_data['PeriodEnd'] = pd.to_datetime(combined_data['PeriodEnd'])
    combined_data['PublicationDate'] = pd.to_datetime(combined_data['PublicationDate'])

    # Обработка и сохранение данных для каждого тикера
    for ticker, group in combined_data.groupby('Ticker'):
        print(f"--- Формирование ежедневного ряда для тикера: {ticker} ---")
        # Сортируем данные по дате публикации, затем по концу периода
        # Удаляем дубликаты для комбинации PeriodEnd/Metric/ReportType, оставляя последнее обновление
        group = group.sort_values(by=['PublicationDate', 'PeriodEnd']).drop_duplicates(
            subset=['PeriodEnd', 'Metric', 'ReportType'], keep='last'
        )

        # Создаем сводную таблицу с мультииндексом: строки - PublicationDate, столбцы - [Metric, ReportType]
        try:
             pivot_multi_df = group.pivot_table(
                 index='PublicationDate', 
                 columns=['Metric', 'ReportType'], 
                 values='Value'
             )
        except Exception as e:
             print(f"  Ошибка при создании сводной таблицы для {ticker}: {e}")
             continue

        if pivot_multi_df.empty:
             print(f"  Предупреждение: Нет данных для {ticker} после пивотирования.")
             continue

        # Определяем полный диапазон дат
        start_date = pivot_multi_df.index.min()
        end_date = datetime.now().date() # До сегодняшнего дня
        daily_index = pd.date_range(start=start_date, end=end_date, freq='D')

        # Переиндексируем и заполняем пропуски вперед
        daily_filled_multi = pivot_multi_df.reindex(daily_index).ffill()

        # Создаем финальный DataFrame
        daily_features = pd.DataFrame(index=daily_index)

        # processed_metrics = set() # Больше не нужно
        final_metric_columns = []

        # Создаем колонки Metric_q и Metric_y напрямую из daily_filled_multi
        for (metric, report_type) in daily_filled_multi.columns:
            suffix = report_type[0] # 'q' или 'y'
            new_col_name = f"{metric}_{suffix}"
            # Проверяем, есть ли хоть одно не NaN значение перед добавлением
            if daily_filled_multi[(metric, report_type)].notna().any():
                 daily_features[new_col_name] = daily_filled_multi[(metric, report_type)]
                 final_metric_columns.append(new_col_name)

        if not final_metric_columns:
            print(f"  Предупреждение: Не найдено данных ни для одной метрики для {ticker} после пивотирования и ffill.")
            continue

        # Рассчитываем Чистый долг _q и _y, если есть компоненты
        netdebt_q_added = False
        if 'Debt_q' in daily_features.columns and 'Cash_q' in daily_features.columns:
            valid_netdebt_q_index = daily_features[['Debt_q', 'Cash_q']].dropna().index
            daily_features['NetDebt_q'] = np.nan
            daily_features.loc[valid_netdebt_q_index, 'NetDebt_q'] = daily_features.loc[valid_netdebt_q_index, 'Debt_q'] - daily_features.loc[valid_netdebt_q_index, 'Cash_q']
            if daily_features['NetDebt_q'].notna().any():
                 final_metric_columns.append('NetDebt_q')
                 netdebt_q_added = True
        
        netdebt_y_added = False
        if 'Debt_y' in daily_features.columns and 'Cash_y' in daily_features.columns:
             valid_netdebt_y_index = daily_features[['Debt_y', 'Cash_y']].dropna().index
             daily_features['NetDebt_y'] = np.nan
             daily_features.loc[valid_netdebt_y_index, 'NetDebt_y'] = daily_features.loc[valid_netdebt_y_index, 'Debt_y'] - daily_features.loc[valid_netdebt_y_index, 'Cash_y']
             if daily_features['NetDebt_y'].notna().any():
                 final_metric_columns.append('NetDebt_y')
                 netdebt_y_added = True

        # Убираем строки до первой полной строки (где хотя бы одна метрика _q или _y не NaN)
        if not final_metric_columns:
             print(f" Предупреждение: Нет колонок с финальными метриками (включая NetDebt) для {ticker}.")
             continue
         
        # Пересортируем колонки для лучшей читаемости (опционально)
        daily_features = daily_features[sorted(final_metric_columns)]
        
        # Ищем первую дату, когда есть хотя бы одно значение
        first_valid_date = daily_features.dropna(how='all').index.min()

        if pd.notna(first_valid_date):
             daily_features = daily_features[daily_features.index >= first_valid_date]
        else:
             print(f"  Предупреждение: Не удалось найти строку хотя бы с одной метрикой для {ticker}. Возможно, данных недостаточно.")
             continue # Пропускаем сохранение, если нет данных

        # --- Приводим все числовые значения к миллионам рублей ---
        numeric_cols = daily_features.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
             # print(f"DEBUG: Converting columns {list(numeric_cols)} to millions for {ticker}")
             daily_features[numeric_cols] = daily_features[numeric_cols] / 1e6
        # --- Конец приведения к миллионам ---

        # --- Округляем числовые значения до 1 знака после запятой ---
        numeric_cols_to_round = daily_features.select_dtypes(include=np.number).columns
        if not numeric_cols_to_round.empty:
             daily_features[numeric_cols_to_round] = daily_features[numeric_cols_to_round].round(1)
        # --- Конец округления ---

        # Сохраняем результат
        output_path = os.path.join(PROCESSED_FEATURES_DIR, f"{ticker}_features.csv")
        try:
            daily_features.to_csv(output_path)
            print(f"Результат для {ticker} сохранен в {output_path}")
        except Exception as e:
            print(f"Ошибка сохранения файла {output_path}: {e}")


if __name__ == "__main__":
    main() 