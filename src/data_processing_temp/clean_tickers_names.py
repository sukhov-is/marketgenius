import pandas as pd
import json
import numpy as np
from fuzzywuzzy import process # Для нечеткого сопоставления

# Пути к файлам
config_file_path = 'configs/all_companies_config.json'
input_csv_path = 'data/processed/gpt/results_gpt_blogs.csv' # Убедитесь, что это правильный путь к вашему файлу
output_csv_path = 'data/processed/gpt/results_gpt_blogs.csv' # Изменено имя выходного файла

# Параметр для нечеткого сопоставления
FUZZY_MATCH_SCORE_CUTOFF = 85 # Порог схожести (0-100)

def normalize_for_matching(text):
    """Нормализует строку для сопоставления: кириллица -> латиница, удаление разделителей, нижний регистр."""
    text = str(text)
    replacements = {
        'А': 'A', 'а': 'a', 'В': 'B', 'в': 'v', 'Е': 'E', 'е': 'e', 'К': 'K', 'к': 'k',
        'М': 'M', 'м': 'm', 'Н': 'H', 'н': 'n', 'О': 'O', 'о': 'o', 'Р': 'P', 'р': 'r',
        'С': 'C', 'с': 's', 'Т': 'T', 'т': 't', 'Х': 'X', 'х': 'h', 'У': 'U', 'у': 'u',
        'Г': 'G', 'г': 'g', 'Д': 'D', 'д': 'd', 'З': 'Z', 'з': 'z', 'И': 'I', 'и': 'i',
        'Л': 'L', 'л': 'l', 'П': 'P', 'п': 'p', 'Ф': 'F', 'ф': 'f', 'Ц': 'TS', 'ц': 'ts',
        'Ч': 'CH', 'ч': 'ch', 'Ш': 'SH', 'ш': 'sh', 'Щ': 'SCH', 'щ': 'sch',
        'Э': 'E', 'э': 'e', 'Ю': 'YU', 'ю': 'yu', 'Я': 'YA', 'я': 'ya', 'Й': 'Y', 'й': 'y',
        'Ж': 'ZH', 'ж': 'zh', 'Б': 'B', 'б': 'b', 'Ь': '', 'ь': '', 'Ъ': '', 'ъ': '',
        'Ё': 'E', 'ё': 'e'
    }
    for cyr, lat_equiv in replacements.items():
        text = text.replace(cyr, lat_equiv)
    
    text = text.replace('-', '').replace('_', '').replace('.', '').replace(' ', '')
    return text.lower()

try:
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    print(f"Конфигурационный файл '{config_file_path}' успешно загружен.")

    try:
        df = pd.read_csv(input_csv_path)
        print(f"CSV файл '{input_csv_path}' успешно загружен. Количество строк: {len(df)}, количество колонок: {len(df.columns)}")
    except FileNotFoundError:
        print(f"Ошибка: CSV файл '{input_csv_path}' не найден.")
        exit()
    except Exception as e:
        print(f"Ошибка при чтении CSV файла '{input_csv_path}': {e}")
        exit()

    name_to_canonical_map = {}
    for canonical_ticker, company_info in config_data.get('companies', {}).items():
        normalized_canon_ticker = normalize_for_matching(canonical_ticker)
        name_to_canonical_map[normalized_canon_ticker] = canonical_ticker
        for name_variant in company_info.get('names', []):
            name_to_canonical_map[normalize_for_matching(name_variant)] = canonical_ticker
    for canonical_ticker, description in config_data.get('indices', {}).items():
        normalized_canon_ticker = normalize_for_matching(canonical_ticker)
        name_to_canonical_map[normalized_canon_ticker] = canonical_ticker
    
    print(f"Карта сопоставления (на основе нормализованных имен) создана. Уникальных записей в карте: {len(name_to_canonical_map)}")

    known_normalized_canonical_tickers_map = {}
    all_canonical_tickers_companies = list(config_data.get('companies', {}).keys())
    all_canonical_tickers_indices = list(config_data.get('indices', {}).keys())
    for ticker in all_canonical_tickers_companies + all_canonical_tickers_indices:
        known_normalized_canonical_tickers_map[normalize_for_matching(ticker)] = ticker

    preserved_columns = ['date', 'summary']
    cleaned_df_columns = {} 

    for col in preserved_columns:
        if col in df.columns:
            cleaned_df_columns[col] = df[col]
        else:
            print(f"Предупреждение: Ожидаемая колонка '{col}' не найдена в CSV.")

    column_groups = {} 
    unmatched_csv_columns = []
    potential_instrument_cols = [col for col in df.columns if col not in preserved_columns]
    
    for original_csv_col_name in potential_instrument_cols:
        normalized_csv_col_name = normalize_for_matching(original_csv_col_name)
        canonical_ticker = name_to_canonical_map.get(normalized_csv_col_name)
        match_type = "точное"

        if not canonical_ticker and known_normalized_canonical_tickers_map:
            prospective_fuzzy_match = process.extractOne(
                normalized_csv_col_name,
                list(known_normalized_canonical_tickers_map.keys()), # Передаем список ключей
                score_cutoff=FUZZY_MATCH_SCORE_CUTOFF
            )
            
            if prospective_fuzzy_match:
                matched_norm_canon_key = prospective_fuzzy_match[0]
                score = prospective_fuzzy_match[1]
                
                allow_this_fuzzy_match = True
                if len(matched_norm_canon_key) == 1 and len(normalized_csv_col_name) > 2:
                    if score < 96:
                        allow_this_fuzzy_match = False
                        # print(f"INFO: Отвергнуто нечеткое сопоставление (правило короткого тикера): CSV норм.='{normalized_csv_col_name}' (исх: '{original_csv_col_name}') к канон. норм.='{matched_norm_canon_key}' с оценкой {score} < 96.")
                
                if allow_this_fuzzy_match:
                    canonical_ticker = known_normalized_canonical_tickers_map[matched_norm_canon_key]
                    match_type = f"нечеткое (схожесть: {score}% к канон.норм.форме '{matched_norm_canon_key}')"
        
        if canonical_ticker:
            if canonical_ticker not in column_groups:
                column_groups[canonical_ticker] = {'originals': [], 'normalized_found_as': set()}
            column_groups[canonical_ticker]['originals'].append(original_csv_col_name)
            column_groups[canonical_ticker]['normalized_found_as'].add(f"'{normalized_csv_col_name}' (исходное: '{original_csv_col_name}', тип: {match_type})")
        else:
            unmatched_csv_columns.append(original_csv_col_name)

    print(f"Найдено {len(column_groups)} групп колонок для объединения.")
    print("\nИнформация об объединенных колонках:")
    if column_groups:
        for canonical, group_info in column_groups.items():
            originals = group_info['originals']
            normalized_details = ", ".join(list(group_info['normalized_found_as']))
            if len(originals) > 0:
                print(f"  Канонический тикер: '{canonical}'")
                print(f"    <- Исходные колонки CSV: {originals}")
                print(f"    <- Сопоставлено через нормализованные/нечеткие формы: {normalized_details}")
    else:
        print("  Не было найдено групп колонок для объединения.")
    
    if unmatched_csv_columns:
        print(f"\nПредупреждение: Следующие {len(unmatched_csv_columns)} колонки из CSV не были сопоставлены ни с одним каноническим тикером и будут проигнорированы:")
        print(f"  {unmatched_csv_columns[:20]}{'...' if len(unmatched_csv_columns) > 20 else ''}")

    for canonical_ticker, group_info in column_groups.items():
        original_cols_list = group_info['originals']
        existing_cols_in_group = [col for col in original_cols_list if col in df.columns]
        
        if not existing_cols_in_group:
            continue

        numeric_data = df[existing_cols_in_group].apply(pd.to_numeric, errors='coerce')

        def get_max_abs_value(row_series):
            if row_series.isnull().all():
                return np.nan 
            else:
                return row_series.loc[row_series.abs().idxmax()]

        cleaned_df_columns[canonical_ticker] = numeric_data.apply(get_max_abs_value, axis=1)
        
    final_df = pd.DataFrame(cleaned_df_columns)
    
    final_cols_order = [col for col in preserved_columns if col in final_df.columns]
    instrument_cols = sorted([col for col in final_df.columns if col not in preserved_columns])
    final_df = final_df[final_cols_order + instrument_cols]

    print(f"\nИтоговый DataFrame содержит {len(final_df.columns)} колонок.")

    try:
        final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Очищенный DataFrame успешно сохранен в '{output_csv_path}'")
        print("\nПервые 5 строк очищенного файла:")
        print(final_df.head().to_string())
        print(f"\nПоследние 5 строк очищенного файла:")
        print(final_df.tail().to_string())

    except Exception as e:
        print(f"Ошибка при сохранении итогового CSV файла: {e}")

except FileNotFoundError as e:
    if 'config_file_path' in locals() and e.filename == config_file_path:
        print(f"Критическая ошибка: Конфигурационный файл '{config_file_path}' не найден. Скрипт не может продолжить работу.")
    elif 'input_csv_path' in locals() and e.filename == input_csv_path:
         print(f"Критическая ошибка: Входной CSV файл '{input_csv_path}' не найден. Скрипт не может продолжить работу.")
    else:
        print(f"Критическая ошибка: Файл не найден - {e.filename}. Скрипт не может продолжить работу.")
except json.JSONDecodeError:
    print(f"Критическая ошибка: Не удалось декодировать JSON из файла '{config_file_path}'. Проверьте его структуру. Скрипт не может продолжить работу.")
except ImportError:
    print("Критическая ошибка: Необходимая библиотека 'fuzzywuzzy' не найдена. Пожалуйста, установите ее: pip install fuzzywuzzy python-Levenshtein")
except Exception as e:
    print(f"Произошла непредвиденная ошибка во время выполнения скрипта: {e}") 