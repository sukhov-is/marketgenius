import requests
import pandas as pd
import os
import re
import time
from datetime import date

# --- Константы ---
# Получаем директорию, где находится сам скрипт
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Предполагаем, что корень проекта на один уровень выше папки scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MERGED_FEATURES_DIR = os.path.join(PROJECT_ROOT, "data/processed/merged_features")
REFERENCE_DATA_DIR = os.path.join(PROJECT_ROOT, "data/reference")
OUTPUT_FILE = os.path.join(REFERENCE_DATA_DIR, "issue_sizes.csv")

MOEX_API_URL = "https://iss.moex.com/iss/securities/{security}.json"
REQUEST_DELAY_SECONDS = 0.5 # Задержка между запросами к API

def get_tickers_from_features(directory):
    """Получает список уникальных тикеров из имен файлов в указанной директории."""
    tickers = set()
    try:
        for filename in os.listdir(directory):
            if filename.endswith("_merged.csv"):
                match = re.match(r"([A-Z0-9]+)_merged.csv", filename)
                if match:
                    tickers.add(match.group(1))
    except FileNotFoundError:
        print(f"Ошибка: Директория не найдена: {directory}")
    return sorted(list(tickers))

def get_issue_size_from_moex(ticker):
    """Получает текущее значение IssueSize для тикера с API Мосбиржи."""
    url = MOEX_API_URL.format(security=ticker)
    print(f"  Запрос для {ticker}...")
    try:
        response = requests.get(url)
        response.raise_for_status() # Проверка на HTTP ошибки
        data = response.json()

        # Ищем IssueSize в блоке 'description'
        # Структура ответа ISS: {'block_name': {'columns': [...], 'data': [[...], ...]}}
        if 'description' in data and 'columns' in data['description'] and 'data' in data['description']:
            try:
                columns = data['description']['columns']
                rows = data['description']['data']
                # Находим индексы нужных колонок
                name_idx = columns.index('name')
                value_idx = columns.index('value')

                for row in rows:
                    if row[name_idx] == 'ISSUESIZE':
                        issue_size_str = row[value_idx]
                        if issue_size_str:
                             try:
                                 # Преобразуем в число (может быть float)
                                 return float(issue_size_str)
                             except (ValueError, TypeError):
                                  print(f"    Предупреждение: Не удалось преобразовать IssueSize '{issue_size_str}' в число для {ticker}.")
                                  return None
                        else:
                             print(f"    Предупреждение: Пустое значение IssueSize для {ticker}.")
                             return None
                
                print(f"    Предупреждение: Поле 'ISSUESIZE' не найдено в description для {ticker}.")
                return None

            except (ValueError, IndexError, KeyError) as e:
                 print(f"    Ошибка парсинга description для {ticker}: {e}")
                 return None
        else:
             print(f"    Предупреждение: Блок 'description' не найден или пуст для {ticker}.")
             return None

    except requests.exceptions.RequestException as e:
        print(f"  Ошибка запроса к API для {ticker}: {e}")
        return None
    except Exception as e:
        print(f"  Неизвестная ошибка при обработке {ticker}: {e}")
        return None

def main():
    print("--- Загрузка количества акций (IssueSize) --- ")
    tickers = get_tickers_from_features(MERGED_FEATURES_DIR)
    if not tickers:
        print("Ошибка: Не найдены тикеры в папке merged_features. Прерывание.")
        return

    print(f"Найдено тикеров: {len(tickers)}")
    
    issue_size_data = []
    today_date = date.today().strftime('%Y-%m-%d') # Дата загрузки

    for ticker in tickers:
        issue_size = get_issue_size_from_moex(ticker)
        if issue_size is not None:
            issue_size_data.append({
                "DATE": today_date,
                "SECID": ticker,
                "IssueSize": issue_size
            })
        else:
             print(f"    Пропуск тикера {ticker} из-за отсутствия данных IssueSize.")
        
        time.sleep(REQUEST_DELAY_SECONDS) # Пауза между запросами

    if not issue_size_data:
         print("Ошибка: Не удалось получить IssueSize ни для одного тикера.")
         return

    # Создаем DataFrame
    issue_size_df = pd.DataFrame(issue_size_data)

    # Создаем папку для справочных данных, если ее нет
    os.makedirs(REFERENCE_DATA_DIR, exist_ok=True)

    # Сохраняем в CSV
    try:
        issue_size_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nДанные по IssueSize успешно сохранены в: {OUTPUT_FILE}")
        print(f"Обработано тикеров с IssueSize: {len(issue_size_df)}")
    except Exception as e:
        print(f"Ошибка сохранения файла {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main() 