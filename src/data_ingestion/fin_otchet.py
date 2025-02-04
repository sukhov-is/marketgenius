import json
import requests
import os
from time import sleep

def load_config(config_file='companies_config.json'):
    """Загружает конфигурацию из JSON-файла."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def download_reports(companies):
    # Создаем папки для отчетов (используем os.makedirs с exist_ok=True)
    quarterly_dir = os.path.join('data', 'reports', 'quarterly')
    yearly_dir = os.path.join('data', 'reports', 'yearly')
    os.makedirs(quarterly_dir, exist_ok=True)
    os.makedirs(yearly_dir, exist_ok=True)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Скачиваем квартальные отчеты
    print("Скачиваем квартальные отчеты...")
    for ticker, name in companies.items():
        url = f"https://smart-lab.ru/q/{ticker}/f/q/MSFO/download/"
        filename = os.path.join(quarterly_dir, f"{ticker}_quarterly.csv")
        try:
            print(f"Скачиваем квартальный отчет {name} ({ticker})...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Успешно сохранено в {filename}")
        except Exception as e:
            print(f"Ошибка при скачивании {ticker}: {e}")
        sleep(1)  # Пауза между запросами

    # Скачиваем годовые отчеты
    print("\nСкачиваем годовые отчеты...")
    for ticker, name in companies.items():
        url = f"https://smart-lab.ru/q/{ticker}/f/y/MSFO/download/"
        filename = os.path.join(yearly_dir, f"{ticker}_yearly.csv")
        try:
            print(f"Скачиваем годовой отчет {name} ({ticker})...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Успешно сохранено в {filename}")
        except Exception as e:
            print(f"Ошибка при скачивании {ticker}: {e}")
        sleep(2)  # Пауза между запросами

    print("\nЗагрузка завершена!")

if __name__ == "__main__":
    # Загружаем конфигурацию из файла
    config = load_config()
    # Предполагается, что в JSON-конфиге данные компаний лежат под ключом "companies"
    companies = config.get("companies", {})
    if not companies:
        print("Список компаний не найден в конфигурации!")
    else:
        download_reports(companies)