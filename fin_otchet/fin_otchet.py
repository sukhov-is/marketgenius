import requests
import os
from time import sleep

def download_reports():
    companies = {
        'GAZP': 'Газпром',
        'SBER': 'Сбербанк',
        'LKOH': 'Лукойл',
        'ROSN': 'Роснефть',
        'GMKN': 'Норникель',
        'NVTK': 'Новатэк',
        'VTBR': 'ВТБ',
        'MTSS': 'МТС',
        'PLZL': 'Полюс'
    }
    
    # Создаем папки для отчетов
    os.makedirs('reports/quarterly', exist_ok=True)
    os.makedirs('reports/yearly', exist_ok=True)
    print("Скачиваем квартальные отчеты...")
    for ticker, name in companies.items():
        url = f"https://smart-lab.ru/q/{ticker}/f/q/MSFO/download/"
        filename = f"reports/quarterly/{ticker}_quarterly.csv"
        
        try:
            print(f"Скачиваем квартальный отчет {name} ({ticker})...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Успешно сохранено в {filename}")
        except Exception as e:
            print(f"Ошибка при скачивании {ticker}: {str(e)}")
        
        sleep(2)  # Пауза между запросами

    # Скачиваем годовые отчеты
    print("\nСкачиваем годовые отчеты...")
    for ticker, name in companies.items():
        url = f"https://smart-lab.ru/q/{ticker}/f/y/MSFO/download/"
        filename = f"reports/yearly/{ticker}_yearly.csv"
        
        try:
            print(f"Скачиваем годовой отчет {name} ({ticker})...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Успешно сохранено в {filename}")
        except Exception as e:
            print(f"Ошибка при скачивании {ticker}: {str(e)}")
        
        sleep(2)  # Пауза между запросами

    print("\nЗагрузка завершена!")

if __name__ == "__main__":
    download_reports()