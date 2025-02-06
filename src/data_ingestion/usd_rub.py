import requests
import pandas as pd
from datetime import datetime, timedelta

# Функция для загрузки данных о курсе доллара за заданный диапазон дат
def get_usd_to_rub_exchange_rate(start_date, end_date):
    base_url = "https://www.cbr.ru/scripts/XML_dynamic.asp"

    # Форматирование дат для запроса
    start_date_str = start_date.strftime('%d/%m/%Y')
    end_date_str = end_date.strftime('%d/%m/%Y')

    # Параметры запроса
    params = {
        'date_req1': start_date_str,
        'date_req2': end_date_str,
        'VAL_NM_RQ': 'R01235'  # Код валюты для USD
    }

    # Запрос к API ЦБ РФ
    response = requests.get(base_url, params=params)

    if response.status_code != 200:
        raise Exception(f"Ошибка при запросе данных: {response.status_code}")

    # Парсинг ответа XML
    data = pd.read_xml(response.content, xpath="//Record")

    # Преобразование данных в DataFrame
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
    data['Value'] = data['Value'].str.replace(',', '.').astype(float)

    # Переименование столбцов для удобства
    data = data.rename(columns={
        'Date': 'Дата',
        'Value': 'Курс USD к RUB'
    })

    return data

def main():
    # Получение данных за последние 10 лет
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10*365)

    # Загрузка данных
    try:
        exchange_rate_data = get_usd_to_rub_exchange_rate(start_date, end_date)
        print("Данные успешно загружены!")
        
        # Сохранение данных в CSV
        output_file = "data/external/macro/usd_to_rub.csv"
        exchange_rate_data.to_csv(output_file, index=False)
        print("Данные сохранены в файл 'usd_to_rub.csv'")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()