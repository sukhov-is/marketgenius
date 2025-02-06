import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import pandas as pd
import csv
import sys


def fetch_data(from_date, to_date):
    headers = {
        'Content-Type': 'text/xml; charset=utf-8',
        'SOAPAction': 'http://web.cbr.ru/KeyRateXML',
    }

    body = f'''<?xml version="1.0" encoding="utf-8"?> 
    <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"> 
        <soap:Body> 
            <KeyRateXML xmlns="http://web.cbr.ru/"> 
                <fromDate>{from_date}</fromDate> 
                <ToDate>{to_date}</ToDate> 
            </KeyRateXML> 
        </soap:Body> 
    </soap:Envelope> 
    '''

    response = requests.post('http://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx',
                             headers=headers, data=body)

    if response.status_code == 200:
        return ET.fromstring(response.content)
    else:
        response.raise_for_status()


def parse_response(root):
    return [
        {
            'date': item.find('DT').text,
            'rate': float(item.find('Rate').text.replace(',', '.'))
        }
        for item in root.iter('KR')
    ]


def create_dataframe(data):
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def fill_missing_dates(df, start_date, end_date):
    all_dates = pd.date_range(start=start_date, end=end_date).date
    df = df.set_index('date').reindex(all_dates, method='ffill').reset_index()
    df.columns = ['date', 'rate']
    return df


def save_to_csv(df):
    df.to_csv(sys.stdout, index=False, encoding='UTF-8', quoting=csv.QUOTE_NONNUMERIC, sep=";")


def main():
    # Получаем сегодняшнюю дату и дату 10 лет назад
    today = datetime.now()
    ten_years_ago = today - timedelta(days=365 * 10)  # приблизительно 10 лет назад

    # Форматируем даты в формате YYYY-MM-DD
    from_date = ten_years_ago.strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')

    # Запрашиваем данные через SOAP API
    root = fetch_data(from_date, to_date)
    data = parse_response(root)
    df = create_dataframe(data)

    # Заполняем пропущенные даты (если данные не приходят за каждый день)
    df = fill_missing_dates(df, ten_years_ago.date(), today.date())

    # Сортируем по дате и выводим CSV в стандартный вывод
    df = df.sort_values(by='date')
    df.to_csv("../data/external/macro/key_rate.csv", index=False, encoding='UTF-8', quoting=csv.QUOTE_NONNUMERIC, sep=";")


if __name__ == "__main__":
    main()