import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def get_moex_index_data(index_name, start_date, end_date):
    """
    Получает данные по индексу с MOEX ISS API
    
    Args:
        index_name (str): Тикер индекса (например, 'IMOEX', 'MOEXBC', etc.)
        start_date (str): Дата начала в формате 'YYYY-MM-DD'
        end_date (str): Дата окончания в формате 'YYYY-MM-DD'
    """
    base_url = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities"
    url = f"{base_url}/{index_name}.json"
    
    params = {
        "from": start_date,
        "till": end_date,
        "history.columns": "TRADEDATE,CLOSE",
        "iss.meta": "off",
        "iss.json": "extended"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data and len(data) > 1 and "history" in data[1]:
            df = pd.DataFrame(data[1]["history"])
            df.columns = ["DATE", index_name]
            return df
        return None
    except Exception as e:
        print(f"Ошибка при получении данных для {index_name}: {e}")
        return None

def main():
    # Список индексов Мосбиржи (исключая RTS)
    indices = [
        "MRBC",     # Индекс Мосбиржи15
        "RTSI",     # Индекс РТС
        "MCXSM",    # Индекс средней и малой капитализации
        "IMOEX",    # Индекс МосБиржи
        "MOEXBC",   # Индекс голубых фишек
        "MOEXBMI",  # Индекс широкого рынка
        "MOEXCN",   # Индекс потребительского сектора
        "MOEXIT",   # Индекс IT
        "MOEXRE",   # Индекс строительных компаний
        "MOEXEU",   # Индекс электроэнергетики
        "MOEXFN",   # Индекс финансов
        "MOEXINN",  # Индекс инноваций
        "MOEXMM",   # Индекс металлов и добычи
        "MOEXOG",   # Индекс нефти и газа
        "MOEXTL",   # Индекс телекоммуникаций
        "MOEXTN",   # Индекс транспорта
        "MOEXCH",   # Индекс химии и нефтехимии
        "MOEXBMI",  # Индекс широкого рынка
    ]
    
    # Даты для загрузки (например, последний год)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Создаем пустой DataFrame с датами как индекс
    result_df = None
    
    for index in indices:
        print(f"Загрузка данных для индекса {index}...")
        df = get_moex_index_data(index, start_date, end_date)
        
        if df is not None:
            if result_df is None:
                result_df = df
            else:
                # Объединяем данные по дате
                result_df = pd.merge(result_df, df, on='DATE', how='outer')
        
        time.sleep(1)  # Задержка между запросами
    
    if result_df is not None:
        # Сортируем по дате
        result_df = result_df.sort_values('DATE')
        
        # Сохраняем в CSV
        output_file = "data/external/macro/moex_indices.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nДанные сохранены в файл: {output_file}")
    else:
        print("Не удалось получить данные индексов")

if __name__ == "__main__":
    main()