import pandas as pd

# Загрузка данных из Excel файла
excel_file = 'data/CPI.xlsx'
df = pd.read_excel(excel_file, usecols=['Дата', 'Инфляция, % г/г'])

# Переименование колонок
df.rename(columns={'Дата': 'DATE', 'Инфляция, % г/г': 'CPI'}, inplace=True)

# Преобразование колонки DATE в datetime объекты
# Предполагается, что формат дат в Excel "mm.yyyy"
df['DATE'] = pd.to_datetime(df['DATE'], format='%m.%Y')

# Установка DATE как индекса для ресемплинга
df.set_index('DATE', inplace=True)
df.sort_index(inplace=True)

# Создание диапазона ежедневных дат от минимальной до максимальной
start_date = df.index.min()
end_date = df.index.max()
daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Создание нового DataFrame с ежедневными датами
# df_daily = pd.DataFrame(index=daily_dates)

# Объединение с исходным DataFrame и заполнение пропусков (forward fill)
# df_daily['CPI'] = pd.Series(df_daily.index.map(df['CPI'])).ffill()

# Реиндексация для получения ежедневных дат и заполнение CPI
df_daily = df.reindex(daily_dates)
df_daily['CPI'] = df_daily['CPI'].ffill()

# Сброс индекса, чтобы DATE стала обычной колонкой
df_daily.reset_index(inplace=True)
df_daily.rename(columns={'index': 'DATE'}, inplace=True)

# Сохранение в CSV файл
output_csv_file = 'data/cpi_daily.csv'
df_daily.to_csv(output_csv_file, index=False, float_format='%.2f')

print(f"Данные успешно обработаны и сохранены в {output_csv_file}") 