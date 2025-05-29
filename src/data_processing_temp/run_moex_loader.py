from src.data_ingestion.indexes_moex import MoexIndexLoader
from datetime import date
import pandas as pd # Добавим импорт pandas для работы с DataFrame

def main():
    # Задаем параметры для загрузки
    start_date = date(2018, 1, 1)
    end_date = date.today() # Загружаем данные по сегодняшний день
    output_file = "data/raw/macro/moex_indices.csv"

    print(f"Запуск загрузки индексов MOEX за период с {start_date} по {end_date}.")
    print(f"Результаты будут сохранены в {output_file}")

    # Инициализируем загрузчик. Можно передать путь к конфигу, если он нужен.
    # loader = MoexIndexLoader(config_path='path/to/your/config.json')
    loader = MoexIndexLoader() # Используем индексы по умолчанию

    # Запускаем загрузку
    # Увеличим задержку до 1.5 секунд, чтобы быть более аккуратными с API MOEX
    results_df = loader.download_indices(
        output_file=output_file,
        start_date=start_date,
        end_date=end_date,
        delay=1.5 # Задержка между запросами к API
    )

    # Выводим отчет о загрузке
    print("\nОтчет о загрузке:")
    if not results_df.empty:
        print(results_df.to_string()) # to_string() для лучшего отображения в консоли
    else:
        print("Отчет о загрузке пуст или произошла ошибка до его формирования.")

    print(f"\nЗагрузка завершена. Проверьте файл: {output_file}")

if __name__ == "__main__":
    # Убедимся, что pandas выводит все колонки DataFrame при печати
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000) # Чтобы строки не переносились сильно
    main() 