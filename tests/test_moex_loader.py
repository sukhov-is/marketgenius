import sys
import os
from datetime import date, timedelta

# Добавляем корень проекта в PYTHONPATH, чтобы можно было импортировать src
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data_ingestion.indexes_moex import MoexIndexLoader

def run_test():
    """Запускает тестовую загрузку данных по индексам MOEX."""
    print("Запуск тестового скрипта для MoexIndexLoader...")

    # 1. Создаем экземпляр загрузчика
    # Можно передать путь к config.json, если он у вас есть
    # loader = MoexIndexLoader(config_path='config.json')
    loader = MoexIndexLoader() # Используем индексы по умолчанию

    # 2. Определяем период загрузки (например, последние 30 дней)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    print(f"Период загрузки: с {start_date} по {end_date}")

    # 3. Определяем путь для выходного файла
    output_file = os.path.join(project_root, "data", "test_moex_indices.csv")
    print(f"Выходной файл: {output_file}")

    # 4. Запускаем загрузку
    # Можно передать список конкретных индексов: indices_list=['IMOEX', 'RTSI']
    try:
        report_df = loader.download_indices(
            output_file=output_file,
            start_date=start_date,
            end_date=end_date,
            # indices_list=['IMOEX'] # Раскомментируйте для теста одного индекса
            delay=0.5 # Небольшая задержка между запросами
        )

        # 5. Выводим отчет
        print("\n--- Отчет о загрузке ---")
        print(report_df.to_string())
        print("------------------------\n")

        if os.path.exists(output_file):
            print(f"Данные успешно сохранены в {output_file}")
        else:
            print("Файл с данными не был создан (возможно, не удалось загрузить ни один индекс).")

    except Exception as e:
        print(f"\n--- Произошла ошибка во время выполнения скрипта ---")
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        print("-----------------------------------------------------")

if __name__ == "__main__":
    run_test() 