from src.data_ingestion.moex_parser import MoexLoader
from datetime import date, timedelta
import logging
import os

# Настройка базового логирования для скрипта
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Определяем даты: последние 8 лет
    end_date = date.today()
    start_date = end_date - timedelta(days=8*365) # Приблизительно 8 лет

    # Директория для сохранения данных
    output_dir = "data/raw/moex_shares"

    # Путь к конфигурационному файлу компаний MOEX.
    # !!! ВАЖНО: Убедитесь, что этот файл существует и содержит список компаний,
    # либо измените путь на правильный.
    # Пример структуры см. в описании этого скрипта или в классе MoexLoader.
    config_path = "configs/all_companies_config.json" # <--- ЗАМЕНИТЕ ПРИ НЕОБХОДИМОСТИ

    print(f"Запуск загрузки данных по акциям MOEX за период с {start_date} по {end_date}.")
    print(f"Данные будут сохранены в директорию: {output_dir}")
    print(f"Используется конфигурационный файл: {config_path}")

    if not os.path.exists(config_path):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Конфигурационный файл {config_path} не найден!")
        print("Пожалуйста, создайте его или укажите правильный путь в скрипте.")
        print("Без него загрузчик не сможет определить список компаний для обработки.")
        # Можно либо прервать выполнение, либо дать загрузчику попытаться (он вызовет исключение).
        # return # Раскомментируйте для прерывания, если файл обязателен до старта.

    try:
        # Инициализируем загрузчик
        loader = MoexLoader(config_path=config_path)

        # Запускаем загрузку для всех компаний из конфига
        # Можно также передать список тикеров: tickers_list=['SBER', 'GAZP']
        loader.download_historical_range(
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            delay=1.1 # Задержка между запросами к API MOEX (в секундах)
        )
        print(f"\nЗагрузка данных по акциям MOEX завершена.")
        print(f"Проверьте результаты в директории: {output_dir}")
        report_file = os.path.join(output_dir, 'download_report.csv')
        if os.path.exists(report_file):
            print(f"Отчет о загрузке находится здесь: {report_file}")

    except FileNotFoundError as fnf_error:
        # Это исключение может быть выброшено из MoexLoader, если конфиг не найден
        print(f"Ошибка: Файл конфигурации не найден. {fnf_error}")
        print("Пожалуйста, убедитесь, что файл существует и путь указан верно.")
    except Exception as e:
        print(f"\nВо время выполнения скрипта произошла ошибка: {e}")
        logging.exception("Детали ошибки:")

if __name__ == "__main__":
    main() 