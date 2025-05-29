from src.data_ingestion.usd_rub_loader import UsdRubLoader
from src.data_ingestion.oil_prices_loader import AlphaVantageBrentLoader
from datetime import date
import logging
import os
from dotenv import load_dotenv # Импортируем load_dotenv

# Настройка базового логирования для скрипта, если загрузчики не настроят его сами
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    load_dotenv() # Загружаем переменные из .env файла

    start_date = date(2018, 1, 1)
    end_date = date.today()

    output_dir_macro = "data/raw/macro"
    usd_rub_output_file = os.path.join(output_dir_macro, "usd_rub_rate.csv")
    brent_output_dir = output_dir_macro

    print(f"Запуск загрузки макроэкономических данных за период с {start_date} по {end_date}.")

    # --- Загрузка курса USD/RUB ---
    print("\n--- Загрузка курса USD/RUB ---")
    usd_loader = UsdRubLoader()
    usd_success = usd_loader.download_rates(
        output_file=usd_rub_output_file,
        start_date=start_date,
        end_date=end_date
    )
    if usd_success:
        print(f"Данные по курсу USD/RUB успешно обработаны. Результат в: {usd_rub_output_file}")
    else:
        print("Во время загрузки данных по курсу USD/RUB возникли ошибки.")

    # --- Загрузка цен на нефть Brent ---
    print("\n--- Загрузка цен на нефть Brent (Alpha Vantage) ---")
    alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not alpha_vantage_api_key:
        print("Ключ ALPHA_VANTAGE_API_KEY не найден в .env файле или переменных окружения.")
        print("Пожалуйста, добавьте его в .env файл в корне проекта: ALPHA_VANTAGE_API_KEY=ваш_ключ")
        brent_success = False
    else:
        try:
            brent_loader = AlphaVantageBrentLoader(alpha_vantage_api_key=alpha_vantage_api_key)
            brent_success = brent_loader.download_prices(
                output_dir=brent_output_dir, # передаем директорию
                start_date=start_date,
                end_date=end_date
            )
            if brent_success:
                print(f"Данные по ценам на нефть Brent успешно обработаны. Результат в: {os.path.join(brent_output_dir, 'brent_prices.csv')}")
            else:
                print("Во время загрузки данных по ценам на нефть Brent возникли ошибки или нет данных для сохранения.")
        except ValueError as e:
            print(f"Ошибка инициализации загрузчика Brent: {e}") # Сообщение об ошибке от загрузчика уже содержит информацию о ключе
            brent_success = False
        except Exception as e:
            print(f"Непредвиденная ошибка при загрузке Brent: {e}")
            brent_success = False

    print(f"\nЗагрузка макроэкономических данных завершена.")
    if usd_success:
        print(f"USD/RUB: Успешно. Файл: {usd_rub_output_file}")
    else:
        print("USD/RUB: Были ошибки.")

    if brent_success:
        print(f"Brent: Успешно. Файл: {os.path.join(brent_output_dir, 'brent_prices.csv')}")
    else:
        print("Brent: Были ошибки или ключ не указан/не найден.")

if __name__ == "__main__":
    main() 