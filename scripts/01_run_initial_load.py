import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import os # Добавляем os
from dotenv import load_dotenv # Добавляем dotenv
import sys # <-- Добавляем sys

# Определяем корневую директорию проекта
project_root = Path(__file__).resolve().parent.parent

# Добавляем корневую директорию в sys.path для корректных импортов
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Импортируем все наши классы-загрузчики
from src.data_ingestion.moex_parser import MoexLoader
from src.data_ingestion.fin_otchet import FinancialReportLoader
from src.data_ingestion.indexes_moex import MoexIndexLoader
from src.data_ingestion.oil_prices_loader import AlphaVantageBrentLoader
from src.data_ingestion.usd_rub_loader import UsdRubLoader
from src.data_ingestion.key_rate_loader import KeyRateLoader

# Загружаем переменные окружения из .env файла
# Ищем .env в текущей директории или выше
dotenv_path = project_root / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Если .env не найден относительно скрипта, пробуем стандартный поиск
    load_dotenv()

def setup_logging(log_level_str: str):
    """Настраивает базовое логирование."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Установлен уровень логирования: {log_level_str.upper()}")

def main():
    parser = argparse.ArgumentParser(description="Запуск начальной загрузки всех исторических данных.")
    parser.add_argument(
        '--years',
        type=int,
        default=7,
        help="Количество лет истории для загрузки (по умолчанию: 7)."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help="Корневая директория для сохранения сырых данных (по умолчанию: data/raw)."
    )
    parser.add_argument(
        '--companies-config',
        type=str,
        default='configs/all_companies_config.json',
        help="Путь к JSON файлу конфигурации компаний (по умолчанию: configs/companies_config.json)."
    )
    parser.add_argument(
        '--indices-config',
        type=str,
        default=None, # По умолчанию загрузчик использует встроенный список
        help="(Опционально) Путь к JSON файлу конфигурации индексов MOEX."
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Уровень логирования (по умолчанию: INFO)."
    )

    args = parser.parse_args()

    # Настройка логирования
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__) # Получаем логгер для этого скрипта

    # Определение дат
    end_date = date.today()
    start_date = end_date - timedelta(days=args.years * 365) # Упрощенно, можно точнее
    logger.info(f"Период загрузки: с {start_date} по {end_date} ({args.years} лет)")

    # Создание базовой выходной директории
    base_output_path = Path(args.output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Данные будут сохраняться в директорию: {base_output_path.resolve()}")

    # Получаем ключ Alpha Vantage из окружения
    alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not alpha_vantage_api_key:
        logger.warning("Ключ ALPHA_VANTAGE_API_KEY не найден в переменных окружения. Загрузка цен Brent (Alpha Vantage) не будет выполнена.")
        # Не прерываем выполнение, просто пропускаем этот шаг
    else:
        logger.info("Ключ ALPHA_VANTAGE_API_KEY успешно загружен.")

    # --- Этап 1: Загрузка данных по акциям MOEX --- #
    logger.info("--- Начало загрузки исторических данных акций MOEX ---")
    try:
        moex_loader = MoexLoader(config_path=args.companies_config)
        moex_output_dir = base_output_path / 'moex_shares'
        # Вызываем метод для загрузки исторического диапазона
        # Он сам сохранит файлы и отчет внутри moex_output_dir
        moex_loader.download_historical_range(
            output_dir=str(moex_output_dir),
            start_date=start_date,
            end_date=end_date
            # delay можно добавить в аргументы, если нужно
        )
        logger.info("--- Загрузка данных акций MOEX завершена ---")
    except FileNotFoundError:
        logger.error(f"Ошибка: Не найден файл конфигурации компаний: {args.companies_config}")
    except Exception as e:
        logger.exception(f"Ошибка при загрузке данных акций MOEX: {e}")

    # --- Этап 2: Загрузка финансовых отчетов --- #
    logger.info("--- Начало загрузки финансовых отчетов (Smart-Lab) ---")
    try:
        report_loader = FinancialReportLoader(config_path=args.companies_config)
        reports_output_dir = base_output_path / 'financial_reports'
        report_loader.download_reports(
            output_dir=str(reports_output_dir)
            # tickers_list=None (загружаем все из конфига)
            # delay можно настроить
        )
        logger.info("--- Загрузка финансовых отчетов завершена ---")
    except FileNotFoundError:
        logger.error(f"Ошибка: Не найден файл конфигурации компаний: {args.companies_config}")
    except Exception as e:
        logger.exception(f"Ошибка при загрузке финансовых отчетов: {e}")

    # --- Этап 3: Загрузка индексов MOEX --- #
    logger.info("--- Начало загрузки индексов MOEX ---")
    try:
        index_loader = MoexIndexLoader(config_path=args.indices_config)
        indices_output_file = base_output_path / 'macro' / 'moex_indices.csv'
        index_loader.download_indices(
            output_file=str(indices_output_file),
            start_date=start_date,
            end_date=end_date
        )
        logger.info("--- Загрузка индексов MOEX завершена ---")
    except Exception as e:
        logger.exception(f"Ошибка при загрузке индексов MOEX: {e}")

    # --- Этап 4: Загрузка цен на нефть (Urals, Brent) --- #
    logger.info("--- Начало загрузки цен на нефть (Brent / Alpha Vantage) ---")
    # Выполняем только если ключ Alpha Vantage есть
    if alpha_vantage_api_key:
        try:
            # Инициализируем новый загрузчик с ключом Alpha Vantage
            oil_loader = AlphaVantageBrentLoader(alpha_vantage_api_key=alpha_vantage_api_key)
            oil_output_dir = base_output_path / 'macro'
            # Метод download_prices теперь только для Brent и возвращает bool
            success = oil_loader.download_prices(
                output_dir=str(oil_output_dir),
                start_date=start_date,
                end_date=end_date
            )
            if success:
                 logger.info("--- Загрузка цен на нефть (Brent / Alpha Vantage) завершена ---")
            else:
                 logger.error("--- Ошибка при загрузке цен на нефть (Brent / Alpha Vantage). См. логи выше. ---")
        except ValueError as ve: # Ловим ошибку, если ключ пустой (хотя мы проверили)
             logger.error(f"Ошибка инициализации AlphaVantageBrentLoader: {ve}")
        except Exception as e:
            logger.exception(f"Ошибка при загрузке цен на нефть (Brent / Alpha Vantage): {e}")
    else:
        logger.warning("Пропуск загрузки цен на нефть из-за отсутствия ALPHA_VANTAGE_API_KEY.")

    # --- Этап 5: Загрузка курса USD/RUB --- #
    logger.info("--- Начало загрузки курса USD/RUB ---")
    try:
        usd_rub_loader = UsdRubLoader()
        usd_rub_output_file = base_output_path / 'macro' / 'usd_rub_rate.csv'
        usd_rub_loader.download_rates(
            output_file=str(usd_rub_output_file),
            start_date=start_date,
            end_date=end_date
        )
        logger.info("--- Загрузка курса USD/RUB завершена ---")
    except Exception as e:
        logger.exception(f"Ошибка при загрузке курса USD/RUB: {e}")

    # --- Этап 6: Загрузка ключевой ставки --- #
    logger.info("--- Начало загрузки истории ключевой ставки ---")
    try:
        key_rate_loader = KeyRateLoader()
        key_rate_output_file = base_output_path / 'macro' / 'key_rate.csv'
        key_rate_loader.download_key_rate(
            output_file=str(key_rate_output_file),
            start_date=start_date,
            end_date=end_date
        )
        logger.info("--- Загрузка ключевой ставки завершена ---")
    except Exception as e:
        logger.exception(f"Ошибка при загрузке ключевой ставки: {e}")

    logger.info("=== Начальная загрузка данных завершена ===")

if __name__ == "__main__":
    main() 