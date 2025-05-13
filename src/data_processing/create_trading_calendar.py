import pandas as pd
from datetime import date, timedelta
import holidays
import os
from pathlib import Path
from typing import Set, Tuple, List, Dict

# Настройка логирования для отладки и информации
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)

def generate_trading_calendar(
    historical_data_path: str | os.PathLike,
    output_path: str | os.PathLike,
    start_date_overall: date = date(2019, 1, 1),
    end_date_overall: date = date(2025, 12, 31),
    date_column_name: str = "TRADEDATE",
) -> None:
    """
    Создает торговый календарь на основе исторических данных и прогноза на будущее.

    Историческая часть календаря определяется наличием даты в предоставленном файле котировок.
    Будущая часть календаря (после последней даты в исторических данных) помечает
    стандартные выходные (Сб, Вс) и официальные российские праздники как неторговые дни.

    Args:
        historical_data_path (str | os.PathLike): Путь к CSV-файлу с историческими котировками
                                                   (например, по акции GAZP). Ожидается колонка с датой.
        output_path (str | os.PathLike): Путь для сохранения сгенерированного CSV-файла календаря.
        start_date_overall (date, optional): Начальная дата для всего календаря.
                                             По умолчанию 1 января 2019 года.
        end_date_overall (date, optional): Конечная дата для всего календаря.
                                           По умолчанию 31 декабря 2025 года.
        date_column_name (str, optional): Название колонки с датами в историческом файле.
                                          По умолчанию "TRADEDATE".

    Returns:
        None: Файл календаря сохраняется по указанному output_path.

    Raises:
        FileNotFoundError: Если файл с историческими данными не найден.
        KeyError: Если колонка с датой не найдена в исторических данных.
        ValueError: Если в колонке с датами содержатся некорректные значения.
    """
    historical_data_path = Path(historical_data_path)
    output_path = Path(output_path)

    _logger.info(f"Начало генерации торгового календаря.")
    _logger.info(f"Используются исторические данные из: {historical_data_path}")
    _logger.info(f"Общий диапазон календаря: с {start_date_overall} по {end_date_overall}")

    # 1. Загрузка исторических торговых дат
    try:
        df_historical = pd.read_csv(historical_data_path)
    except FileNotFoundError:
        _logger.error(f"Файл с историческими данными не найден: {historical_data_path}")
        raise
    except Exception as e:
        _logger.error(f"Ошибка чтения файла {historical_data_path}: {e}")
        raise

    if date_column_name not in df_historical.columns:
        _logger.error(f"Колонка '{date_column_name}' не найдена в файле {historical_data_path}.")
        raise KeyError(f"Колонка '{date_column_name}' не найдена.")

    try:
        # Преобразуем в datetime объекты и берем только дату, затем в set для быстрого поиска
        # Добавляем errors='coerce', чтобы невалидные даты стали NaT, затем удаляем их.
        historical_trading_dates: Set[date] = set(
            pd.to_datetime(df_historical[date_column_name], errors='coerce').dt.date.dropna()
        )
    except Exception as e: # Ловим более общую ошибку на случай проблем с форматом дат
        _logger.error(f"Ошибка преобразования дат в колонке '{date_column_name}': {e}. "
                      f"Убедитесь, что даты в формате, который может распознать pandas (например, YYYY-MM-DD).")
        raise ValueError(f"Некорректный формат дат в колонке '{date_column_name}'.") from e


    if not historical_trading_dates:
        _logger.warning("Не найдено ни одной валидной торговой даты в исторических данных. "
                        "Календарь будет основан только на выходных и праздниках.")
        min_hist_date = end_date_overall # Чтобы вся логика ушла в "будущее"
        max_hist_date = start_date_overall - timedelta(days=1) # Гарантирует, что start_date_overall > max_hist_date
    else:
        min_hist_date = min(historical_trading_dates)
        max_hist_date = max(historical_trading_dates)
        _logger.info(f"Исторические данные охватывают диапазон: с {min_hist_date} по {max_hist_date}.")
        _logger.info(f"Всего уникальных торговых дат в истории: {len(historical_trading_dates)}.")


    # 2. Подготовка праздников для РФ на весь период
    # (включая исторический, т.к. праздники используются для будущих дат,
    # а "будущее" может начаться и в 2024, если CSV до 2023)
    try:
        ru_holidays: holidays.HolidayBase = holidays.RU(
            years=list(range(start_date_overall.year, end_date_overall.year + 1))
        )
        _logger.info(f"Загружены российские праздники для периода {start_date_overall.year}-{end_date_overall.year}.")
    except Exception as e:
        _logger.error(f"Не удалось загрузить библиотеку праздников: {e}. Убедитесь, что 'holidays' установлена.")
        # Можно продолжить без праздников или прервать выполнение
        ru_holidays = {} # Пустой словарь, если загрузка не удалась, чтобы проверки не падали


    # 3. Генерация календаря
    calendar_data: List[Dict[str, Any]] = []
    current_date = start_date_overall

    while current_date <= end_date_overall:
        is_trading_day: bool

        if current_date <= max_hist_date and current_date >= min_hist_date:
            # Логика для дат, покрываемых историческими данными
            # Если CSV не покрывает начало start_date_overall, то эти дни будут False
            is_trading_day = current_date in historical_trading_dates
        elif current_date > max_hist_date : # Строго после последней даты в CSV - это "будущее"
            # Логика для будущих дат
            if current_date.weekday() >= 5:  # Суббота (5) или Воскресенье (6)
                is_trading_day = False
            elif current_date in ru_holidays:
                is_trading_day = False
            else:
                is_trading_day = True
        else: # Даты до min_hist_date (если start_date_overall раньше min_hist_date)
              # По логике пользователя "если нет в таблице, то не торговый"
            is_trading_day = False


        calendar_data.append({"date": current_date.strftime("%Y-%m-%d"), "is_trading_day": is_trading_day})
        current_date += timedelta(days=1)

    # 4. Сохранение результата
    df_calendar = pd.DataFrame(calendar_data)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_calendar.to_csv(output_path, index=False)
        _logger.info(f"Торговый календарь успешно сгенерирован и сохранен в: {output_path}")
        _logger.info(f"Всего записей в календаре: {len(df_calendar)}")
    except Exception as e:
        _logger.error(f"Ошибка сохранения календаря в файл {output_path}: {e}")
        raise


if __name__ == "__main__":
    # Определяем корневую директорию проекта относительно текущего файла
    # Предполагается, что скрипт находится в src/data_processing/
    project_root = Path(__file__).resolve().parent.parent.parent

    # Пути к файлам относительно корня проекта
    # Используем файл, указанный пользователем
    DEFAULT_HISTORICAL_DATA_CSV = project_root / "data" / "raw" / "moex_shares" / "GAZP_moex_data.csv"
    DEFAULT_OUTPUT_CALENDAR_CSV = project_root / "data" / "processed" / "moex_trading_calendar.csv"

    # Пример вызова функции
    try:
        generate_trading_calendar(
            historical_data_path=DEFAULT_HISTORICAL_DATA_CSV,
            output_path=DEFAULT_OUTPUT_CALENDAR_CSV,
            # start_date_overall=date(2023,12,25), # Можно для теста задать меньший диапазон
            # end_date_overall=date(2024,1,15)
        )
        _logger.info("Скрипт успешно завершил работу.")
    except FileNotFoundError:
        _logger.error(f"Один из файлов не найден. Проверьте пути.")
        _logger.error(f"Ожидался файл исторических данных: {DEFAULT_HISTORICAL_DATA_CSV.resolve()}")
        _logger.error(f"Календарь должен был быть сохранен в: {DEFAULT_OUTPUT_CALENDAR_CSV.resolve()}")
    except KeyError as e:
        _logger.error(f"Ошибка конфигурации: {e}")
    except ValueError as e:
        _logger.error(f"Ошибка в данных: {e}")
    except Exception as e:
        _logger.exception(f"Произошла непредвиденная ошибка: {e}") 