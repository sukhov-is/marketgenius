"""
run_processing_pipeline.py
----------------------------------
Оркестратор, который после обновления сырых данных (run_daily_update.py)
последовательно выполняет весь конвейер обработки:

1. Расчёт технических индикаторов (scripts/02_technicalIndicators.py)
2. Обработка финансовых отчётов (scripts/03_process_financial_reports.py)
3. Объединение тех./фин./макро-признаков (scripts/04_merge_features.py)
4. Расчёт финансовых коэффициентов (scripts/05_calculate_financial_ratios.py)
5. Предобработка признаков для обучения (scripts/06_preprocess_features.py)

Каждый этап вызывается как отдельный подпроцесс Python, что позволяет
изолировать зависимости и использовать уже существующую логику скриптов.

Скрипт старается не перезаписывать логику объединения данных –
все подскрипты рассчитаны на повторный запуск и формируют полные
актуальные файлы. Если необходимо более избирательное объединение,
его можно добавить отдельными функциями merge_csv_by_date, оставив
поведение по умолчанию перезаписывать выходные файлы.
"""

import subprocess
import sys
import logging
import shlex
from pathlib import Path
import argparse

# --- Вспомогательные функции -------------------------------------------------

def run_step(args_list: list[str], cwd: Path) -> None:
    """Запускает подпроцесс и пишет stdout/stderr в лог."""
    logging.info(f"Запуск: {' '.join(args_list)}")

    result = subprocess.run(
        args_list, cwd=str(cwd), capture_output=True, text=True
    )
    if result.stdout:
        logging.debug(result.stdout)
    if result.stderr:
        logging.debug(result.stderr)
    if result.returncode != 0:
        logging.error(
            f"Команда завершилась с кодом {result.returncode}: {' '.join(args_list)}\n"
            "См. вывод выше для подробностей."
        )
        raise RuntimeError(f"Ошибка выполнения шага: {' '.join(args_list)}")
    logging.info("Шаг завершён успешно.")

# ----------------------------------------------------------------------------

DEFAULT_STEPS = [
    ("02_technicalIndicators.py", "--input data/raw/moex_shares --output data/processed/technical_indicators"),
    ("03_process_financial_reports.py", ""),
    ("04_merge_features.py", "--tech-dir data/processed/technical_indicators "
                               "--fin-dir data/processed/financial_features "
                               "--macro-dir data/raw/macro "
                               "--out-dir data/processed/merged_features"),
    ("05_calculate_financial_ratios.py", ""),
    ("06_preprocess_features.py", ""),
]


def main():
    parser = argparse.ArgumentParser(
        description="Оркестратор полного конвейера обработки данных"
    )
    parser.add_argument(
        "--scripts-dir",
        type=str,
        default="scripts",
        help="Директория, где находятся скрипты 02-06_*.py (по умолчанию: scripts)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логирования (по умолчанию: INFO)"
    )
    args = parser.parse_args()

    # Настройка логирования
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # scripts_path изначально указывает на директорию, переданную в --scripts-dir (по умолчанию 'scripts')
    # project_dir будет корневой директорией проекта
    scripts_dir_path = Path(args.scripts_dir).resolve()
    project_dir = scripts_dir_path.parent

    if not scripts_dir_path.exists() and args.scripts_dir == "scripts": # Проверяем стандартное расположение, если оно указано
        # Если стандартная папка 'scripts' не существует, это может быть проблемой,
        # но project_dir все равно будет вычислен как родитель предполагаемого args.scripts_dir
        logging.warning(f"Директория {scripts_dir_path}, указанная в --scripts-dir, не найдена, но попытка продолжить, используя ее родителя как корень проекта.")
    elif not scripts_dir_path.exists():
        logging.critical(f"Указанная директория --scripts-dir не найдена: {scripts_dir_path}")
        sys.exit(1)

    # Последовательно выполняем шаги
    for script_filename, extra_args in DEFAULT_STEPS:
        # Формируем путь к скрипту относительно корневой директории проекта
        # Предполагаем, что целевые скрипты теперь всегда в 'src/data_ingestion/'
        target_script_path = project_dir / "src" / "data_ingestion" / script_filename
        
        if not target_script_path.exists():
            logging.error(f"Не найден скрипт {target_script_path}. Пропуск.")
            continue
        
        args_list = [sys.executable, str(target_script_path), *shlex.split(extra_args, posix=False)]
        
        try:
            run_step(args_list, cwd=project_dir)  # cwd = корень проекта
        except RuntimeError:
            logging.error("Конвейер остановлен из-за ошибки на предыдущем шаге.")
            sys.exit(1)

    logging.info("=== Полный конвейер обработки данных выполнен успешно ===")


if __name__ == "__main__":
    main()