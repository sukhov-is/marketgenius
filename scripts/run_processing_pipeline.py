#!/usr/bin/env python
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

def run_step(command: str, cwd: Path) -> None:
    """Запускает подпроцесс и пишет stdout/stderr в лог."""
    logging.info(f"Запуск: {command}")
    result = subprocess.run(
        shlex.split(command), cwd=str(cwd), capture_output=True, text=True
    )
    if result.stdout:
        logging.debug(result.stdout)
    if result.stderr:
        logging.debug(result.stderr)
    if result.returncode != 0:
        logging.error(
            f"Команда завершилась с кодом {result.returncode}: {command}\n"
            "См. вывод выше для подробностей."
        )
        raise RuntimeError(f"Ошибка выполнения шага: {command}")
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

    scripts_path = Path(args.scripts_dir).resolve()
    if not scripts_path.exists():
        logging.critical(f"Директория со скриптами не найдена: {scripts_path}")
        sys.exit(1)

    # Последовательно выполняем шаги
    for script_name, extra_args in DEFAULT_STEPS:
        script_path = scripts_path / script_name
        if not script_path.exists():
            logging.error(f"Не найден скрипт {script_path}. Пропуск.")
            continue
        cmd = f"{sys.executable} {script_path} {extra_args}".strip()
        try:
            run_step(cmd, cwd=scripts_path.parent)  # cwd = корень проекта
        except RuntimeError:
            logging.error("Конвейер остановлен из-за ошибки на предыдущем шаге.")
            sys.exit(1)

    logging.info("=== Полный конвейер обработки данных выполнен успешно ===")


if __name__ == "__main__":
    main() 