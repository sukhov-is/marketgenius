import argparse
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd
from src.utils.ticker_cleaner import clean_results_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

# Корень проекта — три уровня вверх от текущего файла (src/etl)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Общие настройки для двух пайплайнов (можно расширить через аргументы CLI)
DEFAULT_DATASETS = {
    "blogs": {
        "input_csv": "data/external/text/blogs_clusters/representative_blogs.csv",
        "prompt_type": "blogs",
        "file_prefix": "batch_input_blogs_history_part",
        "batch_input_jsonl": "data/external/text/batch/batch_input_blogs_history.jsonl",
        "results_dir": "data/processed/gpt/blogs",
        "output_csv": "data/processed/gpt/results_gpt_blogs.csv",
        "cluster_dir": "data/external/text/blogs_clusters",
    },
    "news": {
        "input_csv": "data/external/text/news_clusters/representative_news.csv",
        "prompt_type": "news",
        "file_prefix": "batch_input_news_history_part",
        "batch_input_jsonl": "data/external/text/batch/batch_input_news_history.jsonl",
        "results_dir": "data/processed/gpt/news",
        "output_csv": "data/processed/gpt/results_gpt_news.csv",
        "cluster_dir": "data/external/text/news_clusters",
    },
}


# ---------- Вспомогательные функции ----------

def run_subprocess(cmd: list[str]):
    """Запускает subprocess и выводит логирование."""
    _logger.info("RUN " + " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        _logger.error("Ошибка выполнения: %s", " ".join(cmd))
        _logger.error(result.stdout)
        raise RuntimeError(f"Команда завершилась с кодом {result.returncode}")
    _logger.info(result.stdout)


def prepare_batch(dataset_cfg: dict):
    """Шаг 1: формирование batch_input файла"""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src/etl/04_prepare_batch_file.py"),
        "--input-csv",
        dataset_cfg["input_csv"],
        "--output-jsonl",
        dataset_cfg["batch_input_jsonl"],
        "--prompt-type",
        dataset_cfg["prompt_type"],
    ]
    run_subprocess(cmd)

    # После успешной подготовки файла переименовываем его в *_part1.jsonl,
    # чтобы он соответствовал схеме, ожидаемой скриптом 05_process_batch_sequence.
    original_path = Path(dataset_cfg["batch_input_jsonl"])
    if original_path.is_file():
        part1_path = original_path.with_name(f"{original_path.stem}_part1{original_path.suffix}")
        try:
            # shutil.move перезапишет существующий файл назначения
            shutil.move(str(original_path), str(part1_path))
            _logger.info(
                f"Файл {original_path.name} перемещён (с заменой) в {part1_path.name} для скрипта 05."
            )
        except Exception as e:
            _logger.error(f"Не удалось переместить {original_path} в {part1_path}: {e}")
            raise
        dataset_cfg["part1_path"] = str(part1_path)
    else:
        _logger.warning(
            f"Ожидался файл {original_path}, но он не найден после подготовки. Пайплайн может завершиться ошибкой."
        )


def process_sequence(dataset_cfg: dict):
    """Шаг 2: отправка батчей и ожидание результатов"""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src/etl/05_process_batch_sequence.py"),
        "--input-dir",
        str(Path(dataset_cfg["batch_input_jsonl"]).parent),
        "--file-prefix",
        dataset_cfg["file_prefix"],
        "--start",
        "1",
        "--end",
        "1",
        "--output-dir",
        dataset_cfg["results_dir"],
        "--save-info-dir",
        "data/external/text/batch",  # сохраняем batch_info JSON в одну папку
    ]
    run_subprocess(cmd)


def process_results(dataset_cfg: dict):
    """Шаг 3: обработка результатов и дозапись к CSV"""
    # Обходим все batch_info_*.json в save-info-dir, относящиеся к результатам этого набора
    save_info_dir = Path("data/external/text/batch")
    for info_file in save_info_dir.glob("batch_info_*.json"):
        # Читаем batch_id и input_file, проверяем что input_file содержит file_prefix данной выборки
        try:
            import json

            with info_file.open("r", encoding="utf-8") as f:
                info_data = json.load(f)
            input_file_path = info_data.get("input_file", "")
            batch_id = info_data.get("batch_id")
            if dataset_cfg["file_prefix"] not in input_file_path:
                continue  # это не нужная нам группа
        except Exception as e:
            _logger.warning(f"Не удалось прочитать {info_file}: {e}")
            continue

        results_file = Path(dataset_cfg["results_dir"]) / f"results_{batch_id}.jsonl"
        errors_file = Path(dataset_cfg["results_dir"]) / f"errors_{batch_id}.jsonl"

        if not results_file.is_file():
            _logger.warning(f"Файл результатов не найден: {results_file}")
            continue

        # 1) Сохраняем «сырые» результаты в директорию cluster_dir без объединения
        cluster_dir = Path(dataset_cfg["cluster_dir"])
        cluster_dir.mkdir(parents=True, exist_ok=True)
        raw_csv_path = cluster_dir / f"results_gpt_{dataset_cfg['prompt_type']}.csv"

        # Конструируем команду с условным добавлением флага --errors-file
        def build_cmd(output_path: str, append: bool = False) -> list[str]:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src/etl/06_process_batch_results.py"),
                "--results-file",
                str(results_file),
            ]
            if errors_file.is_file():
                cmd += ["--errors-file", str(errors_file)]
            cmd += ["--output-csv", output_path]
            if append:
                cmd.append("--append")
            return cmd

        # 1. Сохраняем «сырые» результаты (06_process_batch_results)
        run_subprocess(build_cmd(str(raw_csv_path)))

        # 2. Очищаем файл тикеров
        try:
            cleaned_df = clean_results_file(
                input_csv=raw_csv_path,
                output_csv=raw_csv_path,  # перезаписываем тем же именем
            )
            _logger.info(
                f"Файл {raw_csv_path.name} очищен и содержит {len(cleaned_df.columns)} колонок."
            )
        except Exception as e:
            _logger.error(f"Не удалось очистить файл {raw_csv_path}: {e}")

        # 3. Дозаписываем очищенные данные в общий CSV
        if cleaned_df is not None and not cleaned_df.empty:
            # если итоговый CSV существует — объединяем
            agg_path = Path(dataset_cfg["output_csv"])
            if agg_path.is_file():
                try:
                    existing = pd.read_csv(agg_path)
                    combined = pd.concat([existing, cleaned_df], ignore_index=True)
                except Exception as e:
                    _logger.error(f"Ошибка чтения/объединения {agg_path}: {e}")
                    combined = cleaned_df
            else:
                combined = cleaned_df

            agg_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(agg_path, index=False, encoding="utf-8-sig")
            _logger.info(f"Итоговый CSV обновлён: {agg_path} (строк: {len(combined)})")


# ---------- Основная логика ----------

def run_pipeline(dataset_key: str, dataset_cfg: dict):
    _logger.info(f"=== Запуск пайплайна для: {dataset_key} ===")
    try:
        prepare_batch(dataset_cfg)
        process_sequence(dataset_cfg)
        process_results(dataset_cfg)
    except Exception as e:
        _logger.error(f"Пайплайн {dataset_key} завершился с ошибкой: {e}")
    _logger.info(f"=== Завершение пайплайна для: {dataset_key} ===")


def main():
    parser = argparse.ArgumentParser(
        description="Параллельный запуск пайплайнов Batch API для блогов и новостей."
    )
    parser.add_argument(
        "--datasets",
        default="blogs,news",
        help="Список пайплайнов для запуска, через запятую (blogs,news)",
    )
    args = parser.parse_args()

    selected = [k.strip() for k in args.datasets.split(",") if k.strip()]
    unknown = [k for k in selected if k not in DEFAULT_DATASETS]
    if unknown:
        print(f"Неизвестные датасеты: {', '.join(unknown)}. Доступные: {', '.join(DEFAULT_DATASETS)}")
        sys.exit(1)

    threads: list[threading.Thread] = []
    for key in selected:
        t = threading.Thread(target=run_pipeline, args=(key, DEFAULT_DATASETS[key]), daemon=False)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("\n=== Все пайплайны завершены ===")


if __name__ == "__main__":
    main() 