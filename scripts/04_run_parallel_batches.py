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
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

# Корень проекта — три уровня вверх от текущего файла (src/etl)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Добавляем PROJECT_ROOT в sys.path, чтобы корректно работали импорты вида 'from src...'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ticker_cleaner import clean_results_file
# Общие настройки для двух пайплайнов (можно расширить через аргументы CLI)
DEFAULT_DATASETS = {
    "blogs": {
        "input_csv": "data/external/text/representative_blogs.csv",
        "prompt_type": "blogs",
        "file_prefix": "batch_input_blogs_history_part",
        "batch_input_jsonl": "data/external/text/batch/batch_input_blogs_history.jsonl",
        "results_dir": "data/processed/gpt/blogs",
        "output_csv": "data/processed/gpt/results_gpt_blogs.csv",
        "cluster_dir": "data/external/text/blogs_clusters",
        "history_jsonl": "data/processed/gpt/gpt_blogs_history.jsonl",
    },
    "news": {
        "input_csv": "data/external/text/representative_news.csv",
        "prompt_type": "news",
        "file_prefix": "batch_input_news_history_part",
        "batch_input_jsonl": "data/external/text/batch/batch_input_news_history.jsonl",
        "results_dir": "data/processed/gpt/news",
        "output_csv": "data/processed/gpt/results_gpt_news.csv",
        "cluster_dir": "data/external/text/news_clusters",
        "history_jsonl": "data/processed/gpt/gpt_news_history.jsonl",
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
    # Возвращаем stdout, чтобы вызывающий код мог при необходимости разобрать вывод
    return result.stdout


def update_jsonl_history(history_file_path: Path, new_results_file_path: Path):
    """
    Обновляет исторический JSONL файл новыми результатами.
    Записи с совпадающим 'custom_id' в файле истории заменяются.
    Новые записи добавляются.
    """
    _logger.info(f"Обновление файла истории {history_file_path} данными из {new_results_file_path}")
    history_data_map = {}

    # 1. Чтение существующей истории
    if history_file_path.is_file():
        try:
            with history_file_path.open("r", encoding="utf-8") as f_hist:
                for line_num, line in enumerate(f_hist, 1):
                    try:
                        record = json.loads(line)
                        custom_id = record.get("custom_id")
                        if custom_id:
                            history_data_map[custom_id] = record
                        else:
                            _logger.warning(
                                f"Запись в {history_file_path} (строка {line_num}) не имеет 'custom_id'. Пропущена."
                            )
                    except json.JSONDecodeError:
                        _logger.warning(
                            f"Ошибка декодирования JSON в {history_file_path} (строка {line_num}). Пропущена."
                        )
        except Exception as e:
            _logger.error(f"Ошибка при чтении файла истории {history_file_path}: {e}")
            # В случае ошибки чтения основного файла истории, лучше не продолжать,
            # чтобы не потерять его содержимое.
            return

    # 2. Обработка новых результатов
    if new_results_file_path.is_file():
        try:
            with new_results_file_path.open("r", encoding="utf-8") as f_new:
                for line_num, line in enumerate(f_new, 1):
                    try:
                        record = json.loads(line)
                        custom_id = record.get("custom_id")
                        if custom_id:
                            if custom_id in history_data_map:
                                _logger.info(f"Замена записи с custom_id='{custom_id}' в {history_file_path.name}")
                            else:
                                _logger.info(f"Добавление новой записи с custom_id='{custom_id}' в {history_file_path.name}")
                            history_data_map[custom_id] = record
                        else:
                            _logger.warning(
                                f"Запись в {new_results_file_path} (строка {line_num}) не имеет 'custom_id'. Пропущена."
                            )
                    except json.JSONDecodeError:
                        _logger.warning(
                            f"Ошибка декодирования JSON в {new_results_file_path} (строка {line_num}). Пропущена."
                        )
        except Exception as e:
            _logger.error(f"Ошибка при чтении файла новых результатов {new_results_file_path}: {e}")
            # Если новые результаты не читаются, то и обновлять нечем.
            return
    else:
        _logger.warning(f"Файл новых результатов {new_results_file_path} не найден. Обновление JSONL истории не будет выполнено.")
        return


    # 3. Запись обновленной истории
    # Создаем родительскую директорию, если она не существует
    history_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_file_path = history_file_path.with_suffix(history_file_path.suffix + ".tmp")
    try:
        with temp_file_path.open("w", encoding="utf-8") as f_out:
            for record in history_data_map.values():
                # Записываем каждую запись на отдельной строке, как того требует формат JSONL
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        shutil.move(str(temp_file_path), str(history_file_path))
        _logger.info(f"Файл истории {history_file_path} успешно обновлен. Всего записей: {len(history_data_map)}.")
    except Exception as e:
        _logger.error(f"Ошибка при записи обновленного файла истории {history_file_path}: {e}")
        if temp_file_path.is_file():
            try:
                os.remove(temp_file_path)
            except OSError as e_rm:
                _logger.error(f"Не удалось удалить временный файл {temp_file_path}: {e_rm}")


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
        "",  # передаём пустую строку, чтобы batch_info файлы не сохранялись
        "--prompt-type",
        dataset_cfg["prompt_type"],
    ]
    # Запускаем подпроцесс и пытаемся поймать batch_id из его вывода
    sp_output = run_subprocess(cmd)

    # Ищем строку вида "BATCH_SUMMARY_JSON>>> { ... }"
    if sp_output:
        for _line in sp_output.splitlines():
            if _line.startswith("BATCH_SUMMARY_JSON>>>"):
                try:
                    summary_json = _line.split(">>>", 1)[1].strip()
                    summary_data = json.loads(summary_json)
                    dataset_cfg["batch_id"] = summary_data.get("batch_id")
                    _logger.info(
                        f"Получен batch_id={dataset_cfg.get('batch_id')} из вывода subprocess для {dataset_cfg['prompt_type']}"
                    )
                except Exception as e:
                    _logger.warning(f"Не удалось распарсить BATCH_SUMMARY_JSON из вывода: {e}")
                break


def process_results(dataset_cfg: dict):
    """Шаг 3: обработка результатов и дозапись к CSV"""
    save_info_dir = Path("data/external/text/batch")

    # --- НАЧАЛО: Получение имени актуального part1 файла ---
    current_part1_path_str = dataset_cfg.get("part1_path")
    if not current_part1_path_str:
        _logger.error(f"Ключ 'part1_path' отсутствует в конфигурации для {dataset_cfg.get('prompt_type', 'N/A')}. Невозможно определить актуальный batch_info файл.")
        return # или raise Exception, в зависимости от желаемого поведения
    
    current_part1_filename = Path(current_part1_path_str).name
    _logger.info(f"Ожидается обработка результатов для файла: {current_part1_filename}")
    # === Попытка обработать результаты напрямую по batch_id, если он уже сохранён в dataset_cfg ===
    direct_batch_id = dataset_cfg.get("batch_id")
    if direct_batch_id:
        _logger.info(f"Используем batch_id из конфигурации: {direct_batch_id}")

        batch_id = direct_batch_id  # для совместимости со старыми переменными
        results_file = Path(dataset_cfg["results_dir"]) / f"results_{batch_id}.jsonl"
        errors_file = Path(dataset_cfg["results_dir"]) / f"errors_{batch_id}.jsonl"

        if not results_file.is_file():
            _logger.warning(
                f"Файл результатов не найден по пути {results_file}. Переходим к поиску batch_info_* как резервному варианту."
            )
        else:
            # --- Начало: обновление исторического JSONL ---
            history_jsonl_path_str = dataset_cfg.get("history_jsonl")
            if history_jsonl_path_str:
                target_history_jsonl_path = PROJECT_ROOT / history_jsonl_path_str
                _logger.info(
                    f"Подготовка к обновлению JSONL истории для {dataset_cfg['prompt_type']}: {target_history_jsonl_path}"
                )
                update_jsonl_history(target_history_jsonl_path, results_file)
            else:
                _logger.warning(
                    f"Ключ 'history_jsonl' не найден в конфигурации для {dataset_cfg['prompt_type']}. Пропуск обновления JSONL истории."
                )

            # 1) Сохраняем «сырые» результаты без объединения
            cluster_dir = Path(dataset_cfg["cluster_dir"])
            cluster_dir.mkdir(parents=True, exist_ok=True)
            raw_csv_path = cluster_dir / f"results_gpt_{dataset_cfg['prompt_type']}.csv"

            # Удаляем файл, если он уже существовал в рамках текущей сессии
            if raw_csv_path.is_file():
                try:
                    os.remove(raw_csv_path)
                    _logger.info(
                        f"Удалён существующий файл {raw_csv_path.name} перед обработкой батча {batch_id} ({dataset_cfg['prompt_type']})"
                    )
                except OSError as e_rm:
                    _logger.error(
                        f"Не удалось удалить {raw_csv_path.name} для батча {batch_id}: {e_rm}. Результаты могут быть некорректными."
                    )

            def _build_cmd(output_path: str, append: bool = False) -> list[str]:
                _cmd = [
                    sys.executable,
                    str(PROJECT_ROOT / "src/etl/06_process_batch_results.py"),
                    "--results-file",
                    str(results_file),
                ]
                if errors_file.is_file():
                    _cmd += ["--errors-file", str(errors_file)]
                _cmd += ["--output-csv", output_path]
                if append:
                    _cmd.append("--append")
                return _cmd

            # Конвертируем JSONL → CSV
            run_subprocess(_build_cmd(str(raw_csv_path)))

            # Очищаем тикеры
            try:
                cleaned_df = clean_results_file(
                    input_csv=raw_csv_path,
                    output_csv=raw_csv_path,
                )
                _logger.info(
                    f"Файл {raw_csv_path.name} очищен и содержит {len(cleaned_df.columns)} колонок. Строк: {len(cleaned_df)}"
                )
            except Exception as e:
                _logger.error(f"Не удалось очистить файл {raw_csv_path}: {e}")
                cleaned_df = None

            # Дозапись в агрегированный CSV
            if cleaned_df is not None and not cleaned_df.empty:
                agg_path = Path(dataset_cfg["output_csv"])
                combined_df_to_save = None

                if agg_path.is_file():
                    try:
                        _logger.info(f"Чтение существующего файла: {agg_path}")
                        existing_df = pd.read_csv(agg_path)

                        # Условное удаление последней строки, если дата совпадает
                        can_compare_dates = (
                            not existing_df.empty and 'date' in existing_df.columns and 'date' in cleaned_df.columns and not cleaned_df.empty
                        )
                        if can_compare_dates:
                            last_existing_date_val = existing_df['date'].iloc[-1]
                            if cleaned_df['date'].astype(str).isin([str(last_existing_date_val)]).any():
                                existing_df = existing_df.iloc[:-1]
                                _logger.info("Удалена последняя строка существующего файла из-за совпадающей даты.")

                        combined_df_to_save = pd.concat([existing_df, cleaned_df], ignore_index=True)
                    except Exception as e:
                        _logger.error(f"Ошибка объединения CSV: {e}. Файл будет перезаписан.")
                        combined_df_to_save = cleaned_df
                else:
                    combined_df_to_save = cleaned_df

                try:
                    agg_path.parent.mkdir(parents=True, exist_ok=True)
                    combined_df_to_save.to_csv(agg_path, index=False, encoding="utf-8-sig")
                    _logger.info(f"Агрегированный CSV обновлён: {agg_path} (строк: {len(combined_df_to_save)})")
                except Exception as e_write:
                    _logger.error(f"Не удалось записать итоговый CSV: {e_write}")
            else:
                _logger.info("Нет данных для добавления в агрегированный CSV (cleaned_df пуст или None).")

            _logger.info(
                f"Успешно обработаны и добавлены результаты для батча {batch_id} (по batch_id, без batch_info файлов)."
            )
            return  # завершили обработку напрямую, выходим
    else:
        _logger.error(
            f"batch_id отсутствует в конфигурации для {dataset_cfg['prompt_type']}. Обработка результатов невозможна."
        )
        return


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