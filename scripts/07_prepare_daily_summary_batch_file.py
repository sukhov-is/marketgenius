import argparse
import logging
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Any
import csv
from datetime import datetime, timedelta

from dotenv import load_dotenv

# -----------------------------------------------------------
# Вспомогательные функции
# -----------------------------------------------------------

def get_last_date_from_csv(csv_file_path: Path, logger: logging.Logger) -> str | None:
    """Извлекает последнюю дату из колонки 'date' CSV файла."""
    if not csv_file_path.is_file():
        logger.info("Файл CSV для определения последней даты не найден: %s", csv_file_path)
        return None

    last_date_obj: datetime.date | None = None
    try:
        with csv_file_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "date" not in reader.fieldnames:
                logger.warning("В CSV файле %s отсутствует колонка 'date'.", csv_file_path)
                return None

            for row_num, row in enumerate(reader, 1):
                date_str = row.get("date")
                if date_str:
                    try:
                        current_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                        if last_date_obj is None or current_date_obj > last_date_obj:
                            last_date_obj = current_date_obj
                    except ValueError:
                        logger.warning("Некорректный формат даты '%s' в строке %d файла %s.", date_str, row_num, csv_file_path)
                        continue
        if last_date_obj:
            return last_date_obj.strftime("%Y-%m-%d")
        logger.info("В файле CSV %s не найдено дат или файл пуст.", csv_file_path)
        return None
    except Exception as e:
        logger.error("Ошибка при чтении CSV файла %s: %s", csv_file_path, e)
        return None


def get_next_day_str(date_str: str, logger: logging.Logger) -> str | None:
    """Возвращает строку со следующим днем в формате YYYY-MM-DD."""
    try:
        current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        next_day = current_date + timedelta(days=1)
        return next_day.strftime("%Y-%m-%d")
    except ValueError:
        logger.error("Некорректный формат строки даты '%s' для get_next_day_str.", date_str)
        return None


def _extract_date(custom_id: str) -> str | None:
    """Извлекает YYYY-MM-DD из custom_id вида date_YYYY-MM-DD_chunk_N"""
    parts = custom_id.split("_")
    if len(parts) >= 3 and parts[0] == "date":
        return parts[1]
    return None


def _group_chunk_summaries(input_path: Path) -> Dict[str, List[str]]:
    """Собирает список саммари чанков по каждому дню."""
    day_to_chunks: Dict[str, List[str]] = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning("Некорректный JSON в строке %d: %s", line_num, e)
                continue

            custom_id = obj.get("custom_id")
            if not custom_id:
                continue
            date_str = _extract_date(custom_id)
            if not date_str:
                continue

            summary = obj.get("summary")
            if not summary:
                # Попытка достать из raw ответа Batch API, если ещё не распакован
                body = obj.get("response", {}).get("body", {})
                summary = body.get("choices", [{}])[0].get("message", {}).get("content")
            if not summary:
                continue

            day_to_chunks.setdefault(date_str, []).append(str(summary).strip())
    return day_to_chunks


def _load_prompt(prompt_file: Path):
    system_marker = "####################  SYSTEM  ####################"
    user_marker = "#####################  USER  #####################"
    end_marker = "##################################################"
    text = prompt_file.read_text(encoding="utf-8")
    sys_start = text.index(system_marker) + len(system_marker)
    sys_end = text.index(end_marker, sys_start)
    user_start = text.index(user_marker) + len(user_marker)
    user_end = text.index(end_marker, user_start)
    system_part = text[sys_start:sys_end].strip()
    user_part = text[user_start:user_end].strip()
    return system_part, user_part

# -----------------------------------------------------------
# Функция обработки одного источника данных
# -----------------------------------------------------------
def process_single_source(
    input_file_str: str,
    output_file_str: str,
    custom_id_prefix: str,
    prompt_path: Path,
    model_name: str,
    args_namespace: argparse.Namespace,
    logger: logging.Logger,
    start_date_to_use: str | None,
    end_date_to_use: str | None
):
    """Обрабатывает один источник данных (например, новости или блоги)."""
    input_path = Path(input_file_str)
    output_path = Path(output_file_str)

    if not input_path.is_file():
        logger.error("[%s] Файл с чанковыми саммари не найден: %s", custom_id_prefix, input_path)
        return

    logger.info("[%s] Начало обработки файла: %s. Заданный диапазон дат: Старт=%s, Конец=%s",
                 custom_id_prefix, input_path, start_date_to_use or "Не задана", end_date_to_use or "Не задана")
    day_chunks = _group_chunk_summaries(input_path)

    # --- Фильтрация по датам ---
    if args_namespace.dates:
        specified = {d.strip() for d in args_namespace.dates.split(',') if d.strip()}
        day_chunks = {d: s for d, s in day_chunks.items() if d in specified}
        logger.info("[%s] Отфильтровано по списку дат (--dates '%s'). Осталось %d дней.", custom_id_prefix, args_namespace.dates, len(day_chunks))
    else:
        if start_date_to_use or end_date_to_use:
            start_dt_obj = datetime.fromisoformat(start_date_to_use).date() if start_date_to_use else None
            end_dt_obj = datetime.fromisoformat(end_date_to_use).date() if end_date_to_use else None
            
            initial_day_count = len(day_chunks)
            def _in_range(ds: str) -> bool:
                try:
                    d_obj = datetime.fromisoformat(ds).date()
                    if start_dt_obj and d_obj < start_dt_obj:
                        return False
                    if end_dt_obj and d_obj > end_dt_obj:
                        return False
                    return True
                except ValueError:
                    logger.warning("[%s] Некорректная дата '%s' в данных, пропускается при фильтрации диапазона.", custom_id_prefix, ds)
                    return False

            day_chunks = {d: s for d, s in day_chunks.items() if _in_range(d)}
            logger.info("[%s] Отфильтровано по диапазону дат (старт: %s, конец: %s). Исходно: %d, Осталось: %d дней.",
                         custom_id_prefix, start_date_to_use or "N/A", end_date_to_use or "N/A", initial_day_count, len(day_chunks))

    if not day_chunks:
        logger.warning("[%s] Нет дат, удовлетворяющих заданным условиям. Пропускаем.", custom_id_prefix)
        return

    logger.info("[%s] Будет подготовлено %d запросов (по дням).", custom_id_prefix, len(day_chunks))

    system_prompt, user_template = _load_prompt(prompt_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for date_str, summaries in sorted(day_chunks.items()):
            user_prompt = user_template.format(CHUNK_SUMMARIES="\\n".join(summaries))
            request_body: Dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            batch_line = {
                "custom_id": f"date_{date_str}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body,
            }
            fout.write(json.dumps(batch_line, ensure_ascii=False) + "\n")
            written += 1

    logger.info("[%s] Файл %s успешно создан. Записано %d запросов.", custom_id_prefix, output_path, written)
    print(f"[{custom_id_prefix}] Успех! Batch input файл создан: {output_path}, запросов: {written}")


# -----------------------------------------------------------
# Основной скрипт
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Готовит .jsonl для Batch API с ежедневными агрегированными саммари для новостей и/или блогов.")
    
    # Аргументы для новостей
    parser.add_argument("--input-news-jsonl", default="data/processed/gpt/gpt_news_history.jsonl", help="Файл *.jsonl с саммари чанков новостей.")
    parser.add_argument("--output-news-jsonl", default="data/external/text/batch/daily_summary_news_batch.jsonl", help="Куда сохранить batch input файл для новостей.")
    
    # Аргументы для блогов
    parser.add_argument("--input-blogs-jsonl", default="data/processed/gpt/gpt_blogs_history.jsonl", help="Файл *.jsonl с саммари чанков блогов/мнений экспертов.")
    parser.add_argument("--output-blogs-jsonl", default="data/external/text/batch/daily_summary_blogs_batch.jsonl", help="Куда сохранить batch input файл для блогов.")

    # Общие аргументы
    parser.add_argument("--prompt-file", default="src/prompts/daily_aggregate_promt.txt", help="Файл *_promt.txt для агрегации (общий для всех источников).")
    parser.add_argument("--model", default="gpt-4.1", help="Модель OpenAI, которая будет указана в каждом запросе.")
    
    # Фильтрация дат (общая для всех источников)
    parser.add_argument("--start-date", default=None, help="Начальная дата YYYY-MM-DD включительно.")
    parser.add_argument("--end-date", default=None, help="Конечная дата YYYY-MM-DD включительно.")
    parser.add_argument("--dates", default=None, help="Запятая-разделённый список конкретных дат вида YYYY-MM-DD, имеет приоритет над диапазоном.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY не найден. Для подготовки файла он не нужен, но может понадобиться позже.")

    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
        logger.error("Файл общего промпта не найден: %s. Завершение.", prompt_path)
        sys.exit(1)

    # --- Определяем конечную дату по умолчанию ---
    effective_end_date = args.end_date
    if effective_end_date is None:
        now = datetime.now()
        if now.hour >= 19:  # 19:00
            effective_end_date = now.strftime("%Y-%m-%d")
            logger.info("Конечная дата не указана (--end-date), используется сегодняшняя дата (т.к. время >= 19:00): %s", effective_end_date)
        else:
            effective_end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info("Конечная дата не указана (--end-date), используется вчерашняя дата (т.к. время < 19:00): %s", effective_end_date)
    else:
        logger.info("Используется конечная дата из аргумента --end-date: %s", effective_end_date)

    processed_any = False

    # Обработка новостей
    if args.input_news_jsonl and Path(args.input_news_jsonl).is_file():
        effective_news_start_date = args.start_date
        if effective_news_start_date is None:
            news_history_csv = project_root / "data/processed/gpt/telegram_news.csv"
            last_news_date_str = get_last_date_from_csv(news_history_csv, logger)
            if last_news_date_str:
                effective_news_start_date = get_next_day_str(last_news_date_str, logger)
                if effective_news_start_date:
                    logger.info("Начальная дата для новостей не указана (--start-date), рассчитана по %s: %s", news_history_csv.name, effective_news_start_date)
                else:
                    logger.warning("Не удалось рассчитать следующую начальную дату для новостей по %s.", news_history_csv.name)
            else:
                logger.info("Не удалось найти последнюю дату в %s для новостей. Начальная дата для новостей не будет ограничена.", news_history_csv.name)
        else:
            logger.info("Используется начальная дата из аргумента --start-date для новостей: %s", effective_news_start_date)
        
        process_single_source(
            input_file_str=args.input_news_jsonl,
            output_file_str=args.output_news_jsonl,
            custom_id_prefix="news",
            prompt_path=prompt_path,
            model_name=args.model,
            args_namespace=args,
            logger=logger,
            start_date_to_use=effective_news_start_date,
            end_date_to_use=effective_end_date
        )
        processed_any = True
    elif args.input_news_jsonl:
        logger.warning("Файл для новостей %s не найден, но был указан. Пропускаем новости.", args.input_news_jsonl)

    # Обработка блогов
    if args.input_blogs_jsonl and Path(args.input_blogs_jsonl).is_file():
        effective_blogs_start_date = args.start_date
        if effective_blogs_start_date is None:
            blogs_history_csv = project_root / "data/processed/gpt/telegram_blogs.csv"
            last_blogs_date_str = get_last_date_from_csv(blogs_history_csv, logger)
            if last_blogs_date_str:
                effective_blogs_start_date = get_next_day_str(last_blogs_date_str, logger)
                if effective_blogs_start_date:
                    logger.info("Начальная дата для блогов не указана (--start-date), рассчитана по %s: %s", blogs_history_csv.name, effective_blogs_start_date)
                else:
                    logger.warning("Не удалось рассчитать следующую начальную дату для блогов по %s.", blogs_history_csv.name)
            else:
                logger.info("Не удалось найти последнюю дату в %s для блогов. Начальная дата для блогов не будет ограничена.", blogs_history_csv.name)
        else:
            logger.info("Используется начальная дата из аргумента --start-date для блогов: %s", effective_blogs_start_date)

        process_single_source(
            input_file_str=args.input_blogs_jsonl,
            output_file_str=args.output_blogs_jsonl,
            custom_id_prefix="blogs",
            prompt_path=prompt_path,
            model_name=args.model,
            args_namespace=args,
            logger=logger,
            start_date_to_use=effective_blogs_start_date,
            end_date_to_use=effective_end_date
        )
        processed_any = True
    elif args.input_blogs_jsonl:
         logger.warning("Файл для блогов %s не найден, но был указан. Пропускаем блоги.", args.input_blogs_jsonl)

    if not processed_any:
        logger.error("Не было обработано ни одного источника данных (проверьте пути к файлам --input-*-jsonl). Завершение.")
        sys.exit(1)
    
    logger.info("Обработка завершена.")


if __name__ == "__main__":
    main() 