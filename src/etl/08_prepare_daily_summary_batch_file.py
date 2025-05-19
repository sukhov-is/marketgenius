import argparse
import logging
import sys
import os
from pathlib import Path
import json
from typing import Dict, List

from dotenv import load_dotenv

# -----------------------------------------------------------
# Вспомогательные функции
# -----------------------------------------------------------

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
# Основной скрипт
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Готовит .jsonl для Batch API с ежедневными агрегированными саммари.")
    parser.add_argument("--input-jsonl", required=True, help="Файл *.jsonl с саммари чанков.")
    parser.add_argument("--prompt-file", required=True, help="Файл *_promt.txt для агрегации.")
    parser.add_argument("--output-jsonl", default="data/external/text/batch/daily_summary_batch.jsonl", help="Куда сохранить batch input файл.")
    parser.add_argument("--model", default="gpt-4.1", help="Модель OpenAI, которая будет указана в каждом запросе.")
    # Фильтрация дат
    parser.add_argument("--start-date", default=None, help="Начальная дата YYYY-MM-DD включительно.")
    parser.add_argument("--end-date", default=None, help="Конечная дата YYYY-MM-DD включительно.")
    parser.add_argument("--dates", default=None, help="Запятая-разделённый список конкретных дат вида YYYY-MM-DD, имеет приоритет над диапазоном.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY не найден. Для подготовки файла он не нужен, но может понадобиться позже.")

    input_path = Path(args.input_jsonl)
    prompt_path = Path(args.prompt_file)
    output_path = Path(args.output_jsonl)

    if not input_path.is_file():
        logger.error("Файл с чанковыми саммари не найден: %s", input_path)
        sys.exit(1)
    if not prompt_path.is_file():
        logger.error("Файл промпта не найден: %s", prompt_path)
        sys.exit(1)

    day_chunks = _group_chunk_summaries(input_path)

    # --- Фильтрация по датам ---
    if args.dates:
        specified = {d.strip() for d in args.dates.split(',') if d.strip()}
        day_chunks = {d: s for d, s in day_chunks.items() if d in specified}
        logger.info("После фильтрации по конкретным датам осталось %d дней.", len(day_chunks))
    else:
        if args.start_date or args.end_date:
            from datetime import datetime as _dt
            start_dt = _dt.fromisoformat(args.start_date) if args.start_date else None
            end_dt = _dt.fromisoformat(args.end_date) if args.end_date else None

            def _in_range(ds: str) -> bool:
                d = _dt.fromisoformat(ds)
                if start_dt and d < start_dt.date():
                    return False
                if end_dt and d > end_dt.date():
                    return False
                return True

            day_chunks = {d: s for d, s in day_chunks.items() if _in_range(d)}
            logger.info("После фильтрации по диапазону осталось %d дней.", len(day_chunks))

    if not day_chunks:
        logger.error("Нет дат, удовлетворяющих заданным условиям. Завершаем.")
        sys.exit(1)
    logger.info("Будет подготовлено %d запросов (по дням).", len(day_chunks))

    system_prompt, user_template = _load_prompt(prompt_path)

    # Создаём выходную директорию
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for date_str, summaries in sorted(day_chunks.items()):
            user_prompt = user_template.format(DATE=date_str, CHUNK_SUMMARIES="\n".join(summaries))
            request_body = {
                "model": args.model,
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

    logger.info("Файл %s успешно создан. Записано %d запросов.", output_path, written)
    print(f"Успех! Batch input файл создан: {output_path}, запросов: {written}")


if __name__ == "__main__":
    main() 