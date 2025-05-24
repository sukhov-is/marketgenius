"""
Отправляет daily_summary_*_batch.jsonl файлы (новости и/или блоги) в OpenAI Batch API,
ожидает выполнения, скачивает результаты, извлекает поле "summary" и формирует
CSV файлы для публикации в Telegram. Далее к этим CSV по дате добавляются данные
из соответствующих results_gpt_*.csv, чтобы итоговые файлы содержали как короткое
саммари для Telegram, так и рассчитанное влияние тикеров.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import json # Добавлено для парсинга строк .jsonl

import pandas as pd
from openai import OpenAI # Новый импорт
from dotenv import load_dotenv

import threading

# --- Подготовка окружения и импорты утилит ---------------------------------
project_root = Path(__file__).resolve().parent.parent  # Исправлено определение project_root
sys.path.insert(0, str(project_root))

# Удаляем импорты, связанные с gpt_batch_analyzer, так как переходим на прямые вызовы
# try:
#     from src.utils.gpt_batch_analyzer import (
#         submit_batch_job,
#         check_batch_status,
#         download_batch_results,
#         process_batch_output,
#     )
# except ImportError as e:
#     print(f"Ошибка импорта gpt_batch_analyzer: {e}")
#     print("Убедитесь, что скрипт запускается из корневой директории проекта или src/etl.")
#     sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

# Объявляем клиент OpenAI глобально, но не инициализируем сразу
client: OpenAI | None = None

# STATUS_TERMINAL больше не нужен для прямых вызовов
# STATUS_TERMINAL = {"completed", "failed", "expired", "cancelled"}


# ---------------------------------------------------------------------------
# Вспомогательные функции для прямых вызовов API
# ---------------------------------------------------------------------------

def _extract_date_from_custom_id(custom_id: str) -> str | None:
    """Извлекает YYYY-MM-DD из custom_id вида date_YYYY-MM-DD..."""
    parts = custom_id.split("_")
    if len(parts) >= 2 and parts[0] == "date":
        # Пробуем извлечь дату, даже если есть другие части в custom_id
        date_candidate = parts[1]
        try:
            # Проверка, что это действительно дата
            pd.to_datetime(date_candidate)
            return date_candidate
        except ValueError:
            _logger.warning(f"Не удалось распознать дату в custom_id '{custom_id}': '{date_candidate}' не является датой.")
            return None
    _logger.warning(f"Не удалось извлечь дату из custom_id: {custom_id}")
    return None

def call_openai_chat_completion(
    request_body: Dict[str, Any],
    custom_id: str, # Для логирования
    max_retries: int = 2,
    retry_delay_seconds: int = 10,
    request_timeout_seconds: int = 120 # Таймаут для одного запроса
) -> str | None:
    """Отправляет один запрос в OpenAI Chat Completions API и возвращает саммари."""
    if not client:
        _logger.error(f"[{custom_id}] Клиент OpenAI не инициализирован. Пропуск запроса.")
        return None

    model = request_body.get("model")
    messages = request_body.get("messages")
    temperature = request_body.get("temperature", 0.7) # Значение по умолчанию, если не указано
    response_format = request_body.get("response_format")

    if not model or not messages:
        _logger.error(f"[{custom_id}] В теле запроса отсутствуют 'model' или 'messages'.")
        return None

    current_attempt = 0
    while current_attempt <= max_retries:
        current_attempt += 1
        try:
            _logger.info(f"[{custom_id}] Попытка {current_attempt}/{max_retries + 1} вызова API для модели {model}...")
            
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format:
                api_params["response_format"] = response_format
            
            completion = client.chat.completions.create(
                **api_params,
                timeout=request_timeout_seconds 
            )
            
            # Проверяем, что есть choices и message
            if completion.choices and completion.choices[0].message:
                summary_content = completion.choices[0].message.content
                if summary_content:
                    # Попытка извлечь summary из JSON, если ответ в JSON формате
                    if response_format and response_format.get("type") == "json_object":
                        try:
                            summary_json = json.loads(summary_content)
                            actual_summary = summary_json.get("summary")
                            if actual_summary:
                                _logger.info(f"[{custom_id}] Успешно извлечено 'summary' из JSON-ответа.")
                                return str(actual_summary).strip()
                            else:
                                _logger.warning(f"[{custom_id}] Ответ в формате JSON, но ключ 'summary' не найден или пуст. Ответ: {summary_content[:200]}...")
                                return summary_content.strip() # Возвращаем весь контент, если summary нет
                        except json.JSONDecodeError:
                            _logger.warning(f"[{custom_id}] Ожидался JSON, но не удалось декодировать: {summary_content[:200]}... Возвращаем как есть.")
                            return summary_content.strip()
                    else: # Обычный текстовый ответ
                        return summary_content.strip()
                else:
                    _logger.warning(f"[{custom_id}] Получен пустой content в ответе от API.")
                    return None # или пустую строку, в зависимости от желаемого поведения
            else:
                _logger.warning(f"[{custom_id}] Ответ API не содержит ожидаемой структуры (choices или message).")

        except Exception as e:
            _logger.error(f"[{custom_id}] Ошибка API на попытке {current_attempt}: {e}")
            if current_attempt > max_retries:
                _logger.error(f"[{custom_id}] Превышено максимальное количество попыток для запроса.")
                return None
            _logger.info(f"[{custom_id}] Ожидание {retry_delay_seconds} сек перед следующей попыткой...")
            time.sleep(retry_delay_seconds)
    return None

# Функция _wait_for_completion и STATUS_TERMINAL больше не нужны и будут удалены
# ... (код _wait_for_completion удаляется) ...

# ---------------------------------------------------------------------------
# Основная логика обработки одного .jsonl файла с прямыми вызовами
# ---------------------------------------------------------------------------

def _process_jsonl_file_directly( # Переименована и изменена
    input_file: Path,
    telegram_csv: Path,
    results_csv: Path | None,
    # interval, max_retries, retry_wait, success_wait - часть этих параметров теперь в call_openai_chat_completion
    # или управляются циклом по строкам
    request_interval_seconds: int = 2 # Интервал между отдельными запросами к API
):
    """Читает .jsonl файл, для каждой строки делает прямой вызов OpenAI API,
    собирает результаты и объединяет с CSV файлами."""

    if not input_file.is_file():
        _logger.error(f"Файл с запросами не найден: {input_file}")
        return

    _logger.info(f"Начало прямой обработки файла: {input_file}")
    
    processed_data = [] # Список для хранения {'date': ..., 'summary': ...}

    line_number = 0
    successful_requests = 0
    failed_requests = 0

    with input_file.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line_number += 1
            line = line.strip()
            if not line:
                continue

            try:
                request_data = json.loads(line)
            except json.JSONDecodeError as e:
                _logger.error(f"Ошибка декодирования JSON в файле {input_file}, строка {line_number}: {e}. Строка: {line[:100]}...")
                failed_requests +=1
                continue

            custom_id = request_data.get("custom_id")
            api_body = request_data.get("body")

            if not custom_id or not api_body:
                _logger.error(f"Пропуск строки {line_number} в {input_file}: отсутствует 'custom_id' или 'body'.")
                failed_requests +=1
                continue
            
            date_str = _extract_date_from_custom_id(custom_id)
            if not date_str:
                _logger.error(f"Не удалось извлечь дату из custom_id '{custom_id}' в {input_file}, строка {line_number}. Пропуск.")
                failed_requests +=1
                continue

            summary = call_openai_chat_completion(api_body, custom_id)

            if summary is not None:
                processed_data.append({"date": date_str, "summary": summary})
                successful_requests += 1
            else:
                _logger.warning(f"Не удалось получить саммари для {custom_id} (дата: {date_str}) после всех попыток.")
                failed_requests += 1
            
            # Соблюдаем интервал между запросами, чтобы не превысить лимиты API
            if request_interval_seconds > 0:
                time.sleep(request_interval_seconds)

    _logger.info(f"Обработка файла {input_file} завершена. Успешных запросов: {successful_requests}, неудачных: {failed_requests}.")

    if not processed_data:
        _logger.warning(f"Нет успешно обработанных данных из {input_file} для записи в {telegram_csv}.")
        # Если нет новых данных, но telegram_csv существует, он не должен быть перезаписан пустым DataFrame
        # Однако, если telegram_csv не существует, то он будет создан пустым ниже, если combined_df останется пустым.
        # Это поведение соответствует предыдущей логике (если daily_df был пуст)
        # Проверим, нужно ли тут выходить или дать логике ниже обработать пустой daily_df
        # if not telegram_csv.exists(): # Если файла нет, и данных нет, то нечего создавать
        #      _logger.info(f"Файл {telegram_csv} не будет создан, так как нет данных.")
        #      return
        daily_df = pd.DataFrame(columns=['date', 'tg_summary']) # Создаем пустой DF с нужными колонками
    else:
        daily_df = pd.DataFrame(processed_data)
        daily_df.rename(columns={"summary": "tg_summary"}, inplace=True)
        daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date # Убедимся, что 'date' это объекты date
    
    # --- Дальнейшая логика объединения с существующим telegram_csv и results_csv ---
    # Эта часть кода взята из предыдущей версии функции _process_single_batch
    # и должна работать аналогично, принимая daily_df.

    combined_df_list = []
    processed_dates_from_existing = set()

    if telegram_csv.is_file():
        try:
            existing_df = pd.read_csv(telegram_csv, encoding="utf-8-sig")
            if not existing_df.empty and 'date' in existing_df.columns:
                existing_df['date'] = pd.to_datetime(existing_df['date']).dt.date
                combined_df_list.append(existing_df)
                processed_dates_from_existing = set(existing_df['date'])
        except pd.errors.EmptyDataError:
            _logger.warning(f"Файл {telegram_csv} существует, но пуст. Будет наполнен новыми данными.")
        except Exception as e:
            _logger.error(f"Не удалось прочитать {telegram_csv}: {e}. Попытка использовать только новые данные.")
    
    if not daily_df.empty:
        daily_df_new_dates_only = daily_df[~daily_df['date'].isin(processed_dates_from_existing)]
        if not daily_df_new_dates_only.empty:
            combined_df_list.append(daily_df_new_dates_only)

    if combined_df_list:
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        if 'date' in combined_df.columns:
             combined_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    elif not daily_df.empty: # Если existing_df не было или оно было пустым, но есть новые данные
        combined_df = daily_df.copy() # daily_df уже содержит только новые, уникальные по дате (если custom_id уникальны)
        # Убедимся, что дубликаты по дате из самого daily_df обработаны (на случай если в input_file были дубликаты custom_id с той же датой)
        if 'date' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    else: # combined_df_list пуст и daily_df пуст
        combined_df = pd.DataFrame() 

    if results_csv and Path(results_csv).is_file():
        try:
            results_df = pd.read_csv(results_csv, encoding="utf-8-sig")
            if 'date' in results_df.columns and not results_df.empty:
                results_df['date'] = pd.to_datetime(results_df['date']).dt.date
                results_df = results_df.drop_duplicates(subset=["date"], keep="last")
                if "summary" in results_df.columns: # summary из results_gpt_* это старое название колонки
                    results_df = results_df.drop(columns=["summary"], errors='ignore')
                if "tg_summary" in results_df.columns: # на случай если там уже есть tg_summary
                     results_df = results_df.drop(columns=["tg_summary"], errors='ignore')


                if not combined_df.empty and 'date' in combined_df.columns:
                    # Сохраняем колонку tg_summary из combined_df перед объединением, если она есть
                    tg_summary_backup = None
                    if 'tg_summary' in combined_df.columns:
                        tg_summary_backup = combined_df[['date', 'tg_summary']].copy().set_index('date')

                    combined_df = combined_df.set_index("date")
                    results_df  = results_df.set_index("date")
                    
                    # Обновляем существующие колонки и добавляем новые из results_df
                    # Колонки, которые есть в combined_df, но нет в results_df, останутся
                    # Колонки, которые есть в results_df, но нет в combined_df, добавятся
                    # Общие колонки (кроме tg_summary) будут взяты из results_df если там не NaN
                    
                    # Сначала объединим все колонки, кроме tg_summary
                    cols_to_update = results_df.columns.difference(combined_df.columns)
                    combined_df = combined_df.merge(results_df[cols_to_update], left_index=True, right_index=True, how='left')
                    
                    common_cols = results_df.columns.intersection(combined_df.columns)
                    if not common_cols.empty:
                         combined_df.update(results_df[common_cols], overwrite=True) # Обновляем общие колонки значениями из results_df

                    combined_df.reset_index(inplace=True)

                    # Восстанавливаем tg_summary из combined_df, если он был
                    if tg_summary_backup is not None:
                        # Удаляем tg_summary, который мог прийти из results_df (хотя мы его дропали)
                        if 'tg_summary' in combined_df.columns:
                            combined_df = combined_df.drop(columns=['tg_summary'])
                        # Мержим сохраненный tg_summary
                        combined_df = pd.merge(combined_df, tg_summary_backup.reset_index(), on='date', how='left')
                        
                elif combined_df.empty and not results_df.empty:
                    _logger.info(
                        f"Основной DataFrame (для {telegram_csv}) был пуст. "
                        f"Создаем его на основе данных из {results_csv}, но без tg_summary."
                    )
                    combined_df = results_df.reset_index()
                    # tg_summary в этом случае не будет, что логично
                elif results_df.empty:
                     _logger.warning(f"Файл {results_csv} пуст. Пропускаем объединение.")
                else: # combined_df не пуст, но нет колонки 'date' или results_df не содержит 'date'
                    _logger.warning(
                        f"Проблемы с колонкой 'date' в combined_df или {results_csv}. Пропускаем объединение."
                    )
            elif results_df.empty:
                _logger.warning(f"Файл {results_csv} пуст. Пропускаем объединение.")
            else:
                _logger.warning(
                    f"Файл {results_csv} не содержит колонку 'date'. Пропускаем объединение."
                )
        except Exception as e:
            _logger.error(f"Ошибка при чтении или обработке файла {results_csv}: {e}. Пропускаем объединение.")
    elif results_csv: # results_csv указан, но не найден
        _logger.warning(f"Файл {results_csv} не найден. Пропускаем объединение.")
    # else: results_csv не указан, это нормально

    if not combined_df.empty and 'date' in combined_df.columns:
        combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.date
        combined_df = combined_df.sort_values("date")
        
        # Убедимся, что колонка tg_summary существует, даже если она вся NaN
        if 'tg_summary' not in combined_df.columns:
            combined_df['tg_summary'] = pd.NA

        # Перемещение колонки 'tg_summary' на второе место, если она существует
        if 'tg_summary' in combined_df.columns and 'date' in combined_df.columns:
            cols = combined_df.columns.tolist()
            if 'tg_summary' in cols: cols.remove('tg_summary')
            if 'date' in cols: cols.remove('date')
            final_cols = ['date', 'tg_summary'] + sorted([col for col in cols if col not in ['date', 'tg_summary']]) # Сохраняем остальные колонки отсортированными
            combined_df = combined_df[final_cols]

    elif combined_df.empty:
        _logger.info(f"Итоговый DataFrame для {telegram_csv} пуст перед сохранением.")
    else:
         _logger.warning(f"Итоговый DataFrame для {telegram_csv} не содержит колонку 'date' перед сортировкой и сохранением.")

    telegram_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(telegram_csv, index=False, encoding="utf-8-sig")
    print(f"Успех! Обновлён файл {telegram_csv} (строк: {len(combined_df)})")
    _logger.info(f"Прямая обработка запросов из {input_file.name} и обновление {telegram_csv} завершены.")
    # success_wait можно оставить, если нужно сделать паузу после обработки файла
    # time.sleep(success_wait) # success_wait не определен в этой функции, убрано
    return # Убрано возвращение значения, т.к. оно не использовалось

# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Обработка .jsonl файлов с запросами к OpenAI API напрямую и формирование telegram CSV.")

    # Аргументы для input .jsonl файлов остались те же
    parser.add_argument("--news-batch", default="data/external/text/batch/daily_summary_news_batch.jsonl",
                        help="*.jsonl файл с запросами для новостей (теперь обрабатывается напрямую)")
    parser.add_argument("--blogs-batch", default="data/external/text/batch/daily_summary_blogs_batch.jsonl",
                        help="*.jsonl файл с запросами для блогов (теперь обрабатывается напрямую)")
    
    # Аргументы для output CSV файлов остались те же
    parser.add_argument("--telegram-news-csv", default="data/processed/gpt/telegram_news.csv",
                        help="CSV для итоговых данных новостей в Telegram")
    parser.add_argument("--telegram-blogs-csv", default="data/processed/gpt/telegram_blogs.csv",
                        help="CSV для итоговых данных блогов в Telegram")
    parser.add_argument("--results-news-csv", default="data/processed/gpt/results_gpt_news.csv",
                        help="CSV с результатами предыдущего анализа новостей (для объединения)")
    parser.add_argument("--results-blogs-csv", default="data/processed/gpt/results_gpt_blogs.csv",
                        help="CSV с результатами предыдущего анализа блогов (для объединения)")

    # --temp-dir больше не нужен для результатов батчей, но может быть полезен для других временных файлов если понадобятся
    # parser.add_argument("--temp-dir", default="data/processed/gpt/temp", help="Директория для временных файлов")
    
    # --interval для проверки статуса батча больше не нужен
    # parser.add_argument("--interval", type=int, default=20, help="Интервал проверки статуса batch (сек)")
    parser.add_argument("--request-interval", type=int, default=1, help="Интервал между прямыми запросами к API (сек) для избежания rate limit")
    
    parser.add_argument("--sequential", action="store_true", help="Обрабатывать файлы новостей и блогов последовательно (по умолчанию параллельно, если есть оба файла)")

    args = parser.parse_args()

    global client # Указываем, что будем изменять глобальную переменную client
    load_dotenv(project_root / ".env")
    
    # Инициализируем клиент OpenAI здесь, после загрузки .env
    try:
        # Убедимся, что OPENAI_API_KEY загружен перед инициализацией
        if not os.getenv("OPENAI_API_KEY"):
            _logger.error("Переменная окружения OPENAI_API_KEY не найдена после load_dotenv().")
            sys.exit(1)
        client = OpenAI()
        _logger.info("Клиент OpenAI успешно инициализирован.")
    except Exception as e:
        _logger.error(f"Ошибка инициализации клиента OpenAI в main(): {e}")
        # client останется None, и последующие проверки это обработают
        # или можно сделать sys.exit(1) прямо здесь, если это критично

    if not client: # Дополнительная проверка, что клиент был успешно создан
        _logger.error("Клиент OpenAI не был инициализирован. Завершение.")
        sys.exit(1)

    # temp_dir = Path(args.temp_dir) # temp_dir больше не используется для результатов батчей

    def _run_news():
        if Path(args.news_batch).is_file():
            _process_jsonl_file_directly(
                input_file=Path(args.news_batch),
                telegram_csv=Path(args.telegram_news_csv),
                results_csv=Path(args.results_news_csv),
                request_interval_seconds=args.request_interval
            )
        else:
            _logger.warning(f"Файл для новостей {args.news_batch} не найден. Пропуск обработки новостей.")


    def _run_blogs():
        if Path(args.blogs_batch).is_file():
            _process_jsonl_file_directly(
                input_file=Path(args.blogs_batch),
                telegram_csv=Path(args.telegram_blogs_csv),
                results_csv=Path(args.results_blogs_csv),
                request_interval_seconds=args.request_interval
            )
        else:
            _logger.warning(f"Файл для блогов {args.blogs_batch} не найден. Пропуск обработки блогов.")


    news_exists = Path(args.news_batch).is_file()
    blogs_exists = Path(args.blogs_batch).is_file()

    if not news_exists and not blogs_exists:
        _logger.error("Не найдены файлы для обработки ни для новостей, ни для блогов. Завершение.")
        sys.exit(1)

    if args.sequential or not (news_exists and blogs_exists): # Если sequential или только один файл для обработки
        if news_exists:
            _run_news()
        if blogs_exists:
            _run_blogs()
    else: # Параллельная обработка, если оба файла существуют и не указан флаг --sequential
        _logger.info("Запуск параллельной обработки для новостей и блогов.")
        t1 = threading.Thread(target=_run_news, daemon=True) # daemon=True, чтобы не блокировать выход, если основной поток завершится
        t2 = threading.Thread(target=_run_blogs, daemon=True)
        
        active_threads = []
        if news_exists:
            t1.start()
            active_threads.append(t1)
        if blogs_exists:
            t2.start()
            active_threads.append(t2)
        
        for t in active_threads:
            t.join()
    
    _logger.info("Обработка всех указанных файлов завершена.")

if __name__ == "__main__":
    main() 