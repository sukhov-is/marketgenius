import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from openai import OpenAI # Добавим импорт OpenAI для будущих функций

# Импортируем класс GPTNewsAnalyzer, чтобы использовать его методы и атрибуты
# Предполагается, что gpt_analyzer.py находится в том же каталоге
try:
    from .gpt_analyzer import GPTNewsAnalyzer
except ImportError:
    # Если запускается как скрипт, а не модуль
    from gpt_analyzer import GPTNewsAnalyzer


_logger = logging.getLogger(__name__)


def prepare_batch_input_file(
    analyzer: GPTNewsAnalyzer,
    df: pd.DataFrame,
    date_col: str,
    text_col: str,
    title_col: str,
    prompt_type: str,
    output_file_path: str | os.PathLike,
) -> int:
    """
    Готовит .jsonl файл для OpenAI Batch API на основе DataFrame и настроек анализатора.

    Args:
        analyzer: Экземпляр GPTNewsAnalyzer для доступа к промптам, модели и т.д.
        df: DataFrame с данными для анализа.
        date_col: Название колонки с датой.
        text_col: Название колонки с текстом новости/сообщения.
        title_col: Название колонки с заголовком/источником.
        prompt_type: Тип используемого промпта (ключ в analyzer._prompts).
        output_file_path: Путь для сохранения .jsonl файла.

    Returns:
        Количество запросов, записанных в файл.
    """
    output_file_path = Path(output_file_path)
    _logger.info(f"Начало подготовки файла для Batch API: {output_file_path}")

    if prompt_type not in analyzer._prompts:
        _logger.error(f"Тип промпта '{prompt_type}' не найден в анализаторе.")
        raise ValueError(f"Тип промпта '{prompt_type}' не найден. Доступные: {list(analyzer._prompts.keys())}")

    prompt_template = analyzer._prompts[prompt_type]
    requests_written = 0
    total_messages = 0

    # Разделяем шаблон промпта на системную и пользовательскую части
    system_marker = "####################  SYSTEM  ####################"
    user_marker = "#####################  USER  #####################"
    end_marker = "##################################################" # Используем общий маркер конца

    try:
        # Извлекаем системную часть
        system_start_idx = prompt_template.index(system_marker) + len(system_marker)
        system_end_idx = prompt_template.index(end_marker, system_start_idx)
        system_template_part = prompt_template[system_start_idx:system_end_idx].strip()

        # Извлекаем пользовательскую часть
        user_start_idx = prompt_template.index(user_marker) + len(user_marker)
        # Ищем следующий end_marker после начала пользовательской части
        user_end_idx = prompt_template.index(end_marker, user_start_idx)
        user_template_part = prompt_template[user_start_idx:user_end_idx].strip()

        # Удаляем старое преждевременное форматирование formatted_system_prompt
        # # Форматируем системную часть один раз (здесь только тикеры)
        # formatted_system_prompt = system_template_part.format(
        #     TICKERS_AND_INDICES=analyzer.tickers_block
        #     # Добавляем "пустышки" для других возможных ключей, чтобы избежать KeyError,
        #     # если они случайно окажутся в системной части шаблона
        #     # DATE="",
        #     # NEWS_LINES=""
        # )
    except (ValueError, KeyError, IndexError) as e:
        _logger.error(f"Ошибка разбора или форматирования шаблона промпта '{prompt_type}': {e}. "
                      f"Убедитесь, что маркеры '{system_marker}', '{user_marker}', '{end_marker}' "
                      f"и плейсхолдеры (как минимум {{TICKERS_AND_INDICES}} в системной части) присутствуют.")
        raise ValueError(f"Ошибка обработки шаблона промпта '{prompt_type}'.") from e

    # Используем 'w' режим и кодировку utf-8
    with output_file_path.open("w", encoding="utf-8") as f:
        for date, group in df.groupby(date_col):
            messages: List[Tuple[str, str]] = list(zip(group[title_col], group[text_col]))
            day_str = str(date) # Преобразуем дату в строку
            total_messages += len(messages)

            if not messages:
                _logger.warning(f"Нет сообщений для даты: {day_str}")
                continue # Переходим к следующей дате

            try:
                # Используем существующий метод для разделения на чанки
                chunks = analyzer._split_into_chunks(messages, prompt_type)
            except Exception as e:
                _logger.error(f"Ошибка при разделении на чанки для даты {day_str}: {e}")
                continue # Пропускаем этот день

            for i, chunk in enumerate(chunks):
                custom_id = f"date_{day_str}_chunk_{i+1}"
                # Важно: Новые строки внутри сообщений должны быть экранированы для JSON
                # Заменяем переносы строк для надежности
                formatted_messages = "\\n".join(f"{title.replace(chr(10), ' ').replace(chr(13), '')} : {text.replace(chr(10), ' ').replace(chr(13), '')}" for title, text in chunk)

                # --- Новая логика: Поиск доп. тикеров для чанка --- 
                try:
                    # Используем метод из экземпляра analyzer
                    additional_tickers = analyzer._find_additional_tickers(chunk) 
                    current_tickers_block = analyzer.tickers_block
                    if additional_tickers:
                        additional_lines = [
                            analyzer._all_tickers_descriptions.get(ticker, f"{ticker} : Описание не найдено")
                            for ticker in additional_tickers
                        ]
                        current_tickers_block = analyzer.tickers_block + "\n" + "\n".join(additional_lines)
                        _logger.info(f"Для чанка {custom_id} добавлены тикеры: {additional_tickers}")

                    # Форматируем системную часть с актуальным блоком тикеров
                    # Используем оригинальный system_template_part для форматирования на каждой итерации чанка
                    formatted_system_prompt_chunk = system_template_part.format(
                         TICKERS_AND_INDICES=current_tickers_block
                         # Добавляем "пустышки" для других возможных ключей, если они присутствуют
                         # в system_template_part и не должны вызывать ошибок, но в данном
                         # случае только TICKERS_AND_INDICES актуален для системной части.
                         # DATE="", # Пример, если бы DATE был в system_template_part
                         # NEWS_LINES="" # Пример, если бы NEWS_LINES был в system_template_part
                    )
                except AttributeError as e:
                     _logger.error(f"Ошибка доступа к атрибутам analyzer: {e}. Убедитесь, что экземпляр analyzer содержит методы/атрибуты _find_additional_tickers, tickers_block, _all_tickers_descriptions.")
                     continue # Пропускаем чанк
                except KeyError as e: # Явно ловим KeyError, который мог возникать здесь
                    _logger.error(f"Ошибка форматирования системного промпта (KeyError) для {custom_id}: {e}. Проверьте плейсхолдеры в системной части шаблона и передаваемые аргументы.")
                    continue # Пропускаем чанк
                except Exception as e:
                     _logger.error(f"Ошибка при поиске доп. тикеров или форматировании системного промпта для {custom_id}: {e}")
                     continue # Пропускаем чанк
                 # --- Конец новой логики ---

                # Формируем пользовательскую часть промпта для этого чанка
                try:
                    # Используем user_template_part вместо всего prompt_template
                    prompt_content_user = user_template_part.format(
                        DATE=day_str,
                        NEWS_LINES=formatted_messages,
                        # Добавляем "пустышку" для TICKERS, если он случайно есть в user части
                        # TICKERS_AND_INDICES=""
                    )
                except KeyError as e:
                    _logger.error(f"Ошибка форматирования пользовательской части промпта {prompt_type} для {custom_id}: отсутствует ключ {e}")
                    continue # Пропускаем этот чанк

                # Создаем тело запроса для Chat Completions API с разделенными ролями
                request_body = {
                    "model": analyzer.model,
                    "messages": [
                        {"role": "system", "content": formatted_system_prompt_chunk}, # Используем отформатированную системную часть ДЛЯ ЧАНКА
                        {"role": "user", "content": prompt_content_user}      # Используем отформатированную пользовательскую часть
                    ],
                    "temperature": 0, # Обычно 0 для задач анализа
                    "response_format": {"type": "json_object"}, # <--- Добавлено для Batch API
                }

                # Создаем полную строку для .jsonl файла
                batch_request_line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body,
                }

                # Записываем JSON строку в файл
                try:
                    # Используем ensure_ascii=False для корректной записи кириллицы
                    f.write(json.dumps(batch_request_line, ensure_ascii=False) + '\n')
                    requests_written += 1
                except Exception as e:
                    _logger.error(f"Ошибка записи JSON для {custom_id}: {e}")
                    # Решаем, стоит ли прерывать весь процесс или пропустить только эту строку
                    # continue # Пропустить эту строку
                    # raise # Прервать выполнение, если ошибка критична

    _logger.info(f"Файл {output_file_path} успешно создан.")
    _logger.info(f"Всего обработано сообщений: {total_messages}")
    _logger.info(f"Всего запросов записано в файл: {requests_written}")

    return requests_written


def submit_batch_job(
    input_file_path: str | os.PathLike,
    metadata: Dict[str, str] | None = None,
) -> str | None:
    """
    Загружает подготовленный .jsonl файл в OpenAI и создает пакетное задание.

    Args:
        input_file_path: Путь к .jsonl файлу с запросами.
        metadata: Опциональный словарь с метаданными для пакета.

    Returns:
        ID созданного пакета (batch_id) или None в случае ошибки.
    """
    client = OpenAI() # Инициализируем клиент, ключ берется из окружения
    input_file_path = Path(input_file_path)

    if not input_file_path.is_file():
        _logger.error(f"Ошибка: Входной файл для Batch API не найден: {input_file_path}")
        return None

    _logger.info(f"Загрузка файла {input_file_path} в OpenAI...")
    try:
        batch_input_file = client.files.create(
            file=input_file_path.open("rb"), # Открываем в бинарном режиме
            purpose="batch"
        )
        _logger.info(f"Файл успешно загружен. File ID: {batch_input_file.id}")
    except Exception as e:
        _logger.error(f"Не удалось загрузить файл в OpenAI: {e}", exc_info=True)
        return None

    _logger.info(f"Создание пакетного задания (Batch Job) для File ID: {batch_input_file.id}...")
    try:
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata # Добавляем метаданные, если они есть
        )
        _logger.info(f"Пакетное задание успешно создано. Batch ID: {batch_job.id}")
        _logger.info(f"Статус задания: {batch_job.status}")
        return batch_job.id
    except Exception as e:
        _logger.error(f"Не удалось создать пакетное задание в OpenAI: {e}", exc_info=True)
        # Попытка удалить загруженный файл, если пакет не создан
        try:
            client.files.delete(batch_input_file.id)
            _logger.info(f"Загруженный файл {batch_input_file.id} удален из-за ошибки создания пакета.")
        except Exception as delete_e:
            _logger.warning(f"Не удалось удалить файл {batch_input_file.id} после ошибки создания пакета: {delete_e}")
        return None


def check_batch_status(batch_id: str) -> Dict[str, Any] | None:
    """
    Проверяет статус пакетного задания по его ID.

    Args:
        batch_id: Идентификатор пакетного задания.

    Returns:
        Словарь с информацией о пакете (Batch object) или None в случае ошибки.
    """
    _logger.info(f"Проверка статуса для Batch ID: {batch_id}...")
    client = OpenAI()
    try:
        batch_info = client.batches.retrieve(batch_id)
        _logger.info(f"Статус задания {batch_id}: {batch_info.status}")
        # Возвращаем весь объект, так как он содержит ID файлов с результатами
        return batch_info.to_dict() # Преобразуем в словарь для совместимости
    except Exception as e:
        _logger.error(f"Не удалось получить статус для Batch ID {batch_id}: {e}", exc_info=True)
        return None


def download_batch_results(
    batch_info: Dict[str, Any],
    output_dir: str | os.PathLike,
) -> Tuple[Path | None, Path | None]:
    """
    Скачивает файлы результатов и ошибок для завершенного пакетного задания.

    Args:
        batch_info: Словарь с информацией о пакете (результат check_batch_status).
        output_dir: Директория для сохранения скачанных файлов.

    Returns:
        Кортеж с путями к файлам (results_path, errors_path). Путь будет None,
        если соответствующий файл не найден или задание не завершено.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Убедимся, что директория существует

    batch_id = batch_info.get("id")
    status = batch_info.get("status")

    if not batch_id:
        _logger.error("Не удалось получить ID пакета из предоставленных данных.")
        return None, None

    if status != "completed":
        _logger.warning(f"Задание {batch_id} еще не завершено (статус: {status}). Результаты пока не могут быть скачаны.")
        return None, None

    client = OpenAI()
    results_file_path: Path | None = None
    errors_file_path: Path | None = None

    # Скачивание файла с результатами
    output_file_id = batch_info.get("output_file_id")
    if output_file_id:
        results_file_path = output_dir / f"results_{batch_id}.jsonl"
        _logger.info(f"Скачивание файла результатов (ID: {output_file_id}) в {results_file_path}...")
        try:
            file_response = client.files.content(output_file_id)
            results_file_path.write_text(file_response.text, encoding='utf-8')
            _logger.info("Файл результатов успешно скачан.")
        except Exception as e:
            _logger.error(f"Не удалось скачать файл результатов {output_file_id}: {e}", exc_info=True)
            results_file_path = None # Сбрасываем путь при ошибке
    else:
        _logger.warning(f"ID файла результатов (output_file_id) не найден для задания {batch_id}.")

    # Скачивание файла с ошибками
    error_file_id = batch_info.get("error_file_id")
    if error_file_id:
        errors_file_path = output_dir / f"errors_{batch_id}.jsonl"
        _logger.info(f"Скачивание файла ошибок (ID: {error_file_id}) в {errors_file_path}...")
        try:
            file_response = client.files.content(error_file_id)
            errors_file_path.write_text(file_response.text, encoding='utf-8')
            _logger.info("Файл ошибок успешно скачан.")
        except Exception as e:
            _logger.error(f"Не удалось скачать файл ошибок {error_file_id}: {e}", exc_info=True)
            errors_file_path = None # Сбрасываем путь при ошибке
    else:
        # Это нормально, если ошибок не было
        _logger.info(f"Файл ошибок (error_file_id) не найден для задания {batch_id} (вероятно, ошибок не было).")

    return results_file_path, errors_file_path


# --- Вспомогательные функции для парсинга JSON (адаптировано из GPTNewsAnalyzer) ---

def _repair_json(text: str) -> str:
    """Грубый фикс JSON: отрезаем всё до первой '{' и после последней '}'"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start > end:
        _logger.warning("Не удалось найти '{' и '}' для исправления JSON.")
        return text
    return text[start : end + 1]

def _safe_json(raw: str | None) -> dict:
    """Гарантирует валидный JSON, пытаемся поправить мелкие ошибки."""
    if raw is None:
        # Если контента нет, возвращаем структуру ошибки
        return {"summary": "ERROR: No content received", "impact": {}}

    # Предварительная обработка для удаления знака '+' у чисел, т.к. это невалидный JSON
    # Пример: "TICKER": +1  -> "TICKER": 1
    # Простая замена может быть небезопасной, если ": +" встречается в строковых значениях, но для данного случая может подойти.
    # Более надежно было бы использовать regex, но для простоты начнем с этого.
    # Важно: Сначала заменяем ": +" (с пробелом), потом ":+" (без пробела), чтобы покрыть оба варианта.
    # И добавляем пробел после двоеточия, если его там не было, чтобы ":+1" превратилось в ": 1", а не ":1"
    processed_raw_plus_fixed = raw.replace(': +', ': ').replace(':+', ': ')

    # 1. Попытка с экранированием символов новой строки (предпочтительный метод)
    # Экранируем уже обработанную строку (processed_raw_plus_fixed), где исправлен '+'
    escaped_raw = processed_raw_plus_fixed.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\r')
    try:
        return json.loads(escaped_raw)
    except json.JSONDecodeError as e1:
        _logger.warning(f"Парсинг JSON не удался после исправления '+' и экранирования НС: {e1}.")
        _logger.debug(f"Исходная RAW строка: <<<\n{raw}\n>>>")
        _logger.debug(f"Строка после исправления '+' (перед экранированием НС): <<<\n{processed_raw_plus_fixed}\n>>>")
        _logger.debug(f"Строка после экранирования НС (неудачная попытка): <<<\n{escaped_raw}\n>>>")

        # 2. Попытка с удалением символов новой строки из строки, где уже исправлен '+'
        _logger.info("Попытка парсинга JSON после удаления символов новой строки (из строки с исправленным '+').")
        deleted_newlines_raw = processed_raw_plus_fixed.replace('\r\n', '').replace('\n', '').replace('\r', '')
        try:
            parsed_obj = json.loads(deleted_newlines_raw)
            _logger.warning(f"JSON успешно разобран после исправления '+' и УДАЛЕНИЯ символов новой строки. Исходная ошибка была: {e1}. Содержимое могло измениться.")
            return parsed_obj
        except json.JSONDecodeError as e2:
            _logger.warning(f"Парсинг JSON не удался после исправления '+' и удаления НС: {e2}.")
            _logger.debug(f"Строка после удаления НС (неудачная попытка): <<<\n{deleted_newlines_raw}\n>>>")

            # 3. Попытка с агрессивным исправлением (_repair_json) для строки, где уже исправлен '+'
            _logger.info("Попытка парсинга JSON после агрессивного исправления (_repair_json) строки с исправленным '+'.")
            repaired_after_plus_fix_raw = _repair_json(processed_raw_plus_fixed) 
            try:
                parsed_obj = json.loads(repaired_after_plus_fix_raw)
                _logger.warning(f"JSON успешно разобран после исправления '+' и агрессивного исправления (_repair_json). Исходная ошибка была: {e1}. Содержимое могло значительно измениться.")
                return parsed_obj
            except Exception as e3:
                _logger.warning(f"Парсинг JSON не удался после исправления '+' и _repair_json: {e3}. Попробуем _repair_json на исходной строке.")
                
                # 4. Попытка с агрессивным исправлением (_repair_json) для САМОЙ ИСХОДНОЙ raw строки
                _logger.info("Попытка парсинга JSON после агрессивного исправления (_repair_json) на ИСХОДНОЙ строке.")
                repaired_original_raw = _repair_json(raw)
                try:
                    parsed_obj = json.loads(repaired_original_raw)
                    _logger.warning(f"JSON успешно разобран после агрессивного исправления (_repair_json) ИСХОДНОЙ строки. Исходная ошибка была: {e1}. Содержимое могло значительно измениться.")
                    return parsed_obj
                except Exception as e4:
                    _logger.error(f"Не удалось разобрать JSON ни одним из методов. Ошибка после исправления '+' и экранирования НС: {e1}, после удаления НС: {e2}, после _repair_json(+): {e3}, после _repair_json(исх): {e4}")
                    return {"summary": f"ERROR: Failed to parse JSON response - {e1}", "impact": {}}

# -----------------------------------------------------------------------------

def process_batch_output(
    results_file: str | os.PathLike | None,
    errors_file: str | os.PathLike | None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Обрабатывает скачанные .jsonl файлы с результатами и ошибками от Batch API,
    объединяет результаты по дням и возвращает итоговый DataFrame и список ID с ошибками.

    Args:
        results_file: Путь к .jsonl файлу с успешными результатами.
        errors_file: Путь к .jsonl файлу с ошибками.

    Returns:
        Кортеж: (DataFrame с агрегированными результатами по дням, Список custom_id с ошибками).
    """
    all_results: Dict[str, Any] = {}
    problematic_custom_ids: List[str] = [] # <--- Список для ID с ошибками
    processed_count = 0
    error_count = 0

    # 1. Обработка файла с ошибками (если есть)
    if errors_file and Path(errors_file).is_file():
        _logger.info(f"Обработка файла ошибок: {errors_file}")
        with Path(errors_file).open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    error_data = json.loads(line.strip())
                    custom_id = error_data.get("custom_id")
                    error_details = error_data.get("error", {})
                    response = error_data.get("response", {}) # Иногда ошибка может быть в ответе
                    status_code = response.get("status_code")

                    if not custom_id:
                        _logger.warning(f"Отсутствует custom_id в строке {line_num} файла ошибок.")
                        continue

                    err_code = error_details.get('code', 'UnknownErrorCode')
                    err_msg = error_details.get('message', 'Unknown error')
                    summary = f"ERROR ({err_code}, status: {status_code}): {err_msg}"
                    all_results[custom_id] = {"summary": summary, "impact": {}}
                    error_count += 1
                except json.JSONDecodeError as e:
                    _logger.error(f"Ошибка парсинга JSON в файле ошибок, строка {line_num}: {e}")
                except Exception as e:
                    _logger.error(f"Неизвестная ошибка при обработке строки {line_num} файла ошибок: {e}")
        _logger.info(f"Обработано ошибок: {error_count}")
    else:
        _logger.info("Файл ошибок не найден или не указан.")

    # 2. Обработка файла с результатами
    if results_file and Path(results_file).is_file():
        _logger.info(f"Обработка файла результатов: {results_file}")
        with Path(results_file).open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                custom_id = None # Сбрасываем для каждой строки на случай ошибки парсинга
                try:
                    result_data = json.loads(line.strip())
                    custom_id = result_data.get("custom_id")
                    response = result_data.get("response")

                    if not custom_id:
                        _logger.warning(f"Отсутствует custom_id в строке {line_num} файла результатов.")
                        continue

                    # Пропускаем, если результат для этого ID уже был обработан как ошибка
                    if custom_id in all_results:
                         _logger.warning(f"Результат для {custom_id} уже помечен как ошибка, пропускаем строку {line_num} из файла результатов.")
                         continue

                    if not response or response.get("status_code") != 200:
                        status_code = response.get("status_code", "N/A") if response else "N/A"
                        err_msg = f"Non-200 status code: {status_code}"
                        # Пытаемся получить тело ошибки, если оно есть
                        error_body = response.get("body", {}).get("error", {}).get("message", "No details") if response else "No details"
                        summary = f"ERROR (status: {status_code}): {err_msg} - {error_body}"
                        all_results[custom_id] = {"summary": summary, "impact": {}}
                        error_count += 1
                        continue

                    # Извлекаем тело ответа GPT
                    gpt_body = response.get("body")
                    if not gpt_body:
                         _logger.warning(f"Отсутствует 'body' в успешном ответе для {custom_id} (строка {line_num}). Пропускаем.")
                         all_results[custom_id] = {"summary": "ERROR: Missing 'body' in successful response", "impact": {}}
                         error_count += 1
                         continue

                    gpt_content_raw = gpt_body.get("choices", [{}])[0].get("message", {}).get("content")

                    # Парсим внутренний JSON ответа GPT
                    parsed_gpt_content = _safe_json(gpt_content_raw)
                    all_results[custom_id] = parsed_gpt_content
                    processed_count += 1

                except json.JSONDecodeError as e:
                    _logger.error(f"Ошибка парсинга JSON в файле результатов, строка {line_num}: {e}")
                    if custom_id: # Помечаем как ошибку, если успели извлечь ID
                         all_results[custom_id] = {"summary": f"ERROR: Failed to parse outer JSON - {e}", "impact": {}}
                         error_count += 1
                except IndexError:
                     _logger.error(f"Неожиданная структура ответа в файле результатов, строка {line_num}: 'choices' отсутствуют или пусты.")
                     if custom_id:
                         all_results[custom_id] = {"summary": "ERROR: Invalid response structure (choices)", "impact": {}}
                         error_count += 1
                except Exception as e:
                    _logger.error(f"Неизвестная ошибка при обработке строки {line_num} файла результатов: {e}")
                    if custom_id:
                         all_results[custom_id] = {"summary": f"ERROR: Unknown processing error - {e}", "impact": {}}
                         error_count += 1
        _logger.info(f"Обработано успешных результатов: {processed_count}")
    else:
        _logger.warning("Файл результатов не найден или не указан.")

    # Собираем ID с ошибками из all_results
    for cid, result_data in all_results.items():
        if isinstance(result_data, dict) and result_data.get("summary", "").strip().startswith("ERROR"):
            problematic_custom_ids.append(cid)

    if not all_results:
        _logger.error("Не найдено ни одного результата или ошибки для обработки.")
        return pd.DataFrame(), problematic_custom_ids # Возвращаем пустой DF и список ID

    # 3. Группировка и объединение результатов по дням
    daily_data: Dict[str, List[Dict[str, Any]]] = {}
    _logger.info("Группировка и объединение результатов по дням...")
    for custom_id, result in all_results.items():
        try:
            # Ожидаемый формат custom_id: date_YYYY-MM-DD_chunk_N
            parts = custom_id.split("_")
            if len(parts) >= 3 and parts[0] == 'date':
                date_str = parts[1]
                daily_data.setdefault(date_str, []).append(result)
            else:
                 _logger.warning(f"Не удалось извлечь дату из custom_id '{custom_id}'. Результат будет пропущен при агрегации.")
        except Exception as e:
            _logger.error(f"Ошибка при извлечении даты из custom_id '{custom_id}': {e}")

    final_rows = []
    for date, parts in daily_data.items():
        # Используем логику, похожую на GPTNewsAnalyzer._merge_parts
        valid_parts = []
        error_summaries = []
        for i, part in enumerate(parts):
             # Проверяем, что это словарь и есть нужные ключи, и summary не начинается с ERROR:
            if isinstance(part, dict) and "impact" in part and "summary" in part and not str(part["summary"]).strip().startswith("ERROR"):
                valid_parts.append(part)
            else:
                error_summary = part.get("summary", f"Unknown error in part {i+1}") if isinstance(part, dict) else f"Invalid data structure in part {i+1}: {part}"
                error_summaries.append(str(error_summary))

        if not valid_parts and not error_summaries:
             _logger.warning(f"Нет ни валидных частей, ни ошибок для даты {date}? Странная ситуация.")
             continue

        merged_impact: Dict[str, List[float]] = {}
        summaries = []

        if valid_parts:
            for part in valid_parts:
                summaries.append(str(part.get("summary", "")).strip())
                impact_data = part.get("impact", {})
                if isinstance(impact_data, dict):
                    for ticker, value in impact_data.items():
                        try:
                            num_value = float(value)
                            merged_impact.setdefault(str(ticker), []).append(num_value)
                        except (ValueError, TypeError):
                             _logger.warning(f"Некорректное значение '{value}' для тикера '{ticker}' в части для даты {date}. Пропускаем.")
                else:
                    _logger.warning(f"Некорректный формат 'impact' в части для даты {date}: {impact_data}. Пропускаем.")

        # Формируем итоговое саммари
        final_summary_parts = []
        if error_summaries:
            # Добавляем предупреждение и перечисляем ошибки
            # error_prefix = f"WARNING: {len(error_summaries)}/{len(parts)} part(s) failed. Errors: {'; '.join(error_summaries)}"
            # final_summary_parts.append(error_prefix)
            _logger.warning(f"Для даты {date} были ошибки в {len(error_summaries)}/{len(parts)} частях.")

        # Добавляем саммари из валидных частей
        valid_summaries_str = " ".join(filter(None, summaries))
        if valid_summaries_str:
             final_summary_parts.append(valid_summaries_str)

        final_summary = " ".join(final_summary_parts)
        # Ограничение длины саммари (опционально)
        max_summary_len = 2000
        if len(final_summary) > max_summary_len:
            final_summary = final_summary[:max_summary_len-3] + "..."

        # Находим оценку с максимальным абсолютным значением для каждого тикера
        final_impact = {}
        for ticker, values in merged_impact.items():
            if not values:
                continue
            # Находим значение с максимальным абсолютным значением
            max_abs_value = values[0]
            for value in values[1:]:
                if abs(value) >= abs(max_abs_value):
                    max_abs_value = value
            # Используем найденное значение (с его исходным знаком) и округляем
            final_impact[ticker] = round(max_abs_value, 2)

        # Добавляем строку в итоговый список
        row_data = {"date": date, "summary": final_summary}
        row_data.update(final_impact) # Добавляем тикеры как отдельные колонки
        final_rows.append(row_data)

    if not final_rows:
        _logger.error("Не удалось сформировать ни одной итоговой строки после агрегации.")
        return pd.DataFrame(), problematic_custom_ids # Возвращаем пустой DF и список ID

    # 4. Создание DataFrame
    _logger.info("Создание итогового DataFrame...")
    results_df = pd.DataFrame(final_rows)

    # Переносим колонку 'date' и 'summary' в начало
    if 'date' in results_df.columns and 'summary' in results_df.columns:
        cols_order = ['date', 'summary'] + [col for col in results_df.columns if col not in ['date', 'summary']]
        results_df = results_df[cols_order]
    else:
        _logger.warning("Колонки 'date' или 'summary' отсутствуют в итоговом DataFrame перед переупорядочиванием.")

    # Сортировка по дате (опционально)
    if 'date' in results_df.columns:
        try:
            results_df['date'] = pd.to_datetime(results_df['date']).dt.date
            results_df = results_df.sort_values(by='date')
        except Exception as e:
             _logger.warning(f"Не удалось отсортировать DataFrame по дате: {e}")

    _logger.info(f"Обработка завершена. Итоговый DataFrame содержит {len(results_df)} строк.")
    if problematic_custom_ids:
        _logger.warning(f"Список custom_id, при обработке которых возникли ошибки: {problematic_custom_ids}")

    return results_df, problematic_custom_ids


# --- Пример использования (можно будет вынести в основной скрипт) ---
