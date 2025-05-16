import json
import re
import os

# Константы для путей к файлам
PROBLEM_RESULTS_FILE = "data/processed/gpt/gpt_news_history.jsonl"
ORIGINAL_REQUESTS_FILE = "data/external/text/batch/batch_input_news_history.jsonl"
RETRY_BATCH_OUTPUT_FILE = "data/external/text/batch/batch_retry_input_news_history.jsonl"

def get_custom_id_from_raw_line(line_str: str) -> str | None:
    """
    Извлекает custom_id из сырой строки с помощью регулярного выражения.
    Используется как fallback, если строка не парсится как JSON.
    """
    match = re.search(r'"custom_id":\\s*"([^"]*)"', line_str)
    if match:
        return match.group(1)
    return None

def check_line_and_get_id(line_str: str) -> str | None:
    """
    Проверяет строку из файла результатов и возвращает custom_id, если строка проблемная.
    Возвращает None, если строка считается обработанной корректно.
    """
    custom_id_via_regex = get_custom_id_from_raw_line(line_str)

    try:
        data = json.loads(line_str)
        # Если парсинг успешен, предпочитаем custom_id из данных JSON
        custom_id = data.get("custom_id", custom_id_via_regex)

        # Критерий 1: Ошибка верхнего уровня от API OpenAI
        if data.get("error") is not None:
            return custom_id

        # Критерий 2: Проблемы со структурой или внутренним JSON в ответе
        try:
            # Проверка наличия и корректности вложенного JSON-ответа от модели
            content_str = data["response"]["body"]["choices"][0]["message"]["content"]
            json.loads(content_str)  # Попытка распарсить внутренний JSON
            # Если внешний и внутренний JSON распарсились и нет поля "error", строка считается нормальной
            return None
        except (KeyError, TypeError, IndexError):
            # Нарушена структура пути к 'content' (например, отсутствует 'response', 'choices' пуст и т.д.)
            # Такая строка не может быть корректно обработана далее.
            return custom_id
        except json.JSONDecodeError:
            # Внутренний JSON (фактический ответ LLM) некорректен
            return custom_id
        except AttributeError:
            # Например, если choices[0] или message не являются словарями
            return custom_id


    except json.JSONDecodeError:
        # Критерий 3: Сама строка не является валидным JSON
        return custom_id_via_regex # Возвращаем ID, полученный через regex, если он есть

    return None # Если ни один из критериев проблемы не сработал


def create_retry_batch_file(
    problematic_results_file: str,
    original_requests_file: str,
    retry_batch_output_file: str
):
    """
    Основная функция для создания файла с запросами для повторной отправки.
    """
    problematic_custom_ids = set()
    
    print(f"Анализ файла с результатами: {problematic_results_file}")
    try:
        with open(problematic_results_file, 'r', encoding='utf-8') as f_results:
            for line_num, line_str in enumerate(f_results):
                line_str = line_str.strip()
                if not line_str:
                    continue
                
                custom_id = check_line_and_get_id(line_str)
                if custom_id:
                    problematic_custom_ids.add(custom_id)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл с результатами '{problematic_results_file}' не найден.")
        return
    except Exception as e:
        print(f"ОШИБКА: Не удалось прочитать файл с результатами '{problematic_results_file}': {e}")
        return

    if not problematic_custom_ids:
        print("Не найдено проблемных custom_id для повторной обработки.")
        # Создаем пустой файл для консистентности
        output_dir = os.path.dirname(retry_batch_output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(retry_batch_output_file, 'w', encoding='utf-8') as f_retry_empty:
            pass
        print(f"Создан пустой файл для повторных запросов: {retry_batch_output_file}")
        return

    print(f"Найдено {len(problematic_custom_ids)} уникальных проблемных custom_id: {problematic_custom_ids}")

    requests_to_retry_count = 0
    print(f"Формирование файла для повторной отправки: {retry_batch_output_file}")
    
    output_dir = os.path.dirname(retry_batch_output_file)
    if output_dir: # Создаем директорию, если она не пустая (т.е. не корневой файл)
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(original_requests_file, 'r', encoding='utf-8') as f_orig_requests, \
             open(retry_batch_output_file, 'w', encoding='utf-8') as f_retry:
            for line_str in f_orig_requests:
                line_str = line_str.strip()
                if not line_str:
                    continue
                try:
                    request_data = json.loads(line_str)
                    original_custom_id = request_data.get("custom_id")
                    if original_custom_id in problematic_custom_ids:
                        f_retry.write(line_str + '\n')
                        requests_to_retry_count += 1
                except json.JSONDecodeError:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось распарсить строку как JSON в файле оригинальных запросов: {line_str[:100]}...")
                except Exception as e: # Более общая ошибка при обработке строки из файла запросов
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при обработке строки '{line_str[:100]}...' в файле оригинальных запросов: {e}")
    
    except FileNotFoundError:
        print(f"ОШИБКА: Файл с оригинальными запросами '{original_requests_file}' не найден.")
        return
    except Exception as e:
        print(f"ОШИБКА: Не удалось прочитать файл оригинальных запросов или записать в файл повторных запросов: {e}")
        return

    if requests_to_retry_count > 0:
        print(f"Успешно! {requests_to_retry_count} запросов для повторной отправки сохранено в: {retry_batch_output_file}")
    else:
        # Это может случиться, если custom_ids были найдены, но не совпали ни с одним в файле оригинальных запросов,
        # или если файл оригинальных запросов пуст.
        print(f"Не найдено оригинальных запросов, соответствующих проблемным custom_id. Файл '{retry_batch_output_file}' может быть пустым.")

if __name__ == "__main__":
    print("--- Запуск скрипта для извлечения неудачных запросов ---")
    create_retry_batch_file(
        PROBLEM_RESULTS_FILE,
        ORIGINAL_REQUESTS_FILE,
        RETRY_BATCH_OUTPUT_FILE
    )
    print("--- Работа скрипта завершена ---") 