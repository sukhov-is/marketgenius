import json
import os
from datetime import datetime
import glob # Хотя glob не используется напрямую в основной логике, оставлю для возможного расширения

def parse_custom_id(custom_id_str):
    """
    Разбирает custom_id вида "date_YYYY-MM-DD_chunk_N" 
    и возвращает кортеж (дата_объект, номер_чанка).
    Возвращает (None, None) в случае ошибки разбора.
    """
    if not isinstance(custom_id_str, str):
        print(f"Предупреждение: custom_id не является строкой: {custom_id_str}")
        return None, None
    try:
        parts = custom_id_str.split('_')
        if len(parts) == 4 and parts[0] == 'date' and parts[2] == 'chunk':
            date_str = parts[1]
            chunk_num_str = parts[3]
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            chunk_num = int(chunk_num_str)
            return date_obj, chunk_num
        else:
            print(f"Предупреждение: Некорректный формат custom_id '{custom_id_str}'")
            return None, None
    except ValueError as ve:
        print(f"Ошибка разбора значения в custom_id '{custom_id_str}': {ve}")
        return None, None
    except Exception as e:
        print(f"Общая ошибка разбора custom_id '{custom_id_str}': {e}")
        return None, None

def load_jsonl_to_dict(filepath):
    """
    Загружает JSONL файл и возвращает словарь {custom_id: record_dict}.
    """
    data_dict = {}
    if not os.path.exists(filepath):
        print(f"Файл не найден: {filepath}. Будет создан новый, если это целевой файл.")
        return data_dict # Возвращаем пустой словарь, если файл не найден

    print(f"Чтение файла: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    record = json.loads(line_stripped)
                    custom_id = record.get('custom_id')
                    if custom_id:
                        if custom_id in data_dict:
                            print(f"Предупреждение: Дублирующийся custom_id '{custom_id}' в файле {filepath} (строка {i+1}). Используется последняя запись.")
                        data_dict[custom_id] = record
                    else:
                        print(f"Предупреждение: Запись в файле {filepath} (строка {i+1}) не содержит 'custom_id'. Строка: {line_stripped}")
                except json.JSONDecodeError:
                    print(f"Предупреждение: Не удалось декодировать JSON из строки в файле {filepath} (строка {i+1}). Строка: {line_stripped}")
                except Exception as e:
                    print(f"Неожиданная ошибка при обработке строки {i+1} из файла {filepath}: {e}. Строка: {line_stripped}")
    except FileNotFoundError: # Этот блок уже покрыт os.path.exists, но для надежности
        print(f"Ошибка: Файл не найден при попытке открытия {filepath}")
    except Exception as e:
        print(f"Неожиданная ошибка при чтении файла {filepath}: {e}")
    return data_dict

def update_and_sort_jsonl(target_filepath, update_filepath):
    """
    Обновляет целевой JSONL файл записями из файла обновлений, сортирует и сохраняет.
    """
    print(f"--- Начало обновления файла: {target_filepath} ---")
    print(f"Файл обновлений: {update_filepath}")

    # 1. Загрузить данные из файла обновлений
    update_data_dict = load_jsonl_to_dict(update_filepath)
    if not update_data_dict:
        print(f"Файл обновлений {update_filepath} не содержит данных или не найден. Обновление не будет выполнено.")
        # Если целевой файл существует, но файл обновлений пуст, можно решить не изменять целевой.
        # Если и целевой, и файл обновлений пусты/не существуют, то ничего не делать.
        if not os.path.exists(target_filepath):
             print(f"Целевой файл {target_filepath} также не существует. Никаких действий не предпринято.")
             return
        print(f"Содержимое файла {target_filepath} останется без изменений.")
        return

    # 2. Загрузить существующие данные из целевого файла (если он есть)
    # Используем final_records_map для хранения итоговых записей.
    # Ключи - custom_id, значения - сами записи (словари).
    # Это позволит легко заменять записи и добавлять новые.
    final_records_map = {}

    if os.path.exists(target_filepath):
        print(f"Загрузка существующего целевого файла: {target_filepath}")
        try:
            with open(target_filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    try:
                        record = json.loads(line_stripped)
                        custom_id = record.get('custom_id')
                        if custom_id:
                            final_records_map[custom_id] = record
                        else:
                            print(f"Предупреждение: Запись в целевом файле {target_filepath} (строка {i+1}) без 'custom_id' будет проигнорирована при слиянии.")
                    except json.JSONDecodeError:
                        print(f"Предупреждение: Ошибка декодирования JSON в целевом файле {target_filepath} (строка {i+1}). Строка будет пропущена.")
        except Exception as e:
            print(f"Критическая ошибка при чтении целевого файла {target_filepath}: {e}. Прерывание.")
            return
        print(f"Загружено {len(final_records_map)} записей из целевого файла.")
    else:
        print(f"Целевой файл {target_filepath} не найден. Он будет создан с данными из файла обновлений.")

    # 3. Обновить/добавить записи из update_data_dict в final_records_map
    updated_count = 0
    added_count = 0
    for custom_id, record in update_data_dict.items():
        if custom_id in final_records_map:
            updated_count += 1
        else:
            added_count += 1
        final_records_map[custom_id] = record # Заменяет существующую или добавляет новую

    print(f"Из файла обновлений: {updated_count} записей обновлено, {added_count} записей добавлено.")
    
    # 4. Подготовить записи для сортировки
    records_to_sort = list(final_records_map.values())

    # 5. Сортировка
    def sort_key_func(record_item):
        custom_id_str = record_item.get('custom_id')
        parsed_date, parsed_chunk_num = parse_custom_id(custom_id_str)
        if parsed_date is None or parsed_chunk_num is None:
            return (datetime.max.date(), float('inf')) # Некорректные custom_id в конец
        return parsed_date, parsed_chunk_num

    records_to_sort.sort(key=sort_key_func)
    print(f"Всего записей для записи после слияния и сортировки: {len(records_to_sort)}")

    # 6. Запись результата в целевой файл
    try:
        output_dir = os.path.dirname(target_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Создана директория для выходного файла: {output_dir}")

        with open(target_filepath, 'w', encoding='utf-8') as outfile:
            for record in records_to_sort:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Обновленный и отсортированный файл сохранен: {target_filepath}")

    except IOError as e:
        print(f"Ошибка записи в файл {target_filepath}: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка при записи выходного файла: {e}")
    
    print(f"--- Обновление файла {target_filepath} завершено ---")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    TARGET_JSONL_FILENAME = 'gpt_news_history.jsonl'
    # Имя файла обновлений, как указано пользователем
    UPDATE_JSONL_FILENAME = 'results_batch_6826532428fc8190b0d9d22f23de70e6.jsonl' 

    # Формируем пути относительно корня проекта
    # Целевой файл в data/processed/gpt/
    TARGET_FILE = os.path.join(project_root, 'data', 'processed', 'gpt', TARGET_JSONL_FILENAME)
    # Файл обновлений предполагается в data/external/gpt/ (аналогично merge_jsonl_script)
    # Если он в другом месте, измените 'data', 'external', 'gpt' соответственно
    UPDATE_FILE = os.path.join(project_root, 'data', 'external', 'gpt', UPDATE_JSONL_FILENAME)

    print(f"Целевой файл: {TARGET_FILE}")
    print(f"Файл обновлений: {UPDATE_FILE}")
    
    if not os.path.isfile(UPDATE_FILE):
        print(f"Ошибка: Файл обновлений '{UPDATE_FILE}' не найден.")
        print("Пожалуйста, проверьте путь и имя файла обновлений.")
    else:
        # Перед запуском основной функции, можно добавить проверку существования директорий
        target_dir = os.path.dirname(TARGET_FILE)
        if not os.path.exists(target_dir):
            print(f"Предупреждение: Директория для целевого файла '{target_dir}' не существует. Она будет создана.")
            # os.makedirs(target_dir, exist_ok=True) # Создание директории можно перенести в update_and_sort_jsonl

        update_and_sort_jsonl(TARGET_FILE, UPDATE_FILE)
        print("\nСкрипт завершил работу.") 