import json
import os
import glob
from datetime import datetime

def parse_custom_id(custom_id_str):
    """
    Разбирает custom_id вида "date_YYYY-MM-DD_chunk_N" 
    и возвращает кортеж (дата_объект, номер_чанка).
    Возвращает (None, None) в случае ошибки разбора.
    """
    try:
        # Пример: "date_2021-07-19_chunk_6"
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

def merge_and_process_jsonl(input_dir, output_file, specific_filenames=None):
    """
    Объединяет JSONL файлы из input_dir (или указанные specific_filenames), 
    сортирует по custom_id (дата, чанк) и удаляет дубликаты, 
    сохраняя результат в output_file.
    """
    
    raw_data_with_id = []
    files_to_process = []

    if specific_filenames and isinstance(specific_filenames, list) and len(specific_filenames) > 0:
        print(f"Получен список из {len(specific_filenames)} файлов для выборочной обработки.")
        for filename in specific_filenames:
            if not isinstance(filename, str) or not filename.strip():
                print("Предупреждение: Обнаружено некорректное имя файла в списке (пустое или не строка), пропускается.")
                continue

            if not filename.lower().endswith('.jsonl'):
                print(f"Предупреждение: Файл '{filename}' из списка не имеет расширения .jsonl и будет пропущен.")
                continue
            
            filepath = os.path.join(input_dir, filename)
            if os.path.isfile(filepath):
                files_to_process.append(filepath)
            else:
                print(f"Предупреждение: Файл '{filepath}', указанный в списке, не найден в директории '{input_dir}' и будет пропущен.")
        
        if not files_to_process:
            print(f"Ни один из указанных файлов не был найден или не является корректным .jsonl файлом в директории: {input_dir}")
            return
    else:
        if specific_filenames: # Если был передан пустой список или некорректный тип
             print("Предупреждение: Список specific_filenames был предоставлен, но он пуст или некорректен. Поиск всех .jsonl файлов.")
        print(f"Список конкретных файлов не предоставлен или пуст. Поиск всех .jsonl файлов в '{input_dir}'")
        file_pattern = os.path.join(input_dir, '*.jsonl')
        files_to_process = glob.glob(file_pattern)
        
        if not files_to_process:
            print(f"Не найдено *.jsonl файлов в директории: {input_dir}")
            return

    print(f"Найдено {len(files_to_process)} .jsonl файлов для обработки.")

    for filepath in files_to_process:
        print(f"Чтение файла: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line_stripped = line.strip()
                    if not line_stripped: # Пропустить пустые строки
                        continue
                    try:
                        record = json.loads(line_stripped)
                        custom_id = record.get('custom_id')
                        if custom_id:
                            raw_data_with_id.append(record)
                        else:
                            print(f"Предупреждение: Запись в файле {filepath} (строка {i+1}) не содержит 'custom_id'. Строка: {line_stripped}")
                    except json.JSONDecodeError:
                        print(f"Предупреждение: Не удалось декодировать JSON из строки в файле {filepath} (строка {i+1}). Строка: {line_stripped}")
                    except Exception as e:
                        print(f"Неожиданная ошибка при обработке строки {i+1} из файла {filepath}: {e}. Строка: {line_stripped}")
        except FileNotFoundError:
            print(f"Ошибка: Файл не найден {filepath}")
        except Exception as e:
            print(f"Неожиданная ошибка при чтении файла {filepath}: {e}")


    if not raw_data_with_id:
        print("Не найдено записей с 'custom_id' для обработки.")
        return

    # Шаг 2: Дедупликация (сохраняем первый встреченный экземпляр)
    deduplicated_records = []
    seen_custom_ids = set()
    for record in raw_data_with_id:
        custom_id = record['custom_id'] # 'custom_id' должен существовать на этом этапе
        if custom_id not in seen_custom_ids:
            deduplicated_records.append(record)
            seen_custom_ids.add(custom_id)
            
    print(f"После дедупликации осталось {len(deduplicated_records)} записей из {len(raw_data_with_id)} исходных записей с custom_id.")

    # Шаг 3: Сортировка дедуплицированных записей
    def sort_key_func(record_item):
        custom_id_str = record_item.get('custom_id')
        parsed_date, parsed_chunk_num = parse_custom_id(custom_id_str)
        
        # Для записей с некорректным custom_id, помещаем их в конец
        if parsed_date is None or parsed_chunk_num is None:
            return (datetime.max.date(), float('inf')) 
        return parsed_date, parsed_chunk_num

    deduplicated_records.sort(key=sort_key_func)
    
    # Шаг 4: Запись результата в выходной файл
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Создана директория для выходного файла: {output_dir}")

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for record in deduplicated_records:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        num_source_files = len(files_to_process)
        print(f"\nОбработано записей (с custom_id): {len(raw_data_with_id)} из {num_source_files} файлов.")
        print(f"Найдено уникальных custom_id: {len(seen_custom_ids)}.")
        print(f"Записано {len(deduplicated_records)} отсортированных и дедуплицированных записей в файл: {output_file}")

    except IOError as e:
        print(f"Ошибка записи в файл {output_file}: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка при записи выходного файла: {e}")


if __name__ == '__main__':
    # Определяем директорию, где находится текущий скрипт
    # __file__ дает путь к текущему файлу. os.path.realpath resolves any symlinks.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Поднимаемся на два уровня вверх от src/etl/ до корня проекта
    # (например, если скрипт в /path/to/project/src/etl, project_root будет /path/to/project)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Формируем пути относительно корня проекта
    INPUT_DIRECTORY = os.path.join(project_root, 'data', 'external', 'gpt')
    OUTPUT_JSONL_FILE = os.path.join(project_root, 'data', 'processed', 'gpt', 'gpt_blogs_history.jsonl')

    # --- УКАЖИТЕ КОНКРЕТНЫЕ ФАЙЛЫ ДЛЯ ОБРАБОТКИ ЗДЕСЬ ---
    # Чтобы обработать только определенные файлы, раскомментируйте следующую строку
    # и перечислите имена файлов в списке. Например:
    # SPECIFIC_FILES = ["results_batch_6825fb6710ac8190a84a0c525ab415ec.jsonl", "results_batch_6825fc48badc81908b154c8a5f744be0.jsonl"]
    SPECIFIC_FILES = [] 
    # Если SPECIFIC_FILES остается пустым (как сейчас), скрипт обработает все .jsonl файлы в INPUT_DIRECTORY.
    # ---------------------------------------------------------
    
    print(f"Входная директория: {INPUT_DIRECTORY}")
    if SPECIFIC_FILES:
        print(f"Будут обработаны только указанные файлы: {SPECIFIC_FILES}")
    else:
        print("Будут обработаны все .jsonl файлы из входной директории (список SPECIFIC_FILES пуст).")
    print(f"Выходной файл: {OUTPUT_JSONL_FILE}")

    if not os.path.isdir(INPUT_DIRECTORY):
        print(f"Ошибка: Входная директория '{INPUT_DIRECTORY}' не найдена.")
    else:
        merge_and_process_jsonl(INPUT_DIRECTORY, OUTPUT_JSONL_FILE, specific_filenames=SPECIFIC_FILES)
        print("\nСкрипт завершил работу.") 