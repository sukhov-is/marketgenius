import json
import os

def get_script_and_project_paths():
    """Определяет путь к скрипту и корню проекта."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    return script_dir, project_root

def load_custom_ids_from_jsonl(filepath):
    """
    Загружает все 'custom_id' из JSONL файла.
    Возвращает set 'custom_id' или пустой set в случае ошибки.
    """
    custom_ids = set()
    if not os.path.exists(filepath):
        print(f"Ошибка: Файл не найден: {filepath}")
        return custom_ids
        
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
                        custom_ids.add(custom_id)
                    else:
                        print(f"Предупреждение: Запись в {filepath} (строка {i+1}) не содержит 'custom_id'.")
                except json.JSONDecodeError:
                    print(f"Предупреждение: Не удалось декодировать JSON в {filepath} (строка {i+1}).")
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
    return custom_ids

def find_missing_records(processed_ids_filepath, source_records_filepath, output_filepath):
    """
    Находит записи в source_records_filepath, чьи custom_id отсутствуют 
    в processed_ids_filepath, и сохраняет их в output_filepath.
    """
    print(f"Загрузка custom_id из файла обработанных результатов: {processed_ids_filepath}")
    processed_custom_ids = load_custom_ids_from_jsonl(processed_ids_filepath)
    
    if not processed_custom_ids:
        print("Не удалось загрузить custom_id из файла обработанных результатов или файл пуст. Проверьте предыдущие сообщения.")
        # Можно решить, продолжать ли, если processed_ids пуст. 
        # Если processed_ids пуст, то все записи из source будут "отсутствующими".
        # Для безопасности, прервем, если это не ожидаемое поведение.
        # print("Так как custom_id из файла результатов не найдены, все записи из исходного файла будут считаться отсутствующими.")
    
    print(f"Найдено {len(processed_custom_ids)} уникальных custom_id в файле обработанных результатов.")

    missing_records = []
    
    if not os.path.exists(source_records_filepath):
        print(f"Ошибка: Исходный файл с записями не найден: {source_records_filepath}")
        return

    print(f"Поиск отсутствующих записей в файле: {source_records_filepath}")
    try:
        with open(source_records_filepath, 'r', encoding='utf-8') as f_source:
            for i, line in enumerate(f_source):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    record = json.loads(line_stripped)
                    custom_id = record.get('custom_id')
                    
                    if not custom_id:
                        print(f"Предупреждение: Запись в {source_records_filepath} (строка {i+1}) не содержит 'custom_id'. Пропускается.")
                        continue
                        
                    if custom_id not in processed_custom_ids:
                        missing_records.append(record)
                        
                except json.JSONDecodeError:
                    print(f"Предупреждение: Не удалось декодировать JSON в {source_records_filepath} (строка {i+1}). Пропускается.")
    except Exception as e:
        print(f"Ошибка при чтении исходного файла {source_records_filepath}: {e}")
        return

    if not missing_records:
        print("Не найдено записей в исходном файле, которые отсутствуют в файле результатов.")
        return

    print(f"Найдено {len(missing_records)} записей, отсутствующих в файле результатов.")

    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Создана директория для выходного файла: {output_dir}")
            
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            for record in missing_records:
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Отсутствующие записи ({len(missing_records)} шт.) сохранены в: {output_filepath}")
    except Exception as e:
        print(f"Ошибка при записи выходного файла {output_filepath}: {e}")


if __name__ == '__main__':
    script_dir, project_root = get_script_and_project_paths()

    # Путь к файлу с результатами (откуда берем обработанные custom_id)
    PROCESSED_RESULTS_FILE = os.path.join(project_root, 'data', 'processed', 'gpt', 'gpt_news_history.jsonl')
    
    # Путь к исходному файлу, из которого будем выбирать "недостающие" записи
    SOURCE_HISTORY_FILE = os.path.join(project_root, 'data', 'external', 'text', 'batch', 'batch_input_news_history.jsonl')
    
    # Путь для сохранения нового файла с "недостающими" записями
    MISSING_RECORDS_OUTPUT_FILE = os.path.join(project_root, 'data', 'external', 'text', 'batch', 'missing_from_results_batch.jsonl')

    print(f"Файл с обработанными custom_id: {PROCESSED_RESULTS_FILE}")
    print(f"Исходный файл с записями: {SOURCE_HISTORY_FILE}")
    print(f"Выходной файл для недостающих записей: {MISSING_RECORDS_OUTPUT_FILE}")

    find_missing_records(PROCESSED_RESULTS_FILE, SOURCE_HISTORY_FILE, MISSING_RECORDS_OUTPUT_FILE)
    
    print("\nСкрипт завершил работу.") 