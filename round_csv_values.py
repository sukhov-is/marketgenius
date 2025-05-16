import os
import pandas as pd

def round_csv_files_in_folder(folder_path, decimal_places=3):
    """
    Округляет все числовые значения во всех CSV-файлах в указанной папке
    до заданного количества знаков после запятой.

    Args:
        folder_path (str): Путь к папке с CSV-файлами.
        decimal_places (int): Количество знаков после запятой для округления.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                print(f"Обработка файла: {file_path}")
                # Используем стандартный разделитель, но можно добавить detection если нужно
                df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')

                # Округляем только числовые столбцы
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].round(decimal_places)
                    else:
                        # Пытаемся преобразовать в числовой тип и округлить, если возможно
                        try:
                            # Создаем копию для попытки преобразования
                            temp_col = pd.to_numeric(df[col], errors='raise')
                            df[col] = temp_col.round(decimal_places)
                        except (ValueError, TypeError):
                            # Если не удалось преобразовать, оставляем как есть
                            pass

                # Сохраняем изменения, перезаписывая исходный файл
                df.to_csv(file_path, index=False, encoding='utf-8')
                print(f"Файл {filename} успешно обработан.")
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")

if __name__ == "__main__":
    # Используем абсолютный путь, который вы указали
    target_folder = r"C:\Users\Admin\Documents\MarketGenius\data\features_final"
    round_csv_files_in_folder(target_folder)
    print("Все CSV файлы в папке обработаны.")