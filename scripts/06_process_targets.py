import pandas as pd
import numpy as np
import os
import glob
import re

def calculate_target(df, target_days):
    """
    Рассчитывает таргет как процентное изменение цены закрытия через target_days дней.
    Возвращает Series с таргетами.
    """
    if 'CLOSE' not in df.columns:
        print(f"Предупреждение: колонка 'CLOSE' не найдена. Не могу рассчитать таргеты.")
        return pd.Series([np.nan] * len(df), index=df.index)
    
    # Рассчитываем процентное изменение цены через target_days дней
    future_price = df['CLOSE'].shift(-target_days)
    current_price = df['CLOSE']
    
    # Процентное изменение: (future_price - current_price) / current_price * 100
    target = ((future_price - current_price) / current_price * 100).round(4)
    
    return target

def process_csv_file(file_path):
    """
    Обрабатывает один CSV-файл:
    1. Рассчитывает/обновляет таргет-колонки (target_1d, target_3d, target_7d, target_30d, target_180d).
    2. Создает/обновляет бинарные таргет-колонки (target_Xd_binary).
    3. Проверяет наличие и создает только отсутствующие колонки для предсказаний.
    4. Перемещает все таргет-колонки и колонки предсказаний в конец DataFrame.
    5. Сохраняет изменения в тот же CSV-файл.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Загружен файл {file_path}, размер: {df.shape}")
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return

    # Определяем периоды таргетов для расчета
    target_periods = [1, 3, 7, 30, 180, 365]
    
    original_target_cols = []
    binary_target_cols = []
    original_pred_cols = []
    binary_pred_cols = []

    # Рассчитываем/обновляем таргеты
    print(f"Рассчитываем/Обновляем таргеты для периодов: {target_periods}")
    for period in target_periods:
        target_col_name = f"target_{period}d"
        
        # Всегда рассчитываем/пересчитываем таргет.
        # Если колонка не существует, она будет создана при присвоении.
        # Если существует, будет обновлена.
        print(f"Расчет/Обновление колонки {target_col_name}...")
        
        target_values = calculate_target(df, period)
        df[target_col_name] = target_values # Создает или обновляет колонку
        
        # Добавляем имя колонки в список для дальнейшей обработки (бинарные, предикты, перемещение)
        # Проверка на дубликаты перед добавлением
        if target_col_name not in original_target_cols:
            original_target_cols.append(target_col_name)
            
        print(f"Рассчитан/Обновлен {target_col_name}: {target_values.notna().sum()} значений из {len(target_values)}")

    # Создаем/обновляем бинарные таргеты
    print("Создаем/обновляем бинарные таргеты...")
    for target_col in original_target_cols: # original_target_cols теперь содержит все актуальные таргет-колонки
        binary_target_name = f"{target_col}_binary"
        
        # Создание/обновление бинарной таргет-колонки: 1 если таргет > 0, иначе 0
        df[binary_target_name] = df[target_col].apply(
            lambda x: 1 if pd.notna(x) and x > 0 else 0
        )
        if binary_target_name not in binary_target_cols:
            binary_target_cols.append(binary_target_name)
        
        positive_count = (df[binary_target_name] == 1).sum()
        print(f"Создан/Обновлен {binary_target_name}: {positive_count} положительных из {len(df)} записей")

    # Проверяем и создаем только отсутствующие колонки для предсказаний
    print("Проверяем наличие колонок для предсказаний...")
    
    for target_col in original_target_cols: # Используем обновленный original_target_cols
        # Колонка для предикта исходного таргета
        original_pred_name = f"{target_col}_pred"
        if original_pred_name not in df.columns:
            df[original_pred_name] = np.nan
            print(f"Создана колонка для предсказаний: {original_pred_name}")
        # else: # Оставляем текущую логику - если колонка есть, не трогаем
            # print(f"Колонка {original_pred_name} уже существует, пропускаем")
        if original_pred_name not in original_pred_cols:
             original_pred_cols.append(original_pred_name)
        
        # Колонка для предикта бинарного таргета
        binary_pred_name = f"{target_col}_binary_pred"
        if binary_pred_name not in df.columns:
            df[binary_pred_name] = np.nan
            print(f"Создана колонка для предсказаний: {binary_pred_name}")
        # else: # Оставляем текущую логику - если колонка есть, не трогаем
            # print(f"Колонка {binary_pred_name} уже существует, пропускаем")
        if binary_pred_name not in binary_pred_cols:
            binary_pred_cols.append(binary_pred_name)

    # Перемещение колонок в конец DataFrame
    # Порядок: исходные таргеты → бинарные таргеты → предикты исходных → предикты бинарных
    cols_to_move_to_end = (
        original_target_cols +
        binary_target_cols +
        original_pred_cols +
        binary_pred_cols
    )
    
    remaining_cols = [col for col in df.columns if col not in cols_to_move_to_end]
    new_col_order = remaining_cols + cols_to_move_to_end
    df = df[new_col_order]

    # Сохранение изменений
    try:
        df.to_csv(file_path, index=False)
        print(f"Файл {file_path} успешно обработан и сохранен.")
        print(f"Итоговые размеры: {df.shape}")
        print("="*50)
    except Exception as e:
        print(f"Ошибка при сохранении файла {file_path}: {e}")

def main():
    """
    Главная функция для поиска и обработки всех CSV-файлов в папке data/features_final/.
    """
    base_path = 'data/features_final/'
    search_pattern = os.path.join(base_path, '*.csv')
    
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"В папке {base_path} не найдено CSV-файлов.")
        return
        
    print(f"Найдено CSV-файлов: {len(csv_files)}")
    print("Начинаем обработку...")
    print("="*50)
    
    for i, file_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Обработка файла: {os.path.basename(file_path)}")
        process_csv_file(file_path)
    
    print("\n" + "="*50)
    print("Обработка всех файлов завершена!")

if __name__ == "__main__":
    main() 