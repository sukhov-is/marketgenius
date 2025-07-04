import pandas as pd
import numpy as np
import os
import glob

# --- Конфигурация ---
# Пути должны быть указаны относительно корня проекта или абсолютные
DIR_TICKER_DATA = 'data/processed/ready_for_training/'
FILE_BLOG_SCORES = 'data/processed/gpt/results_gpt_blogs.csv'
FILE_NEWS_SCORES = 'data/processed/gpt/results_gpt_news.csv'
DIR_OUTPUT = 'data/features_final/'

# Веса для агрегации оценок (рабочий день vs нерабочие)
WEIGHT_NON_WORKING = 0.6
WEIGHT_WORKING = 0.4 # Должно быть 1.0 - WEIGHT_NON_WORKING

# Новые конфигурационные переменные для индексов
FILE_TICKERS_INDICES = 'data/processed/tickers_indices.csv'

# Окна для скользящих средних
ROLLING_AVERAGE_WINDOWS = [15, 50]

def function_A_load_scores(score_file_path: str) -> pd.DataFrame | None:
    """Загружает и предварительно обрабатывает DataFrame с оценками."""
    try:
        df = pd.read_csv(score_file_path)
        if 'date' not in df.columns:
            print(f"Ошибка: столбец 'date' не найден в файле оценок {score_file_path}")
            return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        if 'summary' in df.columns:
            df = df.drop(columns=['summary'])
        print(f"Файл оценок {score_file_path} успешно загружен. Количество строк: {len(df)}, столбцов: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Ошибка: Файл оценок не найден: {score_file_path}")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке файла оценок {score_file_path}: {e}")
        return None

def function_B_load_ticker_data(ticker_file_path: str) -> pd.DataFrame | None:
    """Загружает данные тикера и устанавливает дату как индекс."""
    try:
        df = pd.read_csv(ticker_file_path)
        date_col = None
        if 'DATE' in df.columns:
            date_col = 'DATE'
        elif 'date' in df.columns:
            date_col = 'date'
        
        if date_col is None:
            print(f"Ошибка: Столбец с датой ('date' или '<DATE>') не найден в {ticker_file_path}")
            return None
            
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        if df.index.name != 'date':
             df.index.name = 'date'
        print(f"Данные тикера {ticker_file_path} успешно загружены. Индекс установлен на столбец '{date_col}'.")
        return df
    except FileNotFoundError:
        print(f"Ошибка: Файл тикера не найден: {ticker_file_path}")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке файла тикера {ticker_file_path}: {e}")
        return None

def function_C_aggregate_scores_weighted(
    ticker_working_dates_index: pd.DatetimeIndex,
    score_series_all_dates: pd.Series,
    weight_non_working: float,
    weight_working: float
) -> pd.Series:
    """
    Агрегирует оценки одного тикера на рабочие дни с использованием взвешенного среднего.
    Оценки нерабочих дней имеют больший вес.
    """
    original_scores_on_working_days = score_series_all_dates.reindex(ticker_working_dates_index)
    result_series = pd.Series(index=ticker_working_dates_index, dtype=float)
    buffer_non_working_scores = []

    for current_date in reversed(score_series_all_dates.index):
        current_score_value = score_series_all_dates.loc[current_date]

        if current_date in ticker_working_dates_index:
            original_score_this_working_day = original_scores_on_working_days.loc[current_date]
            final_score_for_this_date = original_score_this_working_day

            valid_buffered_scores = [s for s in buffer_non_working_scores if pd.notna(s)]

            if valid_buffered_scores:
                avg_buffered_score = np.mean(valid_buffered_scores)
                
                if pd.isna(original_score_this_working_day) or original_score_this_working_day == 0:
                    final_score_for_this_date = avg_buffered_score
                else:
                    final_score_for_this_date = (original_score_this_working_day * weight_working) + \
                                                (avg_buffered_score * weight_non_working)
            
            result_series.loc[current_date] = round(final_score_for_this_date, 2) if pd.notna(final_score_for_this_date) else final_score_for_this_date
            buffer_non_working_scores = []
        else:
            buffer_non_working_scores.append(current_score_value)
            
    return result_series

def function_D_calculate_rolling_averages(
    df: pd.DataFrame,
    score_column_name: str,
    window_sizes: list[int]
) -> pd.DataFrame:
    """Рассчитывает скользящие средние по ненулевым значениям для указанной колонки."""
    for window in window_sizes:
        temp_series = df[score_column_name].replace(0, np.nan)
        rolling_avg = temp_series.rolling(window=window, min_periods=1).mean()
        df[f'{score_column_name}_roll_avg_{window}'] = rolling_avg.round(2)
    return df

def function_E_load_ticker_to_indices_map(file_path: str) -> dict[str, list[str]]:
    """Загружает и преобразует CSV файл связей тикеров с индексами в словарь."""
    ticker_to_indices = {}
    try:
        df = pd.read_csv(file_path)
        if 'Ticker' not in df.columns or 'Indices' not in df.columns:
            print(f"Ошибка: В файле {file_path} отсутствуют необходимые столбцы 'Ticker' или 'Indices'.")
            return ticker_to_indices

        for _, row in df.iterrows():
            ticker = row['Ticker']
            indices_str = row['Indices']
            if pd.notna(indices_str) and isinstance(indices_str, str):
                indices_list = [idx.strip() for idx in indices_str.split(';') if idx.strip()]
                if indices_list:
                    ticker_to_indices[ticker] = indices_list
            # else:
            #     print(f"Предупреждение: Для тикера {ticker} в {file_path} не указаны индексы или неверный формат.")
        print(f"Карта 'тикер -> индексы' успешно загружена из {file_path}. Найдено связей: {len(ticker_to_indices)}")
    except FileNotFoundError:
        print(f"Ошибка: Файл связей тикеров с индексами не найден: {file_path}")
    except Exception as e:
        print(f"Ошибка при загрузке файла связей тикеров с индексами {file_path}: {e}")
    return ticker_to_indices

def main():
    """Основной управляющий скрипт."""
    print("Запуск скрипта генерации признаков...")

    os.makedirs(DIR_OUTPUT, exist_ok=True)
    print(f"Результаты будут сохранены в: {DIR_OUTPUT}")

    print("\n--- Загрузка файлов оценок ---")
    blog_scores_df = function_A_load_scores(FILE_BLOG_SCORES)
    news_scores_df = function_A_load_scores(FILE_NEWS_SCORES)

    # Загрузка карты тикеров к индексам
    print("\n--- Загрузка карты тикер -> индексы ---")
    ticker_indices_map = function_E_load_ticker_to_indices_map(FILE_TICKERS_INDICES)
    if not ticker_indices_map:
        print("Предупреждение: Карта тикеров к индексам пуста или не загружена. Признаки на основе индексов не будут созданы.")

    if blog_scores_df is None and news_scores_df is None:
        print("Ошибка: Не удалось загрузить ни один из файлов оценок. Завершение работы.")
        return

    score_sources = []
    if blog_scores_df is not None:
        score_sources.append({'df': blog_scores_df, 'suffix': '_blog', 'name': 'Блоги'})
    if news_scores_df is not None:
        score_sources.append({'df': news_scores_df, 'suffix': '_news', 'name': 'Новости'})
    
    ticker_files = glob.glob(os.path.join(DIR_TICKER_DATA, '*_processed.csv'))
    if not ticker_files:
        print(f"В директории {DIR_TICKER_DATA} не найдены файлы тикеров '*_processed.csv'.")
        return
    
    print(f"\nНайдено {len(ticker_files)} файлов тикеров для обработки.")

    for ticker_file_path in ticker_files:
        base_name = os.path.basename(ticker_file_path)
        ticker_name = base_name.replace('_processed.csv', '')
        print(f"\n--- Обработка тикера: {ticker_name} ({base_name}) ---")

        ticker_df = function_B_load_ticker_data(ticker_file_path)
        if ticker_df is None:
            print(f"Пропуск тикера {ticker_name} из-за ошибки загрузки данных.")
            continue

        # Получаем список индексов для текущего тикера
        related_indices = ticker_indices_map.get(ticker_name, [])

        for source in score_sources:
            scores_df = source['df']
            suffix = source['suffix']
            source_name = source['name']
            
            print(f"  Добавление признаков из источника: {source_name}")

            # 1. Признаки на основе собственных оценок тикера
            if ticker_name in scores_df.columns:
                single_ticker_score_series_all_dates = scores_df[ticker_name]
                
                aggregated_scores_series = function_C_aggregate_scores_weighted(
                    ticker_df.index, 
                    single_ticker_score_series_all_dates,
                    WEIGHT_NON_WORKING,
                    WEIGHT_WORKING
                )
                
                agg_score_col_name = f'score{suffix}'
                ticker_df[agg_score_col_name] = aggregated_scores_series
                print(f"    Добавлена колонка агрегированных оценок тикера: {agg_score_col_name}")

                ticker_df = function_D_calculate_rolling_averages(
                    ticker_df, 
                    agg_score_col_name, 
                    ROLLING_AVERAGE_WINDOWS
                )
                print(f"    Рассчитаны скользящие средние для {agg_score_col_name}")
            else:
                print(f"    Предупреждение: Тикер {ticker_name} не найден в файле оценок {source_name}. Пропуск добавления собственных признаков тикера.")

            # 2. Признаки на основе оценок связанных индексов (каждый индекс - отдельный признак)
            if related_indices and ticker_indices_map:
                print(f"    Добавление признаков на основе оценок связанных индексов для {ticker_name} из {source_name}")
                
                other_indices_aggregated_scores_list = []
                moex_processed_for_source = False

                for index_name in related_indices:
                    if index_name in scores_df.columns:
                        index_score_series_all_dates = scores_df[index_name]

                        aggregated_index_series = function_C_aggregate_scores_weighted(
                            ticker_df.index,  # Выравниваем по рабочим дням ТИКЕРА
                            index_score_series_all_dates,
                            WEIGHT_NON_WORKING,
                            WEIGHT_WORKING
                        )

                        if not aggregated_index_series.isna().all():
                            if index_name == "MOEX":
                                moex_score_col_name = f'Index_MOEX{suffix}_score'
                                ticker_df[moex_score_col_name] = aggregated_index_series.round(2)
                                print(f"      Добавлена колонка оценок для индекса MOEX: {moex_score_col_name}")
                                # Скользящие средние для MOEX не рассчитываем по условию
                                print(f"      Пропуск расчета скользящих средних для MOEX: {moex_score_col_name}")
                                moex_processed_for_source = True
                            else:
                                # Собираем серии для последующего усреднения, переименовываем, чтобы избежать конфликтов имен
                                other_indices_aggregated_scores_list.append(aggregated_index_series.rename(f'{index_name}{suffix}_temp'))
                        else:
                            print(f"      Предупреждение: Агрегированные оценки для индекса {index_name} (источник: {source_name}) для тикера {ticker_name} содержат только NaN. Индекс не будет учтен.")
                    else:
                        print(f"      Предупреждение: Индекс {index_name} не найден в {source_name} для тикера {ticker_name}. Признаки для этого индекса не будут добавлены.")

                # Обработка собранных "других" индексов
                if other_indices_aggregated_scores_list:
                    other_indices_df = pd.concat(other_indices_aggregated_scores_list, axis=1)
                    avg_other_indices_score = other_indices_df.mean(axis=1)
                    
                    avg_other_indices_col_name = f'Avg_Other_Indices{suffix}_score'
                    ticker_df[avg_other_indices_col_name] = avg_other_indices_score.round(2)
                    print(f"      Добавлена колонка средних оценок для остальных индексов: {avg_other_indices_col_name}")

                    ticker_df = function_D_calculate_rolling_averages(
                        ticker_df,
                        avg_other_indices_col_name,
                        ROLLING_AVERAGE_WINDOWS
                    )
                    print(f"      Рассчитаны скользящие средние для {avg_other_indices_col_name}")
                elif not moex_processed_for_source : # Если не обработан MOEX и список других индексов пуст
                     print(f"    Предупреждение: Не найдено данных по индексам (MOEX или другие) из {source_name} для тикера {ticker_name} для создания признаков.")

            elif not related_indices and ticker_indices_map :
                 print(f"    Предупреждение: Для тикера {ticker_name} не найдены связанные индексы в {FILE_TICKERS_INDICES}. Пропуск признаков по индексам.")
            elif not ticker_indices_map:
                 print(f"    Предупреждение: Карта тикеров к индексам не загружена или пуста. Пропуск признаков по индексам.")

        output_file_path = os.path.join(DIR_OUTPUT, f'{ticker_name}_final.csv')
        try:
            ticker_df.to_csv(output_file_path)
            print(f"  Результат для тикера {ticker_name} сохранен в: {output_file_path}")
        except Exception as e:
            print(f"  Ошибка при сохранении файла для {ticker_name}: {e}")

    print("\n--- Генерация признаков завершена ---")

if __name__ == '__main__':
    main()
