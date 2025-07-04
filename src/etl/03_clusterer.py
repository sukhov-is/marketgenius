import argparse
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class Clusterer:
    """Класс для кластеризации новостей"""
    
    def __init__(self, config_path: str):
        assert os.path.exists(config_path), f"Config file {config_path} not found"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Инициализируем параметры кластеризации один раз
        self.clustering_params = self.config.get("clustering", {}).copy()
        self.clustering_params.update({
            "metric": "precomputed",
            "linkage": "average"
        })

        # Параметры для вычисления расстояний
        self.same_channels_penalty = self.config['distances']['same_channels_penalty']
        self.time_penalty_modifier = self.config['distances']['time_penalty_modifier']
        self.time_shift_hours = self.config['distances']['time_shift_hours']

        self.all_clusters = []  # Список для хранения всех кластеров

    def cluster_documents(self, df: pd.DataFrame) -> None:
        """
        Кластеризация новостей
        Args:
            df: DataFrame с новостями
        """     
        if df.empty or len(df) < 5:
            return
        
        # Предварительно вычисляем эмбеддинги для всего датафрейма
        embedding = np.vstack(df['embedding'].apply(json.loads).values)
        
        # Вычисляем базовую матрицу расстояний
        distances = pairwise_distances(embedding, metric="cosine", n_jobs=-1)
        
        # Применяем штраф за одинаковые каналы
        same_channels = np.array([[1 if df.iloc[i]['channel_name'] == df.iloc[j]['channel_name'] 
                                else 0 for j in range(len(df))] for i in range(len(df))])
        distances = np.minimum(1.0, 
                            distances * (1 + (self.same_channels_penalty - 1) * same_channels))

        # Применяем временные штрафы
        if self.time_penalty_modifier > 1.0:
            times = df['datetime'].values.reshape(-1, 1)
            time_diffs = np.abs(times - times.T) / np.timedelta64(1, 'h')
            hours_shifted = time_diffs - self.time_shift_hours
            time_penalties = 1.0 + expit(hours_shifted) * (self.time_penalty_modifier - 1.0)
            distances = np.minimum(1.0, distances * time_penalties)

        # Используем предварительно настроенные параметры
        clustering = AgglomerativeClustering(**self.clustering_params)
        labels = clustering.fit_predict(distances)

        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_df = df[labels == label]
            min_datetime = cluster_df['datetime'].min()
            cluster_date = min_datetime.date()
            datetimes_str = cluster_df['datetime'].apply(lambda x: x.isoformat())
            
            cluster_dict = {
                'news': cluster_df['news'].tolist(),
                'datetime': datetimes_str.tolist(),
                'channel_name': cluster_df['channel_name'].tolist(),
                'message_link': cluster_df['message_link'].tolist(),
                'cluster_date': cluster_date.isoformat(),
                'cluster_id': f"{min_datetime.isoformat()}_{label}"
            }
            
            self.all_clusters.append(cluster_dict)

def _get_representative_idx(news_list: List[str]) -> int:
    """Возвращает индекс наиболее информативной новости (простая эвристика: макс. длина текста)."""
    # Если несколько записей имеют одинаковую длину — выбираем первую
    return int(np.argmax([len(text or "") for text in news_list]))


def save_representative_news_to_csv(clusters: List[Dict], output_file: str) -> None:
    """
    Сохраняет по одной «наиболее информативной» новости из каждого кластера (эвристика — длина текста).
    Args:
        clusters: список кластеров
        output_file: путь к выходному CSV файлу
    """
    if not clusters:
        return
        
    rows = []
    for cluster in clusters:
        idx = _get_representative_idx(cluster['news'])
        rows.append({
            'datetime': pd.to_datetime(cluster['datetime'][idx]),
            'channel_name': cluster['channel_name'][idx],
            'message_link': cluster['message_link'][idx],
            'news': cluster['news'][idx],
            'cluster_id': cluster['cluster_id']
        })

    all_news = pd.DataFrame(rows)
    # Сортируем для наглядности — по времени публикации
    all_news.sort_values('datetime', inplace=True)
    all_news.to_csv(output_file, index=False, encoding='utf-8')

# --- Обратная совместимость ---
def save_earliest_news_to_csv(clusters: List[Dict], output_file: str) -> None:  # noqa: D401
    """Alias для сохранения информативных новостей (раннее имя функции)."""
    save_representative_news_to_csv(clusters, output_file)

def process_news_file(args):
    logger.info("Начало обработки файла с новостями")
    clusterer = Clusterer(args.config)

    # Минимальная дата для кластеризации, если указана
    min_dt: Optional[pd.Timestamp] = None
    if getattr(args, "min_datetime", None):
        try:
            # Делаем дату явно tz-aware в UTC, чтобы избежать ошибок сравнения
            min_dt = pd.to_datetime(args.min_datetime, utc=True)
        except Exception as exc:
            logger.error("Некорректный формат --min-datetime: %s", args.min_datetime)
            raise exc

    # Читаем весь файл для определения временного диапазона
    df_dates = pd.read_csv(args.csv_file, usecols=['datetime'], encoding='utf-8')
    df_dates['datetime'] = pd.to_datetime(df_dates['datetime'], utc=True)
    if min_dt is not None:
        df_dates = df_dates[df_dates['datetime'] >= min_dt]
    
    start_date = df_dates['datetime'].min().normalize() if not df_dates.empty else None
    if min_dt is not None and start_date is not None:
        start_date = max(start_date, min_dt.normalize())
    if start_date is None:
        logger.warning("Нет данных для кластеризации после указанной даты")
        return
    
    # Вычисляем общее количество окон
    window_size = timedelta(days=args.window_days)
    window_step = timedelta(days=args.window_step)
    total_windows = int((df_dates['datetime'].max() - start_date).total_seconds() / window_step.total_seconds())
    
    logger.info(f"Общее количество окон для обработки: {total_windows}")
    
    # Задаем размер чанка и инициализируем буфер
    chunksize = 50000
    buffer_df = pd.DataFrame()
    windows_processed = 0
    
    # Создаем прогресс-бар для окон
    with tqdm(total=total_windows, desc="Обработка окон") as window_pbar:
        for chunk in pd.read_csv(args.csv_file, chunksize=chunksize, encoding='utf-8'):
            # Приводим дату к нужному типу
            chunk['datetime'] = pd.to_datetime(chunk['datetime'], utc=True)
            
            # Отбрасываем строки раньше min_dt, если указано
            if min_dt is not None:
                chunk = chunk[chunk['datetime'] >= min_dt]

            # Если после фильтрации чанка нет данных – переходим к следующему
            if chunk.empty:
                continue
            
            # Объединяем с данными из буфера
            if not buffer_df.empty:
                chunk = pd.concat([buffer_df, chunk])
                chunk = chunk.sort_values('datetime')
            
            # Определяем временные границы для текущего чанка
            chunk_start_date = chunk['datetime'].min().normalize()
            chunk_end_date = chunk['datetime'].max()
            
            # Определяем последнее полное окно в текущем чанке
            last_complete_window = chunk_end_date - window_size
            
            # Обрабатываем полные окна
            current_start = chunk_start_date
            while current_start <= last_complete_window:
                current_end = current_start + window_size
                window_df = chunk[
                    (chunk['datetime'] >= current_start) & 
                    (chunk['datetime'] < current_end)
                ]
                
                if not window_df.empty:
                    clusterer.cluster_documents(window_df)
                
                current_start += window_step
                windows_processed += 1
                window_pbar.update(1)
            
            # Определяем данные для буфера
            if current_start is not None:
                buffer_df = chunk[chunk['datetime'] >= current_start].copy()
            else:
                buffer_df = chunk.copy()

    # Обрабатываем оставшиеся данные в буфере
    if not buffer_df.empty:
        logger.info(f"Обработка оставшихся {len(buffer_df)} записей из буфера")
        clusterer.cluster_documents(buffer_df)
        if clusterer.all_clusters:
            logger.debug(f"Найдено {len(clusterer.all_clusters)} кластеров в финальном буфере")
    
    # Сохранение результатов
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.prefix:
            all_clusters_file = os.path.join(args.output_dir, f"all_clusters_{args.prefix}.json")
        else:
            all_clusters_file = os.path.join(args.output_dir, "all_clusters.json")
        
        # Получаем все кластеры
        final_clusters = clusterer.all_clusters
        
        with open(all_clusters_file, 'w', encoding='utf-8') as f:
            result = {
                "total_clusters": len(final_clusters),
                "clustering_date": datetime.now().isoformat(),
                "window_size_days": args.window_days,
                "window_step_days": args.window_step,
                "clusters": final_clusters
            }
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Найдено всего кластеров: {len(final_clusters)}")
        logger.info(f"Результаты сохранены в директорию: {args.output_dir}")

        if args.prefix:
            csv_output_file = os.path.join(args.output_dir, f"representative_{args.prefix}.csv")
        else:
            csv_output_file = os.path.join(args.output_dir, "representative_news.csv")
        save_representative_news_to_csv(final_clusters, csv_output_file)
        logger.info(f"CSV файл с информативными новостями сохранен в {csv_output_file}")
    
    logger.info("Обработка файла завершена")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Кластеризация новостей")
    parser.add_argument("--input-path", type=str, 
                      default="data/external/text/news_with_emb.csv",
                      help="Путь к CSV-файлу с новостями и эмбеддингами")
    parser.add_argument("--config-path", type=str, 
                      default="configs/clusterer_config.json",
                      help="Путь к конфигурационному файлу")
    parser.add_argument("--output-dir", type=str,
                      default="data/external/text/",
                      help="Директория для сохранения результатов")
    parser.add_argument("--window-days", type=float, 
                      default=1.0,
                      help="Размер окна в днях")
    parser.add_argument("--window-step", type=float, 
                      default=1.0,
                      help="Шаг окна в днях")
    parser.add_argument("--dup-threshold", type=float, 
                      default=0.  ,
                      help="Порог определения дубликатов")
    parser.add_argument("--min-datetime", type=str, default=None,
                      help="Минимальная дата (YYYY-MM-DD), с которой выполнять кластеризацию")
    parser.add_argument("--prefix", type=str, default="", help="Префикс для имени выходных файлов (например news или blogs)")
    
    args = parser.parse_args()
    
    # Обновляем имена аргументов для совместимости с process_news_file
    args.csv_file = args.input_path
    args.config = args.config_path
    
    process_news_file(args)