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

def save_earliest_news_to_csv(clusters: List[Dict], output_file: str) -> None:
    """
    Сохранение самых ранних новостей из каждого кластера в CSV файл
    Args:
        clusters: список кластеров
        output_file: путь к выходному CSV файлу
    """
    if not clusters:
        return
        
    # Создаем DataFrame сразу для всех кластеров
    all_news = pd.DataFrame({
        'datetime': [pd.to_datetime(cluster['datetime'][0]) for cluster in clusters],
        'channel_name': [cluster['channel_name'][0] for cluster in clusters],
        'message_link': [cluster['message_link'][0] for cluster in clusters],
        'news': [cluster['news'][0] for cluster in clusters],
        'cluster_id': [cluster['cluster_id'] for cluster in clusters]
    })
    
    # Сортируем по времени и сохраняем
    all_news.sort_values('datetime', inplace=True)
    all_news.to_csv(output_file, index=False, encoding='utf-8')

def process_news_file(args):
    logger.info("Начало обработки файла с новостями")
    clusterer = Clusterer(args.config)

    # Читаем весь файл для определения временного диапазона
    df_dates = pd.read_csv(args.csv_file, usecols=['datetime'], encoding='utf-8')
    df_dates['datetime'] = pd.to_datetime(df_dates['datetime'])
    
    start_date = df_dates['datetime'].min().normalize()
    end_date = df_dates['datetime'].max()
    
    # Вычисляем общее количество окон
    window_size = timedelta(days=args.window_days)
    window_step = timedelta(days=args.window_step)
    total_windows = int((end_date - start_date).total_seconds() / window_step.total_seconds())
    
    logger.info(f"Общее количество окон для обработки: {total_windows}")
    
    # Задаем размер чанка и инициализируем буфер
    chunksize = 50000
    buffer_df = pd.DataFrame()
    windows_processed = 0
    
    # Создаем прогресс-бар для окон
    with tqdm(total=total_windows, desc="Обработка окон") as window_pbar:
        for chunk in pd.read_csv(args.csv_file, chunksize=chunksize, encoding='utf-8'):
            # Приводим дату к нужному типу
            chunk['datetime'] = pd.to_datetime(chunk['datetime'])
            
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

        csv_output_file = os.path.join(args.output_dir, "earliest_news.csv")
        save_earliest_news_to_csv(final_clusters, csv_output_file)
        logger.info(f"CSV файл с самыми ранними новостями сохранен в {csv_output_file}")
    
    logger.info("Обработка файла завершена")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Кластеризация новостей")
    parser.add_argument("--input-path", type=str, 
                      default="data/raw/news_with_emb.csv",
                      help="Путь к CSV-файлу с новостями и эмбеддингами")
    parser.add_argument("--config-path", type=str, 
                      default="configs/clusterer_config.json",
                      help="Путь к конфигурационному файлу")
    parser.add_argument("--output-dir", type=str,
                      default="data/raw/",
                      help="Директория для сохранения результатов")
    parser.add_argument("--window-days", type=float, 
                      default=1.0,
                      help="Размер окна в днях")
    parser.add_argument("--window-step", type=float, 
                      default=1.0,
                      help="Шаг окна в днях")
    parser.add_argument("--dup-threshold", type=float, 
                      default=0.9,
                      help="Порог определения дубликатов")
    
    args = parser.parse_args()
    
    # Обновляем имена аргументов для совместимости с process_news_file
    args.csv_file = args.input_path
    args.config = args.config_path
    
    process_news_file(args)