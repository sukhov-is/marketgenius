import argparse
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict

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
        
        # История кластеров для отслеживания дубликатов
        self.historical_centroids = []
        self.historical_creation_times = []
        self.max_history_days = self.config.get('max_history_days', 7)

    def cluster_documents(self, df: pd.DataFrame) -> List[Dict]:
        """
        Кластеризация новостей
        Args:
            df: DataFrame с новостями
        Returns:
            список кластеров в виде словарей
        """
        if df.empty:
            return []

        # Получение параметров из конфигурации
        clustering_params = self.config.get("clustering", {})
        distances_config = self.config.get("distances", {})
        same_channels_penalty = distances_config.get("same_channels_penalty", 1.0)
        time_penalty_modifier = distances_config.get("time_penalty_modifier", 1.0)
        time_shift_hours = distances_config.get("time_shift_hours", 4)

        # Формирование матрицы эмбеддингов
        embeddings = np.array([json.loads(emb) for emb in df['embedding']])
        
        # Вычисление попарных расстояний
        distances = pairwise_distances(embeddings, metric="cosine")

        # Применение штрафов
        n = len(df)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                    
                # Штраф за одинаковый канал
                if df.iloc[i]['channel_name'] == df.iloc[j]['channel_name']:
                    distances[i, j] *= same_channels_penalty

                # Временной штраф
                if time_penalty_modifier > 1.0:
                    time_diff = abs((df.iloc[i]['datetime'] - df.iloc[j]['datetime']).total_seconds())
                    hours_diff = time_diff / 3600.0
                    hours_shifted = hours_diff - time_shift_hours
                    penalty = 1.0 + expit(hours_shifted) * (time_penalty_modifier - 1.0)
                    distances[i, j] *= penalty

        # Настройка параметров кластеризации
        clustering_params.update({
            "affinity": "precomputed",
            "linkage": "average"
        })

        # Кластеризация
        clustering = AgglomerativeClustering(**clustering_params)
        labels = clustering.fit_predict(distances)

        # Формирование кластеров
        clusters = []
        unique_labels = set(labels)
        current_time = datetime.now()
        
        for label in unique_labels:
            cluster_df = df[labels == label]
            if len(cluster_df) >= self.config.get('min_cluster_size', 2):
                cluster_dict = {
                    'news': cluster_df['news'].tolist(),
                    'datetime': cluster_df['datetime'].tolist(),
                    'channel_name': cluster_df['channel_name'].tolist(),
                    'message_link': cluster_df['message_link'].tolist(),
                    'create_time': current_time.isoformat(),
                    'cluster_id': f"{cluster_df['datetime'].min()}_{label}"
                }
                
                # Проверка на дубликаты
                cluster_centroid = np.mean([json.loads(emb) for emb in cluster_df['embedding']], axis=0)
                if not self.is_duplicate_cluster(cluster_centroid):
                    clusters.append(cluster_dict)
                    self.historical_centroids.append(cluster_centroid)
                    self.historical_creation_times.append(current_time)

        return clusters

    def is_duplicate_cluster(self, centroid: np.ndarray, similarity_threshold: float = 0.9) -> bool:
        """Проверка является ли кластер дубликатом"""
        for hist_centroid in self.historical_centroids:
            similarity = 1 - np.linalg.norm(centroid - hist_centroid)
            if similarity >= similarity_threshold:
                return True
        return False

    def clean_historical_clusters(self, current_time: datetime) -> None:
        """Очистка старых кластеров из истории"""
        valid_indices = [
            i for i, time in enumerate(self.historical_creation_times)
            if (current_time - time).days <= self.max_history_days
        ]
        self.historical_centroids = [self.historical_centroids[i] for i in valid_indices]
        self.historical_creation_times = [self.historical_creation_times[i] for i in valid_indices]

def save_earliest_news_to_csv(clusters: List[Dict], output_file: str) -> None:
    """
    Сохранение самых ранних новостей из каждого кластера в CSV файл
    Args:
        clusters: список кластеров
        output_file: путь к выходному CSV файлу
    """
    earliest_news = []
    
    for cluster in clusters:
        # Создаем временный DataFrame для кластера
        cluster_df = pd.DataFrame({
            'datetime': pd.to_datetime(cluster['datetime']),
            'channel_name': cluster['channel_name'],
            'message_link': cluster['message_link'],
            'news': cluster['news']
        })
        
        # Находим самую раннюю новость в кластере
        earliest_idx = cluster_df['datetime'].idxmin()
        earliest_news.append({
            'datetime': cluster_df.loc[earliest_idx, 'datetime'],
            'channel_name': cluster_df.loc[earliest_idx, 'channel_name'],
            'message_link': cluster_df.loc[earliest_idx, 'message_link'],
            'news': cluster_df.loc[earliest_idx, 'news']
        })
    
    # Создаем DataFrame и сохраняем в CSV
    if earliest_news:
        result_df = pd.DataFrame(earliest_news)
        result_df.sort_values('datetime', inplace=True)
        result_df.to_csv(output_file, index=False, encoding='utf-8')

def process_news_file(args):
    """Основная функция обработки файла новостей"""
    
    logger.info("Начало обработки файла с новостями")
    clusterer = Clusterer(args.config)
    
    df = pd.read_csv(args.csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    logger.info(f"Загружено {len(df)} новостей за период {df['datetime'].min()} - {df['datetime'].max()}")
    
    window_size = timedelta(days=args.window_days)
    window_step = timedelta(days=args.window_step)
    
    start_date = df['datetime'].min()
    end_date = df['datetime'].max()
    current_start = start_date
    
    all_clusters = []
    
    # Подсчет количества итераций для прогресс-бара
    total_windows = int((end_date - start_date) / window_step) + 1
    
    # Создаем прогресс-бар
    with tqdm(total=total_windows, desc="Обработка временных окон") as pbar:
        while current_start <= end_date:
            current_end = current_start + window_size
            window_df = df[(df['datetime'] >= current_start) & (df['datetime'] < current_end)]
            
            clusters = clusterer.cluster_documents(window_df)
            all_clusters.extend(clusters)
            
            clusterer.clean_historical_clusters(datetime.now())
            current_start += window_step
            pbar.update(1)

    if args.output_dir and all_clusters:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Сохранение всех кластеров в один JSON файл
        all_clusters_file = os.path.join(args.output_dir, "all_clusters.json")
        with open(all_clusters_file, 'w', encoding='utf-8') as f:
            result = {
                "total_clusters": len(all_clusters),
                "clustering_date": datetime.now().isoformat(),
                "window_size_days": args.window_days,
                "window_step_days": args.window_step,
                "clusters": all_clusters
            }
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Найдено всего кластеров: {len(all_clusters)}")
        logger.info(f"Результаты сохранены в директорию: {args.output_dir}")
    
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
                      default="data/raw",
                      help="Директория для сохранения результатов")
    parser.add_argument("--window-days", type=float, 
                      default=2.0,
                      help="Размер окна в днях")
    parser.add_argument("--window-step", type=float, 
                      default=1.0,
                      help="Шаг окна в днях")
    parser.add_argument("--dup-threshold", type=float, 
                      default=0.9,
                      help="Порог определения дубликатов")
    
    args = parser.parse_args()
    
    # Обновляем имена аргументов для совместимости с функцией process_news_file
    args.csv_file = args.input_path
    args.config = args.config_path
    
    process_news_file(args)