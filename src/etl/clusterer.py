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
from joblib import parallel_backend

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
        self.historical_centroids = []       # Список центроидов кластеров
        self.historical_creation_times = []  # Время создания кластеров
        self.historical_clusters = []        # Сами кластеры (словарь с данными)
        self.max_history_days = self.config.get('max_history_days', 3)
        
        # Порог определения дубликатов (можно задать в конфиге)
        self.dup_threshold = self.config.get('dup_threshold', 0.9)

        # Параметры для вычисления расстояний
        self.same_channels_penalty = self.config['distances']['same_channels_penalty']
        self.time_penalty_modifier = self.config['distances']['time_penalty_modifier']
        self.time_shift_hours = self.config['distances']['time_shift_hours']

    def cluster_documents(self, df: pd.DataFrame) -> List[Dict]:
        """
        Кластеризация новостей
        Args:
            df: DataFrame с новостями
        Returns:
            список кластеров в виде словарей
        """     
        if len(df) < 5:
            logger.debug(f"Пропуск окна: недостаточно новостей ({len(df)})")
            return

        if df.empty:
            return

        # Константы для ограничения расстояний
        min_distance = 0.0
        max_distance = 1.0

        # Предварительно вычисляем эмбеддинги для всего датафрейма
        embedding = np.array([json.loads(emb) for emb in df['embedding']])
        
        # Вычисляем базовую матрицу расстояний (используем метрику cosine)
        with parallel_backend('loky', n_jobs=-1):
            distances = pairwise_distances(embedding, metric="cosine")
        
        # Применяем штраф за одинаковые каналы
        same_channels = np.array([[1 if df.iloc[i]['channel_name'] == df.iloc[j]['channel_name'] 
                                else 0 for j in range(len(df))] for i in range(len(df))])
        distances = np.minimum(max_distance, 
                            distances * (1 + (self.same_channels_penalty - 1) * same_channels))

        # Применяем временные штрафы
        if self.time_penalty_modifier > 1.0:
            times = df['datetime'].values.reshape(-1, 1)
            time_diffs = np.abs(times - times.T) / np.timedelta64(1, 'h')
            hours_shifted = time_diffs - self.time_shift_hours
            time_penalties = 1.0 + expit(hours_shifted) * (self.time_penalty_modifier - 1.0)
            distances = np.minimum(max_distance, distances * time_penalties)

        clustering_params = self.config.get("clustering", {}).copy()
        clustering_params.update({
            "metric": "precomputed",
            "linkage": "average"
        })

        # Выполняем агломеративную кластеризацию
        clustering = AgglomerativeClustering(**clustering_params)
        labels = clustering.fit_predict(distances)

        clusters = []
        unique_labels = set(labels)
        current_time = datetime.now()
        
        for label in unique_labels:
            cluster_df = df[labels == label]
            # Преобразование столбца datetime в строковый формат ISO
            datetimes_str = cluster_df['datetime'].apply(
                lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
            ).tolist()
            # Получаем минимальную дату в виде строки
            min_datetime = (cluster_df['datetime'].min().isoformat() 
                            if hasattr(cluster_df['datetime'].min(), 'isoformat') 
                            else str(cluster_df['datetime'].min()))
            
            cluster_dict = {
                'news': cluster_df['news'].tolist(),
                'datetime': datetimes_str,
                'channel_name': cluster_df['channel_name'].tolist(),
                'message_link': cluster_df['message_link'].tolist(),
                'create_time': current_time.isoformat(),
                'cluster_id': f"{min_datetime}_{label}"
            }
            
            # Вычисляем центроид кластера
            cluster_centroid = np.mean([json.loads(emb) for emb in cluster_df['embedding']], axis=0)
            # Проверяем, является ли этот кластер дубликатом ранее найденного
            dup_index = self.find_duplicate_cluster(cluster_centroid, self.dup_threshold)
            if dup_index is None:
                clusters.append(cluster_dict)
                self.historical_clusters.append(cluster_dict)
                self.historical_centroids.append(cluster_centroid)
                self.historical_creation_times.append(current_time)
            else:
                existing_cluster = self.historical_clusters[dup_index]
                existing_cluster['news'].extend(cluster_dict['news'])
                existing_cluster['datetime'].extend(cluster_dict['datetime'])
                existing_cluster['channel_name'].extend(cluster_dict['channel_name'])
                existing_cluster['message_link'].extend(cluster_dict['message_link'])
                logger.info(f"Объединение кластера {cluster_dict['cluster_id']} с существующим кластером {existing_cluster['cluster_id']}")
        
        return clusters

    def find_duplicate_cluster(self, centroid: np.ndarray, similarity_threshold: float = 0.9):
        """
        Поиск дублирующегося кластера по центроиду.
        Возвращает индекс найденного дубликата в истории, или None, если дубликат не найден.
        """
        for i, hist_centroid in enumerate(self.historical_centroids):
            similarity = 1 - np.linalg.norm(centroid - hist_centroid)
            if similarity >= similarity_threshold:
                return i
        return None

    def clean_historical_clusters(self, current_time: datetime) -> None:
        """Очистка старых кластеров из истории"""
        valid_indices = [
            i for i, time in enumerate(self.historical_creation_times)
            if (current_time - time).days <= self.max_history_days
        ]
        self.historical_centroids = [self.historical_centroids[i] for i in valid_indices]
        self.historical_creation_times = [self.historical_creation_times[i] for i in valid_indices]
        # self.historical_clusters = [self.historical_clusters[i] for i in valid_indices]

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
    # Если порог дубликатов передан через аргументы, переопределяем значение из конфига
    if hasattr(args, 'dup_threshold'):
        clusterer.dup_threshold = args.dup_threshold
    
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

        # Сохранение самых ранних новостей в CSV файл
        csv_output_file = os.path.join(args.output_dir, "earliest_news.csv")
        save_earliest_news_to_csv(all_clusters, csv_output_file)
        logger.info(f"CSV файл с самыми ранними новостями сохранен в {csv_output_file}")
    
    logger.info("Обработка файла завершена")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Кластеризация новостей")
    parser.add_argument("--input-path", type=str, 
                      default="data/5000_with_emb.csv",
                      help="Путь к CSV-файлу с новостями и эмбеддингами")
    parser.add_argument("--config-path", type=str, 
                      default="configs/clusterer_config.json",
                      help="Путь к конфигурационному файлу")
    parser.add_argument("--output-dir", type=str,
                      default="data/",
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