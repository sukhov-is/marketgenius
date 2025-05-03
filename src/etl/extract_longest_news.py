import argparse
import json
import os
import logging
from typing import List, Dict

import numpy as np
import pandas as pd

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def _get_longest_news_idx(news_list: List[str]) -> int:
    """Возвращает индекс самой длинной новости в списке."""
    # Если несколько записей имеют одинаковую максимальную длину — выбираем первую
    lengths = [len(text or "") for text in news_list]
    if not lengths:
        return -1 # Возвращаем -1, если список новостей пуст
    return int(np.argmax(lengths))


def extract_and_save_longest_news(input_json_path: str, output_csv_path: str) -> None:
    """
    Читает файл all_clusters.json, извлекает самую длинную новость из каждого кластера
    и сохраняет их в CSV файл.

    Args:
        input_json_path: Путь к входному JSON файлу (all_clusters.json).
        output_csv_path: Путь к выходному CSV файлу.
    """
    if not os.path.exists(input_json_path):
        logger.error(f"Входной файл не найден: {input_json_path}")
        return

    logger.info(f"Чтение файла кластеров: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    clusters = data.get("clusters", [])
    if not clusters:
        logger.warning("В файле не найдено кластеров.")
        return

    logger.info(f"Найдено {len(clusters)} кластеров. Извлечение самых длинных новостей...")

    rows = []
    for cluster in clusters:
        news_list = cluster.get('news', [])
        if not news_list:
            logger.warning(f"Кластер {cluster.get('cluster_id', 'N/A')} не содержит новостей.")
            continue

        idx = _get_longest_news_idx(news_list)
        if idx == -1:
             logger.warning(f"Не удалось найти новость в кластере {cluster.get('cluster_id', 'N/A')}.")
             continue

        # Проверяем наличие всех необходимых ключей и их длину
        required_keys = ['datetime', 'channel_name', 'message_link', 'news']
        valid_cluster = True
        for key in required_keys:
            if key not in cluster or len(cluster[key]) <= idx:
                logger.warning(f"Некорректные данные или недостающая длина для ключа '{key}' в кластере {cluster.get('cluster_id', 'N/A')}. Пропуск кластера.")
                valid_cluster = False
                break
        
        if not valid_cluster:
            continue

        try:
            rows.append({
                'datetime': pd.to_datetime(cluster['datetime'][idx]),
                'channel_name': cluster['channel_name'][idx],
                'message_link': cluster['message_link'][idx],
                'news': cluster['news'][idx],
                'cluster_id': cluster.get('cluster_id', 'N/A') # Используем .get для cluster_id на случай отсутствия
            })
        except IndexError:
             logger.warning(f"IndexError при доступе к элементу {idx} в кластере {cluster.get('cluster_id', 'N/A')}. Пропуск.")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при обработке кластера {cluster.get('cluster_id', 'N/A')}: {e}")


    if not rows:
        logger.warning("Не найдено подходящих новостей для сохранения.")
        return

    longest_news_df = pd.DataFrame(rows)
    # Сортируем для наглядности — по времени публикации
    longest_news_df.sort_values('datetime', inplace=True)

    # Создаем директорию, если она не существует
    output_dir = os.path.dirname(output_csv_path)
    if output_dir: # Убедимся, что output_dir не пустой (если путь - просто имя файла)
        os.makedirs(output_dir, exist_ok=True)

    longest_news_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    logger.info(f"CSV файл с самыми длинными новостями сохранен в {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Извлечение самых длинных новостей из кластеров")
    parser.add_argument("--input-json", type=str,
                        default="data/external/text/all_clusters.json",
                        help="Путь к входному JSON-файлу с кластерами (all_clusters.json)")
    parser.add_argument("--output-csv", type=str,
                        default="data/external/text/representative_news",
                        help="Путь к выходному CSV-файлу для сохранения самых длинных новостей")

    args = parser.parse_args()

    extract_and_save_longest_news(args.input_json, args.output_csv) 