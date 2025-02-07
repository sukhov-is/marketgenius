import argparse
import json
import pandas as pd
import torch
from tqdm.auto import tqdm

from embedder import Embedder


def compute_embeddings(texts, embedder):
    """
    Вычисляет эмбеддинги для списка текстов.
    Args:
        texts: список текстов для обработки
        embedder: модель для создания эмбеддингов
    Returns:
        список векторов (эмбеддингов) для каждого текста
    """
    embeddings = embedder(texts)  # Получаем тензор эмбеддингов
    return embeddings.cpu().numpy().tolist()  # Преобразуем тензор в список


def process_news(config_path, input_path, output_path="news_with_emb.csv", news_column="news"):
    """
    Основная функция обработки новостей: загружает данные и создает эмбеддинги.
    Args:
        config_path: путь к конфигурационному файлу
        input_path: путь к входному CSV/TSV файлу
        output_path: путь для сохранения результатов
        news_column: название колонки с текстами новостей
    """
    # Загружаем конфигурацию
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Инициализируем модель эмбеддинга
    embedder = Embedder(**config.get("embedder", {}))

    # Определяем разделитель на основе расширения файла
    sep = "\t" if input_path.lower().endswith(".tsv") else ","
    df = pd.read_csv(input_path, sep=sep)

    # Обрабатываем тексты
    texts = df[news_column].astype(str).tolist()
    embeddings = compute_embeddings(texts, embedder)

    # Сохраняем результаты
    df["embedding"] = [json.dumps(emb) for emb in embeddings]
    df.to_csv(output_path, index=False)
    print(f"Результат сохранён в {output_path}")


def main(**kwargs):

    process_news(
        config_path=kwargs["annotator_config"],
        input_path=kwargs["input_path"],
        output_path=kwargs.get("output_path", "news_with_emb.csv"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка сообщений в CSV/TSV-файле.")
    parser.add_argument("--input-path", type=str, default="data/processed/telegram_news.csv")
    parser.add_argument("--annotator-config", type=str, default="configs/annotator_config.json")
    parser.add_argument("--output-path", type=str, default="data/processed/news_with_emb.csv")
    
    args = parser.parse_args()
    main(**vars(args))