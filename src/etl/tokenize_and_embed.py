import argparse
import json
import pandas as pd
import torch
from tqdm.auto import tqdm

from embedder import Embedder
from tokenizer import Tokenizer


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


def tokenize_texts(texts, tokenizer):
    """
    Токенизирует список текстов с преобразованием в формат "лемма_часть_речи".
    Args:
        texts: список текстов для токенизации
        tokenizer: токенизатор для обработки текстов
    Returns:
        список токенизированных текстов
    """
    tokenized = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer(text)
        # Преобразуем каждый токен в формат "лемма_часть_речи"
        tokens_transformed = [
            f"{t.lemma.lower().replace('_', '')}_{t.pos}"
            for t in tokens if hasattr(t, "lemma") and hasattr(t, "pos")
        ]
        tokenized.append(" ".join(tokens_transformed))
    return tokenized


def process_news(config_path, input_path, output_path="news_with_emb_tokens.csv", news_column="news"):
    """
    Основная функция обработки новостей: загружает данные, создает эмбеддинги и токены.
    Args:
        config_path: путь к конфигурационному файлу
        input_path: путь к входному CSV/TSV файлу
        output_path: путь для сохранения результатов
        news_column: название колонки с текстами новостей
    """
    # Загружаем конфигурацию
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Инициализируем модели
    embedder = Embedder(**config.get("embedder", {}))
    tokenizer = Tokenizer(**config.get("tokenizer", {})) if config.get("tokenizer") else Tokenizer()

    # Определяем разделитель на основе расширения файла
    sep = "\t" if input_path.lower().endswith(".tsv") else ","
    df = pd.read_csv(input_path, sep=sep)
    if news_column not in df.columns:
        raise ValueError(f"Колонка '{news_column}' отсутствует в файле {input_path}")

    # Обрабатываем тексты
    texts = df[news_column].astype(str).tolist()
    embeddings = compute_embeddings(texts, embedder)
    tokens = tokenize_texts(texts, tokenizer)

    # Сохраняем результаты
    df["embedding"] = [json.dumps(emb) for emb in embeddings]
    df["tokens"] = tokens
    df.to_csv(output_path, index=False)
    print(f"Результат сохранён в {output_path}")


def main(**kwargs):
    """
    Точка входа в программу, обрабатывает аргументы командной строки.
    Args:
        kwargs: словарь с аргументами командной строки
    """
    process_news(
        config_path=kwargs["annotator_config"],
        input_path=kwargs["input_path"],
        output_path=kwargs.get("output_path", "news_with_emb_tokens.csv"),
        news_column=kwargs.get("news_column", "news")
    )


if __name__ == "__main__":
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(description="Обработка сообщений в CSV/TSV-файле.")
    parser.add_argument("--input-path", type=str, default="data/processed/telegram_news.csv")
    parser.add_argument("--annotator-config", type=str, default="configs/annotator_config.json")
    parser.add_argument("--output-path", type=str, default="data/processed/news_with_emb_tokens.csv")
    parser.add_argument("--news-column", type=str, default="news")
    
    args = parser.parse_args()
    main(**vars(args))