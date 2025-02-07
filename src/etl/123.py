import argparse
import json
import pandas as pd
import torch
from tqdm.auto import tqdm

from embedder import Embedder
from tokenizer import Tokenizer


def compute_embeddings(texts, embedder):
    # Вычисляет эмбеддинги и возвращает их в виде списка списков
    return embedder(texts).tolist()


def tokenize_texts(texts, tokenizer):
    tokenized = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer(text)
        tokens_transformed = [
            f"{t.lemma.lower().replace('_', '')}_{t.pos}"
            for t in tokens if hasattr(t, "lemma") and hasattr(t, "pos")
        ]
        tokenized.append(" ".join(tokens_transformed))
    return tokenized


def process_news(config_path, input_path, output_path="news_with_emb_tokens.csv", news_column="news"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    embedder = Embedder(**config.get("embedder", {}))
    tokenizer = Tokenizer(**config.get("tokenizer", {})) if config.get("tokenizer") else Tokenizer()

    sep = "\t" if input_path.lower().endswith(".tsv") else ","
    df = pd.read_csv(input_path, sep=sep)
    if news_column not in df.columns:
        raise ValueError(f"Колонка '{news_column}' отсутствует в файле {input_path}")

    texts = df[news_column].astype(str).tolist()
    embeddings = compute_embeddings(texts, embedder)
    tokens = tokenize_texts(texts, tokenizer)

    df["embedding"] = [json.dumps(emb) for emb in embeddings]
    df["tokens"] = tokens
    df.to_csv(output_path, index=False)
    print(f"Результат сохранён в {output_path}")


def main(**kwargs):
    process_news(
        config_path=kwargs["annotator_config"],
        input_path=kwargs["input_path"],
        output_path=kwargs.get("output_path", "news_with_emb_tokens.csv"),
        news_column=kwargs.get("news_column", "news")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка сообщений в CSV/TSV-файле.")
    parser.add_argument("--input-path", type=str, default="data/external/news_tg_csv/telegram_news.tsv")
    parser.add_argument("--annotator-config", type=str, default="configs/annotator_config.json")
    # Необязательные аргументы для изменения выходного файла и названия колонки с текстом
    parser.add_argument("--output-path", type=str, default="news_with_emb_tokens.csv")
    parser.add_argument("--news-column", type=str, default="news")
    
    args = parser.parse_args()
    main(**vars(args))