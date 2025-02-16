import argparse
import pandas as pd
from tqdm.auto import tqdm

from tokenizer import Tokenizer


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
    total_tokens = 0
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer(text)
        tokens_transformed = [
            f"{t.lemma.lower().replace('_', '')}_{t.pos}"
            for t in tokens if hasattr(t, "lemma") and hasattr(t, "pos")
        ]
        total_tokens += len(tokens_transformed)
        tokenized.append(" ".join(tokens_transformed))
    print(f"Всего токенов в датасете: {total_tokens}")
    return tokenized


def process_news(input_path, output_path="news_with_tokens.csv", news_column="news"):
    """
    Основная функция обработки новостей: токенизация текстов.
    Args:
        input_path: путь к входному CSV/TSV файлу
        output_path: путь для сохранения результатов
        news_column: название колонки с текстами новостей
    """
    # Инициализируем токенизатор
    tokenizer = Tokenizer()

    # Определяем разделитель на основе расширения файла
    sep = "\t" if input_path.lower().endswith(".tsv") else ","
    df = pd.read_csv(input_path, sep=sep)
    if news_column not in df.columns:
        raise ValueError(f"Колонка '{news_column}' отсутствует в файле {input_path}")

    # Токенизируем тексты
    texts = df[news_column].astype(str).tolist()
    tokens = tokenize_texts(texts, tokenizer)

    # Сохраняем результаты
    df["tokens"] = tokens
    df.to_csv(output_path, index=False)
    print(f"Результат сохранён в {output_path}")


def main(**kwargs):
    """
    Точка входа в программу, обрабатывает аргументы командной строки.
    """
    process_news(
        input_path=kwargs["input_path"],
        output_path=kwargs.get("output_path", "news_with_tokens.csv"),
        news_column=kwargs.get("news_column", "news")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Токенизация сообщений в CSV/TSV-файле.")
    parser.add_argument("--input-path", type=str, default="data/raw/earliest_news.csv")
    parser.add_argument("--output-path", type=str, default="data/raw/news_with_tokens.csv")
    parser.add_argument("--news-column", type=str, default="news")
    
    args = parser.parse_args()
    main(**vars(args))