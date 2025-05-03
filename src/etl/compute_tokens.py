#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для токенизации текстов из CSV/TSV-файлов с использованием Natasha.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class Tokenizer:
    """
    Класс-обёртка для токенизации текста с использованием Natasha.
    """
    def __init__(self) -> None:
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.embedding = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.embedding)

    def __call__(self, text: str) -> list:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        return doc.tokens


def tokenize_texts(texts: list, tokenizer: Tokenizer) -> list:
    """
    Токенизирует список текстов и возвращает токены в формате 'лемма_POS'.
    """
    tokenized = []
    total_tokens = 0
    for text in tqdm(texts, desc="Tokenizing", unit="doc"):
        tokens = tokenizer(text)
        formatted = [
            f"{token.lemma.lower().replace('_','')}_{token.pos}"
            for token in tokens
            if getattr(token, "lemma", None) and getattr(token, "pos", None)
        ]
        total_tokens += len(formatted)
        tokenized.append(" ".join(formatted))
    logger.info("Всего токенов: %d", total_tokens)
    return tokenized


def process_file(
    input_path: Path,
    output_path: Path,
    news_column: str,
    tokenizer: Tokenizer
) -> None:
    """
    Загружает данные, токенизирует столбец с новостями и сохраняет результат.
    """
    if not input_path.exists():
        logger.error("Входной файл '%s' не найден", input_path)
        sys.exit(1)

    sep = "\t" if input_path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(input_path, sep=sep)
    except Exception as e:
        logger.error("Ошибка чтения '%s': %s", input_path, e)
        sys.exit(1)

    if news_column not in df.columns:
        logger.error("Колонка '%s' отсутствует в файле '%s'", news_column, input_path)
        sys.exit(1)

    texts = df[news_column].astype(str).tolist()
    df['tokens'] = tokenize_texts(texts, tokenizer)

    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        logger.error("Ошибка сохранения '%s': %s", output_path, e)
        sys.exit(1)

    logger.info("Результат сохранен в '%s'", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Токенизация текстов CSV/TSV"
    )
    parser.add_argument(
        "-i", "--input-path", type=Path, required=True,
        help="Путь к входному CSV/TSV файлу"
    )
    parser.add_argument(
        "-o", "--output-path", type=Path,
        default=Path("news_with_tokens.csv"),
        help="Путь для сохранения результата"
    )
    parser.add_argument(
        "-c", "--news-column", type=str, default="news",
        help="Название колонки с текстами новостей"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer()
    process_file(
        input_path=args.input_path,
        output_path=args.output_path,
        news_column=args.news_column,
        tokenizer=tokenizer
    )

if __name__ == "__main__":
    main()