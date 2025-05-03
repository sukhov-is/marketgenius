# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный скрипт для вычисления эмбеддингов текстов из CSV/TSV файлов.
Пример использования:
    python compute_embeddings.py \
        --config configs/annotator_config.json \
        --input data/raw/news_filtred.csv \
        --column news \
        --output data/raw/news_with_emb.csv
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

# Детерминированность и конфигурация CUBLAS (если CUDA доступна)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_random_seed(seed: int) -> None:
    """
    Устанавливает seed для всех генераторов для воспроизводимости.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def gen_batch(records: List, batch_size: int) -> Iterable[List]:
    """
    Разбивает список на батчи заданного размера.
    """
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


class Embedder:
    """
    Класс для создания эмбеддингов текста с использованием трансформер-моделей HuggingFace.
    """
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        max_length: int = 128,
        device: Optional[str] = None,
        pooling_method: str = "default",
        normalize: bool = True,
        text_prefix: str = "",
        seed: int = 56154,
    ) -> None:
        set_random_seed(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling_method = pooling_method
        self.normalize = normalize
        self.text_prefix = text_prefix

    def __call__(self, texts: List[str]) -> torch.Tensor:
        """
        Вычисляет эмбеддинги для списка текстов.
        """
        self.model.eval()
        batches = list(gen_batch(texts, self.batch_size))
        outputs = []
        for batch in tqdm(batches, desc="Вычисление эмбеддингов", unit="batch"):
            batch = [self.text_prefix + text for text in batch]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                result = self.model(**inputs)
                if self.pooling_method == "mean":
                    hidden = result.last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    summed = (hidden * mask).sum(dim=1)
                    counts = mask.sum(dim=1)
                    batch_emb = summed / counts
                elif self.pooling_method == "cls":
                    batch_emb = result.last_hidden_state[:, 0]
                else:  # default pooler_output
                    batch_emb = result.pooler_output
                if self.normalize:
                    batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            outputs.append(batch_emb.cpu())
        return torch.cat(outputs, dim=0)


def compute_embeddings(texts: List[str], embedder: Embedder) -> List[List[float]]:
    """
    Обёртка для получения эмбеддингов в виде списка списков.
    """
    tensor = embedder(texts)
    return tensor.numpy().tolist()


def process_texts(
    config_path: Path,
    input_path: Path,
    output_path: Path,
    text_column: str,
) -> None:
    """
    Загружает данные, вычисляет эмбеддинги и сохраняет результаты в CSV.
    """
    if not config_path.is_file():
        logging.error("Конфигурационный файл не найден: %s", config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    embed_cfg = config.get("embedder", config)
    embedder = Embedder(**embed_cfg)

    if not input_path.is_file():
        logging.error("Входной файл не найден: %s", input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")

    sep = "\t" if input_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(input_path, sep=sep)
    if text_column not in df.columns:
        logging.error("Колонка '%s' не найдена в файле %s", text_column, input_path)
        raise KeyError(f"Column '{text_column}' not found.")

    texts = df[text_column].astype(str).tolist()
    embeddings = compute_embeddings(texts, embedder)

    df["embedding"] = [json.dumps(e, ensure_ascii=False) for e in embeddings]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Результат сохранён в: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скрипт вычисления эмбеддингов текстов.")
    parser.add_argument(
        "--config", "-c", type=Path, default=os.path.join("configs", "annotator_config.json"), help="Путь к конфигу модели (JSON)."
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=os.path.join("data", "external", "text", "news_cleaned.csv"), help="Входной CSV/TSV файл."
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=os.path.join("data", "external", "text", "news_with_emb.csv"), help="Выходной CSV файл."
    )
    parser.add_argument(
        "--column", "-col", type=str, default="news", help="Название колонки с текстом."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        process_texts(args.config, args.input, args.output, args.column)
    except Exception:
        logging.exception("Ошибка при выполнении скрипта")
        sys.exit(1)

if __name__ == "__main__":
    main()