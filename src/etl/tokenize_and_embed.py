import pandas as pd
import json
import argparse
from tokenizer import Tokenizer
from embedder import Embedder
from tqdm import tqdm


def load_config(config_path):
    """Загружает конфигурационный файл JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_batch(texts, tokenizer, embedder):
    """Обрабатывает батч текстов: токенизация и эмбеддинги."""
    # Токенизация
    tokens = []
    for text in texts:
        text_tokens = tokenizer(text)
        tokens.append(" ".join([
            f"{t.lemma.lower().replace('_', '')}_{t.pos}" for t in text_tokens
        ]))
    
    # Получение эмбеддингов
    embeddings = embedder(texts)
    
    return tokens, embeddings.cpu().numpy().tolist()


def process_dataframe(df, tokenizer, embedder, batch_size=32):
    """Обрабатывает весь датафрейм батчами."""
    all_tokens = []
    all_embeddings = []
    
    # Обработка батчами с progress bar
    for i in tqdm(range(0, len(df), batch_size), desc="Обработка текстов"):
        batch_texts = df['news'].iloc[i:i + batch_size].tolist()
        tokens, embeddings = process_batch(batch_texts, tokenizer, embedder)
        all_tokens.extend(tokens)
        all_embeddings.extend(embeddings)
    
    df['tokens'] = all_tokens
    df['embedding'] = all_embeddings
    
    return df


def main(input_path, output_path, config_path):
    """Основная функция обработки файла."""
    # Загрузка конфигурации
    config = load_config(config_path)
    
    # Инициализация моделей
    tokenizer = Tokenizer()
    embedder = Embedder(**config["embedder"])
    
    # Определяем разделитель на основе расширения файла
    separator = "\t" if input_path.endswith(".tsv") else ","
    
    # Загрузка файла
    print(f"Загрузка файла {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8", sep=separator)
    
    # Проверка наличия нужного столбца
    if "news" not in df.columns:
        raise ValueError("В файле отсутствует столбец 'news'")
    
    # Заполнение пустых значений
    df["news"] = df["news"].astype(str).fillna("")
    
    # Обработка текстов
    print("Начало обработки текстов...")
    df = process_dataframe(df, tokenizer, embedder)
    
    # Сохранение результатов
    print(f"Сохранение результатов в {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print("Обработка завершена успешно!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Токенизация и эмбеддинги для текстов.")
    parser.add_argument("--input-path", type=str, default="data/external/news_tg_csv/blogs.csv",
                      help="Путь к входному файлу CSV/TSV")
    parser.add_argument("--output-path", type=str, default="data/raw/news_tokens.csv",
                      help="Путь для сохранения обработанного файла")
    parser.add_argument("--config-path", type=str, default="configs/annotator_config.json",
                      help="Путь к конфигурационному файлу")

    args = parser.parse_args()
    main(**vars(args)) 