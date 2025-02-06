import pandas as pd
import json
import argparse
from text import TextProcessor


def load_config(config_path):
    """Загружает конфигурационный файл JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_messages(df, text_processor):
    """Обрабатывает сообщения в датафрейме."""
    df["news"] = df["news"].astype(str).fillna("")

    df["news"] = df["news"].apply(text_processor)

    # Удаление пустых сообщений
    before_removal = len(df)
    df = df[df["news"].str.strip() != ""]
    empty_messages_removed = before_removal - len(df)

    # Удаление дубликатов
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["news"])
    duplicates_removed = before_duplicates - len(df)

    return df, empty_messages_removed, duplicates_removed


def main(input_path, annotator_config):
    """Основная функция обработки CSV/TSV-файла."""
    # Загрузка конфигурации
    config = load_config(annotator_config)
    text_processor = TextProcessor(config["text_processor"])

    # Определяем разделитель на основе расширения файла
    separator = "\t" if input_path.endswith(".tsv") else ","
    
    # Загрузка файла с соответствующим разделителем
    df = pd.read_csv(input_path, encoding="utf-8", sep=separator)

    # Проверка наличия нужного столбца
    if "news" not in df.columns:
        raise ValueError("В файле отсутствует столбец 'news'.")

    # Обработка сообщений
    df, empty_messages_removed, duplicates_removed = process_messages(df, text_processor)

    # Перезапись оригинального файла с тем же разделителем
    df.to_csv(input_path, index=False, encoding="utf-8", sep=separator)

    # Вывод статистики
    print(f"Файл успешно обработан и сохранен в {input_path} (оригинальный файл заменен).")
    print(f"Удалено пустых сообщений: {empty_messages_removed}")
    print(f"Удалено дубликатов: {duplicates_removed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обработка сообщений в CSV-файле.")
    parser.add_argument("--input-path", type=str, default="data/external/news_tg_csv/telegram_news.tsv")
    parser.add_argument("--annotator-config", type=str, default="configs/annotator_config.json")

    args = parser.parse_args()
    main(**vars(args))