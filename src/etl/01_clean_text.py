import re
import json
import argparse
import logging
from typing import Any, Dict, Tuple

import pandas as pd

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Регулярные выражения для очистки текста
EMOJI_PATTERN = re.compile(
    (
        "["
        "\U0001F1E0-\U0001F1FF"
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002300-\U00002BFF"
        "\U0000FE0E-\U0000FE0F"
        "\U0001F195"  
        "]"
    ),
    flags=re.UNICODE,
)
URL_PATTERN = re.compile(r"(http\S+|www\.\S+)", flags=re.UNICODE)
URL_WITHOUT_HTTP_PATTERN = re.compile(r"[\S]+\.(ru|me|com|org)[/][\S]+", flags=re.UNICODE)
USERS_PATTERN = re.compile(r"\s@(\w+)", flags=re.UNICODE)
HASHTAG_PATTERN = re.compile(r"#(\w+)", flags=re.UNICODE)


def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def remove_hashtags(text: str) -> str:
    return HASHTAG_PATTERN.sub("", text)


def remove_urls(text: str) -> str:
    cleaned = URL_PATTERN.sub("", text)
    return URL_WITHOUT_HTTP_PATTERN.sub("", cleaned)


def remove_users(text: str) -> str:
    return USERS_PATTERN.sub("", text)


def remove_bad_punct(text: str) -> str:
    text = text.replace(". .", ".").replace("..", ".")
    text = text.replace("« ", "«").replace(" »", "»").replace(" :", ":")
    return text.replace("\xa0", " ")


def fix_paragraphs(text: str) -> str:
    paragraphs = [" ".join(p.split()).strip() for p in text.splitlines()]
    paragraphs = [p for p in paragraphs if len(p) >= 3]
    return "\n".join(paragraphs)


class TextCleaner:
    """
    Класс последовательной очистки текста на основе конфигурации.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.pipeline = [
            remove_emoji,
            remove_hashtags,
            remove_users,
            remove_urls,
            remove_bad_punct,
            fix_paragraphs,
        ]
        self.skip_substrings = set(config.get("skip_substrings", []))
        self.rm_substrings = config.get("rm_substrings", [])
        self.obscene_substrings = set(config.get("obscene_substrings", []))
        self.filter_words = set(w.lower() for w in config.get("filter_words", []))

    def clean(self, text: str) -> str:
        """
        Полная очистка текста. Возвращает пустую строку, если текст не проходит фильтры.
        """
        if not text:
            return ""
        txt = text.strip()
        if self._should_skip(txt) or self._has_obscene(txt):
            return ""
        txt = self._remove_substrings(txt)
        for func in self.pipeline:
            txt = func(txt)
        if self._has_obscene(txt):
            return ""
        return self._remove_substrings(txt).strip()

    def _should_skip(self, text: str) -> bool:
        lower = text.lower()
        return any(word in lower for word in self.filter_words)

    def _has_obscene(self, text: str) -> bool:
        return any(sub in text for sub in self.obscene_substrings)

    def _remove_substrings(self, text: str) -> str:
        for sub in self.rm_substrings:
            text = text.replace(sub, " ")
        return text



def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурационный файл JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_messages(
    df: pd.DataFrame,
    cleaner: TextCleaner
) -> Tuple[pd.DataFrame, int, int]:
    """
    Обрабатывает сообщения: фильтрация по дате, очистка текста, удаление пустых.
    """
    df = df.copy()
    if "news" not in df.columns:
        logger.error("Отсутствует столбец 'news'.")
        raise KeyError("Отсутствует столбец 'news'.")

    df["news"] = df["news"].astype(str).fillna("")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    before_date = len(df)
    df = df[df["datetime"] >= pd.to_datetime("2019-02-01", utc=True)]
    date_filtered = before_date - len(df)

    df["news"] = df["news"].apply(cleaner.clean)

    before_empty = len(df)
    df = df[df["news"].str.strip() != ""]
    empty_removed = before_empty - len(df)

    return df, empty_removed, date_filtered


def main(input_path: str, output_path: str, config_path: str) -> None:
    """
    Основная функция: читает CSV/TSV, очищает тексты и сохраняет результат.
    """
    cfg = load_config(config_path)
    cleaner = TextCleaner(cfg.get("text_processor", {}))

    sep_in = "\t" if input_path.lower().endswith(".tsv") else ","
    df = pd.read_csv(input_path, sep=sep_in, encoding="utf-8")

    df, empty_rm, date_rm = process_messages(df, cleaner)

    sep_out = "\t" if output_path.lower().endswith(".tsv") else ","
    df.to_csv(output_path, index=False, sep=sep_out, encoding="utf-8")

    logger.info("Processed file saved to %s", output_path)
    logger.info("Records before 01.02.2019 removed: %d", date_rm)
    logger.info("Empty messages removed: %d", empty_rm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Очистка и ETL текстовых сообщений CSV/TSV" )
    parser.add_argument("--input-path", type=str, default='data/external/news_tg_csv/telegram_news.csv', help="Путь к входному файлу CSV/TSV")
    parser.add_argument("--output-path", type=str, default='data/external/text/news_cleaned.csv', help="Путь для сохранения результата")
    parser.add_argument("--config-path", type=str, default='configs/annotator_config.json', help="Путь к конфигурации JSON")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.config_path) 