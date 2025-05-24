"""Telegram publisher for MarketGenius.

Считывает сгенерированные сообщения из текстовых файлов и отправляет их
в канал Telegram с помощью Bot API.

Перед использованием установите переменные окружения:
    TG_BOT_TOKEN  – токен бота @market_genius_bot
    TG_CHAT_ID    – @marketgenius_blog или numeric chat ID
"""
from pathlib import Path
import os
import time
import json
import requests
import argparse
import re
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.telegram.org/bot{token}/{method}"


def load_channel_mapping(cfg_path: Path) -> dict:
    """Загружает JSON и возвращает {name: url}."""
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    mapping = {}
    for section in data.values():
        for url, name in section.items():
            if not url.startswith("http"):
                url_full = "https://" + url.lstrip("/")
            else:
                url_full = url
            mapping[name] = url_full
    return mapping


def linkify(text: str, mapping: dict) -> str:
    """Заменяет упоминания источников на Markdown ссылки."""
    for name, url in mapping.items():
        # Используем регулярное выражение для регистронезависимого поиска
        # re.escape нужен для экранирования специальных символов в названии
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        # Заменяем все вхождения, сохраняя оригинальный текст в ссылке
        text = pattern.sub(lambda m: f"[{m.group(0)}]({url})", text)
    return text


def italicize_disclaimer(text: str) -> str:
    """Оборачивает строки с дисклеймером в курсив."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("Сообщение сгенерировано") or ln.startswith("Оценки (" ):
            if not (ln.startswith("_") or ln.startswith("*")):
                lines[i] = f"_{ln}_"
    return "\n".join(lines)


def bold_headers(text: str) -> str:
    text = text.replace("📝 Анализ настроений:", "📝 *Анализ настроений:*")
    text = text.replace("📰 Самари новостей:", "📰 *Самари новостей:*")
    return text


def send_message(token: str, chat_id: str, text: str, parse_mode: str = "Markdown", max_retries: int = 5) -> bool:
    """Отправка одного сообщения с автоматическим ретраем при 429."""
    url = API_URL.format(token=token, method="sendMessage")
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200 and resp.json().get("ok"):
            return True
        if resp.status_code == 429:
            retry_sec = resp.json().get("parameters", {}).get("retry_after", 30)
            print(f"[RATE LIMIT] waiting {retry_sec}s (attempt {attempt+1}/{max_retries})")
            time.sleep(retry_sec + 1)
            continue
        print("[ERROR]", resp.text)
        return False
    return False


def publish_from_file(file_path: Path, token: str, chat_id: str, delay: float = 1.0, mapping: dict | None = None):
    """Читает файл, разделённый строками из 40+ дефисов, и публикует каждый блок."""
    content = file_path.read_text(encoding="utf-8")
    # Сообщения разделены строкой из 40 дефисов, как в генераторе
    blocks = [b.strip() for b in content.split("-" * 40) if b.strip()]
    for block in blocks:
        if mapping:
            block = linkify(block, mapping)
        block = italicize_disclaimer(block)
        # block = bold_headers(block)
        ok = send_message(token, chat_id, block)
        if not ok:
            print("Failed to send block, aborting.")
            break
        time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Публикация сообщений из файлов в Telegram канал.")
    parser.add_argument("file", type=str, help="Путь к txt-файлу с сообщениями")
    parser.add_argument("--delay", type=float, default=1.0, help="Пауза между сообщениями (сек)")
    parser.add_argument("--config", type=str, default="configs/channels_config.json", help="JSON с источниками")
    args = parser.parse_args()

    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID", "@marketgenius_blog")

    if not token:
        parser.error("Переменная окружения TG_BOT_TOKEN не установлена")

    cfg_path = Path(args.config)
    mapping = load_channel_mapping(cfg_path) if cfg_path.exists() else None

    publish_from_file(Path(args.file), token, chat_id, args.delay, mapping) 