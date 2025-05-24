"""End-to-end pipeline: detect new dates, generate messages, and publish.
"""
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time
import sys
# Определяем корневую директорию проекта
project_root = Path(__file__).resolve().parent.parent

# Добавляем корневую директорию в sys.path для корректных импортов
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.etl.generate_telegram_messages import generate_messages
from src.etl.telegram_publisher import send_message, load_channel_mapping, linkify, italicize_disclaimer

CSV_BLOGS = Path("data/processed/gpt/telegram_blogs.csv")
CSV_NEWS = Path("data/processed/gpt/telegram_news.csv")
OUTPUT_DIR = Path("data/processed/gpt/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID", "@marketgenius_blog")
CFG_PATH = Path("configs/channels_config.json")
MAPPING = load_channel_mapping(CFG_PATH) if CFG_PATH.exists() else {}


def dates_in_output(txt_path: Path) -> set[str]:
    if not txt_path.exists():
        return set()
    dates = set()
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("📅 "):
            parts = line.split(" ", 1)
            if len(parts) > 1:
                date_str = parts[1].strip()
                if len(date_str) == 10 and date_str[2] == '.' and date_str[5] == '.':
                    dates.add(date_str)
    return dates


def main():
    if not TOKEN:
        raise RuntimeError("TG_BOT_TOKEN not set")

    blogs_out = OUTPUT_DIR / "blogs.txt"
    news_out = OUTPUT_DIR / "news.txt"

    existing_blog_dates = dates_in_output(blogs_out)
    existing_news_dates = dates_in_output(news_out)

    blogs_df = pd.read_csv(CSV_BLOGS)
    news_df = pd.read_csv(CSV_NEWS)

    new_blog_dates = sorted({d for d in blogs_df["date"].unique() if datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m.%Y") not in existing_blog_dates})
    new_news_dates = sorted({d for d in news_df["date"].unique() if datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m.%Y") not in existing_news_dates})

    # Generate messages for new dates
    top_n = 33
    blogs_msgs = generate_messages(str(CSV_BLOGS), mode="blogs", n=top_n, start_date=min(new_blog_dates) if new_blog_dates else None, end_date=max(new_blog_dates) if new_blog_dates else None)
    news_msgs = generate_messages(str(CSV_NEWS), mode="news", n=top_n, start_date=min(new_news_dates) if new_news_dates else None, end_date=max(new_news_dates) if new_news_dates else None)

    blog_dict = {d: m for d, m in blogs_msgs}
    news_dict = {d: m for d, m in news_msgs}

    # Merge dates in chronological order
    all_dates = sorted(set(list(news_dict.keys()) + list(blog_dict.keys())), key=lambda x: datetime.strptime(x, "%d.%m.%Y"))

    for d in all_dates:
        # news first
        if d in news_dict:
            msg = news_dict[d]
            msg = linkify(msg, MAPPING)
            msg = italicize_disclaimer(msg)
            if send_message(TOKEN, CHAT_ID, msg):
                time.sleep(2)
        # blogs
        if d in blog_dict:
            msg = blog_dict[d]
            msg = linkify(msg, MAPPING)
            msg = italicize_disclaimer(msg)
            if send_message(TOKEN, CHAT_ID, msg):
                time.sleep(2)

    # Append actually published messages
    with blogs_out.open("a", encoding="utf-8") as fb:
        for date_fmt, msg in blogs_msgs:
            if date_fmt not in existing_blog_dates:
                fb.write(msg + "\n" + ("-" * 40) + "\n")

    with news_out.open("a", encoding="utf-8") as fn:
        for date_fmt, msg in news_msgs:
            if date_fmt not in existing_news_dates:
                fn.write(msg + "\n" + ("-" * 40) + "\n")


if __name__ == "__main__":
    main() 