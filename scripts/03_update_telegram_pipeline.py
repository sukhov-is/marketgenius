from __future__ import annotations

"""
update_pipeline.py
------------------
Скрипт для инкрементальной загрузки новостей и блогов из Telegram,
их последующей очистки, вычисления эмбеддингов и кластеризации.

Алгоритм работы:
1. Определить дату последнего сохранённого сообщения в «сырых» CSV
   (data/external/news_tg_csv/telegram_news.csv, blogs.csv).
2. С помощью модифицированного `TelegramParser` загрузить только те
   сообщения, которые опубликованы **позже** найденной даты.
3. Дописать новые сообщения в исходные CSV (без перезаписи).
4. Очистить тексты (TextCleaner) -> сохранить в *_cleaned.csv.
5. Вычислить эмбеддинги -> сохранить в *_with_emb.csv.
6. Запустить кластеризацию для обновлённого файла с эмбеддингами.

Требования:
- В .env должны быть заданные API_ID, API_HASH, PHONE.
- Файлы конфигураций используются те же, что и в остальных модулях.

Запуск:
    python src/etl/update_pipeline.py --content-type all
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Инициализация окружения и логов
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE")

if not API_ID or not API_HASH or not PHONE:
    logger.critical("API_ID, API_HASH, PHONE должны быть определены в переменных окружения")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Динамический импорт модулей с «невалидными» именами (начинаются с цифры)
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: str):
    """Импортирует модуль из произвольного пути и возвращает его."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось импортировать модуль {module_name} из {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module

# Импортируем необходимые модули
# _etl_root = Path(__file__).resolve().parent # Старый путь
# Определяем корень проекта, предполагая, что скрипт находится в /scripts, а модули в /src/etl
_project_root = Path(__file__).resolve().parent.parent
_etl_root = _project_root / "src" / "etl"

parser_mod = _import_from_path(
    "telegram_parser_mod",
    str(_etl_root / "00_telegram_parser.py"),
)
clean_mod = _import_from_path(
    "clean_text_mod",
    str(_etl_root / "01_clean_text.py"),
)
emb_mod = _import_from_path(
    "compute_embeddings_mod",
    str(_etl_root / "02_compute_embeddings.py"),
)
# Для кластеризатора проще запускать как отдельный процесс — там тяжёлые SciPy/Sklearn

TelegramParser = parser_mod.TelegramParser  # type: ignore[attr-defined]
load_channels = parser_mod.load_channels  # type: ignore[attr-defined]
TextCleaner = clean_mod.TextCleaner  # type: ignore[attr-defined]
load_config = clean_mod.load_config  # type: ignore[attr-defined]
process_messages = clean_mod.process_messages  # type: ignore[attr-defined]
Embedder = emb_mod.Embedder  # type: ignore[attr-defined]
compute_embeddings = emb_mod.compute_embeddings  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/external/news_tg_csv")
TEXT_DIR = Path("data/external/text")
CONFIGS_DIR = Path("configs")

NEWS_RAW_PATH = RAW_DIR / "telegram_news.csv"
BLOGS_RAW_PATH = RAW_DIR / "blogs.csv"

NEWS_CLEAN_PATH = TEXT_DIR / "news_cleaned.csv"
BLOGS_CLEAN_PATH = TEXT_DIR / "blogs_cleaned.csv"

NEWS_EMB_PATH = TEXT_DIR / "news_with_emb.csv"
BLOGS_EMB_PATH = TEXT_DIR / "blogs_with_emb.csv"

CLUSTER_OUTPUT_DIR_NEWS = TEXT_DIR / "news_clusters"
CLUSTER_OUTPUT_DIR_BLOGS = TEXT_DIR / "blogs_clusters"


CHUNK_SIZE_READ_DATETIME = 200_000


def _get_last_datetime(csv_path: Path) -> Optional[pd.Timestamp]:
    """Определяет максимальную дату в столбце `datetime` CSV файла."""
    if not csv_path.exists():
        return None

    last_dt: Optional[pd.Timestamp] = None
    for chunk in pd.read_csv(
        csv_path,
        usecols=["datetime"],
        parse_dates=["datetime"],
        chunksize=CHUNK_SIZE_READ_DATETIME,
        encoding="utf-8",
    ):
        chunk_max = chunk["datetime"].max()
        if last_dt is None or chunk_max > last_dt:
            last_dt = chunk_max
    return last_dt


async def _download_new_messages(
    content_type: str,
    last_dt: Optional[pd.Timestamp],
    *,
    session_name: str,
    remove_session_file: bool,
) -> pd.DataFrame:
    """Скачивает новые сообщения (после last_dt) и возвращает их DataFrame."""

    parser = TelegramParser(
        session_name=session_name,
        api_id=API_ID,
        api_hash=API_HASH,
        phone=PHONE,
        content_type=content_type,
        last_datetime=None if last_dt is None else last_dt.to_pydatetime(),
        remove_session_file=remove_session_file,
    )

    await parser.init_client()
    channels = load_channels(content_type)
    if not channels:
        logger.error("Для типа %s не найдено каналов в конфигурации", content_type)
        return pd.DataFrame()

    tasks = []
    for link, name in channels.items():
        tasks.append(asyncio.create_task(parser.process_channel(link, name)))
        await asyncio.sleep(0.5)  # Небольшая пауза для избежания FloodWait

    await asyncio.gather(*tasks, return_exceptions=True)
    df = parser._prepare_dataframe()
    await parser.cleanup()

    if last_dt is not None and not df.empty:
        df = df[df["datetime"] > last_dt]

    return df


def _append_df_to_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Добавляет DataFrame в CSV, создавая файл при его отсутствии."""
    if df.empty:
        logger.info("Нет новых записей для %s", csv_path.name)
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False, encoding="utf-8")
    logger.info("В %s добавлено %d строк", csv_path, len(df))


def _process_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config(CONFIGS_DIR / "annotator_config.json")
    cleaner = TextCleaner(cfg.get("text_processor", {}))
    cleaned_df, empty_rm, date_rm = process_messages(df_raw, cleaner)
    logger.info("Cleaned: пустых=%d, устаревших=%d", empty_rm, date_rm)
    return cleaned_df


def _process_embeddings(df_clean: pd.DataFrame) -> pd.DataFrame:
    if df_clean.empty:
        return pd.DataFrame()

    cfg = load_config(CONFIGS_DIR / "annotator_config.json")
    embed_cfg = cfg.get("embedder", cfg)
    embedder = Embedder(**embed_cfg)
    embeddings = compute_embeddings(df_clean["news"].astype(str).tolist(), embedder)
    df_emb = df_clean.copy()
    df_emb["embedding"] = [json.dumps(e, ensure_ascii=False) for e in embeddings]
    return df_emb


def _run_clusterer(input_csv: Path, output_dir: Path, *, min_datetime: Optional[pd.Timestamp] = None, prefix: str = ""):
    """Запускает 03_clusterer.py как подпроцесс для указанного файла."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_etl_root / "03_clusterer.py"),
        "--input-path",
        str(input_csv),
        "--config-path",
        str(CONFIGS_DIR / "clusterer_config.json"),
        "--output-dir",
        str(output_dir),
    ]
    if min_datetime is not None:
        cmd.extend(["--min-datetime", min_datetime.strftime("%Y-%m-%d")])
    if prefix:
        cmd.extend(["--prefix", prefix])
    logger.info("Запуск кластеризации: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_dir  # возвращаем директорию с результатами


# ---------------------------------------------------------------------------
# Обновление глобальных файлов результатов
# ---------------------------------------------------------------------------


def _merge_representatives(new_csv: Path, global_csv: Path):
    """Добавляет записи из new_csv в global_csv, избегая дубликатов cluster_id."""
    if not new_csv.exists():
        logger.warning("Файл %s не найден для слияния представительных новостей", new_csv)
        return

    new_df = pd.read_csv(new_csv, encoding="utf-8")
    if new_df.empty:
        return

    if global_csv.exists():
        global_df = pd.read_csv(global_csv, encoding="utf-8")
        combined = pd.concat([global_df, new_df], ignore_index=True)
        # Удаляем дубликаты по cluster_id
        if "cluster_id" in combined.columns:
            combined.drop_duplicates(subset=["cluster_id"], inplace=True)
        else:
            combined.drop_duplicates(inplace=True)
    else:
        combined = new_df

    combined.sort_values("datetime", inplace=True)
    combined.to_csv(global_csv, index=False, encoding="utf-8")
    logger.info("Файл %s обновлён (%d записей)", global_csv, len(combined))


def _merge_all_clusters(new_json: Path, global_json: Path):
    """Объединяет кластеры в формате JSON без дубликатов по cluster_id."""
    if not new_json.exists():
        logger.warning("Файл %s не найден для слияния кластеров", new_json)
        return

    with open(new_json, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    new_clusters = new_data.get("clusters", [])

    if global_json.exists():
        with open(global_json, "r", encoding="utf-8") as f:
            global_data = json.load(f)
        global_clusters = global_data.get("clusters", [])
        # Индекс существующих cluster_id
        exist_ids = {c["cluster_id"] for c in global_clusters if "cluster_id" in c}
        merged_clusters = global_clusters + [c for c in new_clusters if c.get("cluster_id") not in exist_ids]
    else:
        merged_clusters = new_clusters

    final_obj = {
        "total_clusters": len(merged_clusters),
        "updated_at": datetime.now().isoformat(),
        "clusters": merged_clusters,
    }
    global_json.parent.mkdir(parents=True, exist_ok=True)
    with open(global_json, "w", encoding="utf-8") as f:
        json.dump(final_obj, f, ensure_ascii=False, indent=2)
    logger.info("Файл %s обновлён (%d кластеров)", global_json, len(merged_clusters))


async def _process_content_type(
    content_type: str,
    *,
    session_name: Optional[str] = None,
    remove_session_file: bool = True,
    cluster_override: Optional[pd.Timestamp] = None,
):
    logger.info("=== Обработка типа контента: %s ===", content_type)

    raw_csv = NEWS_RAW_PATH if content_type == "news" else BLOGS_RAW_PATH
    clean_csv = NEWS_CLEAN_PATH if content_type == "news" else BLOGS_CLEAN_PATH
    emb_csv = NEWS_EMB_PATH if content_type == "news" else BLOGS_EMB_PATH
    cluster_out = CLUSTER_OUTPUT_DIR_NEWS if content_type == "news" else CLUSTER_OUTPUT_DIR_BLOGS

    last_dt = _get_last_datetime(raw_csv)
    logger.info("Последняя дата в %s: %s", raw_csv.name, last_dt)

    # Определяем имя сессии (если не передано — создаём своё)
    sess_name = session_name or f"Update_{content_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    new_raw_df = await _download_new_messages(
        content_type,
        last_dt,
        session_name=sess_name,
        remove_session_file=remove_session_file,
    )
    if new_raw_df.empty:
        logger.info("Новых сообщений для %s нет", content_type)
        # Если задали cluster_override – всё равно запускаем кластеризацию
        if cluster_override is None:
            return

    cluster_start = (
        cluster_override.normalize() if cluster_override is not None else
        (last_dt.normalize() if last_dt is not None else None)
    )

    if not new_raw_df.empty:
        # 1. Записываем «сырые» данные
        _append_df_to_csv(new_raw_df, raw_csv)

        # 2. Очистка
        cleaned_df = _process_clean(new_raw_df)
        if not cleaned_df.empty:
            _append_df_to_csv(cleaned_df, clean_csv)

            # 3. Эмбеддинги (только если после очистки что-то осталось)
            emb_df = _process_embeddings(cleaned_df)
            if not emb_df.empty:
                _append_df_to_csv(emb_df, emb_csv)

    # 4. Кластеризация: только начиная с полуночи даты последней старой записи
    cluster_dir = _run_clusterer(emb_csv, cluster_out, min_datetime=cluster_start, prefix=content_type)

    # Обновляем глобальные файлы результатов
    rep_path = cluster_dir / f"representative_{content_type}.csv"
    all_clusters_path = cluster_dir / f"all_clusters_{content_type}.json"

    _merge_representatives(rep_path, TEXT_DIR / f"representative_{content_type}.csv")
    _merge_all_clusters(all_clusters_path, TEXT_DIR / f"all_clusters_{content_type}.json")


async def main_async(content_type: str, cluster_override: Optional[pd.Timestamp]):
    if content_type == "all":
        # Общая сессия для обоих типов
        shared_session = f"Update_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await _process_content_type("news", session_name=shared_session, remove_session_file=False, cluster_override=cluster_override)
        await _process_content_type("blogs", session_name=shared_session, remove_session_file=True, cluster_override=cluster_override)
    else:
        await _process_content_type(content_type, cluster_override=cluster_override)


def main():
    parser = argparse.ArgumentParser(
        description="Инкрементальная загрузка и обработка новостей/блогов из Telegram",
    )
    parser.add_argument(
        "--content-type",
        choices=["news", "blogs", "all"],
        default="all",
        help="Тип контента для загрузки/обработки",
    )
    parser.add_argument(
        "--cluster-start",
        type=str,
        default=None,
        help="Явно указать дату (YYYY-MM-DD), с которой начать кластеризацию. Используется даже если новых данных нет.",
    )
    args = parser.parse_args()

    cluster_override_ts: Optional[pd.Timestamp] = None
    if args.cluster_start:
        try:
            cluster_override_ts = pd.to_datetime(args.cluster_start, utc=True)
        except Exception:
            logger.error("Неверный формат --cluster-start, ожидается YYYY-MM-DD")
            sys.exit(1)

    try:
        asyncio.run(main_async(args.content_type, cluster_override_ts))
    except KeyboardInterrupt:
        logger.warning("Операция прервана пользователем")


if __name__ == "__main__":
    main() 