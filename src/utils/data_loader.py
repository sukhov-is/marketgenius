import pathlib
from typing import Dict, Tuple, List

import pandas as pd

# Папка с итоговыми CSV-файлами по тикерам
DEFAULT_FEATURES_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "features_final"

# Колонки медиа-показателей согласно data/final_structure.txt (113–126)
MEDIA_FEATURE_COLUMNS: List[str] = [
    "GAZP_blog_score",
    "GAZP_blog_score_roll_avg_15",
    "GAZP_blog_score_roll_avg_50",
    "Index_MOEX_blog_score",
    "Index_MOEXOG_blog_score",
    "Index_MOEXOG_blog_score_roll_avg_15",
    "Index_MOEXOG_blog_score_roll_avg_50",
    "GAZP_news_score",
    "GAZP_news_score_roll_avg_15",
    "GAZP_news_score_roll_avg_50",
    "Index_MOEX_news_score",
    "Index_MOEXOG_news_score",
    "Index_MOEXOG_news_score_roll_avg_15",
    "Index_MOEXOG_news_score_roll_avg_50",
]


def load_ticker_df(csv_path: pathlib.Path, parse_dates: bool = True) -> pd.DataFrame:
    """Загрузить CSV для одного тикера.

    :param csv_path: путь к файлу
    :param parse_dates: если True – парсить колонку ``date`` как datetime
    :return: DataFrame, индекс = date
    """
    df = pd.read_csv(csv_path)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.set_index("date")
    return df


def load_all_tickers(
    features_dir: pathlib.Path = DEFAULT_FEATURES_DIR,
) -> Dict[str, pd.DataFrame]:
    """Считать все CSV в словарь {ticker: DataFrame}."""
    data = {}
    for csv_file in features_dir.glob("*_final.csv"):
        ticker = csv_file.stem.replace("_final", "")
        data[ticker] = load_ticker_df(csv_file)
    return data


def split_train_val_test(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Временной сплит DataFrame.

    :returns: (train_df, val_df, test_df)
    """
    if not abs(train_size + val_size + test_size - 1.0) < 1e-6:
        raise ValueError("train + val + test должны давать 1.0")

    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def get_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Вернуть две группы колонок: base_features, media_features.

    Если каких-то колонок медиа нет в датафрейме – они игнорируются.
    """
    media_cols = [c for c in MEDIA_FEATURE_COLUMNS if c in df.columns]
    base_cols = [c for c in df.columns if c not in media_cols]
    return base_cols, media_cols


__all__ = [
    "load_ticker_df",
    "load_all_tickers",
    "split_train_val_test",
    "get_feature_sets",
    "MEDIA_FEATURE_COLUMNS",
] 