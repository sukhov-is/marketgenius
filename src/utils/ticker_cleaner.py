import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from fuzzywuzzy import process

__all__ = [
    "clean_results_file",
]

FUZZY_MATCH_SCORE_CUTOFF = 85


# Кириллица → латиница для упрощённого сопоставления
_CYR_TO_LAT = {
    'А': 'A', 'а': 'a', 'В': 'B', 'в': 'v', 'Е': 'E', 'е': 'e', 'К': 'K', 'к': 'k',
    'М': 'M', 'м': 'm', 'Н': 'H', 'н': 'n', 'О': 'O', 'о': 'o', 'Р': 'P', 'р': 'r',
    'С': 'C', 'с': 's', 'Т': 'T', 'т': 't', 'Х': 'X', 'х': 'h', 'У': 'U', 'у': 'u',
    'Г': 'G', 'г': 'g', 'Д': 'D', 'д': 'd', 'З': 'Z', 'з': 'z', 'И': 'I', 'и': 'i',
    'Л': 'L', 'л': 'l', 'П': 'P', 'п': 'p', 'Ф': 'F', 'ф': 'f', 'Ц': 'TS', 'ц': 'ts',
    'Ч': 'CH', 'ч': 'ch', 'Ш': 'SH', 'ш': 'sh', 'Щ': 'SCH', 'щ': 'sch',
    'Э': 'E', 'э': 'e', 'Ю': 'YU', 'ю': 'yu', 'Я': 'YA', 'я': 'ya', 'Й': 'Y', 'й': 'y',
    'Ж': 'ZH', 'ж': 'zh', 'Б': 'B', 'б': 'b', 'Ь': '', 'ь': '', 'Ъ': '', 'ъ': '',
    'Ё': 'E', 'ё': 'e'
}


def _normalize(text: str) -> str:
    text = str(text)
    for cyr, lat in _CYR_TO_LAT.items():
        text = text.replace(cyr, lat)
    return (
        text.replace('-', '')
        .replace('_', '')
        .replace('.', '')
        .replace(' ', '')
        .lower()
    )


def _build_name_map(config: Dict) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for canonical, info in config.get("companies", {}).items():
        m[_normalize(canonical)] = canonical
        for name in info.get("names", []):
            m[_normalize(name)] = canonical
    for canonical in config.get("indices", {}):
        m[_normalize(canonical)] = canonical
    return m


def clean_results_file(
    input_csv: str | Path,
    output_csv: str | Path,
    config_path: str | Path = "configs/all_companies_config.json",
    preserve_columns: List[str] | None = None,
) -> pd.DataFrame:
    """Очищает CSV-файл результатов GPT, объединяя колонки-тикеры по конфигу.

    Возвращает очищенный DataFrame и сохраняет его в *output_csv*.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    config_path = Path(config_path)

    if preserve_columns is None:
        preserve_columns = ["date", "summary"]

    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(input_csv)

    name_map = _build_name_map(config)
    known_norm_canon = { _normalize(k): k for k in list(config.get("companies", {}).keys()) + list(config.get("indices", {}).keys()) }

    cleaned_cols = {}
    for col in preserve_columns:
        if col in df.columns:
            cleaned_cols[col] = df[col]

    column_groups: Dict[str, Dict] = {}
    unmatched = []

    for csv_col in [c for c in df.columns if c not in preserve_columns]:
        norm = _normalize(csv_col)
        canon = name_map.get(norm)
        if not canon:
            # fuzzy match
            match = process.extractOne(norm, list(known_norm_canon.keys()), score_cutoff=FUZZY_MATCH_SCORE_CUTOFF)
            if match:
                canon = known_norm_canon[match[0]]
        if canon:
            column_groups.setdefault(canon, []).append(csv_col)
        else:
            unmatched.append(csv_col)

    for canon, cols in column_groups.items():
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")
        cleaned_cols[canon] = numeric.apply(lambda r: r.loc[r.abs().idxmax()] if not r.isnull().all() else np.nan, axis=1)

    cleaned_df = pd.DataFrame(cleaned_cols)
    ordered_cols = [c for c in preserve_columns if c in cleaned_df.columns] + sorted([c for c in cleaned_df.columns if c not in preserve_columns])
    cleaned_df = cleaned_df[ordered_cols]

    cleaned_df.to_csv(output_csv, index=False, encoding="utf-8")
    return cleaned_df 