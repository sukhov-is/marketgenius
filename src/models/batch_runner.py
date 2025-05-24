from __future__ import annotations

"""Batch runner для выполнения множества экспериментов.

Используется как программно, так и из CLI-скрипта scripts/run_batch_experiments.py.
"""

from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import json

import pandas as pd

from ..utils.data_loader import DEFAULT_FEATURES_DIR
from .experiment import run_single_experiment, TaskType

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
NUMERIC_TARGETS: List[str] = [
    "target_1d",
    "target_3d",
    "target_7d",
    "target_30d",
    "target_180d",
]
BINARY_TARGETS: List[str] = [f"{t}_binary" for t in [
    "target_1d",
    "target_3d",
    "target_7d",
    "target_30d",
    "target_180d",
]]

ALL_TARGETS: List[str] = NUMERIC_TARGETS + BINARY_TARGETS

MODEL_FAMILIES: List[str] = ["xgb", "lgbm"]
FEATURE_SETS: List[str] = ["base", "media"]  # base = без медиа, media = +медиа признаки

# ---------------------------------------------------------------------------


def infer_task(target: str) -> TaskType:
    """Определить тип задачи по названию таргета."""
    return TaskType.classification if target.endswith("_binary") else TaskType.regression



def iter_experiments(
    features_dir: Path = DEFAULT_FEATURES_DIR,
    targets: Iterable[str] | None = None,
    model_families: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Выполняет все комбинации экспериментов и возвращает таблицу метрик.

    Параметры
    ----------
    features_dir: Path
        Папка, содержащая *_final.csv.
    targets: Iterable[str] | None
        Список таргетов для исследования. Если None – берутся ALL_TARGETS.
    model_families: Iterable[str] | None
        ["xgb", "lgbm"].
    """

    targets = list(targets or ALL_TARGETS)
    model_families = list(model_families or MODEL_FAMILIES)

    records: List[Dict[str, Any]] = []

    # Проходим по всем тикерам
    for csv_path in sorted(Path(features_dir).glob("*_final.csv")):
        ticker = csv_path.stem.replace("_final", "")
        for target in targets:
            task = infer_task(target)
            for model_family in model_families:
                for feature_set in FEATURE_SETS:
                    use_media = feature_set == "media"
                    try:
                        metrics, _ = run_single_experiment(
                            csv_path=csv_path,
                            target_col=target,
                            task=task,
                            use_media=use_media,
                            model_family=model_family,
                        )
                    except KeyError:
                        # колонка таргета отсутствует – пропускаем
                        continue
                    record: Dict[str, Any] = {
                        "ticker": ticker,
                        "target": target,
                        "task": task.value,
                        "model_family": model_family,
                        "feature_set": feature_set,
                    }
                    record.update(metrics)
                    records.append(record)
    return pd.DataFrame.from_records(records)



def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Сохранить результаты в CSV и JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path.with_suffix(".csv"), index=False)
    # Также JSON по тикеру/таргету для удобства
    grouped: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = f"{row['ticker']}/{row['target']}/{row['model_family']}/{row['feature_set']}"
        grouped[key] = row.to_dict()
    output_path.with_suffix(".json").write_text(json.dumps(grouped, indent=2, ensure_ascii=False))


__all__: Tuple[str, ...] = (
    "iter_experiments",
    "save_results",
) 