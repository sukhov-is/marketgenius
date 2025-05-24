from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from ..utils.data_loader import (
    load_ticker_df,
    split_train_val_test,
    get_feature_sets,
)
from .trainer import ModelTrainer, TaskType


def run_single_experiment(
    csv_path: Path,
    target_col: str,
    task: TaskType,
    use_media: bool = False,
    model_family: str = "xgb",
    params: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, float], pd.Series]:
    """Запустить одиночный эксперимент (Train → Eval).

    :returns: metrics, feature_importances
    """
    df = load_ticker_df(csv_path)

    base_cols, media_cols = get_feature_sets(df)
    feature_cols = base_cols + media_cols if use_media else base_cols
    # Удаляем таргет из списка признаков, если вдруг совпадение по имени
    feature_cols = [c for c in feature_cols if c != target_col]
    # Оставляем только числовые столбцы (XGB/LGB не работают с object)
    df_numeric = df[feature_cols].select_dtypes(include=["number"])
    feature_cols = list(df_numeric.columns)

    # Отбрасываем строки с NaN в таргете
    df = df.dropna(subset=[target_col])
    X = df_numeric.loc[df.index]
    y = df[target_col]

    train_df, val_df, test_df = split_train_val_test(df)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    trainer = ModelTrainer(task=task, model_family=model_family, params=params)
    trainer.fit(X_train, y_train, X_val, y_val)
    metrics = trainer.evaluate(X_test, y_test)
    fi = pd.Series(index=feature_cols, data=trainer.feature_importances.values)
    return metrics, fi 