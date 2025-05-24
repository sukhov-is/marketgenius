from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier


class TaskType(str, enum.Enum):
    regression = "regression"
    classification = "classification"


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, float]
    feature_importances: pd.Series


REG_DEFAULTS = dict(
    n_estimators=500,
    learning_rate=0.05,
    objective="reg:squarederror",
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

CLS_DEFAULTS = dict(
    n_estimators=500,
    learning_rate=0.05,
    objective="binary:logistic",
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


class ModelTrainer:
    """Унифицированный интерфейс для XGB / LGBM регрессии и классификации."""

    def __init__(
        self,
        task: TaskType,
        model_family: str = "xgb",  # or "lgbm"
        params: Dict[str, Any] | None = None,
    ):
        self.task = TaskType(task)
        self.model_family = model_family.lower()
        self.params = params or {}
        self.model = self._init_model()

    # ------------------------------------------------------------------
    # Основные методы
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "ModelTrainer":
        fit_params: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["early_stopping_rounds"] = 50
            fit_params["verbose"] = False

        # Некоторые версии lightgbm / xgboost могут не поддерживать отдельные ключи.
        # Пробуем несколько раз, постепенно убирая спорные параметры.
        trial_keys_order = ["early_stopping_rounds", "verbose"]
        for _ in range(len(trial_keys_order) + 1):
            try:
                self.model.fit(X_train, y_train, **fit_params)
                break  # успех
            except TypeError as e:
                msg = str(e)
                removed = False
                for key in list(fit_params.keys()):
                    if key in trial_keys_order and key in msg:
                        fit_params.pop(key, None)
                        removed = True
                        break
                if not removed:
                    # Если ошибка не связана с известными ключами – пробрасываем дальше
                    raise
        return self

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        y_pred = self.model.predict(X_test)
        if self.task == TaskType.regression:
            try:
                rmse = mean_squared_error(y_test, y_pred, squared=False)
            except TypeError:
                # Старые версии sklearn (<0.22) не поддерживают параметр squared
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": rmse,
                "R2": r2_score(y_test, y_pred),
                "MAPE": float(np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, np.nan, y_test))) * 100),
            }
        else:
            # Для классификаторов выход может быть либо вероятность, либо логиты
            if y_pred.ndim == 1:
                y_prob = y_pred  # XGB/LGB predict_proba возвращает 1-D для бинар?
            else:
                y_prob = y_pred[:, 1]
            y_pred_label = (y_prob > 0.5).astype(int)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred_label),
                "F1": f1_score(y_test, y_pred_label),
                "ROC_AUC": roc_auc_score(y_test, y_prob),
                "PR_AUC": pr_auc,
            }
        return metrics

    # ------------------------------------------------------------------
    # Служебные
    # ------------------------------------------------------------------
    def _init_model(self):
        params = self._default_params()
        params.update(self.params)
        if self.model_family == "xgb":
            if self.task == TaskType.regression:
                return XGBRegressor(**params)
            else:
                return XGBClassifier(**params)
        elif self.model_family == "lgbm":
            if self.task == TaskType.regression:
                return LGBMRegressor(**params)
            else:
                return LGBMClassifier(**params)
        else:
            raise ValueError(f"Unknown model_family {self.model_family}")

    def _default_params(self):
        if self.task == TaskType.regression:
            return REG_DEFAULTS.copy() if self.model_family == "xgb" else {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "objective": "regression",
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        else:
            return CLS_DEFAULTS.copy() if self.model_family == "xgb" else {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "objective": "binary",
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }

    # ------------------------------------------------------------------
    @property
    def feature_importances(self) -> pd.Series:
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_)
        else:
            return pd.Series(dtype=float)


__all__ = [
    "ModelTrainer",
    "TaskType",
] 