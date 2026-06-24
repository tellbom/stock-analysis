"""
training.model_zoo
==================
Model zoo with a common interface (T3.2).

Every model in the zoo implements ``ModelBase``:
  fit(X_train, y_train)
  predict(X)
  get_params() -> dict
  model_name -> str

Adding a new model = one new subclass + one entry in ``MODEL_REGISTRY``.

Included models
---------------
  LGBMModel   : LightGBM (the P2 baseline)
  XGBModel    : XGBoost
  CatModel    : CatBoost

All wrapped in sklearn Pipelines (imputer → scaler → model) so preprocessing
is always bundled with the model and cannot detach between training and inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModelBase(ABC):
    """Common interface every model in the zoo must implement."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelBase": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def get_params(self) -> dict: ...

    @abstractmethod
    def get_native_model(self):
        """Return the unwrapped native model (for SHAP etc.)."""
        ...


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBMModel(ModelBase):
    model_name = "LightGBM"

    _DEFAULTS = {
        "objective": "regression", "n_estimators": 200,
        "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
        "random_state": 42, "verbose": -1, "n_jobs": -1,
    }

    def __init__(self, **params) -> None:
        import lightgbm as lgb
        merged = {**self._DEFAULTS, **params}
        self._params = merged
        self._pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   lgb.LGBMRegressor(**merged)),
        ])

    def fit(self, X, y):
        self._pipe.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._pipe.predict(X)

    def get_params(self) -> dict:
        return {"model_name": self.model_name, **self._params}

    def get_native_model(self):
        return self._pipe.named_steps["model"]


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBModel(ModelBase):
    model_name = "XGBoost"

    _DEFAULTS = {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "random_state": 42, "verbosity": 0, "n_jobs": -1,
    }

    def __init__(self, **params) -> None:
        import xgboost as xgb
        merged = {**self._DEFAULTS, **params}
        self._params = merged
        self._pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   xgb.XGBRegressor(**merged)),
        ])

    def fit(self, X, y):
        self._pipe.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._pipe.predict(X)

    def get_params(self) -> dict:
        return {"model_name": self.model_name, **self._params}

    def get_native_model(self):
        return self._pipe.named_steps["model"]


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatModel(ModelBase):
    model_name = "CatBoost"

    _DEFAULTS = {
        "iterations": 200, "learning_rate": 0.05, "depth": 6,
        "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": 0,
    }

    def __init__(self, **params) -> None:
        from catboost import CatBoostRegressor
        merged = {**self._DEFAULTS, **params}
        self._params = merged
        # CatBoost handles missing values natively — no imputer needed,
        # but keep scaler for calibration consistency
        self._pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   CatBoostRegressor(**merged)),
        ])

    def fit(self, X, y):
        self._pipe.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._pipe.predict(X)

    def get_params(self) -> dict:
        return {"model_name": self.model_name, **self._params}

    def get_native_model(self):
        return self._pipe.named_steps["model"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[ModelBase]] = {
    "lgbm":     LGBMModel,
    "xgboost":  XGBModel,
    "catboost": CatModel,
    "ridge":    "RidgeModel",  # forward reference; resolved below
}


# ---------------------------------------------------------------------------
# Ridge (P4A-06 — linear diagnostic baseline)
# ---------------------------------------------------------------------------

class RidgeModel(ModelBase):
    """
    Ridge regression baseline (P4A-06).

    Purpose: diagnostic, not performance.

    On Alpha158-style features, linear/ridge models reach ~0.046 Rank IC
    versus LightGBM's ~0.048 — the signal is largely linear.  If the
    platform's GBM OOF substantially exceeds RidgeModel OOF (ratio > 1.3),
    the GBM is likely capturing nonlinear noise rather than signal — a
    red flag that would partly explain an OOF/lockbox gap.

    The ``alpha`` (L2 regularisation) is intentionally left at a
    conservative default.  Do NOT tune it against the lockbox.

    Usage
    -----
        from quant_platform.training.model_zoo import RidgeModel, fit_zoo_model_oof

        ridge = RidgeModel()
        oof_preds, metrics = fit_zoo_model_oof(
            ridge, panel, feature_cols, label_col="ret_fwd_5d",
            n_splits=5, horizon=5,
        )
    """

    model_name = "Ridge"

    _DEFAULTS = {
        "alpha":     100.0,   # conservative L2; do not tune vs lockbox
        "fit_intercept": True,
        "max_iter":  10_000,
    }

    def __init__(self, **params) -> None:
        from sklearn.linear_model import Ridge
        merged = {**self._DEFAULTS, **params}
        self._params = merged
        self._pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   Ridge(**merged)),
        ])

    def fit(self, X, y):
        self._pipe.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._pipe.predict(X)

    def get_params(self) -> dict:
        return {"model_name": self.model_name, **self._params}

    def get_native_model(self):
        return self._pipe.named_steps["model"]


# Resolve the forward reference in MODEL_REGISTRY
MODEL_REGISTRY["ridge"] = RidgeModel


def get_model(name: str, **params) -> ModelBase:
    """
    Instantiate a model by registry name.

    Raises KeyError for unknown names — never silently falls back.
    """
    if name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"Unknown model '{name}'. Known models: {known}. "
            "To add a model, register it in MODEL_REGISTRY."
        )
    return MODEL_REGISTRY[name](**params)


def fit_zoo_model_oof(
    model:        ModelBase,
    panel:        pd.DataFrame,
    feature_cols: list[str],
    label_col:    str,
    n_splits:     int = 5,
    horizon:      int = 20,
    embargo:      int = 5,
) -> tuple[pd.Series, list[dict]]:
    """
    Fit any ModelBase with purged OOF CV.

    Returns (oof_predictions, fold_metrics) — same shape as fit_oof()
    so the leaderboard can aggregate them identically regardless of model type.
    """
    from quant_platform.training.splitter import PurgedKFold

    panel = panel.copy().reset_index(drop=True)
    valid_mask   = panel[label_col].notna()
    train_panel  = panel[valid_mask].reset_index(drop=True)

    X = train_panel[feature_cols]
    y = train_panel[label_col]

    splitter  = PurgedKFold(n_splits=n_splits, horizon=horizon, embargo=embargo)
    oof_preds = np.full(len(train_panel), np.nan)
    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(train_panel)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx],   y.iloc[val_idx]
        if len(X_tr) == 0 or len(X_va) == 0:
            continue

        # Fresh model instance per fold to avoid state contamination
        import copy
        fold_model = copy.deepcopy(model)
        fold_model.fit(X_tr, y_tr)
        preds_val = fold_model.predict(X_va)
        oof_preds[val_idx] = preds_val

        ic = float(pd.Series(preds_val).corr(y_va.reset_index(drop=True)))
        fold_metrics.append({
            "fold": fold_idx, "n_train": len(train_idx),
            "n_val": len(val_idx), "ic_pearson": ic,
            "model": model.model_name,
        })
        logger.info(
            "%s fold %d: n_train=%d, n_val=%d, IC=%.4f",
            model.model_name, fold_idx, len(train_idx), len(val_idx), ic,
        )

    full_oof = pd.Series(np.nan, index=panel.index)
    full_oof[valid_mask.values] = oof_preds
    return full_oof, fold_metrics
