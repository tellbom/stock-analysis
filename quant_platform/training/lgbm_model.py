"""
training.lgbm_model
===================
LightGBM baseline model (T2.2).

Wrapped in a scikit-learn Pipeline so that preprocessing (imputation,
scaling) always travels with the model and cannot accidentally detach
between training and prediction — the most common source of silent
preprocessing leakage.

Design
------
- Objective: regression on raw forward return (ret_fwd_{h}d).
- Cross-sectional ranking objective is available as an alternative.
- Feature columns are determined from the feature_set_id registry.
- ``fit`` returns OOF (out-of-fold) predictions for the full train panel,
  produced by the PurgedKFold splitter.
- No global normalisation is fit before the train/val split — scalers are
  fit inside each fold on train-only data via the Pipeline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from quant_platform.training.splitter import PurgedKFold
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

# Default LightGBM hyperparameters — deliberately conservative for a baseline
_DEFAULT_LGB_PARAMS = {
    "objective":       "regression",
    "n_estimators":    200,
    "learning_rate":   0.05,
    "num_leaves":      31,
    "min_child_samples": 20,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "random_state":    42,
    "verbose":         -1,
    "n_jobs":          -1,
}


def build_lgbm_pipeline(params: dict | None = None) -> Pipeline:
    """
    Build an sklearn Pipeline: impute → scale → LightGBM.

    Imputation uses median (robust to outliers).
    Scaling uses StandardScaler (required for some downstream calibration).
    """
    params = params or _DEFAULT_LGB_PARAMS
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lgbm",    lgb.LGBMRegressor(**params)),
    ])


@dataclass
class OOFResult:
    """Holds out-of-fold predictions and metadata."""
    oof_predictions: pd.Series        # index aligns with train panel
    fold_metrics:    list[dict]       # per-fold IC stats
    feature_names:   list[str]
    n_folds:         int
    label_col:       str
    params:          dict = field(default_factory=dict)


def fit_oof(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    n_splits: int = 5,
    horizon: int = 20,
    embargo: int = 5,
    lgbm_params: dict | None = None,
    seed: int = 42,
) -> OOFResult:
    """
    Fit LightGBM with purged k-fold CV and return OOF predictions.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have columns: date, symbol, <feature_cols>, <label_col>.
        Sorted by date ascending.
    feature_cols : list[str]
        Feature columns to use.  NaN-heavy columns are imputed inside the pipe.
    label_col : str
        Target column (e.g. ``ret_fwd_20d``).
    n_splits : int
        Number of CV folds.
    horizon : int
        Label horizon in trading rows — used for purging.
    embargo : int
        Embargo gap in trading rows.
    lgbm_params : dict | None
        Override default LightGBM params.

    Returns
    -------
    OOFResult
        Contains OOF predictions aligned to the input panel index.
    """
    np.random.seed(seed)
    panel = panel.copy().reset_index(drop=True)

    # Drop rows with NaN label (can't train on them)
    valid_mask = panel[label_col].notna()
    train_panel = panel[valid_mask].reset_index(drop=True)

    X = train_panel[feature_cols]
    y = train_panel[label_col]

    splitter    = PurgedKFold(n_splits=n_splits, horizon=horizon, embargo=embargo)
    oof_preds   = np.full(len(train_panel), np.nan)
    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(train_panel)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx],   y.iloc[val_idx]

        if len(X_tr) == 0 or len(X_va) == 0:
            logger.warning("Fold %d: empty train or val — skipping", fold_idx)
            continue

        pipe = build_lgbm_pipeline(lgbm_params)
        pipe.fit(X_tr, y_tr)
        preds_val = pipe.predict(X_va)
        oof_preds[val_idx] = preds_val

        # Per-fold IC (Pearson on regression output)
        ic = float(pd.Series(preds_val).corr(y_va.reset_index(drop=True)))
        fold_metrics.append({
            "fold":       fold_idx,
            "n_train":    len(train_idx),
            "n_val":      len(val_idx),
            "ic_pearson": ic,
        })
        logger.info(
            "Fold %d: n_train=%d, n_val=%d, IC=%.4f",
            fold_idx, len(train_idx), len(val_idx), ic,
        )

    # Align OOF predictions back to the original (full) panel index
    full_oof = pd.Series(np.nan, index=panel.index)
    full_oof[valid_mask.values] = oof_preds

    logger.info(
        "OOF fit complete: %d folds, %.1f%% predictions filled",
        len(fold_metrics),
        full_oof.notna().mean() * 100,
    )

    return OOFResult(
        oof_predictions=full_oof,
        fold_metrics=fold_metrics,
        feature_names=feature_cols,
        n_folds=len(fold_metrics),
        label_col=label_col,
        params=lgbm_params or _DEFAULT_LGB_PARAMS,
    )


def fit_final_model(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    lgbm_params: dict | None = None,
) -> Pipeline:
    """
    Fit a final model on the full train panel (no CV).
    Used after OOF evaluation to produce the model for the lockbox test.
    """
    valid = panel[panel[label_col].notna()]
    X, y  = valid[feature_cols], valid[label_col]
    pipe  = build_lgbm_pipeline(lgbm_params)
    pipe.fit(X, y)
    logger.info(
        "Final model fit on %d rows, %d features", len(X), len(feature_cols)
    )
    return pipe
