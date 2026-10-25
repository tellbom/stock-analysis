"""Frozen challenger using the same already-normalised factors as Ridge32.

``robust_ridge_v1`` changes only the training window, train-fitted clipping
and Ridge regularisation. Input factors must have the same semantics as the
original baseline. In particular, existing price ratios are not normalised
again. A fitted pipeline accepts ``build_robust_features(panel)`` output.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from quant_platform.selection.ridge_baseline import ORIGINAL_FEATURES, score_frame

STRATEGY_ID = "robust_ridge_v1"
STRATEGY_VERSION = "1.0-window252-clip01-99-ridge10"
ROBUST_FEATURES = list(ORIGINAL_FEATURES)
WINDOW_TRADING_DAYS = 252
MIN_TRAIN_DATES = 60
ALPHA = 10.0
CLIP_QUANTILES = (0.01, 0.99)
SUPPORTED_LABELS = {"ret_fwd_3d": 3, "ret_fwd_5d": 5}


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Learn per-column bounds on training rows and reuse them unchanged."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        values = check_array(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.lower_, self.upper_ = np.quantile(
            values, [self.lower_quantile, self.upper_quantile], axis=0
        )
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        values = check_array(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("QuantileClipper feature count differs from training")
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("QuantileClipper feature order differs from training")
        return np.clip(values, self.lower_, self.upper_)


def build_robust_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Select the shared 32 factors without fitting or renormalising them.

    Preserve the input index so callers can apply common eligibility masks.
    Invalid values become NaN; callers must exclude incomplete rows before
    fitting/predicting. No imputation can conceal missing factors.
    """
    missing = [column for column in ROBUST_FEATURES if column not in panel]
    if missing:
        raise ValueError(f"Robust Ridge missing feature columns: {missing}")
    features = panel[ROBUST_FEATURES].apply(pd.to_numeric, errors="coerce")
    return features.astype(float).replace([np.inf, -np.inf], np.nan)


def build_robust_pipeline() -> Pipeline:
    """Create the frozen estimator; all learned preprocessing is train-only."""
    return Pipeline([
        ("clip", QuantileClipper(*CLIP_QUANTILES)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=ALPHA)),
    ])


def fit_predict_robust_ridge(
    panel: pd.DataFrame,
    as_of: dt.date,
    label: str,
    *,
    trading_days,
) -> tuple[pd.DataFrame, dict, Pipeline]:
    """Fit on mature exact-label rows and score the feature cutoff day.

    T+1-close labels mature on feature date + horizon + 1 trading days.
    The window consists of the last 252 mature *calendar* trading dates,
    without reaching further back to replace dates with missing data.
    At least 60 dates with complete, finite features/labels are required.
    The caller supplies the actual exchange calendar, not observed row dates.
    """
    if label not in SUPPORTED_LABELS:
        raise ValueError("Robust Ridge supports only exact ret_fwd_3d/ret_fwd_5d labels")
    missing = [column for column in ("symbol", "date", label) if column not in panel]
    if missing:
        raise ValueError(f"Robust Ridge missing columns: {missing}")
    as_of = pd.Timestamp(as_of).date()
    days = sorted(set(pd.to_datetime(list(trading_days)).date))
    if as_of not in days:
        raise ValueError(f"Prediction date {as_of} is absent from trading_days")
    dates = pd.to_datetime(panel["date"], errors="raise").dt.date
    if dates.isna().any():
        raise ValueError("Robust Ridge panel contains missing dates")
    unknown_dates = set(dates[dates <= as_of]) - set(days)
    if unknown_dates:
        raise ValueError("trading_days does not cover all panel dates through as_of")
    horizon = SUPPORTED_LABELS[label]
    mature_count = max(0, days.index(as_of) - horizon)
    window_dates = days[max(0, mature_count - WINDOW_TRADING_DAYS):mature_count]
    features = build_robust_features(panel)
    target = pd.to_numeric(panel[label], errors="coerce").astype(float)
    complete = features.notna().all(axis=1)
    train_mask = dates.isin(window_dates) & (dates < as_of) & complete & np.isfinite(target)
    train_dates = sorted(set(dates[train_mask]))
    if len(train_dates) < MIN_TRAIN_DATES:
        raise RuntimeError(
            f"Robust Ridge requires {MIN_TRAIN_DATES} mature training dates; "
            f"got {len(train_dates)} for {label} at {as_of}"
        )
    pred_mask = (dates == as_of) & complete
    if not pred_mask.any():
        raise RuntimeError(f"Robust Ridge: no complete prediction rows for as_of={as_of}")
    model = build_robust_pipeline()
    model.fit(features.loc[train_mask], target.loc[train_mask])
    pred = panel.loc[pred_mask, ["symbol", "date"]].copy()
    pred["date"] = dates[pred_mask]
    scored = score_frame(pred, model.predict(features.loc[pred_mask]), "base")
    clip = model.named_steps["clip"]
    meta = {
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "algorithm": f"Ridge(alpha={ALPHA})",
        "label": label,
        "horizon": horizon,
        "feature_count": len(ROBUST_FEATURES),
        "feature_cols": list(ROBUST_FEATURES),
        "normalize_price_again": False,
        "window_trading_days": WINDOW_TRADING_DAYS,
        "min_train_dates": MIN_TRAIN_DATES,
        "window_first_date": window_dates[0].isoformat(),
        "window_last_date": window_dates[-1].isoformat(),
        "train_dates": [date.isoformat() for date in train_dates],
        "train_first_date": train_dates[0].isoformat(),
        "train_last_date": train_dates[-1].isoformat(),
        "train_date_count": len(train_dates),
        "train_rows": int(train_mask.sum()),
        "label_maturity_rule": "date < as_of and expiry(date+horizon+1) <= as_of",
        "prediction_date": as_of.isoformat(),
        "prediction_rows": int(pred_mask.sum()),
        "prediction_excluded_rows": int(((dates == as_of) & ~complete).sum()),
        "clip_quantiles": list(CLIP_QUANTILES),
        "clip_bounds": {
            name: {"lower": float(lower), "upper": float(upper)}
            for name, lower, upper in zip(ROBUST_FEATURES, clip.lower_, clip.upper_)
        },
        "scaler_mean": model.named_steps["scaler"].mean_.tolist(),
        "scaler_scale": model.named_steps["scaler"].scale_.tolist(),
        "time_weighted": False,
        "hyperparameter_search": False,
    }
    return scored, meta, model
