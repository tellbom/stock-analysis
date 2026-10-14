"""
quant_platform.selection.ridge_baseline
========================================
Shared weekly Ridge32 baseline — the "original" model.

Both weekly entries MUST reuse this exact implementation so that ordinary
and hybrid arms share one identical Ridge32 prediction:

  * scripts/compare_original_hybrid.py  (ordinary / hybrid_no_flow / hybrid_flow / diagnostic arms)
  * scripts/run_dual_horizon_rrf.py     (RRF strategy; --base-model ridge32, the default)

Frozen contract (verified 2026-09-30, docs/reviews/2026-09-30-controlled):
  * 32 technical features (ORIGINAL_FEATURES) — no fundamentals, no flow, no events
  * Pipeline: median imputation -> standard scaling -> Ridge(alpha=1.0)
  * Train: every row with date < as_of and a non-null label
  * Predict: rows with date == as_of

The 35/40-factor LightGBM long-history model is a SEPARATE, independent
strategy. It must never be silently substituted for this baseline; callers
that keep it must label it explicitly (e.g. "lgbm_long_independent").
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from sklearn.pipeline import Pipeline

STRATEGY_ID = "ridge32_weekly_baseline"
STRATEGY_VERSION = "1.0-ridge-alpha1.0"
ALPHA = 1.0

# P0 快进快出适配（2026-10-15 最终裁决）：样本时间衰减权重的半衰期（天）。
# D+3 短线下，3 年前样本与上周样本等权是不合理的；指数衰减让近期 regime
# 占主导。180 天半衰期是保守起点——权重变化平滑，不会推倒重来。
TIME_WEIGHT_HALF_LIFE_DAYS = 180

# The original 32-column Ridge contract, verified against the delivered file.
ORIGINAL_FEATURES = [
    "ma_5", "ma_10", "ma_20", "ma_60", "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j", "rsi_6", "rsi_12", "rsi_24", "boll_upper",
    "boll_mid", "atr_14", "adx_14", "obv", "cci_14", "roc_10", "willr_14",
    "stoch_k", "stoch_d", "reversal_3d", "cs_rank_close", "cs_rank_volume",
    "cs_rank_rsi_6", "cs_rank_roc_10", "cs_rank_ma_5", "cs_zscore_close",
    "cs_zscore_volume", "cs_zscore_rsi_6",
]


def score_frame(pred: pd.DataFrame, values: np.ndarray, prefix: str) -> pd.DataFrame:
    """Attach score/rank/percentile columns to a prediction-day frame.

    Returns columns: symbol, date, trade_date, {prefix}_score, {prefix}_rank,
    {prefix}_pct.  Identical for every caller — this is what "shared
    baseline" means at the frame level.
    """
    if len(values) != len(pred) or not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite predictions")
    result = pred[["symbol", "date"]].copy()
    result["trade_date"] = result["date"]
    result[f"{prefix}_score"] = values
    result[f"{prefix}_rank"] = result[f"{prefix}_score"].rank(ascending=False, method="first").astype(int)
    result[f"{prefix}_pct"] = result[f"{prefix}_score"].rank(ascending=True, pct=True)
    return result


def fit_predict_ridge32(
    panel: pd.DataFrame,
    as_of: dt.date,
    label: str,
    feature_cols: list[str] | None = None,
    use_time_weight: bool = False,
) -> tuple[pd.DataFrame, dict, "Pipeline"]:
    """Fit the frozen Ridge32 baseline and score the as_of prediction day.

    Parameters
    ----------
    panel : pd.DataFrame
        Feature panel with ``symbol``/``date`` columns (date as dt.date).
    as_of : dt.date
        Feature cutoff date. Trains on ``date < as_of``; scores ``date == as_of``.
    label : str
        Forward-return label column, e.g. ``ret_fwd_3d``.
    feature_cols : list[str] | None
        Defaults to ORIGINAL_FEATURES.  Must all exist in *panel*.
    use_time_weight : bool
        P0 快进快出实验开关（默认 False = 冻结基线行为）。
        True 时训练样本按指数衰减加权：weight = 0.5 ** (days_ago / 180)，
        即 180 天前的样本权重衰减一半。近期 regime 占主导，适配 D+3 短线。
        通过 sklearn 的 sample_weight 传入（Pipeline 末端 step 名为 "model"）。

    Returns
    -------
    (scored, meta, model)
        scored: score_frame output with prefix "base".
        meta: reproducibility metadata (strategy id/version, train window,
        row counts, time-weight flag).
        model: fitted sklearn Pipeline.
    """
    feature_cols = list(feature_cols) if feature_cols is not None else list(ORIGINAL_FEATURES)
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Ridge32 baseline missing {len(missing)} feature columns: {missing[:10]}"
            + ("..." if len(missing) > 10 else "")
        )

    train = panel[panel["date"] < as_of].dropna(subset=feature_cols + [label]).copy()
    if train.empty:
        raise RuntimeError(f"Ridge32 baseline: empty training set for as_of={as_of}, label={label}")
    pred = panel[panel["date"] == as_of].dropna(subset=feature_cols).copy()
    if pred.empty:
        raise RuntimeError(f"Ridge32 baseline: no prediction rows for as_of={as_of}")

    # 惰性导入 sklearn：模块常量（ORIGINAL_FEATURES/score_frame）在无 sklearn
    # 环境也可使用；只有实际拟合时才需要。
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=ALPHA)),
    ])
    fit_kwargs: dict = {}
    if use_time_weight:
        # P0：指数时间衰减。days_ago=0（as_of 前一天）权重为 1，
        # 180 天前权重 0.5，3 年前约 0.06。近期样本主导拟合。
        days_ago = np.array([(as_of - d).days for d in train["date"]], dtype=float)
        weights = 0.5 ** (days_ago / TIME_WEIGHT_HALF_LIFE_DAYS)
        fit_kwargs["model__sample_weight"] = weights
    model.fit(train[feature_cols].values, train[label].values, **fit_kwargs)
    scored = score_frame(pred, model.predict(pred[feature_cols].values), "base")

    meta = {
        "strategy_id": STRATEGY_ID,
        "strategy_version": STRATEGY_VERSION,
        "algorithm": f"Ridge(alpha={ALPHA})",
        "feature_count": len(feature_cols),
        "train_rows": int(len(train)),
        "train_first_date": train["date"].min().isoformat(),
        "train_last_date": train["date"].max().isoformat(),
        "prediction_date": as_of.isoformat(),
        "prediction_rows": int(len(pred)),
        "time_weighted": bool(use_time_weight),
        "time_weight_half_life_days": TIME_WEIGHT_HALF_LIFE_DAYS if use_time_weight else None,
    }
    return scored, meta, model
