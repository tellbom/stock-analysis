"""
evaluation.baselines
====================
Baseline gauntlet (T2.4).

Computes four trivial predictors and evaluates each with the standard
metrics module.  The LightGBM baseline must beat ALL of them to claim alpha.

Baselines
---------
  momentum_1d  : yesterday's return  (ret_1d lagged)
  mean_rev_1d  : negative yesterday's return (mean reversion)
  cs_momentum  : cross-sectional rank of trailing 20d return
  random       : uniform random predictions (permuted by date)

Each baseline is cheap to compute from OHLCV — no model training required.

The comparison table format is designed for MLflow logging and the
alpha-verdict document.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.evaluation.metrics import evaluate, EvalReport
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


def _momentum_predictor(panel: pd.DataFrame, label_col: str) -> pd.Series:
    """Yesterday's return as a predictor (1d momentum)."""
    panel = panel.sort_values(["symbol", "date"]).copy()
    # Use the 1d raw return shifted by 1 (prev day's return → predict next day)
    panel["_pred"] = panel.groupby("symbol")[label_col].shift(1)
    return panel["_pred"]


def _mean_reversion_predictor(panel: pd.DataFrame, label_col: str) -> pd.Series:
    """Negative yesterday's return (short-term mean reversion)."""
    return -_momentum_predictor(panel, label_col)


def _cross_sectional_momentum_predictor(
    panel: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Cross-sectional rank of trailing-window sum of returns.
    Uses close price if available, otherwise the label column.
    """
    panel = panel.sort_values(["symbol", "date"]).copy()
    if "close" in panel.columns:
        panel["_ret"] = panel.groupby("symbol")["close"].pct_change()
    else:
        panel["_ret"] = panel.groupby("symbol")[panel.columns[-1]].shift(0)

    panel["_rolling"] = panel.groupby("symbol")["_ret"].transform(
        lambda x: x.rolling(window, min_periods=max(1, window // 2)).sum()
    )
    panel["_pred"] = panel.groupby("date")["_rolling"].rank(pct=True)
    return panel["_pred"]


def _random_predictor(panel: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Uniform random predictions, permuted within each date."""
    rng    = np.random.default_rng(seed)
    result = panel.copy()
    result["_pred"] = 0.0
    for date, grp in result.groupby("date"):
        result.loc[grp.index, "_pred"] = rng.permutation(len(grp)).astype(float)
    return result["_pred"]


def run_baseline_gauntlet(
    panel: pd.DataFrame,
    label_col: str,
    model_pred: pd.Series,
    model_name: str = "LightGBM",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run all baselines and compare against the model.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have: symbol, date, <label_col>.
    label_col : str
        The label column to evaluate against.
    model_pred : pd.Series
        The model's OOF predictions, aligned to panel.index.
    model_name : str
        Label for the model row in the output table.

    Returns
    -------
    pd.DataFrame
        Comparison table with rows: [model, momentum, mean_rev, cs_momentum, random].
        Columns: rank_ic_mean, icir, quantile_spread, precision_at_10, ece.
    """
    valid = panel[panel[label_col].notna()].copy().reset_index(drop=True)
    dates = pd.to_datetime(valid["date"])
    ret   = valid[label_col]

    predictors = {
        model_name:    model_pred.reindex(valid.index) if model_pred is not None else pd.Series(np.nan, index=valid.index),
        "momentum_1d": _momentum_predictor(valid, label_col),
        "mean_rev_1d": _mean_reversion_predictor(valid, label_col),
        "cs_momentum": _cross_sectional_momentum_predictor(valid),
        "random":      _random_predictor(valid, seed=seed),
    }

    rows = []
    for name, pred in predictors.items():
        pred_aligned = pred.reindex(valid.index)
        report = evaluate(pred_aligned, ret, dates, label_col=name)
        d = report.summary_dict()
        rows.append({
            "predictor":       name,
            "rank_ic_mean":    d["rank_ic_mean"],
            "icir":            d["icir"],
            "quantile_spread": d["quantile_spread"],
            "precision_at_10": d["precision_at_10"],
            "ece":             d["ece"],
            "n_dates":         d["n_dates"],
        })
        logger.info(
            "Baseline '%s': Rank IC=%.4f  ICIR=%.3f",
            name, d["rank_ic_mean"], d["icir"],
        )

    table = pd.DataFrame(rows).set_index("predictor")

    # Flag which baselines the model beats
    model_ric = table.loc[model_name, "rank_ic_mean"]
    table["beats_baseline"] = table["rank_ic_mean"] < model_ric
    table.loc[model_name, "beats_baseline"] = True   # model vs itself

    return table
