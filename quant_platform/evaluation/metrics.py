"""
evaluation.metrics
==================
Standardised evaluation module (T2.3).

Single entry point: ``evaluate(pred, fwd_return, dates)`` → ``EvalReport``.

Metrics produced
----------------
  IC            : Pearson correlation of predictions vs forward returns
                  (per-day cross-sectional, then averaged)
  Rank IC       : Spearman rank correlation (the headline metric)
  ICIR          : Rank IC mean / Rank IC std  (information ratio of the signal)
  Quantile spread: mean return per decile; the monotonicity of the staircase
  Precision@k   : fraction of top-k predictions that are in the top-k realisations
  Calibration   : reliability curve + Expected Calibration Error (ECE)
                  (calibrated probabilities for the eventual report layer)

All metrics are computed per-day cross-sectionally and then summarised,
matching the standard equity ML convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.calibration import calibration_curve

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalReport:
    """Complete evaluation report from one (predictions, labels) pair."""
    # --- IC family ---
    ic_mean:     float = float("nan")   # mean daily Pearson IC
    ic_std:      float = float("nan")
    rank_ic_mean: float = float("nan")  # mean daily Spearman Rank IC
    rank_ic_std:  float = float("nan")
    icir:        float = float("nan")   # Rank IC mean / std

    # --- Quantile ---
    quantile_returns: list[float] = field(default_factory=list)  # decile 0..9
    quantile_spread:  float = float("nan")   # top - bottom decile return
    monotonicity_ok:  bool  = False

    # --- Precision@k ---
    precision_at_k:   dict[int, float] = field(default_factory=dict)  # k → precision

    # --- Calibration ---
    ece:              float = float("nan")
    fraction_pos:     list[float] = field(default_factory=list)
    mean_pred_pos:    list[float] = field(default_factory=list)

    # --- Meta ---
    n_dates:          int = 0
    n_predictions:    int = 0
    label_col:        str = ""

    def summary_dict(self) -> dict:
        return {
            "ic_mean":        round(self.ic_mean, 6),
            "ic_std":         round(self.ic_std, 6),
            "rank_ic_mean":   round(self.rank_ic_mean, 6),
            "rank_ic_std":    round(self.rank_ic_std, 6),
            "icir":           round(self.icir, 4),
            "quantile_spread": round(self.quantile_spread, 6),
            "monotonicity_ok": self.monotonicity_ok,
            "precision_at_10": round(self.precision_at_k.get(10, float("nan")), 4),
            "precision_at_20": round(self.precision_at_k.get(20, float("nan")), 4),
            "ece":            round(self.ece, 6),
            "n_dates":        self.n_dates,
            "n_predictions":  self.n_predictions,
        }

    def print_summary(self) -> None:
        d = self.summary_dict()
        print("\n" + "=" * 55)
        print(f"EVALUATION REPORT  [{self.label_col}]")
        print("=" * 55)
        print(f"  Dates:           {d['n_dates']}")
        print(f"  Predictions:     {d['n_predictions']}")
        print(f"  IC (Pearson):    {d['ic_mean']:+.4f}  +/-{d['ic_std']:.4f}")
        print(f"  Rank IC (Spear): {d['rank_ic_mean']:+.4f}  +/-{d['rank_ic_std']:.4f}")
        print(f"  ICIR:            {d['icir']:+.4f}")
        print(f"  Quantile spread: {d['quantile_spread']:+.4f}")
        print(f"  Monotone:        {'PASS' if d['monotonicity_ok'] else 'FAIL'}")
        print(f"  Precision@10:    {d['precision_at_10']:.4f}")
        print(f"  Precision@20:    {d['precision_at_20']:.4f}")
        print(f"  ECE (calib):     {d['ece']:.4f}")
        if self.quantile_returns:
            q_str = "  ".join(f"{q:+.3f}" for q in self.quantile_returns)
            print(f"  Decile returns:  {q_str}")
        print("=" * 55 + "\n")


def evaluate(
    pred:       pd.Series,
    fwd_return: pd.Series,
    dates:      pd.Series,
    label_col:  str = "",
    k_vals:     list[int] = None,
) -> EvalReport:
    """
    Compute the full evaluation report.

    Parameters
    ----------
    pred : pd.Series
        Model predictions (raw scores, not probabilities).
    fwd_return : pd.Series
        Realised forward returns (the label).
    dates : pd.Series
        Date for each prediction (used to group cross-sectionally).
    label_col : str
        Column name for display purposes.
    k_vals : list[int]
        Values of k for precision@k.  Default [10, 20, 30].

    Returns
    -------
    EvalReport
    """
    k_vals = k_vals or [10, 20, 30]
    df = pd.DataFrame({
        "pred":   pred.values,
        "ret":    fwd_return.values,
        "date":   pd.to_datetime(dates.values),
    }).dropna()

    if df.empty:
        logger.warning("evaluate(): all rows are NaN — returning empty EvalReport")
        return EvalReport(label_col=label_col)

    report = EvalReport(
        n_dates=df["date"].nunique(),
        n_predictions=len(df),
        label_col=label_col,
    )

    # --- IC family (per-day cross-sectional) ---
    daily_ic      = []
    daily_rank_ic = []

    for _, grp in df.groupby("date"):
        if len(grp) < 3:
            continue
        ic_val,  _ = pearsonr(grp["pred"], grp["ret"])
        ric_val, _ = spearmanr(grp["pred"], grp["ret"])
        if not np.isnan(ic_val):
            daily_ic.append(ic_val)
        if not np.isnan(ric_val):
            daily_rank_ic.append(ric_val)

    if daily_ic:
        report.ic_mean = float(np.mean(daily_ic))
        report.ic_std  = float(np.std(daily_ic, ddof=1))
    if daily_rank_ic:
        report.rank_ic_mean = float(np.mean(daily_rank_ic))
        report.rank_ic_std  = float(np.std(daily_rank_ic, ddof=1))
        if report.rank_ic_std > 0:
            report.icir = report.rank_ic_mean / report.rank_ic_std

    # --- Quantile spread (global, pooled) ---
    df["decile"] = df.groupby("date")["pred"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False, duplicates="drop")
        if x.notna().sum() >= 10 else np.nan
    )
    decile_returns = (
        df.dropna(subset=["decile"])
          .groupby("decile")["ret"]
          .mean()
          .sort_index()
    )
    if len(decile_returns) >= 2:
        report.quantile_returns = decile_returns.tolist()
        report.quantile_spread  = float(
            decile_returns.iloc[-1] - decile_returns.iloc[0]
        )
        # Monotonicity: kendall-tau of decile returns vs decile rank
        from scipy.stats import kendalltau
        tau, _ = kendalltau(range(len(decile_returns)), decile_returns.values)
        report.monotonicity_ok = tau > 0.5

    # --- Precision@k (per date, then averaged) ---
    prec_by_k: dict[int, list[float]] = {k: [] for k in k_vals}
    for _, grp in df.groupby("date"):
        if len(grp) < max(k_vals):
            continue
        top_pred    = set(grp.nlargest(max(k_vals), "pred").index)
        top_ret     = set(grp.nlargest(max(k_vals), "ret").index)
        for k in k_vals:
            top_k_pred = set(grp.nlargest(k, "pred").index)
            top_k_ret  = set(grp.nlargest(k, "ret").index)
            prec_by_k[k].append(len(top_k_pred & top_k_ret) / k)

    report.precision_at_k = {
        k: float(np.mean(vals)) for k, vals in prec_by_k.items() if vals
    }

    # --- Calibration (ECE) ---
    # Convert raw scores to pseudo-probabilities via percentile rank
    df["prob"] = df.groupby("date")["pred"].rank(pct=True)
    df["bin_label"] = (df["ret"] > df.groupby("date")["ret"].transform("median")).astype(int)

    try:
        frac_pos, mean_pred = calibration_curve(
            df["bin_label"], df["prob"], n_bins=10, strategy="quantile"
        )
        report.fraction_pos  = frac_pos.tolist()
        report.mean_pred_pos = mean_pred.tolist()
        report.ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    except Exception as exc:
        logger.debug("Calibration failed: %s", exc)

    return report
