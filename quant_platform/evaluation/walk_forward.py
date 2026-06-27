"""
evaluation.walk_forward
=======================
Walk-forward / rolling out-of-sample evaluator (P4A-03).

Motivation
----------
A single 12-month static lockbox at a 20-day horizon yields only ~12
independent forward periods.  With daily Rank IC std ~= 0.14 the standard
error of the mean is ~= 0.039, so a lockbox Rank IC of 0.004 has a 95%
confidence interval of [-0.07, +0.08] - statistically indistinguishable
from zero *and* from a healthy 0.05 signal.

``WalkForwardEvaluator`` fixes this by walking a test window sequentially
through the available history.  With n_windows=5 and window_months=12
at a 5-day horizon the evaluator accumulates ~250 independent forward
periods - enough to detect a 0.04 forward Rank IC with reasonable power.

Design
------
- Each window trains on all data strictly before it (minus a purge gap),
  predicts on the window, and yields a ``WalkForwardWindow`` result.
- No data after the test window is used in training - no future leakage.
- The purge gap equals ``horizon + embargo_extra`` trading days before the
  window start, matching the logic in ``PurgedKFold``.
- Windows are non-overlapping (step_months == window_months by default).
- The aggregate IC is computed over the concatenated OOS predictions from
  all windows - this is the primary evaluation metric.

Relationship to lockbox
-----------------------
``make_lockbox_split`` in ``training.splitter`` is now **deprecated** as
the primary evaluation method.  It is retained for backward compatibility
with existing tests.  Walk-forward is the new default.

Usage
-----
    from quant_platform.evaluation.walk_forward import WalkForwardEvaluator
    from quant_platform.training.lgbm_model import build_lgbm_pipeline

    wf = WalkForwardEvaluator(n_windows=5, window_months=12, horizon=5)
    result = wf.run(
        panel=panel,
        feature_cols=feature_cols,
        label_col="ret_fwd_5d",
        model_factory=build_lgbm_pipeline,
    )
    result.print_summary()
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_platform.evaluation.metrics import evaluate, EvalReport
from quant_platform.evaluation.backtest import run_backtest, BacktestResult
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-window result
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardWindow:
    """Results from a single walk-forward test window."""
    window_id:    int
    train_start:  str      # ISO date string
    train_end:    str
    test_start:   str
    test_end:     str
    n_train:      int
    n_test:       int

    # Metrics on this window's OOS predictions
    rank_ic_mean: float = float("nan")
    rank_ic_std:  float = float("nan")
    icir:         float = float("nan")
    sharpe:       float = float("nan")

    # Independent period count: n_test_dates / horizon
    n_independent_periods: int = 0

    def summary_dict(self) -> dict:
        return {
            "window_id":    self.window_id,
            "train_start":  self.train_start,
            "train_end":    self.train_end,
            "test_start":   self.test_start,
            "test_end":     self.test_end,
            "n_train":      self.n_train,
            "n_test":       self.n_test,
            "rank_ic_mean": round(self.rank_ic_mean, 6),
            "icir":         round(self.icir, 4),
            "sharpe":       round(self.sharpe, 4),
            "n_independent_periods": self.n_independent_periods,
        }


# ---------------------------------------------------------------------------
# IC decay helper
# ---------------------------------------------------------------------------

def _ic_at_lag(
    panel: pd.DataFrame,
    pred_col: str,
    close_col: str,
    lags: list[int],
) -> dict[int, float]:
    """
    Compute cross-sectional Rank IC of ``pred_col`` vs the actual
    ``lag``-day forward return from close prices, for each lag.

    Uses close prices from the panel to compute forward returns on the
    fly - so it only requires a 'close' column, not pre-built label cols.
    Rows with missing close or predictions are skipped.
    """
    result: dict[int, float] = {}
    if close_col not in panel.columns or pred_col not in panel.columns:
        return result

    df = panel.sort_values(["symbol", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    for lag in lags:
        # Forward return at lag days (using close price shift within symbol)
        df[f"_fwd_{lag}"] = (
            df.groupby("symbol")[close_col]
              .transform(lambda x: x.shift(-lag) / x.shift(-1) - 1)
        )
        sub = df[[pred_col, f"_fwd_{lag}", "date"]].dropna()
        if sub.empty:
            result[lag] = float("nan")
            continue

        daily_rics = []
        for _, grp in sub.groupby("date"):
            if len(grp) < 3:
                continue
            ric, _ = spearmanr(grp[pred_col], grp[f"_fwd_{lag}"])
            if not np.isnan(ric):
                daily_rics.append(ric)
        result[lag] = float(np.mean(daily_rics)) if daily_rics else float("nan")

    return result


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Aggregated results from all walk-forward windows."""
    windows: list[WalkForwardWindow] = field(default_factory=list)

    # Aggregate over ALL windows' OOS predictions (the headline metric)
    agg_rank_ic_mean: float = float("nan")
    agg_rank_ic_std:  float = float("nan")
    agg_icir:         float = float("nan")
    agg_sharpe:       float = float("nan")
    agg_max_drawdown: float = float("nan")

    # Total independent forward periods across all windows
    total_independent_periods: int = 0

    # IC sign stability: fraction of windows with positive Rank IC
    # A tiny positive IC still counts as positive here, so the summary also
    # flags near-zero positive windows when stability is 1.0.
    ic_sign_stability: float = float("nan")   # [0, 1]; 1.0 = all windows positive

    # Simple average of per-window Sharpe, shown next to pooled-PnL Sharpe.
    per_window_sharpe_mean: float = float("nan")

    # IC decay curve: {lag_days: mean Rank IC across windows}
    ic_decay: dict[int, float] = field(default_factory=dict)

    # Raw OOS predictions concatenated (for downstream use)
    oos_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def n_windows(self) -> int:
        return len(self.windows)

    def window_ic_series(self) -> pd.Series:
        """Per-window Rank IC as a Series indexed by window_id."""
        return pd.Series(
            {w.window_id: w.rank_ic_mean for w in self.windows}
        )

    def print_summary(self) -> None:
        print("\n" + "=" * 65)
        print("WALK-FORWARD EVALUATION SUMMARY")
        print("=" * 65)
        print(f"  Windows evaluated:          {self.n_windows()}")
        print(f"  Total independent periods:  {self.total_independent_periods}")
        print(f"  Agg Rank IC (OOS):          {self.agg_rank_ic_mean:+.4f}  "
              f"+/-{self.agg_rank_ic_std:.4f}")
        print(f"  Agg ICIR (OOS):             {self.agg_icir:+.4f}")
        print(f"  Agg Sharpe (pooled PnL):    {self.agg_sharpe:+.4f}  "
              f"[per-window mean: {self.per_window_sharpe_mean:+.4f}]")
        print(f"  Agg Max Drawdown:           {self.agg_max_drawdown:+.4f}")
        stability_note = _interpret_sign_stability(self.ic_sign_stability)
        if self.ic_sign_stability == 1.0 and self.n_windows() >= 2:
            near_zero = [w for w in self.windows if 0 < w.rank_ic_mean < 0.01]
            if near_zero:
                stability_note += "; latest/near-zero positive window present"
        print(f"  IC sign stability:          {self.ic_sign_stability:.2f}  "
              f"({stability_note})")
        print()
        print("  Per-window breakdown:")
        print(f"  {'Win':>3}  {'Test period':>25}  {'RankIC':>8}  "
              f"{'ICIR':>7}  {'Sharpe':>7}  {'IndepPd':>8}")
        for w in self.windows:
            print(f"  {w.window_id:>3}  "
                  f"{w.test_start} - {w.test_end}  "
                  f"{w.rank_ic_mean:>+8.4f}  "
                  f"{w.icir:>+7.3f}  "
                  f"{w.sharpe:>+7.3f}  "
                  f"{w.n_independent_periods:>8}")
        if self.ic_decay:
            lags = sorted(self.ic_decay)
            decay_str = "  ".join(f"{l}d:{v:+.4f}" for l, v in sorted(self.ic_decay.items()))
            print(f"\n  IC decay: {decay_str}")
        print("=" * 65 + "\n")

    def to_dataframe(self) -> pd.DataFrame:
        """Per-window metrics as a DataFrame."""
        return pd.DataFrame([w.summary_dict() for w in self.windows])


def _interpret_sign_stability(ratio: float) -> str:
    if np.isnan(ratio):
        return "unknown"
    if ratio >= 0.8:
        return "strong - signal is regime-stable"
    if ratio >= 0.6:
        return "moderate - some regime sensitivity"
    return "weak - signal is regime-sensitive, further investigation required"


# ---------------------------------------------------------------------------
# Walk-forward evaluator
# ---------------------------------------------------------------------------

class WalkForwardEvaluator:
    """
    Sequential walk-forward out-of-sample evaluator.

    Parameters
    ----------
    n_windows : int
        Number of sequential test windows.  Default 5.
    window_months : int
        Length of each test window in calendar months.  Default 12.
    step_months : int
        How far to advance between windows.  Default equals window_months
        (non-overlapping).  Use a smaller value for overlapping windows.
    horizon : int
        Label horizon in trading days.  Used to compute the purge gap
        before each test window and to sub-sample the backtest.
    embargo_extra : int
        Additional trading days of buffer beyond ``horizon`` before each
        test window.  Default 2 (matches the ``−2`` in PurgedKFold).
    min_train_months : int
        Minimum training history required before running a window.
        Windows without enough history are skipped.  Default 12.
    cost_bps : float
        One-way transaction cost for the backtest.  Default 10.
    """

    def __init__(
        self,
        n_windows:        int = 5,
        window_months:    int = 12,
        step_months:      int | None = None,
        horizon:          int = 5,
        embargo_extra:    int = 2,
        min_train_months: int = 12,
        cost_bps:         float = 10.0,
    ) -> None:
        self.n_windows        = n_windows
        self.window_months    = window_months
        self.step_months      = step_months if step_months is not None else window_months
        self.horizon          = horizon
        self.embargo_extra    = embargo_extra
        self.min_train_months = min_train_months
        self.cost_bps         = cost_bps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        model_factory: Callable = None,
        decay_lags: list[int] | None = None,
    ) -> WalkForwardResult:
        """
        Run the walk-forward evaluation.

        Parameters
        ----------
        panel : pd.DataFrame
            Must have columns: date, symbol, <feature_cols>, <label_col>.
            Sorted by date ascending.  One row per (symbol, date).
        feature_cols : list[str]
            Feature columns to use.
        label_col : str
            Target column (e.g. ``ret_fwd_5d``).
        model_factory : callable | None
            Callable that returns a fitted sklearn Pipeline (or any object
            with ``fit(X, y)`` and ``predict(X)``).  If None, uses the
            default LightGBM pipeline from ``training.lgbm_model``.
        decay_lags : list[int] | None
            Lags (in trading days) for the IC decay curve.
            Default [1, 2, 3, 5, 10, 20].  Requires a 'close' column.

        Returns
        -------
        WalkForwardResult
        """
        if model_factory is None:
            from quant_platform.training.lgbm_model import build_lgbm_pipeline
            model_factory = build_lgbm_pipeline

        decay_lags = decay_lags or [1, 2, 3, 5, 10, 20]

        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.sort_values("date").reset_index(drop=True)

        all_dates = sorted(panel["date"].unique())
        if len(all_dates) < 2:
            logger.warning("walk_forward: panel has fewer than 2 unique dates")
            return WalkForwardResult()

        # Build window boundaries (calendar-based)
        windows = self._build_windows(all_dates)
        if not windows:
            logger.warning("walk_forward: could not build any valid windows")
            return WalkForwardResult()

        result = WalkForwardResult()
        all_oos_rows: list[pd.DataFrame] = []
        window_rics: list[float] = []

        for win_id, (test_start, test_end) in enumerate(windows):
            win_result = self._run_window(
                panel, feature_cols, label_col,
                test_start, test_end, win_id,
                model_factory,
            )
            if win_result is None:
                continue
            result.windows.append(win_result)
            window_rics.append(win_result.rank_ic_mean)

            # Collect OOS predictions
            # _run_window filters out rows with missing labels before predicting.
            # Keep aggregation on that exact filtered row set so prediction length
            # cannot drift from the test-window row count near series ends.
            if hasattr(win_result, "_oos_df") and win_result._oos_df is not None:
                all_oos_rows.append(win_result._oos_df)

        if not result.windows:
            return result

        # Aggregate OOS predictions
        if all_oos_rows:
            oos_df = pd.concat(all_oos_rows, ignore_index=True)
            result.oos_predictions = oos_df

            oos_valid = oos_df.dropna(subset=["pred", label_col])
            if not oos_valid.empty:
                agg_eval = evaluate(
                    oos_valid["pred"],
                    oos_valid[label_col],
                    oos_valid["date"],
                    label_col=f"wf_oos_{label_col}",
                )
                result.agg_rank_ic_mean = agg_eval.rank_ic_mean
                result.agg_rank_ic_std  = agg_eval.rank_ic_std
                result.agg_icir         = agg_eval.icir

                bt = run_backtest(
                    oos_valid,
                    pred_col="pred",
                    return_col=label_col,
                    cost_bps=self.cost_bps,
                    horizon=self.horizon,
                )
                result.agg_sharpe      = bt.sharpe
                result.agg_max_drawdown = bt.max_drawdown

                # IC decay curve (requires 'close' column)
                if "close" in oos_valid.columns:
                    result.ic_decay = _ic_at_lag(
                        oos_valid, "pred", "close", decay_lags
                    )

        # Aggregate independent period count
        result.total_independent_periods = sum(
            w.n_independent_periods for w in result.windows
        )

        # IC sign stability
        valid_rics = [r for r in window_rics if not np.isnan(r)]
        if valid_rics:
            result.ic_sign_stability = float(
                sum(1 for r in valid_rics if r > 0) / len(valid_rics)
            )
        window_sharpes = [w.sharpe for w in result.windows if not np.isnan(w.sharpe)]
        if window_sharpes:
            result.per_window_sharpe_mean = float(np.mean(window_sharpes))

        logger.info(
            "Walk-forward complete: %d windows, agg Rank IC=%.4f, ICIR=%.4f, "
            "Sharpe=%.4f, per_window_sharpe_mean=%.4f, total_indep_periods=%d",
            len(result.windows),
            result.agg_rank_ic_mean,
            result.agg_icir,
            result.agg_sharpe,
            result.per_window_sharpe_mean,
            result.total_independent_periods,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_windows(
        self,
        all_dates: list,
    ) -> list[tuple]:
        """
        Build (test_start, test_end) pairs working backwards from the
        end of the date range.

        Walk-forward goes chronologically: window 0 is earliest, window
        n-1 is latest.  We require at least ``min_train_months`` of
        history before the first test window.
        """
        max_date = all_dates[-1]
        min_date = all_dates[0]

        windows = []
        # Build from the end backwards, then reverse
        test_end = max_date
        for _ in range(self.n_windows):
            test_start = test_end - pd.DateOffset(months=self.window_months) + pd.Timedelta(days=1)
            # Check we have enough training history
            purge_days   = self.horizon + self.embargo_extra
            train_end_dt = test_start - pd.Timedelta(days=purge_days + 1)
            min_train_end = min_date + pd.DateOffset(months=self.min_train_months)
            if train_end_dt < min_train_end:
                break  # not enough history for this window
            windows.append((test_start, test_end))
            test_end = test_start - pd.Timedelta(days=1)

        windows.reverse()  # chronological order
        return windows

    def _run_window(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        test_start,
        test_end,
        win_id: int,
        model_factory: Callable,
    ) -> WalkForwardWindow | None:
        """Train on pre-window data, predict on window, return metrics."""
        # Purge gap: horizon + embargo_extra trading days
        purge_days = self.horizon + self.embargo_extra
        train_end_dt = test_start - pd.Timedelta(days=purge_days + 1)

        train_mask = panel["date"] < train_end_dt
        test_mask  = (panel["date"] >= test_start) & (panel["date"] <= test_end)

        train = panel[train_mask].copy()
        test  = panel[test_mask].copy()

        if len(train) == 0 or len(test) == 0:
            logger.warning("Window %d: empty train or test - skipping", win_id)
            return None

        train_valid = train[train[label_col].notna()]
        test_valid  = test[test[label_col].notna()]

        if len(train_valid) < 50 or len(test_valid) < 10:
            logger.warning(
                "Window %d: insufficient data (train=%d, test=%d) - skipping",
                win_id, len(train_valid), len(test_valid),
            )
            return None

        # Fit on training data
        X_tr = train_valid[feature_cols]
        y_tr = train_valid[label_col]
        X_te = test_valid[feature_cols]

        try:
            pipe = model_factory()
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
        except Exception as exc:
            logger.error("Window %d: model fit/predict failed: %s", win_id, exc)
            return None

        pred_series = pd.Series(preds, index=test_valid.index)

        # Evaluate
        oos_eval = evaluate(
            pred_series,
            test_valid[label_col],
            test_valid["date"],
            label_col=f"wf_win{win_id}",
        )

        # Backtest on this window
        bt_panel = test_valid[["date", "symbol", label_col]].copy()
        bt_panel["pred"] = pred_series.values
        bt = run_backtest(
            bt_panel,
            pred_col="pred",
            return_col=label_col,
            cost_bps=self.cost_bps,
            horizon=self.horizon,
        )

        n_test_dates = test_valid["date"].nunique()
        n_indep = max(1, n_test_dates // self.horizon)

        win = WalkForwardWindow(
            window_id=win_id,
            train_start=str(train["date"].min().date()),
            train_end=str(train["date"].max().date()),
            test_start=str(test_start.date() if hasattr(test_start, "date") else test_start),
            test_end=str(test_end.date() if hasattr(test_end, "date") else test_end),
            n_train=len(train_valid),
            n_test=len(test_valid),
            rank_ic_mean=oos_eval.rank_ic_mean,
            rank_ic_std=oos_eval.rank_ic_std,
            icir=oos_eval.icir,
            sharpe=bt.sharpe,
            n_independent_periods=n_indep,
        )
        # Attach filtered OOS rows for aggregation (private attribute).
        oos_cols = ["date", "symbol", label_col]
        if "close" in test_valid.columns:
            oos_cols.append("close")
        oos_df = test_valid[oos_cols].copy()
        oos_df["pred"] = pred_series.values
        win._oos_df = oos_df.reset_index(drop=True)

        logger.info(
            "Window %d [%s – %s]: Rank IC=%.4f  ICIR=%.4f  Sharpe=%.4f  "
            "indep_periods=%d",
            win_id,
            win.test_start, win.test_end,
            win.rank_ic_mean, win.icir, win.sharpe,
            win.n_independent_periods,
        )
        return win
