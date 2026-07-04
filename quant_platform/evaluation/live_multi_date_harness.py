"""
quant_platform.evaluation.live_multi_date_harness
====================================================
T4.2 — NEW multi-date live-style validation harness.

evaluation_config_e_20260626.py validates ONE prediction date. This
module generalises that flow to accumulate >= 20 prediction
cross-sections, scoring a single FIXED model (trained once, before the
first prediction date -- this mirrors how a live deployment actually
works: you don't retrain before every single day's prediction) against
each date's realised forward return, then reports mean daily IC and
IC_IR across all dates, plus a cost-aware backtest on the pooled
predictions.

This is genuinely NEW orchestration (task.md marks it [NEW]) but reuses,
unmodified:
  - evaluation.metrics.evaluate       (per-cross-section IC/precision@k)
  - evaluation.backtest.run_backtest  (cost-aware long-short backtest)
  - ingest.ohlcv_collector.OHLCVCollector (incremental fetch of ONLY the
    missing tail -- per the OUT-OF-SCOPE constraint, this harness never
    triggers a large-scale re-collection; it calls the collector's
    already-incremental collect_symbol(), which fetches from
    last_stored_date+1 onward and skips symbols already up to date).

Distinct from WalkForwardEvaluator (T4.1)
-------------------------------------------
WalkForwardEvaluator retrains per rolling window -- the right tool for an
OFFLINE research verdict. This harness trains ONCE and scores many live
dates with the same fixed model -- the right tool for validating what a
deployed model would actually have produced day-by-day, which is what
the Config E single-date script was trying (and failing, at n=1) to do.
Both are valid, complementary evaluation instruments; neither replaces
the other.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.evaluation.metrics import evaluate
from quant_platform.evaluation.backtest import run_backtest, BacktestResult
from quant_platform.ingest.ohlcv_collector import OHLCVCollector, CollectorSummary

logger = get_logger(__name__)

MIN_DATES_REQUIRED = 20  # task.md T4.2: report §7.4 asks for 20+


@dataclass
class LiveHarnessResult:
    n_dates: int = 0
    daily_ic: pd.Series = field(default_factory=pd.Series)   # indexed by date
    mean_daily_ic: float = float("nan")
    ic_ir: float = float("nan")          # mean_daily_ic / std_daily_ic
    backtest: BacktestResult | None = None
    pooled_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def print_summary(self) -> None:
        print("\n" + "=" * 65)
        print("LIVE MULTI-DATE VALIDATION SUMMARY (T4.2)")
        print("=" * 65)
        print(f"  Prediction dates accumulated: {self.n_dates} "
              f"(task.md floor: {MIN_DATES_REQUIRED})")
        print(f"  Mean daily Rank IC:           {self.mean_daily_ic:+.4f}")
        print(f"  IC_IR:                        {self.ic_ir:+.4f}")
        if self.backtest is not None:
            print(f"  Backtest Sharpe (net of cost): {self.backtest.sharpe:+.4f}")
            print(f"  Backtest gross long-short spread: {self.backtest.long_minus_short:+.4f}")
            print(f"  Backtest net long-short spread:    {self.backtest.net_long_minus_short:+.4f}")
        if not self.daily_ic.empty:
            print("\n  Daily IC series (first 10 / last 10):")
            print(self.daily_ic.head(10).to_string())
            if len(self.daily_ic) > 10:
                print("  ...")
                print(self.daily_ic.tail(10).to_string())
        print("=" * 65 + "\n")


def ensure_data_current(
    store_root,
    symbols: list[str],
    end_date: dt.date,
    start_date: dt.date | None = None,
) -> CollectorSummary:
    """
    Bring cached silver/ohlcv up to *end_date*, fetching ONLY the missing
    tail per symbol (OHLCVCollector.collect_symbol is already incremental
    -- it does nothing for a symbol already up to date). Never triggers a
    full re-collection. This is the only network-touching call this
    module makes.
    """
    collector = OHLCVCollector(store_root, start_date=start_date, end_date=end_date)
    summary = collector.run(symbols)
    logger.info(
        "ensure_data_current: %d succeeded, %d failed, %d already up-to-date",
        summary.succeeded, summary.failed, summary.skipped,
    )
    return summary


def run_live_multi_date_harness(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    train_end_date,
    prediction_dates: list,
    model_factory: Callable,
    cost_bps: float = 10.0,
    horizon: int = 5,
) -> LiveHarnessResult:
    """
    Train ONE fixed model on panel[date < train_end_date], score every
    date in *prediction_dates*, and aggregate mean daily IC + IC_IR.

    Parameters
    ----------
    panel : pd.DataFrame
        Feature+label panel, columns [symbol, date, close (optional),
        <feature_cols>, <label_col>].
    train_end_date : date-like
        Model is trained ONLY on rows strictly before this date -- no
        peeking into the validation window.
    prediction_dates : list[date-like]
        The dates to score and evaluate. Must all be >= train_end_date.
        A warning (not an error) is logged if len < MIN_DATES_REQUIRED.
    model_factory : Callable
        Returns a fit/predict object. Called exactly once (T4.2 trains a
        single fixed model, unlike T4.1's per-window retraining).
    """
    if len(prediction_dates) < MIN_DATES_REQUIRED:
        logger.warning(
            "run_live_multi_date_harness: only %d prediction dates supplied "
            "(< %d) -- task.md report §7.4 asks for >= 20; treat this result "
            "as preliminary.", len(prediction_dates), MIN_DATES_REQUIRED,
        )

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    train_end_date = pd.to_datetime(train_end_date)
    pred_dates = sorted(pd.to_datetime(d) for d in prediction_dates)

    train = panel[panel["date"] < train_end_date].dropna(subset=feature_cols + [label_col])
    if train.empty:
        raise ValueError("run_live_multi_date_harness: empty training set before train_end_date")

    model = model_factory()
    model.fit(train[feature_cols], train[label_col])
    logger.info("Fixed model trained once on %d rows (< %s)", len(train), train_end_date.date())

    scored_frames = []
    daily_ic = {}
    for d in pred_dates:
        day = panel[panel["date"] == d].dropna(subset=feature_cols)
        if day.empty:
            logger.debug("run_live_multi_date_harness: no rows for %s -- skipping", d.date())
            continue
        preds = model.predict(day[feature_cols])
        day = day.copy()
        day["pred"] = preds

        day_eval_input = day.dropna(subset=[label_col])
        if len(day_eval_input) >= 5:
            ev = evaluate(day_eval_input["pred"], day_eval_input[label_col],
                          day_eval_input["date"], label_col=label_col)
            daily_ic[d] = ev.rank_ic_mean

        scored_frames.append(day)

    if not scored_frames:
        logger.warning("run_live_multi_date_harness: no scoreable dates -- returning empty result")
        return LiveHarnessResult()

    pooled = pd.concat(scored_frames, ignore_index=True)
    ic_series = pd.Series(daily_ic).sort_index()
    ic_series.index.name = "date"

    mean_ic = float(ic_series.mean()) if not ic_series.empty else float("nan")
    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan")
    ic_ir = mean_ic / ic_std if ic_std and not np.isnan(ic_std) and ic_std > 1e-12 else float("nan")

    bt_input = pooled.dropna(subset=["pred", label_col])
    backtest_result = None
    if not bt_input.empty:
        backtest_result = run_backtest(
            bt_input, pred_col="pred", return_col=label_col,
            cost_bps=cost_bps, horizon=horizon,
        )

    result = LiveHarnessResult(
        n_dates=len(ic_series), daily_ic=ic_series,
        mean_daily_ic=mean_ic, ic_ir=ic_ir,
        backtest=backtest_result, pooled_predictions=pooled,
    )
    result.print_summary()
    return result
