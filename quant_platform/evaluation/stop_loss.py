"""Stop-loss counterfactual arm (``stop-loss-v1``).

This module implements a *paired* stop-loss comparison against the frozen
baseline: the SAME picks, with exit accounting changed from "hold h days"
to "hold up to h days, exit early if price falls X% below entry".

=====================================================================
PRE-DECLARED RULE (fixed 2026-10-15, BEFORE any results were observed)
=====================================================================
- Entry: close(T+1)  [``trade_date`` close from the picks CSV]
- Holding window: the ``h`` trading days AFTER entry (T+2 .. T+1+h)
- Trigger: any holding-day LOW <= entry * (1 - stop_pct)  -> stopped
- Fill on stop: entry * (1 - stop_pct), minus one-way exit slippage
- If never triggered: realized return == fixed-horizon return (unchanged)

Thresholds (pre-declared, NEVER tuned to results):
- Primary:   8%   (thesis-wrong level for a D+3/D+5 fast trade)
- Sensitivity: 5%  (tighter; expect more whipsaw — reported, not hidden)

Both thresholds are ALWAYS reported.  Changing these constants without
bumping ``STOP_LOSS_RULE_VERSION`` breaks ``test_stop_loss.py::
test_thresholds_are_pre_registered`` on purpose.

Why this is not p-hacking / overfitting:
1. Thresholds were written down BEFORE any stop-loss result was computed.
2. Paired design: identical picks, only exit accounting differs.  We do not
   re-select stocks, so there is no selection bias from the stop rule.
3. Full reporting: both thresholds, all metrics, no cherry-picking.
4. The historical counterfactual below is DESCRIPTIVE (in-sample).  The
   confirmatory test is the forward 4-week parallel run.

Known limitations (documented, not hidden):
- Daily bars: intraday fill is assumed at the stop price.  Gap-through and
  limit-down (跌停, incl. 20% STAR/ChiNext boards) are NOT modelled, so the
  counterfactual is slightly optimistic for volatile names.
- 8%/5% are judgment calls for D+3/D+5 快进快出, not optimized values.
- Suspended / missing bars -> NaN (no silent fabrication).

Cost convention: one-way exit slippage 7.5 bps, consistent with the
project's 0.15% two-way (15 bps) convention.  Entry cost is identical on
both legs of the paired comparison and therefore cancels out.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-registered rule parameters — DO NOT EDIT without bumping the version.
# ---------------------------------------------------------------------------
STOP_LOSS_PRIMARY: float = 0.08
STOP_LOSS_LEVELS: Tuple[float, ...] = (0.08, 0.05)  # primary first
STOP_LOSS_RULE_VERSION: str = "stop-loss-v1"
STOP_LOSS_RULE_DECLARED: str = "2026-10-15"

# One-way exit slippage in basis points (7.5 bps == half of 0.15% two-way).
EXIT_COST_BPS: float = 7.5


@dataclass(frozen=True)
class StopLossResult:
    """Outcome of applying the stop-loss rule to one pick's holding window."""

    stopped: bool
    stop_day_offset: Optional[int]  # 1-based offset within holding window
    realized_return: float          # net of exit cost when stopped; NaN if incomputable
    trigger_low: Optional[float]    # the low that breached the stop, if any


def apply_stop_loss(
    entry_px: float,
    lows: Sequence[float],
    stop_pct: float,
    exit_cost_bps: float = EXIT_COST_BPS,
) -> StopLossResult:
    """Pure path-dependent stop-loss accounting (no I/O).

    Parameters
    ----------
    entry_px:
        Entry price = close(T+1).
    lows:
        Daily LOWs for the ``h`` holding days AFTER entry, in order.
    stop_pct:
        Pre-declared stop fraction, e.g. 0.08.
    exit_cost_bps:
        One-way exit slippage applied to the stop fill.

    Returns
    -------
    StopLossResult with ``realized_return`` == ``-stop_pct - exit_cost`` when
    stopped.  When not stopped, ``realized_return`` is NaN and the caller
    should use the fixed-horizon return (identical on both paired legs).
    NaN is also returned for invalid inputs (no silent fabrication).
    """
    if (
        entry_px is None
        or not np.isfinite(entry_px)
        or entry_px <= 0
        or not (0 < stop_pct < 1)
    ):
        return StopLossResult(False, None, math.nan, None)
    try:
        low_arr = np.asarray(lows, dtype=float)
    except (TypeError, ValueError):
        return StopLossResult(False, None, math.nan, None)
    if low_arr.size == 0 or not np.isfinite(low_arr).all() or (low_arr <= 0).any():
        return StopLossResult(False, None, math.nan, None)

    stop_px = entry_px * (1.0 - stop_pct)
    breached = np.nonzero(low_arr <= stop_px)[0]
    if breached.size == 0:
        return StopLossResult(False, None, math.nan, None)

    first = int(breached[0])
    exit_cost = exit_cost_bps / 10_000.0
    realized = -stop_pct - exit_cost
    return StopLossResult(True, first + 1, realized, float(low_arr[first]))


def stop_return_for_pick(
    symbol: str,
    trade_date: date,
    horizon: int,
    stop_pct: float,
    store_root: Path | str,
    exit_cost_bps: float = EXIT_COST_BPS,
) -> Dict[str, object]:
    """Compute fixed-horizon and stop-loss realized returns for one pick.

    Convention (matches ``labels.builder`` T+1 -> T+1+h):
    - entry position  = bar of ``trade_date`` (== T+1), entry = its close
    - holding window  = the ``h`` bars AFTER entry
    - fixed return    = close(entry+h) / close(entry) - 1

    Returns a dict with keys: ``symbol``, ``trade_date``, ``horizon``,
    ``entry_px``, ``ret_fixed``, ``ret_sl``, ``stopped``, ``stop_day_offset``,
    ``stop_pct``, ``rule_version``.  Returns NaN for ``ret_fixed``/``ret_sl``
    when the OHLCV window is incomplete (never fabricate).
    """
    from quant_platform.store.lake import ohlcv_path
    from quant_platform.store.parquet_store import read_ohlcv

    out: Dict[str, object] = {
        "symbol": symbol,
        "trade_date": trade_date,
        "horizon": horizon,
        "entry_px": math.nan,
        "ret_fixed": math.nan,
        "ret_sl": math.nan,
        "stopped": False,
        "stop_day_offset": None,
        "stop_pct": stop_pct,
        "rule_version": STOP_LOSS_RULE_VERSION,
    }
    try:
        df = read_ohlcv(ohlcv_path(store_root, symbol))
    except Exception as exc:  # unreadable store -> NaN, never fabricate
        logger.warning("%s: OHLCV unreadable: %s", symbol, exc)
        return out
    if df.empty:
        return out

    dates = pd.to_datetime(df["date"]).dt.date
    pos = np.nonzero(dates.values == trade_date)[0]
    if pos.size == 0:
        return out
    i = int(pos[0])
    n = len(df)
    # Need entry bar + h holding bars.
    if i + horizon >= n:
        return out

    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    window_close = close[i : i + horizon + 1]
    window_low = low[i + 1 : i + horizon + 1]
    if (
        not np.isfinite(window_close).all()
        or not np.isfinite(window_low).all()
        or (window_close <= 0).any()
        or (window_low <= 0).any()
    ):
        return out

    entry_px = float(window_close[0])
    ret_fixed = float(window_close[horizon] / entry_px - 1.0)
    res = apply_stop_loss(entry_px, window_low, stop_pct, exit_cost_bps)

    out["entry_px"] = entry_px
    out["ret_fixed"] = ret_fixed
    out["stopped"] = res.stopped
    out["stop_day_offset"] = res.stop_day_offset
    # Paired leg: identical when not stopped; stop accounting when stopped.
    out["ret_sl"] = res.realized_return if res.stopped else ret_fixed
    return out


def paired_summary(
    records: List[Dict[str, object]],
) -> Dict[str, object]:
    """Paired comparison statistics: stop-loss leg vs fixed-horizon leg.

    Uses only records where BOTH legs are finite.  Reports mean difference,
    paired t-statistic, trigger rate, and tail metrics.  No p-value library
    needed: t-stat + N is reported; formal inference happens locally.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return {"n": 0}
    pair = df[np.isfinite(df["ret_fixed"]) & np.isfinite(df["ret_sl"])].copy()
    n = len(pair)
    if n == 0:
        return {"n": 0}
    diff = (pair["ret_sl"] - pair["ret_fixed"]).to_numpy(dtype=float)
    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else math.nan
    t_stat = float(mean_diff / (sd / math.sqrt(n))) if sd and sd > 0 else 0.0
    stopped = pair["stopped"].to_numpy(dtype=bool)
    summary: Dict[str, object] = {
        "rule_version": STOP_LOSS_RULE_VERSION,
        "stop_pct": float(pair["stop_pct"].iloc[0]),
        "n": n,
        "n_stopped": int(stopped.sum()),
        "trigger_rate": float(stopped.mean()),
        "mean_ret_fixed": float(pair["ret_fixed"].mean()),
        "mean_ret_sl": float(pair["ret_sl"].mean()),
        "mean_diff_sl_minus_fixed": mean_diff,
        "paired_t_stat": t_stat,
        # Tail: does the stop cut the left tail?
        "min_ret_fixed": float(pair["ret_fixed"].min()),
        "min_ret_sl": float(pair["ret_sl"].min()),
        "worst10_mean_fixed": float(pair["ret_fixed"].nsmallest(max(1, n // 10)).mean()),
        "worst10_mean_sl": float(pair["ret_sl"].nsmallest(max(1, n // 10)).mean()),
    }
    return summary
