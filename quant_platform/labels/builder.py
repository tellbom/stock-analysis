"""
labels.builder
==============
Label builder (T1.6).

Constructs forward-return labels with strict T+1 execution assumption,
matching Qlib's Alpha158 convention:

    label(T) = close(T+1+h) / close(T+1) - 1

where:
  T   = the date we observe features (feature construction date)
  T+1 = earliest possible execution date (next trading day's open/close)
  h   = holding horizon in trading days

This means:
- The label uses ONLY future prices relative to T+1.
- The label at date T does NOT include close(T) in its denominator
  (which would create a trivial look-ahead from the T close itself).
- A label at T is valid only if dates T+1 and T+1+h both exist in the data.

Label types produced
--------------------
  ret_fwd_{h}d      : raw forward return (continuous, signed)
  ret_fwd_{h}d_cs   : cross-sectional quantile decile (0–9) of that return
                       on the same date (day-relative rank, not look-ahead)
  ret_fwd_{h}d_bin  : binary 1/0 = outperforms cross-sectional median return

Risk labels
-----------
  vol_fwd_{h}d      : forward realised volatility (std of daily returns T+1…T+h)
  mdd_fwd_{h}d      : forward max drawdown (T+1…T+h)

NaN policy
----------
- Rows where T+1 or T+1+h do not exist in the data get NaN labels.
- These are the rows at the END of each symbol's history (the embargo zone).
  The leakage harness (T1.7) confirms that these NaN rows are excluded from
  training — they cannot be imputed or forward-filled.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import label_path, label_dir, ohlcv_path
from quant_platform.store.parquet_store import read_ohlcv

logger = get_logger(__name__)

# Default holding horizons (in trading days)
DEFAULT_HORIZONS: list[int] = [1, 5, 20]


def build_labels(
    store_root: Path | str,
    symbols: list[str],
    horizons: list[int] | None = None,
    overwrite: bool = True,
) -> dict[str, int]:
    """
    Compute forward-return labels for all symbols and write to label Parquets.

    Parameters
    ----------
    store_root : Path | str
    symbols : list[str]
    horizons : list[int] | None
        Holding periods in trading days.  Default [1, 5, 20].
    overwrite : bool
        If True, overwrite existing label files.  Default True.

    Returns
    -------
    dict[str, int]
        Mapping symbol → number of rows written (0 if skipped/failed).
    """
    store_root = Path(store_root)
    horizons   = horizons or DEFAULT_HORIZONS
    results: dict[str, int] = {}

    for symbol in symbols:
        try:
            n = _build_symbol_labels(symbol, store_root, horizons, overwrite)
            results[symbol] = n
        except Exception as exc:
            logger.error("Label build failed for %s: %s", symbol, exc)
            results[symbol] = 0

    succeeded = sum(1 for v in results.values() if v > 0)
    logger.info(
        "Label build done: %d/%d symbols, horizons=%s",
        succeeded, len(symbols), horizons,
    )
    return results


def _build_symbol_labels(
    symbol: str,
    store_root: Path,
    horizons: list[int],
    overwrite: bool,
) -> int:
    """Build labels for one symbol; return number of rows written."""
    df = read_ohlcv(ohlcv_path(store_root, symbol))
    if df.empty:
        logger.warning("%s: no OHLCV — skipping labels", symbol)
        return 0

    df["date"]  = pd.to_datetime(df["date"]).dt.date
    df          = df.sort_values("date").reset_index(drop=True)
    close       = df["close"].values
    n           = len(df)

    # Base output: symbol + date
    out = pd.DataFrame({"symbol": symbol, "date": df["date"]})

    for h in horizons:
        ret_col  = f"ret_fwd_{h}d"
        cs_col   = f"ret_fwd_{h}d_cs"
        bin_col  = f"ret_fwd_{h}d_bin"
        vol_col  = f"vol_fwd_{h}d"
        mdd_col  = f"mdd_fwd_{h}d"

        ret_vals = np.full(n, np.nan)
        vol_vals = np.full(n, np.nan)
        mdd_vals = np.full(n, np.nan)

        for i in range(n):
            # T+1 index and T+1+h index
            t1   = i + 1
            t1ph = i + 1 + h
            if t1 >= n or t1ph >= n:
                continue  # not enough future data → NaN
            # Forward return: close(T+1+h) / close(T+1) - 1
            c_t1   = close[t1]
            c_t1ph = close[t1ph]
            if c_t1 <= 0:
                continue
            ret_vals[i] = c_t1ph / c_t1 - 1.0

            # Forward realised volatility: std of daily returns T+1 … T+1+h
            if t1ph > t1:
                window = close[t1 : t1ph + 1]
                daily_rets = np.diff(window) / window[:-1]
                vol_vals[i] = float(np.std(daily_rets, ddof=1)) if len(daily_rets) > 1 else np.nan

            # Forward max drawdown T+1 … T+1+h
            window_mdd = close[t1 : t1ph + 1]
            mdd_vals[i] = _max_drawdown(window_mdd)

        out[ret_col] = ret_vals
        out[vol_col] = vol_vals
        out[mdd_col] = mdd_vals

        # Cross-sectional decile and binary — computed below after all symbols
        # For per-symbol label file, store raw return; CS decile added in panel
        out[cs_col]  = np.nan   # placeholder; filled by build_label_panel()
        out[bin_col] = np.nan   # placeholder

    # Write per-symbol label Parquet
    out_path = label_path(store_root, "forward_returns", symbol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not out_path.exists():
        out.to_parquet(out_path, index=False)
        logger.debug("%s: wrote %d label rows → %s", symbol, len(out), out_path)

    return len(out)


def _max_drawdown(prices: np.ndarray) -> float:
    """Max drawdown of a price series."""
    if len(prices) < 2:
        return np.nan
    peak = np.maximum.accumulate(prices)
    dd   = (prices - peak) / peak
    return float(np.min(dd))


def build_label_panel(
    store_root: Path | str,
    symbols: list[str],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Load per-symbol label Parquets, concatenate, and add cross-sectional
    decile and binary columns.

    Returns a panel sorted by (date, symbol).
    Cross-sectional columns are computed across all symbols for each date.
    """
    store_root = Path(store_root)
    horizons   = horizons or DEFAULT_HORIZONS

    frames = []
    for symbol in symbols:
        p = label_path(store_root, "forward_returns", symbol)
        if p.exists():
            frames.append(pd.read_parquet(p))

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Fill cross-sectional decile and binary columns
    for h in horizons:
        ret_col = f"ret_fwd_{h}d"
        cs_col  = f"ret_fwd_{h}d_cs"
        bin_col = f"ret_fwd_{h}d_bin"

        if ret_col not in panel.columns:
            continue

        # Decile (0–9) within each date
        panel[cs_col] = (
            panel.groupby("date")[ret_col]
                 .transform(lambda x: pd.qcut(
                     x.rank(method="first"), 10,
                     labels=False, duplicates="drop"
                 ) if x.notna().sum() >= 10 else np.nan)
        )

        # Binary: 1 if return > cross-sectional median, else 0
        panel[bin_col] = (
            panel.groupby("date")[ret_col]
                 .transform(lambda x: (x > x.median()).astype(float))
        )

    return panel
