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
  ret_fwd_{h}d           : raw forward return (continuous, signed)
  ret_fwd_{h}d_cs        : cross-sectional quantile decile (0–9) of that return
                            on the same date (day-relative rank, not look-ahead)
  ret_fwd_{h}d_bin       : binary 1/0 = outperforms cross-sectional median return
  excess_vs_csi300_{h}d  : forward return minus CSI 300 index forward return
                            (market-neutral label); NaN when index OHLCV unavailable

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

P4A-04 change
-------------
DEFAULT_HORIZONS updated from [1, 5, 20] to [1, 5, 10, 20].
The 5-day horizon is now the **primary evaluation horizon** for the
platform.  At 5d a 12-month walk-forward window yields ~50 independent
forward periods vs. ~12 at 20d — a 4× improvement in statistical power.

P4A-05 addition
---------------
``build_label_panel`` now adds ``excess_vs_csi300_{h}d`` columns when a
CSI 300 index OHLCV is available in the lake
(``silver/index_ohlcv/000300.parquet``).  The excess return strips
market beta from the training target:

    excess_vs_csi300_{h}d(T) = ret_fwd_{h}d(stock, T)
                               − ret_fwd_{h}d(CSI300_index, T)

Both legs use the identical T+1…T+1+h window — no look-ahead.
The excess label is NaN for any date where the index return is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import label_path, label_dir, ohlcv_path, index_ohlcv_path
from quant_platform.store.parquet_store import read_ohlcv

logger = get_logger(__name__)

# Default holding horizons (in trading days).
# P4A-04: 10d added; 5d is the primary evaluation horizon.
DEFAULT_HORIZONS: list[int] = [1, 5, 10, 20]

# Primary label used throughout the platform (shortest horizon with reasonable
# signal decay and high independent-period count per year).
PRIMARY_LABEL_HORIZON: int = 5
PRIMARY_LABEL_COL: str = f"ret_fwd_{PRIMARY_LABEL_HORIZON}d"

# CSI 300 index code used for excess-return label
CSI300_SYMBOL: str = "000300"


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
        Holding periods in trading days.  Default [1, 5, 10, 20].
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
    add_excess_csi300: bool = True,
) -> pd.DataFrame:
    """
    Load per-symbol label Parquets, concatenate, and add cross-sectional
    decile, binary, and (optionally) excess-vs-CSI300 columns.

    Returns a panel sorted by (date, symbol).
    Cross-sectional columns are computed across all symbols for each date.

    Parameters
    ----------
    add_excess_csi300 : bool
        If True (default), attempt to join CSI 300 index forward returns
        and add ``excess_vs_csi300_{h}d`` columns.  Silently skips if the
        index OHLCV is not yet in the lake.
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

    # P4A-05: excess return vs CSI 300 index
    if add_excess_csi300:
        panel = _add_excess_vs_csi300(panel, store_root, horizons)

    return panel


# ---------------------------------------------------------------------------
# P4A-05: excess-vs-CSI300 label
# ---------------------------------------------------------------------------

def _add_excess_vs_csi300(
    panel: pd.DataFrame,
    store_root: Path,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Join CSI 300 index forward returns and subtract from stock forward returns.

    The index OHLCV is read from ``silver/index_ohlcv/000300.parquet``
    (written by ``ingest.index_collector``).  If the file does not exist,
    this function returns the panel unchanged and logs a warning.

    PIT safety: the index forward return uses the identical T+1…T+1+h
    window as the stock label — no look-ahead.

    Column added per horizon h:
        excess_vs_csi300_{h}d = ret_fwd_{h}d(stock) − ret_fwd_{h}d(index)

    The column is NaN for any date where the index return is NaN.
    Cross-sectional mean of excess_vs_csi300 is ≈ 0 by construction.
    """
    idx_path = index_ohlcv_path(store_root, CSI300_SYMBOL)
    if not idx_path.exists():
        logger.warning(
            "excess_vs_csi300: index OHLCV not found at %s — skipping. "
            "Run ingest.index_collector first.",
            idx_path,
        )
        return panel

    try:
        idx_df = pd.read_parquet(idx_path)
    except Exception as exc:
        logger.warning("excess_vs_csi300: could not read index OHLCV: %s", exc)
        return panel

    if "close" not in idx_df.columns or "date" not in idx_df.columns:
        logger.warning("excess_vs_csi300: index OHLCV missing 'date' or 'close' — skipping")
        return panel

    idx_df["date"] = pd.to_datetime(idx_df["date"]).dt.date
    idx_df = idx_df.sort_values("date").reset_index(drop=True)
    idx_close = idx_df["close"].values
    idx_dates = idx_df["date"].values
    n_idx     = len(idx_df)

    # Build date → index forward return mapping for each horizon
    for h in horizons:
        ret_col    = f"ret_fwd_{h}d"
        excess_col = f"excess_vs_csi300_{h}d"

        if ret_col not in panel.columns:
            continue

        # Compute index forward return for each date in the index series
        idx_fwd = {}
        for i in range(n_idx):
            t1   = i + 1
            t1ph = i + 1 + h
            if t1 >= n_idx or t1ph >= n_idx:
                continue
            c_t1   = idx_close[t1]
            c_t1ph = idx_close[t1ph]
            if c_t1 <= 0:
                continue
            idx_fwd[idx_dates[i]] = c_t1ph / c_t1 - 1.0

        if not idx_fwd:
            panel[excess_col] = np.nan
            continue

        idx_fwd_series = pd.Series(idx_fwd)

        # Subtract index return from each stock's return on the same date
        panel[excess_col] = (
            panel[ret_col]
            - panel["date"].map(idx_fwd_series)
        )

        n_valid = panel[excess_col].notna().sum()
        logger.info(
            "excess_vs_csi300_%dd: added for %d rows (%.1f%% coverage)",
            h, n_valid, n_valid / len(panel) * 100,
        )

    return panel
