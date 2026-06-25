"""
features.event
==============
Event-driven feature builder (P4C-03).

Currently implements lockup expiry features from the silver lockup table.

PIT correctness (critical)
--------------------------
Upcoming lockup expiry dates are announced at IPO or at the time of the
private placement, so they are *public knowledge at all prior dates*.
Feature at date T:
  - ``days_to_next_unlock`` uses unlock_date > T only (strictly future)
  - ``unlock_size_ratio`` uses the NEXT unlock after T

Leakage trap to avoid: do NOT use unlock_date == T (the unlock itself).
The unlock event at date T is a contemporaneous event — we conservatively
treat it as unavailable until T+1.

Feature columns produced
------------------------
  days_to_next_unlock  int    Calendar days until the next lock-up expiry
                               after date T.  Default 999 when no unlock
                               event is scheduled within the next 180 days.
  unlock_size_ratio    float  shares_million(next_unlock) / float_mcap_yi
                               (supply pressure proxy).  0 when no unlock
                               within 30 days.  NaN when float_mcap unavailable.

Signal hypothesis
-----------------
Large imminent unlocks → supply pressure → negative short-horizon return.
Expected IC direction: negative vs ret_fwd_1d to ret_fwd_5d.

Spec list: LOCKUP_SPECS (for the feature registry)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.features.registry import FeatureSpec
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

# Default horizon for unlock_size_ratio (only non-zero within this many days)
_RATIO_WINDOW_DAYS = 30

# Default value when no unlock is scheduled within MAX_FORWARD_DAYS
_MAX_FORWARD_DAYS = 180
_DEFAULT_DAYS     = 999


# ---------------------------------------------------------------------------
# FeatureSpec declarations
# ---------------------------------------------------------------------------

LOCKUP_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "days_to_next_unlock", "event",
        ("unlock_date",), 0,
        f"calendar_days_to_next_unlock (999 if none in {_MAX_FORWARD_DAYS}d)",
        warmup=0,
    ),
    FeatureSpec(
        "unlock_size_ratio", "event",
        ("unlock_date", "shares_million", "float_mcap_yi"), 0,
        f"shares_unlocking/float_mcap_yi (0 if no unlock in {_RATIO_WINDOW_DAYS}d)",
        warmup=0,
    ),
]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_lockup_features(
    panel: pd.DataFrame,
    lockup_panel: pd.DataFrame,
    valuation_panel: pd.DataFrame | None = None,
    max_forward_days: int = _MAX_FORWARD_DAYS,
    ratio_window_days: int = _RATIO_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Add lock-up expiry features to the universe panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Universe panel with [symbol, date, ...].
    lockup_panel : pd.DataFrame
        Concatenated silver lockup data for all symbols.
        Columns: symbol, unlock_date, lock_type, shares_million, ratio_pct.
    valuation_panel : pd.DataFrame | None
        For float_mcap_yi normalisation.  If None, unlock_size_ratio
        uses shares_million directly (unnormalised, less informative).
    max_forward_days : int
        ``days_to_next_unlock`` is capped at this value when no event is
        found within the window.
    ratio_window_days : int
        ``unlock_size_ratio`` is only non-zero when the next unlock is
        within this many days.

    Returns
    -------
    pd.DataFrame
        Panel with lockup feature columns added.
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    if lockup_panel.empty:
        logger.info("build_lockup_features: empty lockup panel — features will default to 999/0")
        df["days_to_next_unlock"] = _DEFAULT_DAYS
        df["unlock_size_ratio"]   = 0.0
        return df

    lk = lockup_panel.copy()
    lk["unlock_date"] = pd.to_datetime(lk["unlock_date"]).dt.date

    # Merge valuation for float_mcap
    if valuation_panel is not None and not valuation_panel.empty:
        val = valuation_panel.copy()
        val["date"] = pd.to_datetime(val["date"]).dt.date
        mcap = val[["symbol", "date", "float_mcap_yi"]].copy()
        df = df.merge(mcap, on=["symbol", "date"], how="left")
    else:
        df["float_mcap_yi"] = np.nan

    # For each (symbol, date) row, find the next upcoming unlock after T
    # Vectorised approach: for each symbol, precompute sorted unlock dates
    # then join by nearest future date.

    days_vals  = np.full(len(df), _DEFAULT_DAYS, dtype=float)
    ratio_vals = np.zeros(len(df), dtype=float)

    # Group lockups by symbol for fast lookup
    lk_by_sym: dict[str, pd.DataFrame] = {
        sym: grp.sort_values("unlock_date")
        for sym, grp in lk.groupby("symbol")
    }

    df_np_date = df["date"].values
    df_np_sym  = df["symbol"].values
    df_np_mcap = df["float_mcap_yi"].values if "float_mcap_yi" in df.columns else np.full(len(df), np.nan)

    for idx in range(len(df)):
        sym  = df_np_sym[idx]
        date = df_np_date[idx]

        if sym not in lk_by_sym:
            continue

        sym_lk = lk_by_sym[sym]
        # Strictly future: unlock_date > date (not >= date)
        future = sym_lk[sym_lk["unlock_date"] > date]
        if future.empty:
            continue

        next_event = future.iloc[0]
        days = (next_event["unlock_date"] - date).days

        if days <= max_forward_days:
            days_vals[idx] = float(days)

        if days <= ratio_window_days:
            shares = float(next_event.get("shares_million", 0) or 0)
            mcap   = float(df_np_mcap[idx])
            if not np.isnan(mcap) and mcap > 0:
                # Convert shares_million to 亿股 for consistency: 1M shares = 0.01亿股
                # float_mcap_yi is in 亿元; assume price ~ 10-50 CNY/share
                # Better: use ratio_pct directly when available
                ratio_pct = float(next_event.get("ratio_pct", 0) or 0)
                ratio_vals[idx] = ratio_pct / 100.0   # as a fraction
            elif shares > 0:
                ratio_vals[idx] = shares   # fallback: absolute millions of shares

    df["days_to_next_unlock"] = days_vals
    df["unlock_size_ratio"]   = ratio_vals

    # Drop working column
    if "float_mcap_yi" in df.columns and "float_mcap_yi" not in panel.columns:
        df = df.drop(columns=["float_mcap_yi"])

    n_upcoming = (df["days_to_next_unlock"] < _DEFAULT_DAYS).sum()
    logger.info(
        "build_lockup_features: %d rows with upcoming unlock (within %dd)",
        n_upcoming, max_forward_days,
    )
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_lockup_panel(
    store_root: Path | str,
    symbols: list[str],
) -> pd.DataFrame:
    """Load and concatenate silver lockup Parquets for all symbols."""
    from quant_platform.store.lake import lockup_path

    store_root = Path(store_root)
    frames = []
    for symbol in symbols:
        p = lockup_path(store_root, symbol)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df["unlock_date"] = pd.to_datetime(df["unlock_date"]).dt.date
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not load lockup for %s: %s", symbol, exc)

    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
          .sort_values(["symbol", "unlock_date"])
          .reset_index(drop=True)
    )
