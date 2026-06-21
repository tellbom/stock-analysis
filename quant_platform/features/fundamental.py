"""
features.fundamental
====================
PIT fundamental feature builder (T1.4).

Joins PIT fundamentals to the OHLCV panel using ``announce_date`` as the
join key — never ``period_end``.  This is the correct PIT join: a feature
at date T uses only fundamental data announced on or before T.

The join is an as-of (LOCF) join: for each (symbol, date) row in the panel,
we find the most-recent fundamental row with announce_date <= date.

Output columns added to the panel:
  fund_revenue        : most recently announced revenue
  fund_net_profit     : most recently announced net profit
  fund_eps            : earnings per share
  fund_roe            : return on equity
  fund_period_end     : the period this announcement refers to
  fund_period_type    : Q1 / H1 / Q3 / annual
  fund_lag_days       : calendar days between announce_date and feature_date
                        (useful for staleness filtering in models)

If no fundamentals Parquet exists for a symbol, all fund_* columns are NaN.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_platform.ingest.fundamentals_collector import query_fundamentals_as_of
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

# Fundamental columns to extract and their renames
_FUND_COLS = {
    "revenue":     "fund_revenue",
    "net_profit":  "fund_net_profit",
    "eps":         "fund_eps",
    "roe":         "fund_roe",
    "period_end":  "fund_period_end",
    "period_type": "fund_period_type",
}


def build_fundamental_features(
    panel: pd.DataFrame,
    store_root: Path | str,
) -> pd.DataFrame:
    """
    Add PIT fundamental features to an OHLCV panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have columns: symbol, date (as dt.date).
    store_root : Path | str
        Root of the Parquet lake (fundamentals live at silver/fundamentals/).

    Returns
    -------
    pd.DataFrame
        Panel with fund_* columns added.  Symbols with no fundamentals
        data get NaN in all fund_* columns.
    """
    store_root = Path(store_root)
    panel = panel.copy()

    # Initialise output columns as NaN
    for col in list(_FUND_COLS.values()) + ["fund_lag_days", "fund_announce_date"]:
        panel[col] = float("nan") if col not in ("fund_period_end", "fund_period_type",
                                                   "fund_announce_date") else None

    symbols = panel["symbol"].unique().tolist()

    for symbol in symbols:
        sym_mask = panel["symbol"] == symbol
        sym_rows = panel[sym_mask].copy()

        # Load all fundamentals for this symbol once
        from quant_platform.store.lake import fundamentals_path
        path = fundamentals_path(store_root, symbol)
        if not path.exists():
            logger.debug("%s: no fundamentals file — fund_* will be NaN", symbol)
            continue

        try:
            fund_df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("%s: cannot read fundamentals: %s", symbol, exc)
            continue

        if fund_df.empty:
            continue

        metric_cols = [c for c in ("revenue", "net_profit", "eps", "roe") if c in fund_df.columns]
        if metric_cols:
            fund_df = fund_df[fund_df[metric_cols].notna().any(axis=1)].copy()
        else:
            fund_df = fund_df.iloc[0:0].copy()
        if fund_df.empty:
            continue

        fund_df["announce_date"] = pd.to_datetime(fund_df["announce_date"]).dt.date
        fund_df = fund_df.sort_values("announce_date").reset_index(drop=True)

        # As-of join per date: find most recent fund row with announce_date <= date
        dates = pd.to_datetime(sym_rows["date"]).dt.date
        idx_in_panel = sym_rows.index

        for panel_idx, feature_date in zip(idx_in_panel, dates):
            eligible = fund_df[fund_df["announce_date"] <= feature_date]
            if eligible.empty:
                continue
            latest = eligible.iloc[-1]

            for src_col, dst_col in _FUND_COLS.items():
                if src_col in latest.index:
                    panel.at[panel_idx, dst_col] = latest[src_col]

            panel.at[panel_idx, "fund_announce_date"] = latest["announce_date"]
            panel.at[panel_idx, "fund_lag_days"] = (
                feature_date - latest["announce_date"]
            ).days

    return panel
