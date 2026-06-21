"""
store.schemas
=============
Canonical column definitions and schema enforcement for all silver tables.

Why this exists
---------------
AKShare returns differently-named columns depending on the endpoint and version.
Normalising to these canonical names at ingest time means every downstream
module (features, labels, DuckDB views) can rely on a stable contract.

Schema enforcement is intentionally lightweight: we validate required columns
are present and cast to the right dtype; we do not reject extra columns.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# OHLCV silver schema
# ---------------------------------------------------------------------------
# One row per (symbol, date).  Prices are forward-adjusted (qfq) by default
# unless the collector is configured to store raw prices separately.

OHLCV_REQUIRED: list[str] = ["symbol", "date", "open", "high", "low", "close", "volume"]

OHLCV_DTYPES: dict[str, Any] = {
    "symbol": "str",
    "date":   "object",   # dt.date — stored as string in Parquet, cast on read
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
    "volume": "float64",  # float to handle NaN; convert to int64 after dropna if needed
    "amount": "float64",  # total turnover in CNY — optional but kept when available
}


def enforce_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate and normalise a DataFrame to the OHLCV silver schema.

    - Checks required columns are present; raises ValueError if not.
    - Ensures 'symbol' column matches *symbol* argument.
    - Casts dtypes.
    - Sorts by date ascending.
    - Drops exact duplicate (symbol, date) rows (keeps first).
    - Returns a clean copy; never modifies in place.
    """
    df = df.copy()

    missing = [c for c in OHLCV_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"OHLCV schema violation for {symbol!r}: missing columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure symbol column is consistent
    df["symbol"] = symbol

    # Cast numeric columns
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Normalise date to dt.date objects
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Sort and deduplicate
    df = df.sort_values("date").drop_duplicates(subset=["symbol", "date"], keep="first")
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Adjustment factor schema
# ---------------------------------------------------------------------------

ADJ_FACTOR_REQUIRED: list[str] = ["symbol", "date", "adj_factor"]

def enforce_adj_factor(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate and normalise an adjustment-factor DataFrame."""
    df = df.copy()
    missing = [c for c in ADJ_FACTOR_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"adj_factor schema violation for {symbol!r}: missing {missing}")

    df["symbol"]     = symbol
    df["date"]       = pd.to_datetime(df["date"]).dt.date
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset=["symbol", "date"], keep="first")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PIT fundamentals schema  (T0.7 will use this)
# ---------------------------------------------------------------------------

FUNDAMENTALS_REQUIRED: list[str] = ["symbol", "announce_date", "period_end"]

def enforce_fundamentals(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate PIT fundamentals: must carry both announce_date and period_end.
    Using period_end as the join key is the single most common PIT mistake —
    this schema makes the correct key (announce_date) explicit.
    """
    df = df.copy()
    missing = [c for c in FUNDAMENTALS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Fundamentals schema violation for {symbol!r}: missing {missing}. "
            "Both announce_date AND period_end are required for PIT correctness. "
            "Do NOT join on period_end alone."
        )

    df["symbol"] = symbol
    announce_date = pd.to_datetime(df["announce_date"], errors="coerce")
    period_end    = pd.to_datetime(df["period_end"], errors="coerce")
    invalid = {
        "announce_date": int(announce_date.isna().sum()),
        "period_end":    int(period_end.isna().sum()),
    }
    invalid = {k: v for k, v in invalid.items() if v > 0}
    if invalid:
        raise ValueError(
            f"Fundamentals schema violation for {symbol!r}: invalid PIT dates {invalid}. "
            "Both announce_date and period_end must be real dates; no estimated or empty "
            "dates may be written."
        )

    df["announce_date"] = announce_date.dt.date
    df["period_end"]    = period_end.dt.date
    df = df.sort_values("announce_date").reset_index(drop=True)
    return df
