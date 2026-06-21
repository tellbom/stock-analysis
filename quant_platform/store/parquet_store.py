"""
store.parquet_store
===================
Low-level Parquet read/write helpers for the silver layer.

Rules
-----
- Every write goes through ``write_ohlcv`` / ``write_adj_factor`` so that
  schema enforcement (``store.schemas``) is always applied.
- Reads return DataFrames with date columns as ``dt.date`` objects (not strings
  or Timestamps) — consistent with the rest of the platform.
- Writes are atomic: data is written to a temp file then renamed, so a crashed
  write never leaves a corrupt Parquet.
- Functions are stateless; they take explicit paths from ``store.lake``.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

import pandas as pd

from quant_platform.store.schemas import enforce_ohlcv, enforce_adj_factor
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def write_ohlcv(df: pd.DataFrame, path: Path, symbol: str) -> None:
    """
    Enforce schema, then write OHLCV DataFrame to *path* atomically.
    Creates parent directories if needed.
    """
    clean = enforce_ohlcv(df, symbol)
    _atomic_write(clean, path)
    logger.info("OHLCV written → %s (%d rows)", path, len(clean))


def read_ohlcv(path: Path) -> pd.DataFrame:
    """
    Read an OHLCV Parquet file.  Returns empty DataFrame with correct columns
    if the file does not exist (so callers can append without special-casing).
    Raises RuntimeError for files that exist but are unreadable.
    """
    if not path.exists():
        return _empty_ohlcv()
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to read OHLCV Parquet at {path}: {exc}") from exc


def read_ohlcv_range(
    path: Path,
    start: dt.date | None = None,
    end:   dt.date | None = None,
) -> pd.DataFrame:
    """Read OHLCV and filter to [start, end] inclusive (both optional)."""
    df = read_ohlcv(path)
    if df.empty:
        return df
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df.reset_index(drop=True)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])


# ---------------------------------------------------------------------------
# Adjustment factors
# ---------------------------------------------------------------------------

def write_adj_factor(df: pd.DataFrame, path: Path, symbol: str) -> None:
    """Enforce schema and write adjustment factor DataFrame atomically."""
    clean = enforce_adj_factor(df, symbol)
    _atomic_write(clean, path)
    logger.info("adj_factor written → %s (%d rows)", path, len(clean))


def read_adj_factor(path: Path) -> pd.DataFrame:
    """Read adjustment factors; returns empty frame if file absent."""
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "date", "adj_factor"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Generic read (for calendar, universe, catalog, etc.)
# ---------------------------------------------------------------------------

def read_parquet(path: Path) -> pd.DataFrame:
    """Read any Parquet file; raises FileNotFoundError if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet atomically; creates parent dirs."""
    _atomic_write(df, path)


# ---------------------------------------------------------------------------
# Internal: atomic write
# ---------------------------------------------------------------------------

def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """
    Write *df* to *path* atomically via a sibling temp file + rename.
    Ensures a crashed write never leaves a corrupt file at the target path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise date objects to ISO strings for clean Parquet storage
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            sample = out[col].dropna()
            if not sample.empty and isinstance(sample.iloc[0], dt.date):
                out[col] = out[col].apply(lambda v: str(v) if v is not None else None)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    try:
        os.close(fd)
        out.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)   # atomic on POSIX
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
