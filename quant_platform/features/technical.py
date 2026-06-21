"""
features.technical
==================
Technical feature builder (T1.2).

Reuses existing ``technical_indicators.py`` as-is for the core five indicators
(MA / MACD / KDJ / RSI / BOLL), then adds breadth indicators from
``pandas_ta_classic`` (ATR, ADX, OBV, CCI, ROC, WillR, Stoch).

Warm-up masking
---------------
Each indicator has a minimum look-back period before its values are reliable.
With ``min_periods=1`` (the current technical_indicators.py setting), early
rows carry unstable values (e.g. MA5 on row 1 = that row's close alone).
``build_technical_features`` masks these warm-up rows to NaN per the
``warmup`` field in each FeatureSpec.

The output column names match the FeatureSpec.name fields in TECHNICAL_SPECS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.features.registry import TECHNICAL_SPECS, FeatureSpec

logger = get_logger(__name__)

# Maximum warm-up rows needed across all technical specs
_MAX_WARMUP = max(s.warmup for s in TECHNICAL_SPECS)


def build_technical_features(
    df: pd.DataFrame,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compute technical features for one symbol's OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date, open, high, low, close, volume.
        Must be sorted by date ascending (enforce_ohlcv guarantees this).
    project_root : str | Path | None
        Path containing the original ``technical_indicators.py``.
        Defaults to the current working directory.

    Returns
    -------
    pd.DataFrame
        Original columns plus feature columns defined in TECHNICAL_SPECS.
        Warm-up rows are NaN for each feature.
    """
    df = df.copy().reset_index(drop=True)
    _ensure_ohlcv(df)

    # --- Core indicators from technical_indicators.py ---
    df = _apply_core_indicators(df, project_root)

    # --- Rename to canonical feature names ---
    rename = {
        "MA5": "ma_5", "MA10": "ma_10", "MA20": "ma_20", "MA60": "ma_60",
        "DIF": "macd_dif", "DEA": "macd_dea", "MACD": "macd_hist",
        "K": "kdj_k", "D": "kdj_d", "J": "kdj_j",
        "RSI6": "rsi_6", "RSI12": "rsi_12", "RSI24": "rsi_24",
        "BOLL_UPPER": "boll_upper", "BOLL_MID": "boll_mid", "BOLL_LOWER": "boll_lower",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # --- Additional breadth indicators from pandas_ta_classic ---
    df = _apply_pta_indicators(df)

    # --- Apply warm-up masks ---
    df = _mask_warmup(df)

    return df


# ---------------------------------------------------------------------------
# Core indicators from technical_indicators.py
# ---------------------------------------------------------------------------

def _apply_core_indicators(df: pd.DataFrame, project_root) -> pd.DataFrame:
    """Import and apply calculate_all_indicators from project technical_indicators.py."""
    _ensure_ti_importable(project_root)
    from technical_indicators import calculate_all_indicators  # type: ignore[import]
    return calculate_all_indicators(df)


def _ensure_ti_importable(project_root) -> None:
    """Add project_root to sys.path so technical_indicators.py is importable."""
    root = str(project_root or Path.cwd())
    if root not in sys.path:
        sys.path.insert(0, root)


# ---------------------------------------------------------------------------
# pandas_ta_classic breadth indicators
# ---------------------------------------------------------------------------

def _apply_pta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ATR, ADX, OBV, CCI, ROC, WillR, Stoch via pandas_ta_classic."""
    try:
        import pandas_ta_classic as pta
    except ImportError:
        logger.warning(
            "pandas_ta_classic not installed — skipping breadth indicators. "
            "Run: pip install pandas-ta-classic"
        )
        return df

    hi, lo, cl, vo = df["high"], df["low"], df["close"], df["volume"]

    # Use reindex to handle pta functions that return fewer rows (e.g. Stoch warm-up)
    n = len(df)
    idx_range = range(n)

    for col_name, fn_result in [
        ("atr_14",   pta.atr(hi, lo, cl, length=14)),
        ("obv",      pta.obv(cl, vo)),
        ("cci_14",   pta.cci(hi, lo, cl, length=14)),
        ("roc_10",   pta.roc(cl, length=10)),
        ("willr_14", pta.willr(hi, lo, cl, length=14)),
    ]:
        if fn_result is not None:
            s = fn_result if isinstance(fn_result, pd.Series) else pd.Series(fn_result)
            df[col_name] = s.reset_index(drop=True).reindex(idx_range).values

    # ADX returns a DataFrame with DMP, DMN, ADX columns
    adx_df = pta.adx(hi, lo, cl, length=14)
    if adx_df is not None and not adx_df.empty:
        adx_col = [c for c in adx_df.columns if c.upper().startswith("ADX")]
        if adx_col:
            df["adx_14"] = adx_df[adx_col[0]].reset_index(drop=True).reindex(idx_range).values

    # Stoch returns K and D — may be shorter than df due to warm-up
    stoch_df = pta.stoch(hi, lo, cl)
    if stoch_df is not None and not stoch_df.empty:
        cols = list(stoch_df.columns)
        if len(cols) >= 2:
            df["stoch_k"] = stoch_df[cols[0]].reset_index(drop=True).reindex(idx_range).values
            df["stoch_d"] = stoch_df[cols[1]].reset_index(drop=True).reindex(idx_range).values

    return df


# ---------------------------------------------------------------------------
# Warm-up masking
# ---------------------------------------------------------------------------

def _mask_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """Set warm-up rows to NaN for each feature column per TECHNICAL_SPECS."""
    spec_map = {s.name: s.warmup for s in TECHNICAL_SPECS}
    for col, warmup in spec_map.items():
        if col in df.columns and warmup > 0:
            df.loc[df.index[:warmup], col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _ensure_ohlcv(df: pd.DataFrame) -> None:
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"build_technical_features: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )
