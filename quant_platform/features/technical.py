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

Feature normalisation (cross-sectional comparability)
------------------------------------------------------
Several raw indicators carry absolute price units (MA60 in CNY, BOLL in CNY,
MACD in CNY, ATR in CNY, OBV in share count) that are not comparable across
stocks in a cross-sectional ranking — a 500-CNY stock's MA60 is always larger
than a 5-CNY stock's MA60, creating spurious cross-sectional IC.

After warm-up masking this builder normalises every absolute-price feature
into a dimensionless, cross-sectionally comparable ratio:

  ma_5/10/20/60  → close/MA - 1          (price distance from moving average)
  boll_upper      → (close-boll_lower)/(boll_upper-boll_lower)  (%B, 0–1)
  boll_mid        → close/boll_mid - 1   (same as ma_20 ratio)
  boll_lower      → dropped (redundant after %B)
  macd_dif/dea/hist → divided by close   (normalise by price level)
  atr_14          → atr_14 / close       (ATR as fraction of price)
  obv             → obv.diff() z-scored per symbol  (rate of change, not level)

Oscillators (RSI, KDJ, CCI, ROC, WillR, Stoch, ADX) are already dimensionless
(percentage or ratio) and require no normalisation.

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
        Defaults to ``/mnt/project`` (the mounted project directory).

    Returns
    -------
    pd.DataFrame
        Original columns plus normalised feature columns defined in TECHNICAL_SPECS.
        Warm-up rows are NaN for each feature.
    """
    df = df.copy().reset_index(drop=True)
    _ensure_ohlcv(df)

    # --- Core indicators from technical_indicators.py ---
    df = _apply_core_indicators(df, project_root)

    # --- Rename raw indicator columns to canonical names ---
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

    # --- T3.1: short-term reversal counter-signal (additive, does not
    # touch any existing momentum feature) ---
    df = _apply_reversal_factor(df)

    # --- Apply warm-up masks ---
    df = _mask_warmup(df)

    # --- Normalise absolute-price features for cross-sectional comparability ---
    df = _normalise_price_features(df)

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
    root = str(project_root or "/mnt/project")
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

    def _align_to_input(obj) -> pd.Series:
        # pandas_ta may return a shortened Series/DataFrame with the original
        # row index preserved. Reindex on that original index; resetting it
        # would move future indicator values onto earlier dates.
        s = obj if isinstance(obj, pd.Series) else pd.Series(obj)
        return s.reindex(idx_range)

    for col_name, fn_result in [
        ("atr_14",   pta.atr(hi, lo, cl, length=14)),
        ("obv",      pta.obv(cl, vo)),
        ("cci_14",   pta.cci(hi, lo, cl, length=14)),
        ("roc_10",   pta.roc(cl, length=10)),
        ("willr_14", pta.willr(hi, lo, cl, length=14)),
    ]:
        if fn_result is not None:
            df[col_name] = _align_to_input(fn_result).values

    # ADX returns a DataFrame with DMP, DMN, ADX columns
    adx_df = pta.adx(hi, lo, cl, length=14)
    if adx_df is not None and not adx_df.empty:
        adx_col = [c for c in adx_df.columns if c.upper().startswith("ADX")]
        if adx_col:
            df["adx_14"] = _align_to_input(adx_df[adx_col[0]]).values

    # Stoch returns K and D — may be shorter than df due to warm-up
    stoch_df = pta.stoch(hi, lo, cl)
    if stoch_df is not None and not stoch_df.empty:
        cols = list(stoch_df.columns)
        if len(cols) >= 2:
            df["stoch_k"] = _align_to_input(stoch_df[cols[0]]).values
            df["stoch_d"] = _align_to_input(stoch_df[cols[1]]).values

    return df


# ---------------------------------------------------------------------------
# T3.1: Short-term reversal factor (additive counter-signal to momentum)
# ---------------------------------------------------------------------------

def _apply_reversal_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``reversal_3d`` = -(close(T) / close(T-3) - 1).

    This is the negative of the trailing 3-day return: a stock that just
    ran up hard gets a low (negative) reversal_3d value; a stock that just
    dropped gets a high (positive) value. Already dimensionless (a return
    ratio) so no further cross-sectional normalisation is needed, unlike
    the absolute-price features handled in _normalise_price_features.

    Additive only — does not modify or remove any existing momentum
    feature (ma_*, macd_*, roc_10, etc.).
    """
    df["reversal_3d"] = -(df["close"].pct_change(3))
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
# Price-feature normalisation
# ---------------------------------------------------------------------------

def _normalise_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert absolute-price indicator columns into dimensionless ratios.

    This makes features cross-sectionally comparable across stocks with
    very different price levels (e.g. 5-CNY bank stocks vs 1500-CNY Moutai).

    Transformations applied in-place on the same column names so that
    downstream code (pipeline, registry) does not need to change:

      ma_5/10/20/60  → close/MA - 1          (positive = price above MA)
      boll_upper     → %B = (close-lower)/(upper-lower)   [0..1, clipped]
      boll_mid       → close/boll_mid - 1    (same as ma_20 ratio)
      boll_lower     → dropped (fully captured by %B)
      macd_dif/dea/hist → value / close      (dimensionless MACD)
      atr_14         → atr_14 / close        (ATR as pct of price)
      obv            → per-symbol rolling z-score of obv.diff()
                        (rate-of-change of cumulative volume, normalised)
    """
    cl = df["close"]
    eps = 1e-9   # avoid division by zero

    # Moving average ratios
    for col in ("ma_5", "ma_10", "ma_20", "ma_60"):
        if col in df.columns:
            df[col] = cl / (df[col] + eps) - 1.0

    # Bollinger %B  (replaces boll_upper; boll_mid becomes MA ratio; boll_lower dropped)
    if "boll_upper" in df.columns and "boll_lower" in df.columns:
        band_width = (df["boll_upper"] - df["boll_lower"]).replace(0, np.nan)
        pct_b = (cl - df["boll_lower"]) / band_width
        df["boll_upper"] = pct_b.clip(-1.0, 2.0)   # reuse column name → %B
        df = df.drop(columns=["boll_lower"])         # redundant after %B

    if "boll_mid" in df.columns:
        df["boll_mid"] = cl / (df["boll_mid"] + eps) - 1.0

    # MACD (all three components) / close
    for col in ("macd_dif", "macd_dea", "macd_hist"):
        if col in df.columns:
            df[col] = df[col] / (cl.abs() + eps)

    # ATR as fraction of close
    if "atr_14" in df.columns:
        df["atr_14"] = df["atr_14"] / (cl + eps)

    # OBV: convert level → rate-of-change, then rolling z-score
    if "obv" in df.columns:
        obv_diff = df["obv"].diff()
        roll_mean = obv_diff.rolling(20, min_periods=5).mean()
        roll_std  = obv_diff.rolling(20, min_periods=5).std(ddof=1).replace(0, np.nan)
        df["obv"] = (obv_diff - roll_mean) / roll_std

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
