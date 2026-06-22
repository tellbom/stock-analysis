"""
features.cross_sectional
========================
Cross-sectional feature builder (T1.3).

Computes per-date rankings and z-scores across the universe using pandas
groupby.  Cross-sectional features are computed *within a single date*
so they carry no look-back leakage across time.

Critical: the universe for each date comes from the point-in-time membership
table (T0.2).  Using today's members for a 2021 date would secretly include
stocks that only joined the index later — a form of survivorship bias.
This builder accepts the full feature panel (all symbols × all dates) and
applies window functions over it, so the universe is automatically whatever
symbols are present for each date in that panel.

Output columns (matching CROSS_SECTIONAL_SPECS):
  cs_rank_volume   : rank of volume among universe on that date (0–1)
  cs_rank_rsi_6    : rank of rsi_6
  cs_rank_roc_10   : rank of roc_10
  cs_rank_ma_5     : rank of normalised ma_5 ratio (close/MA5-1)
  cs_zscore_volume : z-score of volume
  cs_zscore_rsi_6  : z-score of rsi_6

NOTE: cs_rank_close and cs_zscore_close are intentionally excluded.
Raw close price in CNY is not cross-sectionally meaningful — a 500-CNY
stock always ranks above a 5-CNY stock regardless of signal content.
Use normalised features (ma_5 ratio, roc_10, rsi_6) for cross-sectional
ranking instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


def build_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional rank / z-score features to a universe panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns: symbol, date, volume.
        May also contain rsi_6, roc_10, ma_5 (added by technical builder).
        One row per (symbol, date).

    Returns
    -------
    pd.DataFrame
        Original panel plus cs_rank_* and cs_zscore_* columns.
        Dates where fewer than 2 symbols are present get NaN cross-sectional
        features (z-score undefined for N<2; rank undefined for N<1).
    """
    df = panel.copy()

    rank_targets = [
        ("cs_rank_volume",  "volume"),
        ("cs_rank_rsi_6",   "rsi_6"),
        ("cs_rank_roc_10",  "roc_10"),
        ("cs_rank_ma_5",    "ma_5"),    # normalised ratio, safe for cross-section
    ]
    zscore_targets = [
        ("cs_zscore_volume", "volume"),
        ("cs_zscore_rsi_6",  "rsi_6"),
    ]

    # Per-date rank (0 to 1 percentile rank)
    for out_col, src_col in rank_targets:
        if src_col not in df.columns:
            logger.debug("Skipping cs_rank for '%s' — column absent", src_col)
            df[out_col] = float("nan")
            continue
        df[out_col] = (
            df.groupby("date")[src_col]
              .rank(method="average", ascending=True, pct=True)
        )

    # Per-date z-score
    def _zscore_group(g: pd.Series) -> pd.Series:
        std = g.std(ddof=1)
        if std == 0 or pd.isna(std):
            return pd.Series(float("nan"), index=g.index)
        return (g - g.mean()) / std

    for out_col, src_col in zscore_targets:
        if src_col not in df.columns:
            df[out_col] = float("nan")
            continue
        df[out_col] = df.groupby("date")[src_col].transform(_zscore_group)

    return df
