"""
features.valuation
==================
Valuation and size feature builder (P4B-02).

Builds cross-sectional features from the silver valuation table:
PE_TTM, PB, market capitalisation, and turnover rate.

All features are cross-sectional percentile ranks or z-scores within
each date so they are comparable across stocks at different price levels.
A large-cap stock's PE is not inherently higher than a small-cap's —
the cross-sectional rank removes the price-level effect.

PIT safety
----------
All source fields (PE_TTM, PB, market cap, turnover) are derived from
the closing price of the same day T.  No announcement-date lag is needed.
Join on (symbol, date) directly.

Feature columns produced
------------------------
  cs_log_float_mcap      float in [0,1]  scaled log(float_mcap)
  cs_pe_ttm_rank         float in [0,1]  rank of PE_TTM; negatives → rank 0
  cs_pb_rank             float in [0,1]  rank of PB
  cs_turnover_rank       float in [0,1]  rank of turnover_pct
  cs_log_mcap_rank       float in [0,1]  rank of log(total_mcap)  [alias of size]
  pe_momentum_5d         float           5-day change in PE_TTM (expansion/compression)

Spec list: VALUATION_SPECS (for the feature registry)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.features.registry import FeatureSpec
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FeatureSpec declarations — registered into the feature registry
# ---------------------------------------------------------------------------

VALUATION_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "cs_log_float_mcap", "valuation",
        ("float_mcap_yi",), 0, "minmax(log(float_mcap_yi))", warmup=0,
    ),
    FeatureSpec(
        "cs_pe_ttm_rank", "valuation",
        ("pe_ttm",), 0, "pct_rank(pe_ttm; neg→0)", warmup=0,
    ),
    FeatureSpec(
        "cs_pb_rank", "valuation",
        ("pb",), 0, "pct_rank(pb)", warmup=0,
    ),
    FeatureSpec(
        "cs_turnover_rank", "valuation",
        ("turnover_pct",), 0, "pct_rank(turnover_pct)", warmup=0,
    ),
    FeatureSpec(
        "cs_log_mcap_rank", "valuation",
        ("total_mcap_yi",), 0, "pct_rank(log(total_mcap_yi))", warmup=0,
    ),
    FeatureSpec(
        "pe_momentum_5d", "valuation",
        ("pe_ttm",), 5, "pct_rank(PE_TTM.diff(5))", warmup=5,
    ),
]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_valuation_features(
    panel: pd.DataFrame,
    valuation_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cross-sectional valuation and size features to the universe panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Universe panel with columns [symbol, date, ...].  One row per
        (symbol, date).  Sorted by date.
    valuation_panel : pd.DataFrame
        Concatenated silver valuation data for all symbols.
        Columns: symbol, date, pe_ttm, pb, total_mcap_yi, float_mcap_yi,
                 turnover_pct.

    Returns
    -------
    pd.DataFrame
        Original panel plus valuation feature columns.
        Rows with no matching valuation data on a given date get NaN.
    """
    if valuation_panel.empty:
        logger.warning("build_valuation_features: empty valuation panel — returning panel unchanged")
        for spec in VALUATION_SPECS:
            panel[spec.name] = np.nan
        return panel

    val = valuation_panel.copy()
    val["date"] = pd.to_datetime(val["date"]).dt.date

    # Merge valuation onto the universe panel
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    val_cols = ["symbol", "date", "pe_ttm", "pb", "total_mcap_yi", "float_mcap_yi", "turnover_pct"]
    val_sub = val[[c for c in val_cols if c in val.columns]].copy()

    df = df.merge(val_sub, on=["symbol", "date"], how="left")

    # --- Log-transform market caps ---
    df["_log_float_mcap"] = np.log(df["float_mcap_yi"].clip(lower=1e-6))
    df["_log_total_mcap"] = np.log(df["total_mcap_yi"].clip(lower=1e-6))

    # --- PE_TTM winsorisation: negative PE → replaced with NaN before ranking ---
    # Stocks with negative earnings rank at the BOTTOM (most expensive / no earnings)
    df["_pe_rank_input"] = df["pe_ttm"].where(df["pe_ttm"] > 0, np.nan)

    # --- Cross-sectional percentile ranks (per date) ---
    def _pct_rank(series: pd.Series) -> pd.Series:
        """Percentile rank within group (0 to 1); NaN inputs get NaN output."""
        return series.rank(method="average", ascending=True, pct=True)

    def _safe_zscore(g: pd.Series) -> pd.Series:
        std = g.std(ddof=1)
        if std == 0 or pd.isna(std):
            return pd.Series(np.nan, index=g.index)
        return (g - g.mean()) / std

    rank_targets = [
        ("cs_log_float_mcap", "_log_float_mcap"),
        ("cs_pe_ttm_rank",    "_pe_rank_input"),
        ("cs_pb_rank",        "pb"),
        ("cs_turnover_rank",  "turnover_pct"),
        ("cs_log_mcap_rank",  "_log_total_mcap"),
    ]

    for out_col, src_col in rank_targets:
        if src_col not in df.columns:
            df[out_col] = np.nan
            continue
        df[out_col] = df.groupby("date")[src_col].transform(_pct_rank)

    log_min = df["_log_float_mcap"].min()
    log_max = df["_log_float_mcap"].max()
    if pd.isna(log_min) or pd.isna(log_max) or log_max == log_min:
        df["cs_log_float_mcap"] = np.nan
    else:
        df["cs_log_float_mcap"] = (df["_log_float_mcap"] - log_min) / (log_max - log_min)

    # --- PE_TTM momentum (5-day change) — per symbol, cross-sectionally ranked ---
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    if "pe_ttm" in df.columns:
        df["_pe_chg_5d"] = df.groupby("symbol")["pe_ttm"].transform(
            lambda x: x.diff(5)
        )
        df["pe_momentum_5d"] = df.groupby("date")["_pe_chg_5d"].transform(_pct_rank)
    else:
        df["pe_momentum_5d"] = np.nan

    # Drop working columns
    wk = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=wk, errors="ignore")
    df = df.drop(columns=["float_mcap_yi", "total_mcap_yi"], errors="ignore")

    logger.info(
        "build_valuation_features: added %d valuation columns to panel (%d rows)",
        len(VALUATION_SPECS), len(df),
    )
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_valuation_panel(
    store_root: Path | str,
    symbols: list[str],
) -> pd.DataFrame:
    """
    Load and concatenate silver valuation Parquets for all symbols into one panel.
    Returns an empty DataFrame if no files exist.
    """
    from quant_platform.store.lake import valuation_path

    store_root = Path(store_root)
    frames = []
    for symbol in symbols:
        p = valuation_path(store_root, symbol)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not load valuation for %s: %s", symbol, exc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
