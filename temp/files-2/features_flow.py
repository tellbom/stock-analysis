"""
features.flow
=============
Capital flow feature builder (P4B-07).

Builds cross-sectional features from the silver fund flow table.
All flow values are normalised by float market cap so they are
comparable across stocks of different sizes.

PIT safety
----------
Flow data released after market close for date T.
Features at date T use flow data through date T only (same day close).
5-day cumulative features use T−4 through T.

Key A-share pattern
--------------------
  main_net > 0  = institutional / large-order buying (bullish signal)
  small_net > 0 = retail buying  (contrarian — often a sell signal)

Feature columns produced
------------------------
  cs_main_flow_rank_1d    float [0,1]  rank of main_net(T) / float_mcap_yi
  cs_main_flow_rank_5d    float [0,1]  rank of cumulative main_net(T-4:T) / float_mcap_yi
  cs_small_flow_rank_1d   float [0,1]  rank of small_net(T) / float_mcap_yi (contrarian)
  cs_super_flow_rank_1d   float [0,1]  rank of super_net(T) / float_mcap_yi (institutional)
  cs_flow_reversal_5d     float        cs_main_flow_rank_1d − cs_main_flow_rank_5d

Spec list: FLOW_SPECS (for the feature registry)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.features.registry import FeatureSpec
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FeatureSpec declarations
# ---------------------------------------------------------------------------

FLOW_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "cs_main_flow_rank_1d", "flow",
        ("main_net", "float_mcap_yi"), 1, "pct_rank(main_net/float_mcap)", warmup=1,
    ),
    FeatureSpec(
        "cs_main_flow_rank_5d", "flow",
        ("main_net", "float_mcap_yi"), 5, "pct_rank(sum(main_net,5)/float_mcap)", warmup=5,
    ),
    FeatureSpec(
        "cs_small_flow_rank_1d", "flow",
        ("small_net", "float_mcap_yi"), 1, "pct_rank(small_net/float_mcap)", warmup=1,
    ),
    FeatureSpec(
        "cs_super_flow_rank_1d", "flow",
        ("super_net", "float_mcap_yi"), 1, "pct_rank(super_net/float_mcap)", warmup=1,
    ),
    FeatureSpec(
        "cs_flow_reversal_5d", "flow",
        ("main_net", "float_mcap_yi"), 5,
        "cs_main_flow_rank_1d - cs_main_flow_rank_5d", warmup=5,
    ),
]

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_flow_features(
    panel: pd.DataFrame,
    flow_panel: pd.DataFrame,
    valuation_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add cross-sectional capital flow features to the universe panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Universe panel with columns [symbol, date, ...].
    flow_panel : pd.DataFrame
        Concatenated silver fund flow data (all symbols).
        Columns: symbol, date, main_net, small_net, mid_net, large_net, super_net.
    valuation_panel : pd.DataFrame | None
        Used to get float_mcap_yi for normalisation.  If None, flow values
        are normalised by their own cross-sectional mean instead (weaker but
        functional fallback).

    Returns
    -------
    pd.DataFrame
        Panel with flow feature columns added.
    """
    if flow_panel.empty:
        logger.warning("build_flow_features: empty flow panel — all flow features will be NaN")
        for spec in FLOW_SPECS:
            panel[spec.name] = np.nan
        return panel

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # --- Prepare flow data ---
    flow = flow_panel.copy()
    flow["date"] = pd.to_datetime(flow["date"]).dt.date
    flow_cols = ["symbol", "date", "main_net", "small_net", "mid_net", "large_net", "super_net"]
    flow = flow[[c for c in flow_cols if c in flow.columns]]

    # Merge float_mcap for normalisation
    if valuation_panel is not None and not valuation_panel.empty:
        val = valuation_panel.copy()
        val["date"] = pd.to_datetime(val["date"]).dt.date
        mcap = val[["symbol", "date", "float_mcap_yi"]].copy()
        flow = flow.merge(mcap, on=["symbol", "date"], how="left")
    else:
        flow["float_mcap_yi"] = np.nan

    # Fallback: if float_mcap missing, use cross-sectional mean as denominator
    # (this is less informative but avoids NaN cascade)
    flow["_mcap_denom"] = flow["float_mcap_yi"].where(
        flow["float_mcap_yi"] > 0,
        flow.groupby("date")["float_mcap_yi"].transform(
            lambda x: x.clip(lower=1e-6).mean()
        ).clip(lower=1e-6),
    )

    # --- Normalised flow: value / float_mcap_yi ---
    for col in ("main_net", "small_net", "super_net"):
        if col in flow.columns:
            flow[f"_{col}_norm"] = flow[col] / flow["_mcap_denom"]
        else:
            flow[f"_{col}_norm"] = np.nan

    # --- 5-day cumulative main flow (per symbol, then normalise) ---
    flow = flow.sort_values(["symbol", "date"]).reset_index(drop=True)
    flow["_main_cum5"] = (
        flow.groupby("symbol")["main_net"]
            .transform(lambda x: x.rolling(5, min_periods=1).sum())
        / flow["_mcap_denom"]
    )

    # --- Merge flow into panel (left join on symbol, date) ---
    flow_features = flow[[
        "symbol", "date",
        "_main_net_norm", "_small_net_norm", "_super_net_norm", "_main_cum5",
    ]].copy()

    df = df.merge(flow_features, on=["symbol", "date"], how="left")

    # --- Cross-sectional percentile ranks ---
    def _pct_rank(s: pd.Series) -> pd.Series:
        return s.rank(method="average", ascending=True, pct=True)

    rank_pairs = [
        ("cs_main_flow_rank_1d",  "_main_net_norm"),
        ("cs_main_flow_rank_5d",  "_main_cum5"),
        ("cs_small_flow_rank_1d", "_small_net_norm"),
        ("cs_super_flow_rank_1d", "_super_net_norm"),
    ]

    for out_col, src_col in rank_pairs:
        if src_col in df.columns:
            df[out_col] = df.groupby("date")[src_col].transform(_pct_rank)
        else:
            df[out_col] = np.nan

    # --- Flow reversal: 1d rank − 5d rank ---
    if "cs_main_flow_rank_1d" in df.columns and "cs_main_flow_rank_5d" in df.columns:
        df["cs_flow_reversal_5d"] = df["cs_main_flow_rank_1d"] - df["cs_main_flow_rank_5d"]
    else:
        df["cs_flow_reversal_5d"] = np.nan

    # Drop working columns
    wk = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=wk, errors="ignore")

    logger.info(
        "build_flow_features: added %d flow columns to panel (%d rows)",
        len(FLOW_SPECS), len(df),
    )
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_flow_panel(
    store_root: Path | str,
    symbols: list[str],
) -> pd.DataFrame:
    """Load and concatenate silver fund flow Parquets for all symbols."""
    from quant_platform.store.lake import fund_flow_path

    store_root = Path(store_root)
    frames = []
    for symbol in symbols:
        p = fund_flow_path(store_root, symbol)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not load flow for %s: %s", symbol, exc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
