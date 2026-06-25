"""
features.industry
=================
Industry-relative feature builder (P4B-04) and excess-vs-industry label
helper (P4B-05).

Within-industry features capture stock performance relative to its sector
peers, stripping sector beta.  In A-shares where sector rotation is
violent, these features tend to generalise better than market-wide ranks.

PIT industry join
-----------------
Each stock's industry is looked up as-of the feature date using the SCD
table (``silver/industry_map.parquet``).  This ensures that a stock that
changed sectors in 2023 uses its pre-change sector for 2022 dates.

Feature columns produced
------------------------
  ind_rank_rsi_6        float [0,1]  percentile rank of rsi_6 within industry
  ind_rank_turnover     float [0,1]  rank of turnover_pct within industry
  ind_rank_main_flow    float [0,1]  rank of 1-day main-force net inflow / float_mcap
                                     (populated only after P4B-07 flow features exist)
  sector_momentum_10d   float [0,1]  pct rank of 10-day industry-average past return

Excess-vs-industry label helper
---------------------------------
``build_excess_vs_industry_labels`` computes:

    excess_vs_industry_{h}d = ret_fwd_{h}d(stock)
                             - mean(ret_fwd_{h}d, industry peers on date T)

This strips sector beta from the label.  Must be called AFTER the base
forward-return labels are built and AFTER the industry map is available.
The industry average is computed only from stocks present in the panel on
that date (PIT-correct).

Spec list: INDUSTRY_SPECS (for the feature registry)
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

INDUSTRY_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "ind_rank_rsi_6", "industry",
        ("rsi_6", "industry_code"), 0, "pct_rank(rsi_6, within_industry)", warmup=0,
    ),
    FeatureSpec(
        "ind_rank_turnover", "industry",
        ("turnover_pct", "industry_code"), 0, "pct_rank(turnover_pct, within_industry)", warmup=0,
    ),
    FeatureSpec(
        "ind_rank_main_flow", "industry",
        ("cs_main_flow_rank_1d", "industry_code"), 0,
        "pct_rank(main_flow_1d/float_mcap, within_industry)", warmup=0,
    ),
    FeatureSpec(
        "sector_momentum_10d", "industry",
        ("close", "industry_code"), 10, "pct_rank(ind_avg_ret_10d)", warmup=10,
    ),
]


# ---------------------------------------------------------------------------
# Industry join helper
# ---------------------------------------------------------------------------

def _join_industry(
    panel: pd.DataFrame,
    industry_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add ``industry_code`` and ``industry_name`` to the panel using a PIT join.

    For each (symbol, date) row, the industry is the most recent SCD record
    with effective_date <= date and (out_date IS NULL OR out_date > date).

    If no industry record exists for a stock on a given date, ``industry_code``
    is set to "_UNKNOWN".
    """
    if industry_map.empty:
        panel["industry_code"] = "_UNKNOWN"
        panel["industry_name"] = ""
        return panel

    imap = industry_map.copy()
    imap["effective_date"] = pd.to_datetime(imap["effective_date"]).dt.date
    if "out_date" in imap.columns:
        imap["out_date"] = pd.to_datetime(imap["out_date"], errors="coerce").dt.date

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Build a flat (symbol, date) → industry lookup for the dates in the panel
    # Strategy: for each unique (symbol, date) pair, find the active SCD row.
    # This is O(panel_dates × scd_rows_per_symbol) — acceptable for CSI 300.
    unique_sd = df[["symbol", "date"]].drop_duplicates()

    def _lookup_industry(row):
        sym, dt_ = row["symbol"], row["date"]
        candidates = imap[
            (imap["symbol"] == sym)
            & (imap["effective_date"] <= dt_)
            & (imap["out_date"].isna() | (imap["out_date"] > dt_))
        ]
        if candidates.empty:
            return pd.Series({"industry_code": "_UNKNOWN", "industry_name": ""})
        last = candidates.iloc[-1]
        return pd.Series({
            "industry_code": str(last.get("industry_code", "_UNKNOWN")),
            "industry_name":  str(last.get("industry_name", "")),
        })

    lookup = unique_sd.apply(_lookup_industry, axis=1)
    unique_sd = pd.concat([unique_sd.reset_index(drop=True), lookup], axis=1)

    df = df.merge(unique_sd, on=["symbol", "date"], how="left")
    df["industry_code"] = df["industry_code"].fillna("_UNKNOWN")
    df["industry_name"]  = df["industry_name"].fillna("")
    return df


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_industry_features(
    panel: pd.DataFrame,
    industry_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add within-industry rank features and sector momentum to the panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Universe panel with columns [symbol, date, ...].
        Should already contain: rsi_6, turnover_pct (from valuation features),
        and optionally cs_main_flow_rank_1d (from flow features).
    industry_map : pd.DataFrame
        SCD table loaded from ``silver/industry_map.parquet``.

    Returns
    -------
    pd.DataFrame
        Panel with industry feature columns added.
    """
    if industry_map.empty:
        logger.warning(
            "build_industry_features: no industry map — all industry features will be NaN"
        )
        for spec in INDUSTRY_SPECS:
            panel[spec.name] = np.nan
        return panel

    df = _join_industry(panel, industry_map)

    def _pct_rank(s: pd.Series) -> pd.Series:
        return s.rank(method="average", ascending=True, pct=True)

    # --- Within-industry percentile ranks ---
    within_ind = ["date", "industry_code"]

    # ind_rank_rsi_6
    if "rsi_6" in df.columns:
        df["ind_rank_rsi_6"] = df.groupby(within_ind)["rsi_6"].transform(_pct_rank)
    else:
        df["ind_rank_rsi_6"] = np.nan

    # ind_rank_turnover (from valuation collector)
    if "turnover_pct" in df.columns:
        df["ind_rank_turnover"] = df.groupby(within_ind)["turnover_pct"].transform(_pct_rank)
    else:
        df["ind_rank_turnover"] = np.nan

    # ind_rank_main_flow (populated if flow features already added)
    if "cs_main_flow_rank_1d" in df.columns:
        df["ind_rank_main_flow"] = df.groupby(within_ind)["cs_main_flow_rank_1d"].transform(_pct_rank)
    else:
        df["ind_rank_main_flow"] = np.nan

    # --- Sector momentum (10-day industry-average past return) ---
    if "close" in df.columns:
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        # 10-day past return per symbol (using T-10 to T close)
        df["_ret_10d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(10)
        )
        # Industry average 10-day return per date
        df["_ind_avg_ret_10d"] = df.groupby(within_ind)["_ret_10d"].transform("mean")
        # Cross-sectional rank of the industry average
        df["sector_momentum_10d"] = df.groupby("date")["_ind_avg_ret_10d"].transform(_pct_rank)
        df = df.drop(columns=["_ret_10d", "_ind_avg_ret_10d"], errors="ignore")
    else:
        df["sector_momentum_10d"] = np.nan

    logger.info(
        "build_industry_features: added %d industry columns to panel (%d rows)",
        len(INDUSTRY_SPECS), len(df),
    )
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# P4B-05: excess-vs-industry label
# ---------------------------------------------------------------------------

def build_excess_vs_industry_labels(
    label_panel: pd.DataFrame,
    industry_map: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add ``excess_vs_industry_{h}d`` columns to the label panel.

    Computation
    -----------
        excess_vs_industry_{h}d(T) = ret_fwd_{h}d(stock, T)
            - mean(ret_fwd_{h}d, universe peers in same industry at T)

    The industry average uses only stocks present in the panel on date T
    with the same industry code (PIT-correct via SCD join).
    Stocks with ``_UNKNOWN`` industry code get NaN excess labels.

    This strips sector beta from the training target.  Together with
    ``excess_vs_csi300_{h}d`` (P4A-05) it forms the recommended label
    hierarchy: raw → excess_csi300 → excess_industry.

    Parameters
    ----------
    label_panel : pd.DataFrame
        Panel from ``labels.builder.build_label_panel()``.  Must contain
        ``ret_fwd_{h}d`` columns for each requested horizon.
    industry_map : pd.DataFrame
        SCD table.

    Returns
    -------
    pd.DataFrame
        Label panel with ``excess_vs_industry_{h}d`` columns added.
    """
    from quant_platform.labels.builder import DEFAULT_HORIZONS
    horizons = horizons or DEFAULT_HORIZONS

    if industry_map.empty:
        logger.warning(
            "build_excess_vs_industry_labels: no industry map — skipping excess labels"
        )
        for h in horizons:
            label_panel[f"excess_vs_industry_{h}d"] = np.nan
        return label_panel

    df = _join_industry(label_panel, industry_map)

    for h in horizons:
        ret_col    = f"ret_fwd_{h}d"
        excess_col = f"excess_vs_industry_{h}d"

        if ret_col not in df.columns:
            df[excess_col] = np.nan
            continue

        # Industry average forward return on each date
        df["_ind_avg"] = df.groupby(["date", "industry_code"])[ret_col].transform("mean")

        # Excess = stock return - industry average
        df[excess_col] = df[ret_col] - df["_ind_avg"]

        # Stocks in _UNKNOWN industry get NaN
        df.loc[df["industry_code"] == "_UNKNOWN", excess_col] = np.nan

        df = df.drop(columns=["_ind_avg"], errors="ignore")

        n_valid = df[excess_col].notna().sum()
        logger.info(
            "excess_vs_industry_%dd: added for %d rows (%.1f%% coverage)",
            h, n_valid, n_valid / len(df) * 100,
        )

    # Drop the join columns added by _join_industry (industry_code, industry_name)
    # unless they were already in the original panel
    for col in ["industry_code", "industry_name"]:
        if col not in label_panel.columns and col in df.columns:
            df = df.drop(columns=[col], errors="ignore")

    return df.sort_values(["date", "symbol"]).reset_index(drop=True)
