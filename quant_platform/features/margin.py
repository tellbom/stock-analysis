"""
features.margin
===============
Margin trading (融资融券) feature builder (P4B-08).

Builds leverage-sentiment features from the silver margin table.
Margin balance changes signal retail leverage momentum: rapidly increasing
margin balance often precedes overcrowding; sharp declines signal forced
unwinding.

PIT correctness: margin data is released with a 1-business-day delay.
A 1-day lag is applied: features at date T use margin data through T−1.

Feature columns produced
------------------------
  cs_margin_balance_change_5d  float [0,1]  rank of 5-day rate of change in
                                             融资余额 / float_mcap
  cs_rzrq_ratio_rank           float [0,1]  rank of total 融资融券余额 / float_mcap

Spec list: MARGIN_SPECS (for the feature registry)
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

MARGIN_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "cs_margin_balance_change_5d", "margin",
        ("rzye", "float_mcap_yi"), 6,   # 5d change + 1d lag = 6
        "pct_rank(rzye.diff(5)/float_mcap, lag=1)", warmup=6,
    ),
    FeatureSpec(
        "cs_rzrq_ratio_rank", "margin",
        ("rzrqye", "float_mcap_yi"), 1,
        "pct_rank(rzrqye/float_mcap, lag=1)", warmup=1,
    ),
]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_margin_features(
    panel: pd.DataFrame,
    margin_panel: pd.DataFrame,
    valuation_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add cross-sectional margin trading features to the universe panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Universe panel with columns [symbol, date, ...].
    margin_panel : pd.DataFrame
        Concatenated silver margin data (all symbols).
        Columns: symbol, date, rzye, rzmre, rqye, rzrqye, ...
    valuation_panel : pd.DataFrame | None
        For float_mcap_yi normalisation.  If None, uses absolute values.

    Returns
    -------
    pd.DataFrame
        Panel with margin feature columns added.
    """
    if margin_panel.empty:
        logger.info(
            "build_margin_features: empty margin panel — "
            "margin features will be NaN (stocks may not be margin-eligible)"
        )
        for spec in MARGIN_SPECS:
            panel[spec.name] = np.nan
        return panel

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    margin = margin_panel.copy()
    margin["date"] = pd.to_datetime(margin["date"]).dt.date

    # Apply 1-day lag: shift margin dates forward by 1 business day
    # Simplest correct approach: group by symbol, sort, then merge with
    # a shifted date key onto the panel.
    margin = margin.sort_values(["symbol", "date"])
    margin["_join_date"] = margin.groupby("symbol")["date"].shift(-1)
    margin_lagged = margin.dropna(subset=["_join_date"]).copy()
    margin_lagged["_join_date"] = pd.to_datetime(margin_lagged["_join_date"]).dt.date
    margin_lagged = margin_lagged.rename(columns={"_join_date": "date_panel"})

    # Merge float_mcap
    if valuation_panel is not None and not valuation_panel.empty:
        val = valuation_panel.copy()
        val["date"] = pd.to_datetime(val["date"]).dt.date
        mcap = val[["symbol", "date", "float_mcap_yi"]].copy()
        margin_lagged = margin_lagged.merge(
            mcap.rename(columns={"date": "date_panel"}),
            on=["symbol", "date_panel"], how="left",
        )
    else:
        margin_lagged["float_mcap_yi"] = np.nan

    # Fallback denominator
    margin_lagged["_denom"] = margin_lagged["float_mcap_yi"].where(
        margin_lagged["float_mcap_yi"] > 0, 1e8   # 1亿 as fallback
    )

    # 5-day change in rzye / float_mcap (leverage momentum)
    margin_lagged = margin_lagged.sort_values(["symbol", "date_panel"]).reset_index(drop=True)
    margin_lagged["_rzye_norm"] = margin_lagged["rzye"] / margin_lagged["_denom"]
    margin_lagged["_rzye_chg5"] = margin_lagged.groupby("symbol")["_rzye_norm"].transform(
        lambda x: x.diff(5)
    )

    # Total margin ratio
    if "rzrqye" in margin_lagged.columns:
        margin_lagged["_rzrq_norm"] = margin_lagged["rzrqye"] / margin_lagged["_denom"]
    else:
        margin_lagged["_rzrq_norm"] = np.nan

    # Merge into panel on (symbol, date)
    m_join = margin_lagged[["symbol", "date_panel", "_rzye_chg5", "_rzrq_norm"]].rename(
        columns={"date_panel": "date"}
    )
    df = df.merge(m_join, on=["symbol", "date"], how="left")

    # Cross-sectional percentile ranks
    def _pct_rank(s: pd.Series) -> pd.Series:
        return s.rank(method="average", ascending=True, pct=True)

    df["cs_margin_balance_change_5d"] = df.groupby("date")["_rzye_chg5"].transform(_pct_rank)
    df["cs_rzrq_ratio_rank"] = df.groupby("date")["_rzrq_norm"].transform(_pct_rank)

    # Drop working columns
    wk = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=wk, errors="ignore")

    n_valid = df["cs_rzrq_ratio_rank"].notna().sum()
    logger.info(
        "build_margin_features: added margin features; %d non-NaN rows (%.1f%% coverage)",
        n_valid, n_valid / len(df) * 100,
    )
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_margin_panel(
    store_root: Path | str,
    symbols: list[str],
) -> pd.DataFrame:
    """Load and concatenate silver margin Parquets for all symbols."""
    from quant_platform.store.lake import margin_path

    store_root = Path(store_root)
    frames = []
    for symbol in symbols:
        p = margin_path(store_root, symbol)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not load margin for %s: %s", symbol, exc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
