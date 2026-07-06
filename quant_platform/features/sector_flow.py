"""
Sector/concept fund-flow proxy features.

These features are not stock-level main-flow fields.  They are explicitly named
``sector_*`` / ``stock_industry_*`` and should be reported separately by the
coverage gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.features.registry import FeatureSpec
from quant_platform.store.lake import sector_fund_flow_dir

logger = get_logger(__name__)


SECTOR_FLOW_SPECS: list[FeatureSpec] = [
    FeatureSpec(
        "sector_main_flow_rank", "sector_flow",
        ("industry_name", "sector_main_net"), 0,
        "pct_rank(sector_main_net by date)", warmup=0,
    ),
    FeatureSpec(
        "sector_flow_momentum_3d", "sector_flow",
        ("industry_name", "sector_main_net"), 3,
        "sector_main_net pct_change(3)", warmup=3,
    ),
    FeatureSpec(
        "sector_flow_momentum_5d", "sector_flow",
        ("industry_name", "sector_main_net"), 5,
        "sector_main_net pct_change(5)", warmup=5,
    ),
    FeatureSpec(
        "stock_industry_flow_strength", "sector_flow",
        ("industry_name", "sector_main_net_rate"), 0,
        "industry flow strength joined to stock", warmup=0,
    ),
]


def load_sector_fund_flow_panel(store_root: Path | str) -> pd.DataFrame:
    root = sector_fund_flow_dir(store_root)
    frames = []
    if root.exists():
        for path in root.glob("*.parquet"):
            try:
                df = pd.read_parquet(path)
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
                frames.append(df)
            except Exception as exc:
                logger.warning("Could not load sector fund-flow %s: %s", path, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["trade_date"])


def build_sector_flow_features(
    panel: pd.DataFrame,
    sector_flow_panel: pd.DataFrame,
) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for spec in SECTOR_FLOW_SPECS:
        df[spec.name] = np.nan

    if sector_flow_panel.empty or "industry_name" not in df.columns:
        return df

    sf = sector_flow_panel.copy()
    sf["date"] = pd.to_datetime(sf.get("date", sf["trade_date"]), errors="coerce").dt.date
    sf = sf.rename(columns={"name": "industry_name"})
    sf = sf.sort_values(["industry_name", "date"]).reset_index(drop=True)
    sf["sector_flow_momentum_3d"] = sf.groupby("industry_name")["sector_main_net"].transform(lambda x: x.pct_change(3))
    sf["sector_flow_momentum_5d"] = sf.groupby("industry_name")["sector_main_net"].transform(lambda x: x.pct_change(5))
    sf["sector_main_flow_rank"] = sf.groupby("date")["sector_main_net"].transform(
        lambda s: s.rank(method="average", pct=True)
    )
    if "sector_main_net_rate" in sf.columns:
        sf["stock_industry_flow_strength"] = pd.to_numeric(sf["sector_main_net_rate"], errors="coerce")
    join_cols = [
        "industry_name", "date",
        "sector_main_flow_rank", "sector_flow_momentum_3d",
        "sector_flow_momentum_5d", "stock_industry_flow_strength",
    ]
    sf = sf[[c for c in join_cols if c in sf.columns]].drop_duplicates(["industry_name", "date"], keep="last")
    df = df.drop(columns=[s.name for s in SECTOR_FLOW_SPECS], errors="ignore")
    df = df.merge(sf, on=["industry_name", "date"], how="left")
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)
