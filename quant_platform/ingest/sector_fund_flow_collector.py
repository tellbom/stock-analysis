"""
Sector/concept fund-flow collectors.

These are proxy flow sources.  They are intentionally stored separately from
``silver/fund_flow`` so they cannot be mistaken for stock-level main-force
net inflow.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import (
    concept_fund_flow_dir,
    concept_fund_flow_path,
    init_lake,
    sector_fund_flow_dir,
    sector_fund_flow_path,
)

logger = get_logger(__name__)


def _pick_col(df: pd.DataFrame, *names: str) -> str | None:
    by_lower = {str(c).lower(): c for c in df.columns}
    by_raw = {str(c): c for c in df.columns}
    for name in names:
        if name in by_raw:
            return by_raw[name]
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def _normalise_hist(df: pd.DataFrame, name: str, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    date_col = _pick_col(df, "日期", "date", "trade_date")
    main_col = _pick_col(df, "主力净流入-净额", "主力净流入", "main_net")
    rate_col = _pick_col(df, "主力净流入-净占比", "主力净占比", "main_net_rate")
    if not date_col or not main_col:
        raise RuntimeError(f"{source}/{name}: missing date or main flow column: {list(df.columns)}")
    out = pd.DataFrame({
        "name": name,
        "trade_date": pd.to_datetime(df[date_col], errors="coerce").dt.date,
        "sector_main_net": pd.to_numeric(df[main_col], errors="coerce"),
        "sector_main_net_rate": pd.to_numeric(df[rate_col], errors="coerce") if rate_col else pd.NA,
        "source": source,
        "raw_update_time": None,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    out["date"] = out["trade_date"]
    return out.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)


class SectorFundFlowCollector:
    def __init__(self, store_root: Path | str) -> None:
        self.store_root = Path(store_root)
        init_lake(self.store_root)

    def collect_sector_rank(self, sector_type: str = "行业资金流", indicator: str = "今日") -> pd.DataFrame:
        import akshare as ak

        df = ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type=sector_type)
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        out["source"] = f"akshare_stock_sector_fund_flow_rank:{sector_type}:{indicator}"
        out["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        return out

    def run_sectors(self, sector_names: list[str]) -> dict[str, int]:
        import akshare as ak

        sector_fund_flow_dir(self.store_root).mkdir(parents=True, exist_ok=True)
        results: dict[str, int] = {}
        for name in sector_names:
            try:
                raw = ak.stock_sector_fund_flow_hist(symbol=name)
                df = _normalise_hist(raw, name, "akshare_sector_fund_flow_hist")
                if df.empty:
                    results[name] = 0
                    continue
                path = sector_fund_flow_path(self.store_root, name)
                combined = self._upsert(path, df)
                results[name] = len(combined)
            except Exception as exc:
                logger.warning("sector fund-flow failed for %s: %s", name, exc)
                results[name] = 0
        return results

    def run_concepts(self, concept_names: list[str]) -> dict[str, int]:
        import akshare as ak

        concept_fund_flow_dir(self.store_root).mkdir(parents=True, exist_ok=True)
        results: dict[str, int] = {}
        for name in concept_names:
            try:
                raw = ak.stock_concept_fund_flow_hist(symbol=name)
                df = _normalise_hist(raw, name, "akshare_concept_fund_flow_hist")
                if df.empty:
                    results[name] = 0
                    continue
                path = concept_fund_flow_path(self.store_root, name)
                combined = self._upsert(path, df)
                results[name] = len(combined)
            except Exception as exc:
                logger.warning("concept fund-flow failed for %s: %s", name, exc)
                results[name] = 0
        return results

    @staticmethod
    def _upsert(path: Path, df: pd.DataFrame) -> pd.DataFrame:
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce").dt.date
        combined["date"] = combined["trade_date"]
        combined = (
            combined.dropna(subset=["trade_date"])
            # FIXED (review finding #6): drop_duplicates BEFORE sort_values,
            # same reasoning as flow_collector.py -- pandas' default sort
            # is not stable, so keep="last" after sorting doesn't reliably
            # prefer the freshly-fetched row for overlapping trade_dates.
            .drop_duplicates(["name", "trade_date"], keep="last")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)
        return combined
