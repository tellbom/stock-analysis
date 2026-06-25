"""
store.lake
==========
Parquet data lake layout: path conventions and zone definitions.

Medallion layout
----------------
bronze/   Raw API responses — immutable, never modified after write.
          One file per (symbol, source, pull_date) so re-ingestion is safe.
          Not queried by downstream; only the collector writes here.

silver/   Normalised, typed, deduplicated tables.
          ohlcv/<symbol>.parquet      — price/volume, one row per (symbol, date)
          adj_factor/<symbol>.parquet — raw price + adjustment factors
          fundamentals/<symbol>.parquet — PIT fundamentals (T0.7)

gold/     Analysis-ready panels assembled from silver by DuckDB views.
          Not stored as separate files; defined as DuckDB views over silver.

universe/ Membership tables (T0.2).
calendar/ Trading calendar (T0.3).

All path logic lives here; no other module hard-codes paths.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Zone roots
# ---------------------------------------------------------------------------

def bronze_root(store_root: Path | str) -> Path:
    return Path(store_root) / "bronze"

def silver_root(store_root: Path | str) -> Path:
    return Path(store_root) / "silver"

def universe_root(store_root: Path | str) -> Path:
    return Path(store_root) / "universe"

def calendar_path(store_root: Path | str) -> Path:
    return Path(store_root) / "calendar" / "trading_calendar.parquet"


# ---------------------------------------------------------------------------
# Silver: OHLCV
# ---------------------------------------------------------------------------

def ohlcv_dir(store_root: Path | str) -> Path:
    return silver_root(store_root) / "ohlcv"

def ohlcv_path(store_root: Path | str, symbol: str) -> Path:
    return ohlcv_dir(store_root) / f"{symbol}.parquet"


# ---------------------------------------------------------------------------
# Silver: adjustment factors
# ---------------------------------------------------------------------------

def adj_factor_dir(store_root: Path | str) -> Path:
    return silver_root(store_root) / "adj_factor"

def adj_factor_path(store_root: Path | str, symbol: str) -> Path:
    return adj_factor_dir(store_root) / f"{symbol}.parquet"


# ---------------------------------------------------------------------------
# Silver: PIT fundamentals  (T0.7 will populate)
# ---------------------------------------------------------------------------

def fundamentals_dir(store_root: Path | str) -> Path:
    return silver_root(store_root) / "fundamentals"

def fundamentals_path(store_root: Path | str, symbol: str) -> Path:
    return fundamentals_dir(store_root) / f"{symbol}.parquet"


# ---------------------------------------------------------------------------
# Bronze: raw pull snapshots
# ---------------------------------------------------------------------------

def bronze_path(store_root: Path | str, symbol: str, source: str, pull_date: str) -> Path:
    """
    bronze/<source>/<symbol>/<pull_date>.parquet
    e.g. bronze/akshare_daily/600519/2024-01-15.parquet
    """
    return bronze_root(store_root) / source / symbol / f"{pull_date}.parquet"


# ---------------------------------------------------------------------------
# Collector catalog (T0.6)
# ---------------------------------------------------------------------------

def catalog_path(store_root: Path | str) -> Path:
    return Path(store_root) / "catalog" / "collector_catalog.parquet"


# ---------------------------------------------------------------------------
# Directory initialisation
# ---------------------------------------------------------------------------

def init_lake(store_root: Path | str) -> None:
    """Create all lake directories.  Idempotent."""
    root = Path(store_root)
    for sub in (
        "bronze",
        "silver/ohlcv",
        "silver/adj_factor",
        "silver/fundamentals",
        "silver/index_ohlcv",   # P4A-05: CSI 300 and other index OHLCV
        "silver/valuation",     # P4B-01: daily PE/PB/mcap/turnover
        "silver/fund_flow",     # P4B-06: daily capital flow
        "silver/margin",        # P4B-08: daily margin trading
        "silver/lockup",        # P4C-03: lockup expiry calendar
        "universe",
        "calendar",
        "catalog",
        "evaluation",           # P4C-01: feature IC reports and pruning logs
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Features (T1.1-T1.5)
# ---------------------------------------------------------------------------

def features_root(store_root: Path | str) -> Path:
    return Path(store_root) / "features"


def feature_set_dir(store_root: Path | str, feature_set_id: str) -> Path:
    """Directory for one versioned feature set: features/<feature_set_id>/."""
    return features_root(store_root) / feature_set_id


def feature_path(store_root: Path | str, feature_set_id: str, symbol: str) -> Path:
    return feature_set_dir(store_root, feature_set_id) / f"{symbol}.parquet"


def feature_spec_path(store_root: Path | str) -> Path:
    """Registry of all feature-set specs."""
    return features_root(store_root) / "feature_specs.parquet"


# ---------------------------------------------------------------------------
# Labels (T1.6)
# ---------------------------------------------------------------------------

def labels_root(store_root: Path | str) -> Path:
    return Path(store_root) / "labels"


def label_dir(store_root: Path | str, label_name: str) -> Path:
    return labels_root(store_root) / label_name


def label_path(store_root: Path | str, label_name: str, symbol: str) -> Path:
    return label_dir(store_root, label_name) / f"{symbol}.parquet"


# ---------------------------------------------------------------------------
# Silver: index OHLCV (P4A-05 — CSI 300 and other market indices)
# ---------------------------------------------------------------------------

def index_ohlcv_dir(store_root: Path | str) -> Path:
    return silver_root(store_root) / "index_ohlcv"


def index_ohlcv_path(store_root: Path | str, symbol: str) -> Path:
    """
    Path to the silver index OHLCV Parquet for one index symbol.

    Example: silver/index_ohlcv/000300.parquet  (CSI 300)
    """
    return index_ohlcv_dir(store_root) / f"{symbol}.parquet"


# ---------------------------------------------------------------------------
# Silver: valuation (P4B-01 — PE/PB/market cap/turnover from Tencent Finance)
# ---------------------------------------------------------------------------

def valuation_dir(store_root: Path | str) -> Path:
    return silver_root(store_root) / "valuation"


def valuation_path(store_root: Path | str, symbol: str) -> Path:
    """silver/valuation/{symbol}.parquet — daily PE_TTM, PB, mcap, turnover."""
    return valuation_dir(store_root) / f"{symbol}.parquet"


# ---------------------------------------------------------------------------
# Silver: industry classification (P4B-03 — slowly-changing dimension)
# ---------------------------------------------------------------------------

def industry_map_path(store_root: Path | str) -> Path:
    """
    silver/industry_map.parquet — universe-level SCD table.

    Schema: symbol, industry_code, industry_name, concept_tags,
            effective_date, out_date (None = still current)
    """
    return silver_root(store_root) / "industry_map.parquet"


# ---------------------------------------------------------------------------
# Silver: capital flow (P4B-06 — daily 120-day fund flow from Eastmoney)
# ---------------------------------------------------------------------------

def fund_flow_dir(store_root: Path | str) -> Path:
    return silver_root(store_root) / "fund_flow"


def fund_flow_path(store_root: Path | str, symbol: str) -> Path:
    """silver/fund_flow/{symbol}.parquet — daily main/small/mid/large/super net inflow (元)."""
    return fund_flow_dir(store_root) / f"{symbol}.parquet"


# ---------------------------------------------------------------------------
# Silver: margin trading (P4B-08 — daily 融资融券 from Eastmoney datacenter)
# ---------------------------------------------------------------------------

def margin_dir(store_root: Path | str) -> Path:
    return silver_root(store_root) / "margin"


def margin_path(store_root: Path | str, symbol: str) -> Path:
    """silver/margin/{symbol}.parquet — daily margin balance, buy, short."""
    return margin_dir(store_root) / f"{symbol}.parquet"


# ---------------------------------------------------------------------------
# Silver: lockup expiry calendar (P4C-03 — 限售解禁)
# ---------------------------------------------------------------------------

def lockup_dir(store_root: Path | str) -> Path:
    return silver_root(store_root) / "lockup"


def lockup_path(store_root: Path | str, symbol: str) -> Path:
    """
    silver/lockup/{symbol}.parquet — lockup expiry events.

    Schema: symbol, unlock_date, lock_type, shares_million, ratio_pct
    PIT safe: all dates are public knowledge at the time of announcement.
    """
    return lockup_dir(store_root) / f"{symbol}.parquet"
