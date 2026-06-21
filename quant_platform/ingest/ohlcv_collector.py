"""
ingest.ohlcv_collector
======================
Batch OHLCV collector: fetches daily price/volume data for a symbol universe
and writes normalised Parquet to the silver layer.

Design
------
- One symbol at a time (serial) by default; ``max_workers > 1`` enables
  ThreadPoolExecutor parallelism.  Start with 1 worker; increase carefully —
  AKShare/Sina documentation warns that heavy concurrent scraping causes IP bans.
- Each fetch goes through ``core.fetch.safe_call`` (retry + back-off).
- Endpoint priority:
    1. ``ak.stock_zh_a_daily``  (Sina, English columns, has adjust factor support)
    2. ``ak.stock_zh_a_hist``   (EM, Chinese columns, has turnover)
  The first endpoint to return non-empty data wins for a given symbol.
- Column normalisation is done here; ``store.schemas.enforce_ohlcv`` handles
  the final type casting and deduplication.
- **Fail loudly**: if a symbol fetch fails from ALL endpoints, it is recorded
  in the result summary as a failure.  The collector never writes partial or
  fabricated data.

Incremental update
------------------
If a Parquet file already exists for a symbol, the collector reads its latest
date and fetches only the tail (start = last_date + 1 day).  New rows are
appended and the file is overwritten atomically.  This is the generalisation
of the diff logic in ``update_stock_report.py``.

The full-universe catalog (T0.6) sits on top of this module; this module only
handles the per-symbol fetch-and-store logic.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from quant_platform.core.fetch import safe_call
from quant_platform.core.logging import get_logger
from quant_platform.core.market import detect_market
from quant_platform.store.lake import ohlcv_path, init_lake
from quant_platform.store.parquet_store import write_ohlcv, read_ohlcv

logger = get_logger(__name__)

# AKShare date format
_DATE_FMT = "%Y%m%d"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    symbol:    str
    success:   bool
    rows_new:  int = 0
    rows_total: int = 0
    last_date: dt.date | None = None
    error:     str = ""


@dataclass
class CollectorSummary:
    started_at:  str = ""
    finished_at: str = ""
    total:       int = 0
    succeeded:   int = 0
    failed:      int = 0
    skipped:     int = 0
    results:     list[FetchResult] = field(default_factory=list)

    @property
    def failed_symbols(self) -> list[str]:
        return [r.symbol for r in self.results if not r.success and r.error != "skipped"]


# ---------------------------------------------------------------------------
# Per-symbol fetch helpers
# ---------------------------------------------------------------------------

def _normalise_daily(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalise stock_zh_a_daily output → canonical OHLCV columns.
    stock_zh_a_daily returns: date, open, high, low, close, volume[, amount]
    The index may be dates; reset it first.
    """
    df = df.reset_index() if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex) else df.copy()

    # Rename if needed (some akshare versions return Chinese columns)
    rename = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low",  "收盘": "close", "成交量": "volume", "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["symbol"] = symbol
    return df


def _normalise_hist(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalise stock_zh_a_hist output → canonical OHLCV columns.
    stock_zh_a_hist returns Chinese columns: 日期, 开盘, 收盘, 最高, 最低,
    成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    """
    df = df.copy()
    rename = {
        "日期": "date",   "开盘": "open",   "最高": "high",
        "最低": "low",    "收盘": "close",  "成交量": "volume",
        "成交额": "amount", "换手率": "turnover",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["symbol"] = symbol
    return df


def _fetch_symbol_raw(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """
    Attempt to fetch OHLCV for one symbol from AKShare.
    Tries stock_zh_a_daily first, then stock_zh_a_hist.
    Returns a normalised DataFrame or None if all endpoints fail.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare is not installed")
        return None

    mi = detect_market(symbol)

    # --- Endpoint 1: stock_zh_a_daily (Sina, English columns) ---
    df = safe_call(
        ak.stock_zh_a_daily,
        symbol=mi.prefixed,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
        label=f"stock_zh_a_daily {symbol}",
        retries=3,
    )
    if df is not None and not df.empty:
        return _normalise_daily(df, symbol)

    logger.warning("stock_zh_a_daily failed for %s, trying stock_zh_a_hist", symbol)

    # --- Endpoint 2: stock_zh_a_hist (EM, Chinese columns, has turnover) ---
    df = safe_call(
        ak.stock_zh_a_hist,
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
        label=f"stock_zh_a_hist {symbol}",
        retries=3,
    )
    if df is not None and not df.empty:
        return _normalise_hist(df, symbol)

    logger.error(
        "All endpoints failed for %s. Network may be restricted (403). "
        "No data written — no fabricated data.",
        symbol,
    )
    return None


# ---------------------------------------------------------------------------
# Per-symbol collect-and-store
# ---------------------------------------------------------------------------

def collect_symbol(
    symbol: str,
    store_root: Path | str,
    start_date: dt.date,
    end_date: dt.date,
) -> FetchResult:
    """
    Fetch OHLCV for *symbol* and write/append to the silver lake.

    Incremental: if existing data is found, only fetches from
    (last_stored_date + 1) onward and appends.

    Returns a FetchResult describing the outcome.
    """
    store_root = Path(store_root)
    path = ohlcv_path(store_root, symbol)

    # --- Read existing data to determine fetch window ---
    existing = read_ohlcv(path)
    if not existing.empty:
        last_stored = pd.to_datetime(existing["date"]).dt.date.max()
        if last_stored >= end_date:
            logger.info("%s: already up to date (last=%s)", symbol, last_stored)
            return FetchResult(
                symbol=symbol, success=True,
                rows_new=0, rows_total=len(existing),
                last_date=last_stored, error="skipped",
            )
        fetch_start = last_stored + dt.timedelta(days=1)
    else:
        fetch_start = start_date

    fetch_start_s = fetch_start.strftime(_DATE_FMT)
    fetch_end_s   = end_date.strftime(_DATE_FMT)

    logger.info("Fetching %s: %s → %s", symbol, fetch_start_s, fetch_end_s)

    # --- Fetch ---
    new_df = _fetch_symbol_raw(symbol, fetch_start_s, fetch_end_s)

    if new_df is None or new_df.empty:
        return FetchResult(
            symbol=symbol, success=False,
            rows_total=len(existing),
            error="All AKShare endpoints returned no data (network may be blocked)",
        )

    # --- Merge with existing and write ---
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    write_ohlcv(combined, path, symbol)

    rows_new   = len(new_df)
    rows_total = len(pd.read_parquet(path))  # re-read to confirm dedup
    last_date  = pd.to_datetime(combined["date"]).dt.date.max()

    logger.info(
        "%s: +%d new rows, %d total, last=%s",
        symbol, rows_new, rows_total, last_date,
    )
    return FetchResult(
        symbol=symbol, success=True,
        rows_new=rows_new, rows_total=rows_total,
        last_date=last_date,
    )


# ---------------------------------------------------------------------------
# Universe-level batch collector
# ---------------------------------------------------------------------------

class OHLCVCollector:
    """
    Batch OHLCV collector for a symbol universe.

    Parameters
    ----------
    store_root : Path | str
        Root of the Parquet data lake.
    start_date : date | str
        Earliest date to collect.  Default: 3 years ago.
    end_date : date | str
        Latest date to collect.  Default: today.
    max_workers : int
        Thread pool size.  Default 1 (serial).  Increase cautiously —
        AKShare warns that heavy concurrency causes IP bans.
    """

    def __init__(
        self,
        store_root: Path | str,
        start_date: dt.date | str | None = None,
        end_date:   dt.date | str | None = None,
        max_workers: int = 1,
    ) -> None:
        self.store_root  = Path(store_root)
        self.start_date  = _as_date(start_date) if start_date else (
            dt.date.today() - dt.timedelta(days=365 * 3)
        )
        self.end_date    = _as_date(end_date) if end_date else dt.date.today()
        self.max_workers = max_workers
        init_lake(self.store_root)

    def run(self, symbols: list[str]) -> CollectorSummary:
        """
        Fetch OHLCV for all *symbols* and write to the lake.

        Parameters
        ----------
        symbols : list[str]
            6-digit A-share codes, e.g. ["600519", "000858"].

        Returns
        -------
        CollectorSummary
            Contains per-symbol FetchResult objects and aggregate counts.
        """
        summary = CollectorSummary(
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
            total=len(symbols),
        )

        if not symbols:
            logger.warning("collect.run() called with empty symbol list")
            summary.finished_at = dt.datetime.now().isoformat(timespec="seconds")
            return summary

        logger.info(
            "Starting OHLCV collection: %d symbols, %s → %s, workers=%d",
            len(symbols), self.start_date, self.end_date, self.max_workers,
        )

        if self.max_workers == 1:
            results = [self._collect_one(s) for s in symbols]
        else:
            results = self._collect_parallel(symbols)

        summary.results      = results
        summary.succeeded    = sum(1 for r in results if r.success)
        summary.failed       = sum(1 for r in results if not r.success)
        summary.skipped      = sum(1 for r in results if r.error == "skipped")
        summary.finished_at  = dt.datetime.now().isoformat(timespec="seconds")

        logger.info(
            "Collection complete: %d succeeded, %d failed, %d skipped (already up-to-date)",
            summary.succeeded, summary.failed, summary.skipped,
        )
        if summary.failed_symbols:
            logger.error(
                "Failed symbols (%d): %s",
                len(summary.failed_symbols),
                summary.failed_symbols[:10],
            )

        return summary

    def _collect_one(self, symbol: str) -> FetchResult:
        try:
            return collect_symbol(symbol, self.store_root, self.start_date, self.end_date)
        except Exception as exc:
            logger.error("Unexpected error collecting %s: %s", symbol, exc)
            return FetchResult(symbol=symbol, success=False, error=str(exc))

    def _collect_parallel(self, symbols: list[str]) -> list[FetchResult]:
        results: list[FetchResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._collect_one, s): s for s in symbols}
            for future in as_completed(futures):
                results.append(future.result())
        # Restore input order for deterministic output
        order = {s: i for i, s in enumerate(symbols)}
        return sorted(results, key=lambda r: order.get(r.symbol, 0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_date(v: dt.date | str) -> dt.date:
    if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
        return v
    return pd.to_datetime(v).date()
