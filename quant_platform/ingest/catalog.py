"""
ingest.catalog
==============
Persistent collector catalog: tracks per-(symbol, source) fetch state so that
incremental runs and crash recovery work correctly.

Why a separate catalog (not just reading Parquet last dates)?
-------------------------------------------------------------
T0.5's ``collect_symbol`` already reads the OHLCV Parquet file to find the
last stored date — but that only works if the file exists and is valid.
The catalog adds:

1. **Crash recovery**: a symbol marked ``status='in_progress'`` was interrupted
   mid-write.  The next run sees this and re-fetches rather than skipping.
2. **Source tracking**: records which AKShare endpoint was used, so future runs
   can prefer the last-successful endpoint.
3. **Run history**: ``last_run_at`` + ``last_success_date`` separate "when did
   we try" from "what data do we have" — useful for quality reports (T0.8).
4. **Universe-driven scheduling**: ``CatalogDrivenCollector`` reads the universe
   membership, consults the catalog, and only re-fetches symbols that need work.

Catalog schema
--------------
Parquet: ``<store_root>/catalog/collector_catalog.parquet``

Columns:
  symbol           str   6-digit code
  source           str   "ohlcv_daily" (extensible to "fundamentals", etc.)
  status           str   "pending" | "in_progress" | "success" | "failed"
  last_run_at      str   ISO datetime of most recent attempt
  last_success_at  str   ISO datetime of last successful fetch (or "")
  last_success_date str  Last date successfully stored (YYYY-MM-DD or "")
  error_msg        str   Last error message (or "")
  rows_total       int   Total rows in Parquet after last success

Crash recovery
--------------
``in_progress`` rows are detected on startup.  The collector treats them as
``pending`` and re-fetches.  A write is only marked ``success`` **after**
``write_ohlcv`` completes — if the process dies between fetch and write,
the status stays ``in_progress`` and the next run retries cleanly.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.ingest.ohlcv_collector import OHLCVCollector, collect_symbol, FetchResult
from quant_platform.store.lake import catalog_path, ohlcv_path, init_lake
from quant_platform.store.parquet_store import read_ohlcv, write_parquet, read_parquet

logger = get_logger(__name__)

_SOURCE_OHLCV = "ohlcv_daily"

# ---------------------------------------------------------------------------
# Catalog schema helpers
# ---------------------------------------------------------------------------

_CATALOG_COLS = [
    "symbol", "source", "status",
    "last_run_at", "last_success_at", "last_success_date",
    "error_msg", "rows_total",
]

_EMPTY_CATALOG = pd.DataFrame({c: pd.Series(dtype="object") for c in _CATALOG_COLS})


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _empty_row(symbol: str, source: str) -> dict:
    return {
        "symbol":            symbol,
        "source":            source,
        "status":            "pending",
        "last_run_at":       "",
        "last_success_at":   "",
        "last_success_date": "",
        "error_msg":         "",
        "rows_total":        0,
    }


# ---------------------------------------------------------------------------
# CollectorCatalog
# ---------------------------------------------------------------------------

class CollectorCatalog:
    """
    Read/write interface for the collector catalog Parquet.

    Parameters
    ----------
    store_root : Path | str
        Root of the Parquet data lake.
    """

    def __init__(self, store_root: Path | str) -> None:
        self.store_root = Path(store_root)
        self._path = catalog_path(self.store_root)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Return the full catalog DataFrame.
        Returns an empty frame (correct schema) if the catalog does not exist yet.
        """
        if not self._path.exists():
            return _EMPTY_CATALOG.copy()
        try:
            return pd.read_parquet(self._path)
        except Exception as exc:
            logger.error("Failed to read catalog at %s: %s — starting fresh", self._path, exc)
            return _EMPTY_CATALOG.copy()

    def get_row(self, symbol: str, source: str) -> dict | None:
        """Return the catalog row for (symbol, source), or None if absent."""
        df = self.load()
        mask = (df["symbol"] == symbol) & (df["source"] == source)
        rows = df[mask]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def needs_fetch(self, symbol: str, source: str, end_date: dt.date) -> bool:
        """
        Return True if this symbol should be (re-)fetched.

        Criteria:
        - No catalog row yet (never fetched).
        - Status is "pending" or "in_progress" (crashed last time).
        - Status is "failed" (retry).
        - Status is "success" but last_success_date < end_date (stale).
        """
        row = self.get_row(symbol, source)
        if row is None:
            return True
        status = row.get("status", "pending")
        if status in ("pending", "in_progress", "failed"):
            return True
        if status == "success":
            lsd = row.get("last_success_date", "")
            if not lsd:
                return True
            try:
                last = dt.date.fromisoformat(str(lsd))
                return last < end_date
            except ValueError:
                return True
        return True

    def symbols_needing_fetch(
        self,
        symbols: list[str],
        source: str,
        end_date: dt.date,
    ) -> list[str]:
        """Filter *symbols* to those that need a fetch per the catalog."""
        return [s for s in symbols if self.needs_fetch(s, source, end_date)]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def mark_in_progress(self, symbol: str, source: str) -> None:
        """Mark a symbol as in-progress BEFORE the fetch begins (crash safety)."""
        self._upsert(symbol, source, {"status": "in_progress", "last_run_at": _now()})

    def mark_success(self, symbol: str, source: str, result: FetchResult) -> None:
        """Mark a symbol as successfully fetched."""
        self._upsert(symbol, source, {
            "status":            "success",
            "last_success_at":   _now(),
            "last_success_date": str(result.last_date) if result.last_date else "",
            "error_msg":         "",
            "rows_total":        result.rows_total,
        })

    def mark_failed(self, symbol: str, source: str, error: str) -> None:
        """Mark a symbol as failed."""
        self._upsert(symbol, source, {
            "status":    "failed",
            "error_msg": error[:500],   # cap length
        })

    def mark_skipped(self, symbol: str, source: str, result: FetchResult) -> None:
        """Mark a symbol as skipped (already up-to-date) — treated as success."""
        self._upsert(symbol, source, {
            "status":            "success",
            "last_success_at":   _now(),
            "last_success_date": str(result.last_date) if result.last_date else "",
            "error_msg":         "",
            "rows_total":        result.rows_total,
        })

    def reset_in_progress(self) -> int:
        """
        On startup: reset any "in_progress" rows to "pending".
        Returns the number of rows reset (for logging).
        """
        df = self.load()
        if df.empty:
            return 0
        mask = df["status"] == "in_progress"
        count = int(mask.sum())
        if count:
            df.loc[mask, "status"] = "pending"
            df.loc[mask, "error_msg"] = "reset_from_in_progress_on_startup"
            self._save(df)
            logger.warning(
                "Reset %d in_progress rows to pending (crash recovery)", count
            )
        return count

    def summary(self) -> dict:
        """Return a count-by-status dict for quality reports."""
        df = self.load()
        if df.empty:
            return {"total": 0}
        counts = df.groupby("source")["status"].value_counts().to_dict()
        return {
            "total":        len(df),
            "by_status":    df["status"].value_counts().to_dict(),
            "by_source":    counts,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upsert(self, symbol: str, source: str, updates: dict) -> None:
        """Insert or update a catalog row, then save."""
        df = self.load()
        mask = (df["symbol"] == symbol) & (df["source"] == source)

        if mask.any():
            for k, v in updates.items():
                df.loc[mask, k] = v
        else:
            row = _empty_row(symbol, source)
            row.update(updates)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        self._save(df)

    def _save(self, df: pd.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(df, self._path)


# ---------------------------------------------------------------------------
# CatalogDrivenCollector
# ---------------------------------------------------------------------------

class CatalogDrivenCollector:
    """
    Universe-aware, catalog-backed OHLCV collector.

    Wraps ``OHLCVCollector`` and ``CollectorCatalog`` to provide:
    - Universe-driven symbol list (from ``UniverseService``).
    - Catalog-based skip logic (avoids re-fetching up-to-date symbols).
    - Crash recovery (resets ``in_progress`` rows on startup).
    - Per-symbol status tracking written to the catalog Parquet.

    Parameters
    ----------
    store_root : Path | str
        Root of the Parquet data lake.
    universe_key : str
        Universe key (e.g. ``"csi300"``).
    start_date : date | str | None
        Earliest date to collect.  Default: 3 years ago.
    end_date : date | str | None
        Latest date to collect.  Default: today.
    max_workers : int
        Thread concurrency for ``OHLCVCollector``.  Default 1.
    """

    def __init__(
        self,
        store_root: Path | str,
        universe_key: str,
        start_date: dt.date | str | None = None,
        end_date:   dt.date | str | None = None,
        max_workers: int = 1,
    ) -> None:
        self.store_root    = Path(store_root)
        self.universe_key  = universe_key
        self.end_date      = _as_date(end_date) if end_date else dt.date.today()
        self.start_date    = _as_date(start_date) if start_date else (
            self.end_date - dt.timedelta(days=365 * 3)
        )
        self.max_workers   = max_workers
        self.catalog       = CollectorCatalog(store_root)
        self._inner        = OHLCVCollector(
            store_root=store_root,
            start_date=self.start_date,
            end_date=self.end_date,
            max_workers=max_workers,
        )
        init_lake(self.store_root)

    def run(self, symbols: list[str] | None = None) -> "CatalogRunSummary":
        """
        Run the catalog-driven collection pass.

        Parameters
        ----------
        symbols : list[str] | None
            Override the symbol list.  If None, uses UniverseService to get
            current members of ``universe_key``.

        Returns
        -------
        CatalogRunSummary
        """
        # 1. Crash recovery: reset any leftover in_progress rows
        reset_count = self.catalog.reset_in_progress()
        if reset_count:
            logger.info("Crash recovery: reset %d in_progress symbols", reset_count)

        # 2. Resolve symbol list
        if symbols is None:
            symbols = self._get_universe_symbols()

        # 3. Filter to symbols that actually need a fetch
        to_fetch = self.catalog.symbols_needing_fetch(
            symbols, _SOURCE_OHLCV, self.end_date
        )
        skipped_count = len(symbols) - len(to_fetch)
        logger.info(
            "Universe: %d symbols, %d need fetch, %d already up-to-date",
            len(symbols), len(to_fetch), skipped_count,
        )

        # 4. Fetch each symbol with catalog tracking
        fetch_results: list[FetchResult] = []
        for symbol in to_fetch:
            result = self._fetch_one_tracked(symbol)
            fetch_results.append(result)

        return CatalogRunSummary(
            universe_key=self.universe_key,
            total_universe=len(symbols),
            fetched=len(to_fetch),
            skipped_by_catalog=skipped_count,
            succeeded=sum(1 for r in fetch_results if r.success),
            failed=sum(1 for r in fetch_results if not r.success),
            results=fetch_results,
            catalog_summary=self.catalog.summary(),
        )

    def _fetch_one_tracked(self, symbol: str) -> FetchResult:
        """Fetch one symbol with catalog in_progress / success / failed tracking."""
        # Mark in_progress BEFORE fetch (crash safety: if we die here, next run retries)
        self.catalog.mark_in_progress(symbol, _SOURCE_OHLCV)
        try:
            result = collect_symbol(
                symbol, self.store_root, self.start_date, self.end_date
            )
            if result.success:
                if result.error == "skipped":
                    self.catalog.mark_skipped(symbol, _SOURCE_OHLCV, result)
                else:
                    self.catalog.mark_success(symbol, _SOURCE_OHLCV, result)
            else:
                self.catalog.mark_failed(symbol, _SOURCE_OHLCV, result.error)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("Unexpected error for %s: %s", symbol, error)
            self.catalog.mark_failed(symbol, _SOURCE_OHLCV, error)
            return FetchResult(symbol=symbol, success=False, error=error)

    def _get_universe_symbols(self) -> list[str]:
        """Load universe symbols from UniverseService; raises if not initialised."""
        from quant_platform.ingest.universe_service import UniverseService
        svc = UniverseService(self.universe_key, self.store_root)
        return svc.get_symbols_as_of()


# ---------------------------------------------------------------------------
# Run summary dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field

@dataclass
class CatalogRunSummary:
    universe_key:       str
    total_universe:     int
    fetched:            int
    skipped_by_catalog: int
    succeeded:          int
    failed:             int
    results:            list[FetchResult] = dc_field(default_factory=list)
    catalog_summary:    dict              = dc_field(default_factory=dict)

    @property
    def failed_symbols(self) -> list[str]:
        return [r.symbol for r in self.results if not r.success]

    def __str__(self) -> str:
        return (
            f"CatalogRunSummary({self.universe_key}): "
            f"universe={self.total_universe}, fetched={self.fetched}, "
            f"skipped={self.skipped_by_catalog}, "
            f"succeeded={self.succeeded}, failed={self.failed}"
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _as_date(v: dt.date | str) -> dt.date:
    if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
        return v
    return pd.to_datetime(v).date()
