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
With stored QFQ history, updates verify and replace a complete same-source
series before appending any new dates. An explicit deep_verify=False is a
legacy/audit opt-out; automatic updates verify whenever new bars are needed.

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

# QFQ scale consistency check
# ---------------------------
# Forward-adjusted (qfq) prices change scale on ex-rights/ex-dividend dates.
# An incremental tail append cannot detect a scale change that happened
# BEFORE the fetched window: the tail matches (post-ex-date) while the stored
# history is on the old scale.  The 300803 (指南针) case proved this: 896 of
# 908 shared dates differed, only the last 12 matched.
_QFQ_TOL = 0.02  # yuan; matches the audit's 0.020001 threshold
_DEEP_VERIFY_INTERVAL_DAYS = 30  # auto-trigger full history check


def _verify_sidecar_path(store_root: Path | str, symbol: str) -> Path:
    """Sidecar tracking last deep QFQ verification per symbol."""
    return Path(store_root) / "silver" / "ohlcv" / f"{symbol}.verify.json"


def verify_qfq_scale(
    existing: pd.DataFrame,
    fresh: pd.DataFrame,
    tol: float = _QFQ_TOL,
) -> tuple[bool, list, float]:
    """
    Compare stored history vs freshly fetched full history on overlapping dates.

    Unlike a tail-only overlap check, this compares the FULL overlap so an
    ex-rights/ex-dividend outside any tail window is still detected
    (the 300803 case: tail 12 days matched, 896 earlier dates differed).

    Returns
    -------
    (consistent, mismatched_dates, max_abs_diff)
        consistent: True if all overlapping closes agree within tol.
    """
    if fresh.empty:
        return False, [], float("inf")
    if existing.empty:
        return True, [], 0.0
    old = existing[["date", "close"]].copy()
    new = fresh[["date", "close"]].copy()
    old["date"] = pd.to_datetime(old["date"]).dt.date
    new["date"] = pd.to_datetime(new["date"]).dt.date
    joined = old.merge(new, on="date", suffixes=("_old", "_new"), how="inner")
    if joined.empty:
        return False, [], float("inf")
    diff = (joined["close_old"] - joined["close_new"]).abs()
    bad = joined.loc[diff > tol, "date"]
    return (
        len(bad) == 0,
        sorted(bad.tolist()),
        float(diff.max()),
    )


def refresh_symbol_full_history(
    symbol: str,
    store_root: Path | str,
    start_date: dt.date,
    end_date: dt.date,
) -> FetchResult:
    """
    Full-history QFQ refresh for *symbol*.

    Fetches the complete history from the earliest stored date (or
    *start_date* when no data exists), runs the QFQ scale consistency check
    over the FULL overlap, and — only when the scale differs — replaces the
    stored history with the fresh fetch after an integrity check.

    Integrity check before write: fresh data must be non-empty, span
    [earliest_needed, end_date], and contain no null/non-positive OHLCV.
    The write is atomic (temp file + rename via write_ohlcv).

    On ANY failure the old file is kept untouched and success=False
    is returned — never a partial or spliced write.
    """
    store_root = Path(store_root)
    path = ohlcv_path(store_root, symbol)
    existing = read_ohlcv(path)

    if not existing.empty:
        earliest_needed = pd.to_datetime(existing["date"]).dt.date.min()
        earliest_needed = min(earliest_needed, start_date)
    else:
        earliest_needed = start_date

    # --- Fetch full history ---
    new_df = _fetch_symbol_raw(
        symbol,
        earliest_needed.strftime(_DATE_FMT),
        end_date.strftime(_DATE_FMT),
    )
    if new_df is None or new_df.empty:
        return FetchResult(
            symbol=symbol, success=False,
            rows_total=len(existing),
            error="Full-history refetch returned no data; old file kept",
        )

    new_df = new_df.copy()
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    if not required.issubset(new_df.columns):
        return FetchResult(symbol=symbol, success=False, rows_total=len(existing),
                           error="Full-history refetch missing required OHLCV columns; old file kept")
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce").dt.date
    if new_df.date.isna().any() or new_df.date.duplicated().any() or not new_df.symbol.astype(str).str.zfill(6).eq(symbol).all():
        return FetchResult(symbol=symbol, success=False, rows_total=len(existing),
                           error="Invalid or duplicate dates/symbols in full history; old file kept")
    new_df = new_df[new_df.date <= end_date].sort_values("date").reset_index(drop=True)
    if new_df.empty:
        return FetchResult(symbol=symbol, success=False, rows_total=len(existing),
                           error="Full history has no prices through requested end date; old file kept")

    # --- Integrity check: coverage + sane bars ---
    # The fresh fetch must cover the entire stored history range (so the
    # scale comparison is meaningful).  It does not need to reach end_date:
    # on weekends/holidays no new bars exist yet; the incremental path
    # below handles the tail.
    fresh_min = new_df["date"].min()
    fresh_max = new_df["date"].max()
    if not existing.empty:
        need_min = pd.to_datetime(existing["date"]).dt.date.min()
        need_max = pd.to_datetime(existing["date"]).dt.date.max()
        if fresh_min > need_min or fresh_max < need_max:
            return FetchResult(
                symbol=symbol, success=False,
                rows_total=len(existing),
                error=f"Full-history refetch does not cover stored range "
                      f"[{need_min}, {need_max}] (got [{fresh_min}, {fresh_max}]); old file kept",
            )
        missing_dates = set(pd.to_datetime(existing.date).dt.date) - set(new_df.date)
        if missing_dates:
            return FetchResult(symbol=symbol, success=False, rows_total=len(existing),
                               error=f"Full-history refetch missing {len(missing_dates)} stored dates; old file kept")
    ohlc_cols = ["open", "high", "low", "close"]
    if (new_df[ohlc_cols + ["volume"]].isna().any().any()
            or (new_df[ohlc_cols] <= 0).any().any() or (new_df.volume < 0).any()):
        return FetchResult(
            symbol=symbol, success=False,
            rows_total=len(existing),
            error="Full-history refetch has null/non-positive OHLCV; old file kept",
        )

    # --- QFQ scale consistency check over the FULL overlap ---
    consistent, mismatched, max_diff = verify_qfq_scale(existing, new_df)
    if consistent and not existing.empty:
        compare_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        old = existing[compare_cols].copy().sort_values("date").reset_index(drop=True)
        old["date"] = pd.to_datetime(old.date).dt.date
        same_bars = old.equals(new_df[compare_cols])
    else:
        same_bars = False
    if same_bars:
        _record_deep_verify(store_root, symbol)
        return FetchResult(
            symbol=symbol, success=True,
            rows_new=0, rows_total=len(existing),
            last_date=pd.to_datetime(existing["date"]).dt.date.max() if not existing.empty else fresh_max,
            error="scale_ok",
        )

    # Scale differs: replace the whole history with the fresh fetch.
    # The fresh QFQ series is self-consistent; splicing old-scale rows
    # would create an artificial jump (the 300803 defect).
    if not consistent:
        logger.warning("%s: QFQ scale changed (%d mismatched dates, max diff %.2f) — replacing full history",
                       symbol, len(mismatched), max_diff)
    else:
        logger.info("%s: complete history updated, including new/missing bars", symbol)
    write_ohlcv(new_df, path, symbol)
    _record_deep_verify(store_root, symbol)

    rows_total = len(pd.read_parquet(path))
    return FetchResult(
        symbol=symbol, success=True,
        rows_new=len(new_df), rows_total=rows_total,
        last_date=fresh_max,
        error=f"scale_refreshed:{len(mismatched)}" if not consistent else "history_refreshed",
    )


def _record_deep_verify(store_root: Path | str, symbol: str) -> None:
    """Record that a deep QFQ verification completed for *symbol* today."""
    sidecar = _verify_sidecar_path(store_root, symbol)
    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        import json
        sidecar.write_text(
            json.dumps({"last_deep_verify": dt.date.today().isoformat()}),
            encoding="utf-8",
        )
    except Exception as exc:  # non-fatal bookkeeping
        logger.debug("Could not write verify sidecar for %s: %s", symbol, exc)


def _deep_verify_due(store_root: Path | str, symbol: str) -> bool:
    """True when no deep verification ran within the interval."""
    sidecar = _verify_sidecar_path(store_root, symbol)
    if not sidecar.exists():
        return True
    try:
        import json
        last = dt.date.fromisoformat(
            json.loads(sidecar.read_text(encoding="utf-8"))["last_deep_verify"]
        )
        return (dt.date.today() - last).days >= _DEEP_VERIFY_INTERVAL_DAYS
    except Exception:
        return True


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


def qfq_as_of(raw: pd.DataFrame, factors: pd.DataFrame, end_date: str) -> pd.DataFrame:
    """Anchor Sina forward adjustment to the requested cutoff, not fetch day.

    Sina revises historical qfq factors when a later corporate action occurs.
    Its ready-made qfq endpoint then changes even the cutoff day's close.
    Rebuilding from unadjusted bars with factor(cutoff)/factor(bar) cancels
    that later normalization and keeps the cutoff close on its traded scale.
    """
    cutoff = pd.Timestamp(end_date)
    prices = raw.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices.date <= cutoff].sort_values("date")
    factor = factors[["date", "qfq_factor"]].copy()
    factor["date"] = pd.to_datetime(factor.date)
    factor["qfq_factor"] = pd.to_numeric(factor.qfq_factor, errors="raise")
    factor = factor[factor.date <= cutoff].sort_values("date")
    if factor.empty or factor.date.duplicated().any() or factor.qfq_factor.isna().any() or (factor.qfq_factor <= 0).any():
        raise ValueError("Missing, duplicate or invalid as-of QFQ factors")
    anchor = float(factor.qfq_factor.iloc[-1])
    merged = pd.merge_asof(prices, factor, on="date", direction="backward")
    if merged.qfq_factor.isna().any():
        raise ValueError("QFQ factors do not cover the complete requested price history")
    for col in ("open", "high", "low", "close"):
        merged[col] = (merged[col] * anchor / merged.qfq_factor).round(2)
    merged["date"] = merged.date.dt.date
    return merged.drop(columns="qfq_factor")


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

    # Fetch unadjusted bars and explicitly anchor the factor series. Asking
    # the ready-made qfq endpoint for a historical end_date is insufficient.
    df = safe_call(
        ak.stock_zh_a_daily,
        symbol=mi.prefixed,
        start_date=start_date,
        end_date=end_date,
        adjust="",
        label=f"stock_zh_a_daily {symbol}",
        retries=3,
    )
    if df is not None and not df.empty:
        raw = _normalise_daily(df, symbol)
    else:
        raw = None

    if raw is None:
        logger.warning("stock_zh_a_daily failed for %s, trying stock_zh_a_hist", symbol)

        # EM fallback also starts with unadjusted bars; it shares the explicit
        # factor cutoff instead of silently returning fetch-day qfq prices.
        df = safe_call(
            ak.stock_zh_a_hist,
            symbol=symbol, period="daily", start_date=start_date,
            end_date=end_date, adjust="", label=f"stock_zh_a_hist {symbol}", retries=3,
        )
        if df is not None and not df.empty:
            raw = _normalise_hist(df, symbol)

    if raw is not None:
        factors = safe_call(
            ak.stock_zh_a_daily, symbol=mi.prefixed, adjust="qfq-factor",
            label=f"Sina as-of QFQ factors {symbol}", retries=3,
        )
        if factors is not None and not factors.empty:
            try:
                return qfq_as_of(raw, factors, end_date)
            except (ValueError, KeyError, TypeError) as exc:
                logger.error("%s: as-of QFQ conversion failed: %s", symbol, exc)
        # Never use an unanchored qfq fallback after a factor failure.

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
    *,
    force: bool = False,
    deep_verify: bool | None = None,
) -> FetchResult:
    """
    Fetch OHLCV for *symbol* and write/append to the silver lake.

    Incremental: if existing data is found, only fetches from
    (last_stored_date + 1) onward and appends.

    force=True bypasses the up-to-date skip and requests full verification.
    With explicit deep_verify=False only, the legacy 10-day overlap is used.

    deep_verify: when True, run a full-history QFQ scale consistency check
    (detects ex-rights/ex-dividend outside any tail window, e.g. 300803).
    When None (default), the check auto-triggers if no deep verification ran
    within _DEEP_VERIFY_INTERVAL_DAYS.  When False, skip the check.

    Returns a FetchResult describing the outcome.
    """
    store_root = Path(store_root)
    path = ohlcv_path(store_root, symbol)

    # --- Automatic deep QFQ verification (P0: 300803 defect) ---
    # A tail-only overlap cannot detect a scale change whose ex-date is
    # outside the fetched window.  The deep check compares the FULL overlap.
    # Skipped when there is no stored history (nothing to compare against).
    existing_probe = read_ohlcv(path)
    want_deep = False
    if not existing_probe.empty:
        want_deep = deep_verify if deep_verify is not None else (
            force or pd.to_datetime(existing_probe.date).dt.date.max() < end_date
            or _deep_verify_due(store_root, symbol)
        )
    if want_deep:
        deep_result = refresh_symbol_full_history(symbol, store_root, start_date, end_date)
        if not deep_result.success:
            # Full-history refetch failed: keep old file, report failure.
            # Do NOT fall through to a tail append that would splice scales.
            return deep_result
        # The verified full fetch also supplies new/missing bars. Never discard
        # it and issue a second tail fetch on a potentially different scale.
        return deep_result

    # --- Read existing data to determine fetch window ---
    existing = existing_probe
    if not existing.empty:
        last_stored = pd.to_datetime(existing["date"]).dt.date.max()
        if last_stored >= end_date and not force:
            logger.info("%s: already up to date (last=%s)", symbol, last_stored)
            return FetchResult(
                symbol=symbol, success=True,
                rows_new=0, rows_total=len(existing),
                last_date=last_stored, error="skipped",
            )
        if force:
            # Overlap window: re-fetch recent bars so corrections overwrite.
            fetch_start = max(start_date, end_date - dt.timedelta(days=10))
        else:
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
        if force:
            # Fresh rows first: enforce_ohlcv keeps the FIRST row on
            # (symbol, date) duplicates, so re-fetched bars overwrite.
            combined = pd.concat([new_df, existing], ignore_index=True)
        else:
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
    force : bool
        Re-fetch even symbols already up to date (10-day overlap window,
        fresh rows win).  Default False (incremental skip).
    deep_verify : bool | None
        Full-history QFQ scale check per symbol.  None (default) auto-triggers
        when the last deep verification is older than the interval.
    """

    def __init__(
        self,
        store_root: Path | str,
        start_date: dt.date | str | None = None,
        end_date:   dt.date | str | None = None,
        max_workers: int = 1,
        force: bool = False,
        deep_verify: bool | None = None,
    ) -> None:
        self.store_root  = Path(store_root)
        self.start_date  = _as_date(start_date) if start_date else (
            dt.date.today() - dt.timedelta(days=365 * 3)
        )
        self.end_date    = _as_date(end_date) if end_date else dt.date.today()
        self.max_workers = max_workers
        self.force       = force
        self.deep_verify = deep_verify
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
            return collect_symbol(symbol, self.store_root, self.start_date, self.end_date,
                                  force=self.force, deep_verify=self.deep_verify)
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
