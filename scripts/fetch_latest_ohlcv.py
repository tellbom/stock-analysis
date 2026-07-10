"""
fetch_latest_ohlcv.py
=====================
Incrementally fetch the latest OHLCV tail for all CSI 300 symbols and
append to silver/ohlcv/.

FIXED (merge review): this used to hand-roll its own akshare fetch +
market-prefix logic and write parquet directly. That duplicated
``ingest.ohlcv_collector.collect_symbol`` (which already does incremental
fetch, market-prefix resolution via ``core.market.detect_market``, and
retry/back-off via ``core.fetch.safe_call``) and, in the case of
``extend_ohlcv_simple.py``, bypassed ``store.parquet_store.write_ohlcv``'s
schema enforcement and atomic write entirely. This script is now a thin
wrapper around ``ingest.ohlcv_collector.OHLCVCollector``, which is the
single source of truth for "how do we fetch OHLCV" — no more parallel
implementation to keep in sync.

``extend_ohlcv_simple.py`` is now a deprecated shim that calls this file's
``main()`` — kept only so any existing invocation of the old script name
still works.

Usage:
  cd E:/stock-analysis && PYTHONPATH=. python scripts/fetch_latest_ohlcv.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("E:/stock-analysis")
STORE_ROOT = ROOT / "models/data"
sys.path.insert(0, str(ROOT))

from quant_platform.core.logging import get_logger
from quant_platform.ingest.ohlcv_collector import OHLCVCollector
from quant_platform.store.lake import ohlcv_path
from quant_platform.store.parquet_store import read_ohlcv

logger = get_logger(__name__)

# Target: make sure every symbol's OHLCV reaches at least this date.
# OHLCVCollector figures out per-symbol what's actually missing (it reads
# each existing file's last date and only fetches the tail) -- this is
# just the ceiling we report against.
MIN_DATE = dt.date(2026, 7, 3)
# end_date passed to OHLCVCollector -- give it a few extra calendar days of
# room past MIN_DATE so weekends/holidays don't leave it one day short.
FETCH_END_DATE = MIN_DATE + dt.timedelta(days=3)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest CSI 300 OHLCV data")
    parser.add_argument("--store-root", default=str(STORE_ROOT))
    parser.add_argument("--min-date", default=MIN_DATE.isoformat())
    parser.add_argument(
        "--fetch-end-date",
        default="",
        help="Collector end date; defaults to min-date + 3 calendar days.",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def resolve_dates(args: argparse.Namespace) -> tuple[dt.date, dt.date]:
    min_date = pd.to_datetime(args.min_date).date()
    fetch_end_date = (
        pd.to_datetime(args.fetch_end_date).date()
        if args.fetch_end_date
        else min_date + dt.timedelta(days=3)
    )
    if fetch_end_date < min_date:
        raise ValueError("--fetch-end-date must be >= --min-date")
    return min_date, fetch_end_date


def _max_date_or_none(store_root: Path, symbol: str) -> dt.date | None:
    df = read_ohlcv(ohlcv_path(store_root, symbol))
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).dt.date.max()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    store_root = Path(args.store_root)
    min_date, fetch_end_date = resolve_dates(args)

    print("=" * 70)
    print("FETCH LATEST OHLCV DATA (via ingest.ohlcv_collector.OHLCVCollector)")
    print(f"Target: at least {min_date}")
    print("=" * 70)
    print()

    universe_df = pd.read_parquet(store_root / "universe/csi300/membership.parquet")
    symbols = sorted(universe_df["symbol"].tolist())
    print(f"Universe: {len(symbols)} symbols")

    already_ok = [s for s in symbols if (_max_date_or_none(store_root, s) or dt.date.min) >= min_date]
    print(f"Already up to date: {len(already_ok)}/{len(symbols)}")
    print()

    # OHLCVCollector.run() is already incremental per symbol -- symbols
    # already up to date are skipped internally (FetchResult.error == "skipped"),
    # so there's no need to pre-filter the list ourselves.
    collector = OHLCVCollector(
        store_root=store_root,
        end_date=fetch_end_date,
        max_workers=args.workers,  # serial recommended -- AKShare/Sina can rate-limit concurrency
    )
    summary = collector.run(symbols)

    print()
    print(f"Done: {summary.succeeded} succeeded, {summary.failed} failed, "
          f"{summary.skipped} already up to date")
    if summary.failed_symbols:
        print(f"Failed symbols ({len(summary.failed_symbols)}): {summary.failed_symbols[:10]}")

    # Verify
    final_ok = sum(1 for s in symbols if (_max_date_or_none(store_root, s) or dt.date.min) >= min_date)
    print(f"Final coverage: {final_ok}/{len(symbols)} at or beyond {min_date}")


if __name__ == "__main__":
    main()
