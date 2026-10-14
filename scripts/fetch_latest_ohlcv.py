"""
fetch_latest_ohlcv.py
=====================
Incrementally fetch the latest OHLCV tail for all CSI 300 symbols and
append to silver/ohlcv/.

Thin wrapper around ``ingest.ohlcv_collector.OHLCVCollector``, the single
source of truth for "how do we fetch OHLCV" — no parallel implementation.

``extend_ohlcv_simple.py`` is a deprecated shim that calls this file's
``main()`` — kept only so any existing invocation of the old script name
still works.

本机执行命令（Codex / Windows 本机，仓库根目录下）::

    cd <本机 stock-analysis 目录>
    python scripts/fetch_latest_ohlcv.py --as-of 2026-10-14

常用变体::

    # 只检查覆盖度，不实际拉取（dry-run 需配合 --workers 0? 不支持；直接看输出即可）
    python scripts/fetch_latest_ohlcv.py --as-of 2026-10-14 --workers 1

    # 强制重拉最近 10 天（含已是最新），覆盖修正被修订的 bar
    python scripts/fetch_latest_ohlcv.py --as-of 2026-10-14 --force

    # 指定数据湖根目录（默认：仓库下 models/data）
    python scripts/fetch_latest_ohlcv.py --as-of 2026-10-14 --store-root D:/data/models/data

退出码：0 = 全部达标；1 = 有拉取失败或覆盖度未达目标（不要静默跳过）。

环境变量 STOCK_ANALYSIS_ROOT 可覆盖仓库根目录推断。
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("STOCK_ANALYSIS_ROOT", Path(__file__).resolve().parents[1]))
STORE_ROOT = ROOT / "models/data"
sys.path.insert(0, str(ROOT))

from quant_platform.core.logging import get_logger
from quant_platform.ingest.ohlcv_collector import OHLCVCollector
from quant_platform.store.lake import ohlcv_path
from quant_platform.store.parquet_store import read_ohlcv

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest CSI 300 OHLCV data")
    parser.add_argument("--store-root", default=str(STORE_ROOT))
    parser.add_argument(
        "--as-of",
        default="",
        help="目标日期：每个 symbol 的 OHLCV 至少覆盖到该日。默认今天。",
    )
    # 向后兼容：旧参数名 --min-date
    parser.add_argument("--min-date", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fetch-end-date",
        default="",
        help="Collector end date; defaults to target + 3 calendar days.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重拉：即使已是最新也重拉最近 10 天并覆盖（修正被修订的 bar）。",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def resolve_dates(args: argparse.Namespace) -> tuple[dt.date, dt.date]:
    raw = args.as_of or args.min_date
    target = pd.to_datetime(raw).date() if raw else dt.date.today()
    fetch_end_date = (
        pd.to_datetime(args.fetch_end_date).date()
        if args.fetch_end_date
        else target + dt.timedelta(days=3)
    )
    if fetch_end_date < target:
        raise ValueError("--fetch-end-date must be >= --as-of")
    return target, fetch_end_date


def _max_date_or_none(store_root: Path, symbol: str) -> dt.date | None:
    df = read_ohlcv(ohlcv_path(store_root, symbol))
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).dt.date.max()


def main(argv: list[str] | None = None) -> int:
    """返回退出码：0 全部达标；1 有失败或覆盖度不足。"""
    args = parse_args(argv)
    store_root = Path(args.store_root)
    target, fetch_end_date = resolve_dates(args)

    print("=" * 70)
    print("FETCH LATEST OHLCV DATA (via ingest.ohlcv_collector.OHLCVCollector)")
    print(f"Target: at least {target}" + (" [FORCE]" if args.force else ""))
    print("=" * 70)
    print()

    universe_df = pd.read_parquet(store_root / "universe/csi300/membership.parquet")
    symbols = sorted(universe_df["symbol"].tolist())
    print(f"Universe: {len(symbols)} symbols")

    already_ok = [s for s in symbols if (_max_date_or_none(store_root, s) or dt.date.min) >= target]
    print(f"Already up to date: {len(already_ok)}/{len(symbols)}")
    if args.force:
        print("Force mode: already-up-to-date symbols will be re-fetched (10-day overlap).")
    print()

    # OHLCVCollector.run() is already incremental per symbol -- symbols
    # already up to date are skipped internally (FetchResult.error == "skipped")
    # unless --force is given, so there's no need to pre-filter the list.
    collector = OHLCVCollector(
        store_root=store_root,
        end_date=fetch_end_date,
        max_workers=args.workers,  # serial recommended -- AKShare/Sina can rate-limit concurrency
        force=args.force,
    )
    summary = collector.run(symbols)

    print()
    print(f"Done: {summary.succeeded} succeeded, {summary.failed} failed, "
          f"{summary.skipped} already up to date")
    if summary.failed_symbols:
        print(f"Failed symbols ({len(summary.failed_symbols)}): {summary.failed_symbols[:10]}")

    # Verify
    final_ok = sum(1 for s in symbols if (_max_date_or_none(store_root, s) or dt.date.min) >= target)
    print(f"Final coverage: {final_ok}/{len(symbols)} at or beyond {target}")

    # 失败退出逻辑：拉取失败或覆盖度未达目标 -> 非零退出，不静默跳过。
    if summary.failed > 0:
        print(f"ERROR: {summary.failed} symbol(s) failed to fetch.", file=sys.stderr)
        return 1
    if final_ok < len(symbols):
        print(
            f"ERROR: coverage {final_ok}/{len(symbols)} below target {target}; "
            "re-run or investigate failed symbols.",
            file=sys.stderr,
        )
        return 1
    print("OK: all symbols reach the target date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
