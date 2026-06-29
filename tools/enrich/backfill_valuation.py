"""Backfill historical valuation silver data from AKShare stock_value_em."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_platform.ingest.universe_service import UniverseService
from quant_platform.ingest.valuation_collector import ValuationCollector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--write", action="store_true", help="Write to silver; default is dry-run")
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store_root = Path(args.store_root)
    symbols = UniverseService(args.universe, store_root).get_symbols_as_of()

    collector = ValuationCollector(store_root)
    report = collector.backfill_history(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        dry_run=not args.write,
        min_success_rate=args.min_success_rate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
