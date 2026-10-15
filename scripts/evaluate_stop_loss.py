#!/usr/bin/env python3
"""Historical stop-loss counterfactual for frozen-baseline picks.

Reads a picks CSV (``symbol,date,trade_date,close,horizon,...`` as produced
by ``compare_original_hybrid.py``), applies the PRE-DECLARED stop-loss rule
(``stop-loss-v1``: 8% primary + 5% sensitivity, fixed 2026-10-15) to the SAME
picks, and reports the paired comparison against fixed-horizon returns.

This is DESCRIPTIVE (in-sample), not confirmatory.  The confirmatory test is
the forward 4-week parallel run.  Both thresholds are always reported; the
rule constants live in ``quant_platform.evaluation.stop_loss`` and are
guarded by ``tests/test_stop_loss.py::test_thresholds_are_pre_registered``.

Usage:
    python scripts/evaluate_stop_loss.py \
        --picks docs/reviews/2026-10-15-ffae653-acceptance/D3_ordinary_picks.csv \
        --store-root /path/to/ohlcv_store \
        --horizon 3 \
        --out /tmp/sl_D3_ordinary.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from quant_platform.evaluation.stop_loss import (
    STOP_LOSS_LEVELS,
    STOP_LOSS_RULE_VERSION,
    paired_summary,
    stop_return_for_pick,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--picks", required=True, help="Picks CSV with symbol,date,trade_date,horizon")
    p.add_argument("--store-root", required=True, help="OHLCV store root")
    p.add_argument("--horizon", type=int, default=None,
                   help="Holding horizon in trading days (default: from picks 'horizon' col)")
    p.add_argument("--out", required=True, help="Per-pick output CSV path")
    p.add_argument("--summary", default=None, help="Optional JSON summary path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    picks = pd.read_csv(args.picks)
    # BOM-tolerant column names (some picks CSVs carry a BOM).
    picks.columns = [c.lstrip("\ufeff") for c in picks.columns]
    required = {"symbol", "trade_date"}
    missing = required - set(picks.columns)
    if missing:
        raise SystemExit(f"picks CSV missing columns: {sorted(missing)}")

    store_root = Path(args.store_root)
    records = []
    for _, row in picks.iterrows():
        symbol = str(row["symbol"])
        td = row["trade_date"]
        trade_date = td.date() if hasattr(td, "date") else pd.to_datetime(td).date()
        horizon = args.horizon or int(row.get("horizon", 0) or 0)
        if not horizon:
            raise SystemExit("horizon not given and picks CSV has no usable 'horizon' column")
        for stop_pct in STOP_LOSS_LEVELS:
            rec = stop_return_for_pick(
                symbol, trade_date, horizon, stop_pct, store_root
            )
            rec["signal_date"] = str(row.get("date", ""))
            records.append(rec)

    out_df = pd.DataFrame(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("wrote %d rows -> %s", len(out_df), out_path)

    # Paired summaries, one per pre-declared threshold (all reported).
    summaries = []
    for stop_pct in STOP_LOSS_LEVELS:
        sub = [r for r in records if abs(float(r["stop_pct"]) - stop_pct) < 1e-12]
        s = paired_summary(sub)
        summaries.append(s)
        print(f"\n=== stop-loss {stop_pct:.0%} ({STOP_LOSS_RULE_VERSION}) ===")
        print(json.dumps(s, indent=2, default=str))
    if args.summary:
        sp = Path(args.summary)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summaries, indent=2, default=str))
        logger.info("summary -> %s", sp)


if __name__ == "__main__":
    main()
