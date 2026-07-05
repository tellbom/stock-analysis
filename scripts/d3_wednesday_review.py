"""
d3_wednesday_review.py
======================
Wednesday D+3 verification. Run AFTER market close on 2026-07-08.

Reads the D+3 prediction output, fetches or loads realized OHLCV data for
July 6-8, 2026, and computes:
  - D+3 realized returns
  - Rank IC (prediction vs realized)
  - Hit rate (fraction of Top 50 with positive D+3 return)
  - Industry-neutral vs global baseline comparison
  - Pass/fail against criteria

Usage:
  cd E:/stock-analysis && PYTHONPATH=. python scripts/d3_wednesday_review.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("E:/stock-analysis")
STORE_ROOT = ROOT / "models/data"
OUTPUT_DIR = STORE_ROOT / "reports"

PREDICTION_DATE = dt.date(2026, 7, 3)
D1_DATE = dt.date(2026, 7, 6)
D3_DATE = dt.date(2026, 7, 8)

sys.path.insert(0, str(ROOT))

from quant_platform.store.lake import ohlcv_path
from quant_platform.store.parquet_store import read_ohlcv


def main():
    today = dt.date.today()
    print("=" * 70)
    print("WEDNESDAY D+3 VERIFICATION")
    print(f"Prediction: {PREDICTION_DATE}")
    print(f"Review date: {today}")
    print("=" * 70)
    print()

    # 1. Check data availability
    print("[1/5] Data availability check...")
    if today < D3_DATE:
        print(f"  ERROR: D+3 date ({D3_DATE}) has not occurred yet. Today is {today}.")
        print(f"  Run this script on or after {D3_DATE} after market close.")
        return
    print(f"  D+3 date ({D3_DATE}) has passed. Proceeding.")
    print()

    # 2. Load prediction
    print("[2/5] Loading predictions...")
    label = f"D3_Prediction_{PREDICTION_DATE.isoformat()}"
    ranked_path = OUTPUT_DIR / f"{label}_ranked.csv"
    global_path = OUTPUT_DIR / f"{label}_global_top50.csv"

    if not ranked_path.exists():
        print(f"  ERROR: {ranked_path} not found")
        return

    ranked = pd.read_csv(ranked_path, dtype={"symbol": str})
    ranked["symbol"] = ranked["symbol"].astype(str).str.zfill(6)
    selected = set(ranked[ranked["selected"] == True]["symbol"].tolist())
    print(f"  Industry-neutral selected: {len(selected)} stocks")

    global_top = pd.read_csv(global_path, dtype={"symbol": str})
    global_top["symbol"] = global_top["symbol"].astype(str).str.zfill(6)
    global_set = set(global_top["symbol"].tolist())
    print(f"  Global Top-50: {len(global_set)} stocks")
    print()

    # 3. Fetch/load realized OHLCV
    print("[3/5] Loading realized OHLCV data...")
    realized = {}
    all_syms = sorted(set(ranked["symbol"].tolist()))

    # Try reading from silver/ohlcv first
    for sym in all_syms:
        df = read_ohlcv(ohlcv_path(STORE_ROOT, sym))
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.date
        d1_row = df[df["date"] == D1_DATE]
        d3_row = df[df["date"] == D3_DATE]
        if not d1_row.empty and not d3_row.empty:
            c1 = d1_row["close"].iloc[0]
            c3 = d3_row["close"].iloc[0]
            if c1 > 0:
                realized[sym] = c3 / c1 - 1.0

    print(f"  Realized D+3 returns: {len(realized)}/{len(all_syms)} symbols")
    if len(realized) < 10:
        print("  ERROR: Insufficient realized data. Fetch OHLCV for July 6-8 first.")
        print(f"  Run: python scripts/extend_ohlcv_simple.py with dates 20260706-20260709")
        return
    print()

    # 4. Compute metrics
    print("[4/5] Computing verification metrics...")
    eval_df = ranked[["symbol", "model_score"]].drop_duplicates(subset="symbol").copy()
    eval_df["realized_ret_3d"] = eval_df["symbol"].map(realized)
    eval_df = eval_df.dropna(subset=["realized_ret_3d"])

    if len(eval_df) < 5:
        print("  ERROR: Too few symbols with realized returns")
        return

    # Rank IC
    rank_ic, _ = spearmanr(eval_df["model_score"], eval_df["realized_ret_3d"])
    print(f"  D+3 Rank IC: {rank_ic:+.4f}")

    # Industry-neutral selected set
    sel_df = eval_df[eval_df["symbol"].isin(selected)]
    sel_mean = sel_df["realized_ret_3d"].mean()
    sel_hit = (sel_df["realized_ret_3d"] > 0).mean()
    print(f"  Industry-neutral D+3 mean return: {sel_mean:+.4%}")
    print(f"  Industry-neutral hit rate: {sel_hit:.1%} ({int(sel_hit*len(sel_df))}/{len(sel_df)})")

    # Global Top-50
    glob_df = eval_df[eval_df["symbol"].isin(global_set)]
    glob_mean = glob_df["realized_ret_3d"].mean()
    glob_hit = (glob_df["realized_ret_3d"] > 0).mean()
    print(f"  Global Top-50 D+3 mean return: {glob_mean:+.4%}")
    print(f"  Global Top-50 hit rate: {glob_hit:.1%}")
    print()

    # 5. Pass/fail
    print("[5/5] Pass/fail assessment...")
    print()
    checks = {
        "Rank IC > 0": rank_ic > 0,
        "Industry-neutral mean return > 0": sel_mean > 0,
        "Hit rate > 50%": sel_hit > 0.5,
        "Industry-neutral > Global baseline": sel_mean > glob_mean,
    }

    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check}")

    print()
    if all_pass:
        print("  OVERALL: ALL CHECKS PASSED")
    else:
        print("  OVERALL: SOME CHECKS FAILED — review before next prediction")

    # Save report
    result = {
        "prediction_date": str(PREDICTION_DATE),
        "review_date": str(today),
        "d1_date": str(D1_DATE),
        "d3_date": str(D3_DATE),
        "n_symbols": len(eval_df),
        "rank_ic": rank_ic,
        "industry_neutral": {
            "mean_return": sel_mean,
            "hit_rate": sel_hit,
            "n": len(sel_df),
        },
        "global_top50": {
            "mean_return": glob_mean,
            "hit_rate": glob_hit,
            "n": len(glob_df),
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "all_pass": all_pass,
    }
    out_path = OUTPUT_DIR / f"{label}_wednesday_review.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
