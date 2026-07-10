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

PREDICTION_DATE = dt.date(2026, 7, 3)   # T = signal date
# Holding horizon in trading days. Must match the model's training label
# (ret_fwd_3d).  Per labels/builder.py the realized target is
#     close(T+1+HORIZON) / close(T+1) - 1
# so entry = T+1 and exit = T+1+HORIZON, both derived from the trading
# calendar below -- never hardcode the exit date.  Hardcoding it as 07-08
# scored this 3d model on a 2-day window (off-by-one).
HORIZON = 3

sys.path.insert(0, str(ROOT))

from quant_platform.store.lake import ohlcv_path
from quant_platform.store.parquet_store import read_ohlcv


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="D+H forward verification")
    p.add_argument("--horizon", type=int, default=3,
                   help="holding horizon in trading days (default 3)")
    p.add_argument("--prediction-date", type=dt.date.fromisoformat,
                   default=dt.date(2026, 7, 3), help="signal date T (YYYY-MM-DD)")
    return p.parse_args()


def main():
    global HORIZON, PREDICTION_DATE
    args = _parse_args()
    HORIZON = args.horizon
    PREDICTION_DATE = args.prediction_date

    today = dt.date.today()
    print("=" * 70)
    print(f"D+{HORIZON} VERIFICATION")
    print(f"Prediction: {PREDICTION_DATE}")
    print(f"Review date: {today}")
    print("=" * 70)
    print()

    # 1. Plan (entry/exit dates are derived from the trading calendar in step 3)
    print("[1/5] Review plan...")
    print(f"  Signal date T: {PREDICTION_DATE}   Horizon: {HORIZON} trading days")
    print(f"  Target = close(T+1+{HORIZON}) / close(T+1) - 1  (matches ret_fwd_{HORIZON}d)")
    print()

    # 2. Load prediction
    print("[2/5] Loading predictions...")
    label = f"D{HORIZON}_Prediction_{PREDICTION_DATE.isoformat()}"
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

    # 3. Load realized OHLCV and derive entry/exit from the trading calendar
    print("[3/5] Loading realized OHLCV data...")
    all_syms = sorted(set(ranked["symbol"].tolist()))

    closes = {}          # sym -> close Series indexed by trade date
    all_dates: set = set()
    for sym in all_syms:
        df = read_ohlcv(ohlcv_path(STORE_ROOT, sym))
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.date
        s = df.set_index("date")["close"]
        s = s[~s.index.duplicated(keep="last")]
        closes[sym] = s
        all_dates.update(s.index)

    # Trading calendar = post-signal trading days (union across the universe).
    calendar = sorted(d for d in all_dates if d > PREDICTION_DATE)
    if len(calendar) < HORIZON + 1:
        print(f"  ERROR: only {len(calendar)} post-signal trading day(s) available, "
              f"need {HORIZON + 1} (T+1 .. T+1+{HORIZON}).")
        print(f"  Backfill OHLCV after market close, then re-run.")
        return
    entry_date = calendar[0]            # T+1  (execution / denominator)
    exit_date = calendar[HORIZON]       # T+1+HORIZON  (numerator)
    print(f"  Entry (T+1): {entry_date}    Exit (T+1+{HORIZON}): {exit_date}")

    # Guard against a partially-backfilled exit day (e.g. run intraday before
    # close): it would silently shrink n and yield a misleading IC.
    exit_cov = sum(1 for s in closes.values() if exit_date in s.index)
    if exit_cov < 0.9 * len(closes):
        print(f"  ERROR: exit date {exit_date} coverage only {exit_cov}/{len(closes)} "
              f"-- data looks incomplete (market not closed / not backfilled). Aborting.")
        return

    realized = {}
    for sym, s in closes.items():
        if entry_date in s.index and exit_date in s.index:
            c1 = s.loc[entry_date]
            c3 = s.loc[exit_date]
            if c1 > 0:
                realized[sym] = c3 / c1 - 1.0

    print(f"  Realized {HORIZON}d returns: {len(realized)}/{len(all_syms)} symbols")
    if len(realized) < 10:
        print("  ERROR: Insufficient realized data.")
        return
    print()

    # 4. Compute metrics
    print("[4/5] Computing verification metrics...")
    eval_df = ranked[["symbol", "model_score"]].drop_duplicates(subset="symbol").copy()
    eval_df["realized_ret"] = eval_df["symbol"].map(realized)
    eval_df = eval_df.dropna(subset=["realized_ret"])

    if len(eval_df) < 5:
        print("  ERROR: Too few symbols with realized returns")
        return

    # Rank IC
    rank_ic, _ = spearmanr(eval_df["model_score"], eval_df["realized_ret"])
    print(f"  D+{HORIZON} Rank IC: {rank_ic:+.4f}")

    # Industry-neutral selected set
    sel_df = eval_df[eval_df["symbol"].isin(selected)]
    sel_mean = sel_df["realized_ret"].mean()
    sel_hit = (sel_df["realized_ret"] > 0).mean()
    print(f"  Industry-neutral D+{HORIZON} mean return: {sel_mean:+.4%}")
    print(f"  Industry-neutral hit rate: {sel_hit:.1%} ({int(sel_hit*len(sel_df))}/{len(sel_df)})")

    # Global Top-50
    glob_df = eval_df[eval_df["symbol"].isin(global_set)]
    glob_mean = glob_df["realized_ret"].mean()
    glob_hit = (glob_df["realized_ret"] > 0).mean()
    print(f"  Global Top-50 D+{HORIZON} mean return: {glob_mean:+.4%}")
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
        "horizon_days": HORIZON,
        "entry_date": str(entry_date),
        "exit_date": str(exit_date),
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
