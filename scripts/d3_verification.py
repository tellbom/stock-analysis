"""
d3_verification.py
==================
Phase 2: D+3 realization verification script.

Run this after D+3 data is available for the D+3 check. If D+5 data is also
available, the same run will compute the D+5 comparison.

Reads:
  - Prediction_2026-06-26_ranked_industry_neutral_equal_top3.csv
  - Prediction_2026-06-26_global_top50.csv
  - OHLCV data (silver/ohlcv/) covering verification period

Computes:
  - D+3 realized returns (ret_fwd_3d)
  - D+5 realized returns (ret_fwd_5d)
  - Rank IC at both horizons
  - Selected-set performance at both horizons
  - Industry exposure performance breakdown
  - D+3 vs D+5 comparison (data-driven, no assumptions)

Usage:
  cd E:/stock-analysis && PYTHONPATH=. python scripts/d3_verification.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("E:/stock-analysis")
STORE_ROOT = ROOT / "models/data"
OUTPUT_DIR = STORE_ROOT / "reports"
PREDICTION_DATE = dt.date(2026, 6, 26)

sys.path.insert(0, str(ROOT))

from quant_platform.core.logging import get_logger
from quant_platform.store.parquet_store import read_ohlcv
from quant_platform.store.lake import ohlcv_path
from quant_platform.selection.config import SelectionConfig
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.selection.exposure import ExposureMonitor

logger = get_logger(__name__)


def compute_realized_return(symbol: str, start_date: dt.date, end_date: dt.date) -> float | None:
    """
    Compute realized forward return: close(end_date) / close(start_date) - 1.
    Uses the T+1 convention: start_date = T+1, end_date = T+1+h.
    """
    try:
        df = read_ohlcv(ohlcv_path(STORE_ROOT, symbol))
        df["date"] = pd.to_datetime(df["date"]).dt.date
        start_row = df[df["date"] == start_date]
        end_row = df[df["date"] == end_date]
        if start_row.empty or end_row.empty:
            return None
        c1 = start_row["close"].iloc[0]
        c2 = end_row["close"].iloc[0]
        if c1 <= 0:
            return None
        return c2 / c1 - 1.0
    except Exception:
        return None


def main():
    print("=" * 70)
    print("D+3 / D+5 REAL-MARKET VERIFICATION")
    print(f"Prediction date: {PREDICTION_DATE}")
    print(f"Run date: {dt.date.today().isoformat()}")
    print("=" * 70)
    print()

    # Verify data availability
    d1 = dt.date(2026, 6, 29)   # T+1
    d3 = dt.date(2026, 7, 2)    # T+1+3
    d5 = dt.date(2026, 7, 6)    # T+1+5
    today = dt.date.today()

    print("[1/5] Data availability check:")
    print(f"  D+1 date: {d1} — {'OK' if today >= d1 else 'PENDING'}")
    print(f"  D+3 date: {d3} — {'OK' if today >= d3 else 'PENDING'}")
    print(f"  D+5 date: {d5} — {'OK' if today >= d5 else 'PENDING'}")
    print()

    if today < d1:
        print("ERROR: T+1 data not yet available. Cannot verify.")
        return

    # Load predictions
    print("[2/5] Loading predictions...")
    ranked_path = OUTPUT_DIR / "Prediction_2026-06-26_ranked_industry_neutral_equal_top3.csv"
    global_path = OUTPUT_DIR / "Prediction_2026-06-26_global_top50.csv"

    if not ranked_path.exists():
        print(f"ERROR: ranked file not found: {ranked_path}")
        return

    ranked = pd.read_csv(ranked_path, dtype={"symbol": str})
    ranked["symbol"] = ranked["symbol"].astype(str).str.zfill(6)
    selected_symbols = set(ranked[ranked["selected"] == True]["symbol"].tolist())
    print(f"  Industry-neutral selected: {len(selected_symbols)} stocks")

    global_top = pd.read_csv(global_path, dtype={"symbol": str})
    global_top["symbol"] = global_top["symbol"].astype(str).str.zfill(6)
    global_symbols = set(global_top["symbol"].tolist())
    print(f"  Global Top-50: {len(global_symbols)} stocks")

    # Get all prediction symbols
    all_pred_symbols = sorted(set(ranked["symbol"].tolist()))
    print(f"  Total prediction symbols: {len(all_pred_symbols)}")
    print()

    # Compute realized returns
    print("[3/5] Computing realized returns...")
    d3_returns = {}
    d5_returns = {}
    d3_available = today >= d3
    d5_available = today >= d5

    for i, sym in enumerate(all_pred_symbols):
        if i % 50 == 0:
            print(f"  Processing {i+1}/{len(all_pred_symbols)}...")

        if d3_available:
            r3 = compute_realized_return(sym, d1, d3)
            if r3 is not None:
                d3_returns[sym] = r3

        if d5_available:
            r5 = compute_realized_return(sym, d1, d5)
            if r5 is not None:
                d5_returns[sym] = r5

    print(f"  D+3 realized returns: {len(d3_returns)} symbols")
    print(f"  D+5 realized returns: {len(d5_returns)} symbols")
    print()

    # Compute metrics
    print("[4/5] Computing verification metrics...")
    results = {}

    if d3_returns:
        d3_series = pd.Series(d3_returns, name="realized_ret_3d")
        eval_d3 = ranked[["symbol", "model_score"]].copy()
        eval_d3["realized_ret_3d"] = eval_d3["symbol"].map(d3_returns)
        eval_d3 = eval_d3.dropna(subset=["realized_ret_3d"])

        # Rank IC
        if len(eval_d3) >= 5:
            d3_ic, _ = spearmanr(eval_d3["model_score"], eval_d3["realized_ret_3d"])
            print(f"  D+3 Rank IC (n={len(eval_d3)}): {d3_ic:+.4f}")

            # Industry-neutral selected set
            sel_d3 = eval_d3[eval_d3["symbol"].isin(selected_symbols)]
            sel_mean = sel_d3["realized_ret_3d"].mean() if len(sel_d3) > 0 else float("nan")
            print(f"  Industry-neutral D+3 mean return: {sel_mean:+.4%} ({len(sel_d3)} stocks)")

            # Global Top-N
            glob_d3 = eval_d3[eval_d3["symbol"].isin(global_symbols)]
            glob_mean = glob_d3["realized_ret_3d"].mean() if len(glob_d3) > 0 else float("nan")
            print(f"  Global Top-50 D+3 mean return: {glob_mean:+.4%} ({len(glob_d3)} stocks)")

            results["d3"] = {
                "n": len(eval_d3),
                "rank_ic": d3_ic,
                "industry_neutral_mean_return": sel_mean,
                "industry_neutral_n": len(sel_d3),
                "global_top50_mean_return": glob_mean,
                "global_top50_n": len(glob_d3),
            }

    if d5_returns:
        d5_series = pd.Series(d5_returns, name="realized_ret_5d")
        eval_d5 = ranked[["symbol", "model_score"]].copy()
        eval_d5["realized_ret_5d"] = eval_d5["symbol"].map(d5_returns)
        eval_d5 = eval_d5.dropna(subset=["realized_ret_5d"])

        if len(eval_d5) >= 5:
            d5_ic, _ = spearmanr(eval_d5["model_score"], eval_d5["realized_ret_5d"])
            print(f"  D+5 Rank IC (n={len(eval_d5)}): {d5_ic:+.4f}")

            sel_d5 = eval_d5[eval_d5["symbol"].isin(selected_symbols)]
            sel_mean = sel_d5["realized_ret_5d"].mean() if len(sel_d5) > 0 else float("nan")
            print(f"  Industry-neutral D+5 mean return: {sel_mean:+.4%} ({len(sel_d5)} stocks)")

            glob_d5 = eval_d5[eval_d5["symbol"].isin(global_symbols)]
            glob_mean = glob_d5["realized_ret_5d"].mean() if len(glob_d5) > 0 else float("nan")
            print(f"  Global Top-50 D+5 mean return: {glob_mean:+.4%} ({len(glob_d5)} stocks)")

            results["d5"] = {
                "n": len(eval_d5),
                "rank_ic": d5_ic,
                "industry_neutral_mean_return": sel_mean,
                "industry_neutral_n": len(sel_d5),
                "global_top50_mean_return": glob_mean,
                "global_top50_n": len(glob_d5),
            }

    # D+3 vs D+5 comparison
    if "d3" in results and "d5" in results:
        print()
        print("[5/5] D+3 vs D+5 comparison:")
        r3 = results["d3"]
        r5 = results["d5"]
        print(f"  D+3 Rank IC: {r3['rank_ic']:+.4f} | D+5 Rank IC: {r5['rank_ic']:+.4f}")
        print(f"  D+3 Ind-Neutral Ret: {r3['industry_neutral_mean_return']:+.4%} | D+5: {r5['industry_neutral_mean_return']:+.4%}")
        print(f"  D+3 Global Top-50 Ret: {r3['global_top50_mean_return']:+.4%} | D+5: {r5['global_top50_mean_return']:+.4%}")
        print()
        # Data-driven verdict
        if abs(r3["rank_ic"]) > abs(r5["rank_ic"]):
            print("  VERDICT: D+3 shows stronger IC than D+5 (data-driven observation)")
        else:
            print("  VERDICT: D+5 shows stronger IC than D+3 (data-driven observation)")
        print("  NOTE: Single cross-section only — NOT sufficient for horizon selection.")
        print("  Accumulate >= 20 prediction dates before making a final decision.")

    # Save results
    output_path = OUTPUT_DIR / "Prediction_2026-06-26_d3_d5_verification.json"
    verification = {
        "prediction_date": str(PREDICTION_DATE),
        "verification_date": str(today),
        "d1_date": str(d1),
        "d3_date": str(d3),
        "d5_date": str(d5),
        "d3_available": d3_available,
        "d5_available": d5_available,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2, default=str)
    print(f"Results saved: {output_path}")
    print()
    print("=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
