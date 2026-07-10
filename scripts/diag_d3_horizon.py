"""
diag_d3_horizon.py — one-off diagnostic (D3 line, 2026-07-09).

Runs WalkForwardEvaluator on the SAME accumulated panel for ret_fwd_3d and
ret_fwd_5d, so we can decide model-vs-strategy on multi-window evidence rather
than the single 2026-07-03 live window (Rank IC -0.2507).

Reuses the production panel builder (feature_set 80fd2338) and the Ridge
pipeline that the D3 model actually uses. Throwaway — delete after reading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "E:/stock-analysis")

from scripts.current_week_prediction import build_prediction_panel, load_universe_symbols
from scripts.run_walk_forward_verdict import run_primary_verdict

STORE_ROOT = Path("E:/stock-analysis/models/data")
FEATURE_SET_ID = "80fd2338"          # current DEFAULT_SPECS set, has reversal_3d
N_WINDOWS = 5
WINDOW_MONTHS = 6                    # 12m yields only ~2 windows on current history

META_COLS = {
    "symbol", "date", "close", "volume", "open", "high", "low", "amount",
    "outstanding_share", "turnover", "industry_code", "industry_name", "concept_tags",
}
LABEL_PREFIXES = ("ret_fwd_", "vol_fwd_", "mdd_fwd_", "excess_vs_")


def ridge_factory():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])


def main():
    symbols = load_universe_symbols()
    print(f"symbols: {len(symbols)}")
    panel = build_prediction_panel(symbols, FEATURE_SET_ID)
    feature_cols = [
        c for c in panel.columns
        if c not in META_COLS and not c.startswith(LABEL_PREFIXES)
    ]
    # Drop near-empty feature columns (production does the same >90% NaN drop).
    # Without this, dropna over a sparse column nukes the whole panel.
    nan_frac = panel[feature_cols].isna().mean()
    feature_cols = [c for c in feature_cols if nan_frac[c] < 0.90]
    print(f"panel rows: {len(panel)}  feature_cols (after NaN filter): {len(feature_cols)}")
    print(f"date range: {panel['date'].min()} .. {panel['date'].max()}")

    results = {}
    for label_col, horizon in [("ret_fwd_3d", 3), ("ret_fwd_5d", 5),
                               ("ret_fwd_10d", 10), ("ret_fwd_20d", 20)]:
        # dropna on the LABEL only -- the pipeline's SimpleImputer handles
        # residual feature NaNs; dropping feature-NaN rows here would bias
        # coverage between horizons.
        wf_panel = panel.dropna(subset=[label_col]).copy()
        res = run_primary_verdict(
            wf_panel, feature_cols, label_col,
            horizon=horizon, n_windows=N_WINDOWS, window_months=WINDOW_MONTHS,
            model_factory=ridge_factory, store_root=STORE_ROOT,
        )
        results[label_col] = res
        print("\n" + "=" * 64)
        print(f"{label_col} (horizon={horizon})  rows={len(wf_panel)}")
        print("=" * 64)
        df = res.to_dataframe()[["window_id", "test_start", "test_end",
                                 "n_test", "rank_ic_mean", "icir", "sharpe"]]
        print(df.to_string(index=False))
        print(f"  agg_rank_ic_mean : {res.agg_rank_ic_mean:+.4f}")
        print(f"  agg_icir         : {res.agg_icir:+.4f}")
        print(f"  agg_sharpe       : {res.agg_sharpe:+.4f}")
        print(f"  ic_sign_stability: {res.ic_sign_stability:.2f} "
              f"(fraction of windows with positive Rank IC)")
        print(f"  n_windows        : {res.n_windows()}")

    print("\n" + "#" * 64)
    print("VERDICT")
    print("#" * 64)
    for label_col in ["ret_fwd_3d", "ret_fwd_5d", "ret_fwd_10d", "ret_fwd_20d"]:
        r = results[label_col]
        print(f"  {label_col:12s} agg IC {r.agg_rank_ic_mean:+.4f} / "
              f"ICIR {r.agg_icir:+.4f} / sign-stab {r.ic_sign_stability:.2f} / "
              f"sharpe {r.agg_sharpe:+.3f}")


if __name__ == "__main__":
    main()
