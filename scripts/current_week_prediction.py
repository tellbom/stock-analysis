"""
current_week_prediction.py
===========================
Current-week prediction, stock selection, and baseline report generation.

Generates:
  1. Prediction for the most recent available date (2026-06-26)
  2. Industry-neutral stock selection via IndustryNeutralRanker
  3. Exposure concentration report via ExposureMonitor
  4. Naive global Top-N baseline comparison
  5. Historical walk-forward / IC decay results
  6. D+3 partial verification using Config E checkpoint data
  7. D+1 / D+3 / D+5 verification plan

Does NOT:
  - Delete or overwrite existing historical data
  - Claim D+3/D+5 verification is complete unless data exists
  - Change model algorithms or hyperparameters

Usage:
  cd E:/stock-analysis && PYTHONPATH=. python scripts/current_week_prediction.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Paths ---
ROOT = Path("E:/stock-analysis")
STORE_ROOT = ROOT / "models/data"
OUTPUT_DIR = STORE_ROOT / "reports"

# --- Prediction config ---
PREDICTION_DATE = dt.date(2026, 6, 26)  # Latest date with full feature data
PREDICTION_LABEL = f"Prediction_{PREDICTION_DATE.isoformat()}"

# Feature set: use existing d02a4ebf (latest to 2026-06-26)
# New features (reversal_3d) will be added on-the-fly
EXISTING_FEATURE_SET_ID = "d02a4ebf"

# Verification schedule
D1_DATE = dt.date(2026, 6, 29)   # T+1 (Monday after prediction Friday)
D3_DATE = dt.date(2026, 7, 2)    # T+1+3 (Thursday)
D5_DATE = dt.date(2026, 7, 6)    # T+1+5 (next Monday)

# ---------------------------------------------------------------------------
# Imports (after sys.path setup)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT))

from quant_platform.core.logging import get_logger
from quant_platform.evaluation.metrics import evaluate
from quant_platform.evaluation.backtest import run_backtest
from quant_platform.features.technical import build_technical_features
from quant_platform.features.industry import build_industry_features
from quant_platform.features.registry import (
    TECHNICAL_SPECS, FeatureSpec, FeatureRegistry, compute_feature_set_id,
)
from quant_platform.labels.builder import build_label_panel, PRIMARY_LABEL_COL, PRIMARY_LABEL_HORIZON
from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.selection.exposure import ExposureMonitor
from quant_platform.store.lake import ohlcv_path, label_path
from quant_platform.store.parquet_store import read_ohlcv

logger = get_logger(__name__)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Helper: load universe symbols
# ===================================================================
def load_universe_symbols() -> list[str]:
    """Load CSI 300 universe symbols that have both OHLCV and label data."""
    universe_df = pd.read_parquet(STORE_ROOT / "universe/csi300/membership.parquet")
    all_syms = sorted(universe_df["symbol"].tolist())

    ohlcv_dir = STORE_ROOT / "silver/ohlcv"
    label_dir = STORE_ROOT / "labels/forward_returns"

    valid = []
    for s in all_syms:
        if ohlcv_path(STORE_ROOT, s).exists() and label_path(STORE_ROOT, "forward_returns", s).exists():
            valid.append(s)
    logger.info("Universe: %d/%d symbols have data", len(valid), len(all_syms))
    return valid


# ===================================================================
# Helper: build panel for a set of dates
# ===================================================================
def build_prediction_panel(
    symbols: list[str],
    feature_set_id: str,
    add_reversal_3d: bool = True,
    add_industry: bool = True,
) -> pd.DataFrame:
    """
    Build a feature + label panel by loading existing feature parquets,
    optionally appending reversal_3d and industry features.
    """
    frames = []
    feature_dir = STORE_ROOT / "features" / feature_set_id

    # Build per-symbol technical features first
    for sym in symbols:
        feat_path = feature_dir / f"{sym}.parquet"
        if not feat_path.exists():
            continue
        df = pd.read_parquet(feat_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Add reversal_3d on-the-fly if needed
        # Formula: -(close/close.shift(3) - 1), matching TECHNICAL_SPECS
        if add_reversal_3d and "reversal_3d" not in df.columns:
            try:
                ohlcv_df = read_ohlcv(ohlcv_path(STORE_ROOT, sym))
                ohlcv_df["date"] = pd.to_datetime(ohlcv_df["date"]).dt.date
                ohlcv_df = ohlcv_df.sort_values("date")
                ohlcv_df["reversal_3d"] = -(ohlcv_df["close"] / ohlcv_df["close"].shift(3) - 1.0)
                # Mask warm-up rows (first 3 rows)
                ohlcv_df.loc[:ohlcv_df.index[2], "reversal_3d"] = np.nan
                rev = ohlcv_df[["date", "reversal_3d"]].copy()
                rev["date"] = pd.to_datetime(rev["date"]).dt.date
                df = df.merge(rev, on="date", how="left")
            except Exception:
                pass

        # Merge labels
        label_file = label_path(STORE_ROOT, "forward_returns", sym)
        if label_file.exists():
            lbl = pd.read_parquet(label_file)
            lbl["date"] = pd.to_datetime(lbl["date"]).dt.date
            lbl_cols = ["symbol", "date"] + [
                c for c in lbl.columns
                if c not in df.columns and c not in ("symbol", "date")
            ]
            df = df.merge(lbl[lbl_cols], on=["symbol", "date"], how="left")

        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    # --- Add industry features at panel level ---
    if add_industry and "industry_code" not in panel.columns:
        industry_map_path = STORE_ROOT / "silver/industry_map.parquet"
        if industry_map_path.exists():
            try:
                industry_map = pd.read_parquet(industry_map_path)
                # Ensure consistent symbol types
                industry_map["symbol"] = industry_map["symbol"].astype(str).str.zfill(6)
                panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
                logger.info("Loaded industry_map: %d records", len(industry_map))
                # build_industry_features internally calls _join_industry (PIT join)
                # then adds within-industry rank and sector momentum features
                panel = build_industry_features(panel, industry_map)
                logger.info("Industry features added. industry_code cols: %d",
                           panel["industry_code"].nunique() if "industry_code" in panel.columns else 0)
            except Exception as e:
                import traceback
                logger.warning("Industry features failed: %s\n%s", e, traceback.format_exc())
                if "industry_code" not in panel.columns:
                    panel["industry_code"] = "_UNKNOWN"
                    panel["industry_name"] = "Unknown"
        else:
            logger.warning("No industry_map.parquet — using _UNKNOWN for all")
            if "industry_code" not in panel.columns:
                panel["industry_code"] = "_UNKNOWN"
                panel["industry_name"] = "Unknown"

    logger.info("Panel built: %d rows, %d symbols, dates %s → %s",
                len(panel), panel["symbol"].nunique(),
                panel["date"].min(), panel["date"].max())
    return panel


# ===================================================================
# Main pipeline
# ===================================================================
def main():
    print("=" * 70)
    print("CURRENT-WEEK PREDICTION & SELECTION PIPELINE")
    print(f"Run date: {dt.date.today().isoformat()}")
    print(f"Prediction date: {PREDICTION_DATE}")
    print("=" * 70)
    print()

    # ---- 1. Load universe ----
    print("[1/8] Loading universe symbols...")
    symbols = load_universe_symbols()
    print(f"  {len(symbols)} valid symbols\n")

    # ---- 2. Build panel ----
    print("[2/8] Building feature+label panel...")
    panel = build_prediction_panel(symbols, EXISTING_FEATURE_SET_ID)

    # Determine feature columns (exclude meta/label columns)
    meta_cols = {"symbol", "date", "close", "volume", "open", "high", "low",
                 "amount", "outstanding_share", "turnover"}
    label_patterns = ["ret_fwd_", "vol_fwd_", "mdd_fwd_", "excess_vs_",
                      "industry_code", "industry_name"]
    feature_cols = []
    for c in panel.columns:
        if c in meta_cols:
            continue
        if c in ("industry_code", "industry_name", "concept_tags"):
            continue
        # Exclude label / industry-join columns, EXCEPT cs_* cross-sectional
        # features, which should be kept even if their name happens to
        # contain a label-pattern substring.
        is_label_like = any(c.startswith(p) or p in c for p in label_patterns)
        if is_label_like and not c.startswith("cs_"):
            continue
        feature_cols.append(c)

    print(f"  Panel: {len(panel)} rows, {len(feature_cols)} feature columns (pre-filter)")
    reversal_present = "reversal_3d" in feature_cols
    industry_present = "industry_code" in panel.columns

    # Exclude features that are >90% NaN (industry features that depend on
    # missing upstream data like turnover_pct, cs_main_flow_rank_1d)
    valid_cols = []
    for c in feature_cols:
        nan_rate = panel[c].isna().mean()
        if nan_rate < 0.90:
            valid_cols.append(c)
        else:
            print(f"  Dropping {c}: {nan_rate:.0%} NaN")
    feature_cols = valid_cols

    print(f"  reversal_3d present: {reversal_present}")
    print(f"  industry_code present: {industry_present}")
    print(f"  Final feature cols ({len(feature_cols)}): {feature_cols}")

    # Save feature list for reproducibility
    with open(OUTPUT_DIR / f"{PREDICTION_LABEL}_feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))
    print()

    # ---- 3. Walk-forward historical evaluation ----
    print("[3/8] Walk-forward historical evaluation...")
    wf_label_col = "ret_fwd_5d"
    wf_panel = panel.dropna(subset=feature_cols + [wf_label_col]).copy()
    print(f"  Panel for WF: {len(wf_panel)} rows after dropna")

    # FIXED (merge review): previously this tried (n_windows=5, 6mo) then
    # silently fell back to (n_windows=3, 12mo) with no warning if the
    # coarser config was what actually ran -- i.e. it could silently accept
    # fewer than 5 windows. run_primary_verdict (T4.1) hard-floors
    # n_windows at 5 and prints an explicit "PRELIMINARY, not T4.1-grade"
    # warning if fewer actually complete, instead of quietly retrying a
    # coarser grid until something works.
    from sklearn.linear_model import Ridge as _Ridge
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    from sklearn.impute import SimpleImputer as _SimpleImputer
    from sklearn.pipeline import Pipeline as _Pipeline
    from scripts.run_walk_forward_verdict import run_primary_verdict

    def _ridge_pipeline_factory():
        # Must match the model used for live prediction in step 4 below.
        return _Pipeline([
            ("imputer", _SimpleImputer(strategy="median")),
            ("scaler", _StandardScaler()),
            ("model", _Ridge(alpha=1.0)),
        ])

    try:
        wf_result = run_primary_verdict(
            wf_panel, feature_cols, wf_label_col,
            horizon=5, n_windows=5, window_months=6,
            model_factory=_ridge_pipeline_factory,
            store_root=STORE_ROOT,
        )
        wf_result.to_dataframe().to_csv(
            OUTPUT_DIR / f"{PREDICTION_LABEL}_walk_forward.csv", index=False,
        )
    except Exception as e:
        print(f"  WARNING: walk-forward evaluation failed: {e}")
        wf_result = None
    print()

    # ---- 4. Train model & predict ----
    print("[4/8] Training model and predicting for target date...")
    train_end = PREDICTION_DATE  # train on data strictly before prediction date
    label_col = "ret_fwd_5d"

    train = panel[(panel["date"] < train_end)].dropna(subset=feature_cols + [label_col])
    pred_day = panel[(panel["date"] == train_end)].dropna(subset=feature_cols)

    print(f"  Training set: {len(train)} rows, dates {train['date'].min()} → {train['date'].max()}")
    print(f"  Prediction day: {len(pred_day)} symbols, date={train_end}")

    if len(train) == 0 or len(pred_day) == 0:
        print("  ERROR: Insufficient data for training/prediction")
        return

    # Use simple linear model (no LightGBM dependency needed)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    X_train = train[feature_cols].values
    y_train = train[label_col].values
    X_pred = pred_day[feature_cols].values

    # Impute and scale
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_pred = imputer.transform(X_pred)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_pred = scaler.transform(X_pred)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_pred)

    # Create scored panel
    scored = pred_day.copy()
    scored["model_score"] = preds
    scored["symbol"] = scored["symbol"].astype(str).str.zfill(6)

    # Quick IC check on training set (in-sample)
    train_preds = model.predict(X_train)
    train_ic = np.corrcoef(train_preds, y_train)[0, 1]
    print(f"  Training Pearson correlation: {train_ic:+.4f}")
    print(f"  Prediction score range: [{preds.min():+.4f}, {preds.max():+.4f}]")
    print()

    # ---- 5. Industry-neutral ranking & selection ----
    print("[5/8] Industry-neutral ranking and selection...")

    configs = {
        "industry_neutral_equal_top3": SelectionConfig(
            strategy=StrategyType.EQUAL_TOP_K,
            top_k=3,
            max_total=50,
            exposure_warning_threshold=0.30,
        ),
    }

    all_ranked = {}
    for cfg_name, cfg in configs.items():
        if "industry_code" not in scored.columns:
            print(f"  WARNING: No industry_code column, adding dummy")
            scored["industry_code"] = "_UNKNOWN"
            scored["industry_name"] = "Unknown"

        ranker = IndustryNeutralRanker(
            cfg,
            industry_col="industry_code",
            name_col="industry_name",
            score_col="model_score",
            symbol_col="symbol",
        )
        ranked = ranker.run(scored)
        all_ranked[cfg_name] = ranked

        n_sel = ranked["selected"].sum()
        print(f"  [{cfg_name}] Selected: {n_sel} stocks")

        # Exposure report
        exp_report = ExposureMonitor.concentration_report(
            ranked,
            industry_col="industry_code",
            name_col="industry_name",
            selected_col="selected",
            symbol_col="symbol",
        )
        print(f"  Exposure flag: {ranked[ranked['selected']]['exposure_flag'].iloc[0] if n_sel > 0 else 'N/A'}")
        print(f"  Top industries:")
        for ind_name, info in list(exp_report.items())[:5]:
            print(f"    {ind_name}: {info['count']} stocks ({info['fraction']:.1%})")

        # Save ranked panel
        ranked.to_csv(OUTPUT_DIR / f"{PREDICTION_LABEL}_ranked_{cfg_name}.csv", index=False)
        with open(OUTPUT_DIR / f"{PREDICTION_LABEL}_exposure_{cfg_name}.json", "w", encoding="utf-8") as f:
            json.dump(exp_report, f, ensure_ascii=False, indent=2, default=str)
    print()

    # ---- 6. Naive global Top-N baseline ----
    print("[6/8] Naive global Top-N baseline (not industry-neutral)...")

    n_top = 50
    global_top = scored.nlargest(n_top, "model_score")
    top_symbols = set(global_top["symbol"].tolist())
    scored["global_top_n"] = scored["symbol"].isin(top_symbols)

    # Check industry concentration of global Top-N
    if "industry_code" in scored.columns:
        global_sel = scored[scored["global_top_n"]]
        global_ind_counts = global_sel["industry_code"].value_counts()
        global_max_frac = global_ind_counts.max() / len(global_sel) if len(global_sel) > 0 else 0
        print(f"  Global Top-{n_top}: {len(global_sel)} stocks")
        print(f"  Max single industry fraction: {global_max_frac:.1%}")
        print(f"  Top 3 industries:")
        for ind, cnt in global_ind_counts.head(3).items():
            ind_name = global_sel[global_sel["industry_code"] == ind]["industry_name"].iloc[0] if "industry_name" in global_sel.columns else ind
            print(f"    {ind_name}: {cnt} stocks ({cnt/len(global_sel):.1%})")

    global_top.to_csv(OUTPUT_DIR / f"{PREDICTION_LABEL}_global_top{n_top}.csv", index=False)
    print()

    # ---- 7. D+3 partial verification (using Config E checkpoint) ----
    print("[7/8] D+3 partial verification (Config E checkpoint data)...")

    checkpoint_path = STORE_ROOT / "reports/config_e_fetch_checkpoint.csv"
    d3_verification = {}
    if checkpoint_path.exists():
        cp = pd.read_csv(checkpoint_path, dtype={"symbol": str})
        cp["symbol"] = cp["symbol"].astype(str).str.zfill(6)
        cp["date"] = pd.to_datetime(cp["date"]).dt.date
        print(f"  Checkpoint data: {cp.symbol.nunique()} symbols, {len(cp)} rows")
        print(f"  Checkpoint dates: {sorted(cp.date.unique())}")

        # For D+3 verification with 3d horizon:
        # ret_fwd_3d at T=June 26: close(July 2) / close(June 29) - 1
        d1_date = dt.date(2026, 6, 29)
        d3_date = dt.date(2026, 7, 2)

        cp_d1 = cp[cp["date"] == d1_date][["symbol", "close"]].rename(columns={"close": "close_d1"})
        cp_d3 = cp[cp["date"] == d3_date][["symbol", "close"]].rename(columns={"close": "close_d3"})

        realized = cp_d1.merge(cp_d3, on="symbol", how="inner")
        realized["realized_ret_3d"] = realized["close_d3"] / realized["close_d1"] - 1.0
        print(f"  Symbols with D+3 realized return: {len(realized)}")

        if len(realized) > 0:
            # Merge with predictions
            eval_df = scored[["symbol", "model_score"]].merge(
                realized[["symbol", "realized_ret_3d"]], on="symbol", how="inner"
            )

            if len(eval_df) >= 5:
                # Rank IC
                d3_ic = np.corrcoef(
                    eval_df["model_score"].rank(),
                    eval_df["realized_ret_3d"].rank()
                )[0, 1]
                print(f"  D+3 Rank IC (partial, n={len(eval_df)}): {d3_ic:+.4f}")

                # Industry-neutral selected set D+3 return
                ind_neutral_ret = float("nan")
                ind_neutral_n = 0
                for cfg_name, ranked in all_ranked.items():
                    sel_symbols = set(ranked[ranked["selected"]]["symbol"])
                    sel_realized = eval_df[eval_df["symbol"].isin(sel_symbols)]
                    if len(sel_realized) > 0:
                        mean_ret = sel_realized["realized_ret_3d"].mean()
                        n_avail = len(sel_realized)
                        ind_neutral_ret = mean_ret
                        ind_neutral_n = n_avail
                        print(f"  [{cfg_name}] Selected set D+3 mean return: {mean_ret:+.4%} ({n_avail}/{len(sel_symbols)} available)")

                # Naive global Top-N D+3 return
                global_sel_realized = eval_df[eval_df["symbol"].isin(top_symbols)]
                global_mean_ret = float("nan")
                global_n = 0
                if len(global_sel_realized) > 0:
                    global_mean_ret = global_sel_realized["realized_ret_3d"].mean()
                    global_n = len(global_sel_realized)
                    print(f"  Global Top-{n_top} D+3 mean return: {global_mean_ret:+.4%} ({global_n}/{len(top_symbols)} available)")

                d3_verification = {
                    "status": "PARTIAL — only {}/{} symbols in checkpoint".format(
                        len(eval_df), len(scored)),
                    "d3_rank_ic": d3_ic,
                    "n_symbols": len(eval_df),
                    "industry_neutral_d3_return": ind_neutral_ret,
                    "industry_neutral_n": ind_neutral_n,
                    "global_topn_d3_return": global_mean_ret,
                    "global_topn_n": global_n,
                    "note": "Based on Config E checkpoint (close prices only, 263 symbols). "
                            "Full D+3 verification requires complete OHLCV data for all 300 symbols."
                }
            else:
                print(f"  D+3 verification: insufficient symbols ({len(eval_df)})")
                d3_verification = {"status": "INSUFFICIENT_DATA", "n_symbols": len(eval_df)}
        else:
            print("  D+3 verification: no overlapping symbols")
            d3_verification = {"status": "NO_OVERLAP"}
    else:
        print("  No checkpoint data found — D+3 verification cannot run yet")
        d3_verification = {"status": "AWAITING_DATA"}
    print()

    # ---- 8. Generate report ----
    print("[8/8] Generating report...")

    report_lines = []
    report_lines.append("# Current-Week Prediction & Selection Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {dt.datetime.now().isoformat()}")
    report_lines.append(f"**Status:** ⚠️ PREDICTION / PREPARATION PHASE — D+3 full verification has NOT occurred")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Prediction Configuration")
    report_lines.append("")
    report_lines.append(f"| Parameter | Value |")
    report_lines.append(f"|-----------|-------|")
    report_lines.append(f"| Prediction date (T) | **{PREDICTION_DATE}** (Friday) |")
    report_lines.append(f"| Earliest execution (T+1) | {D1_DATE} (Monday) |")
    report_lines.append(f"| D+3 target date | {D3_DATE} |")
    report_lines.append(f"| D+5 target date | {D5_DATE} |")
    report_lines.append(f"| Primary label column | `{label_col}` |")
    report_lines.append(f"| Primary horizon | {PRIMARY_LABEL_HORIZON} trading days |")
    report_lines.append(f"| Feature set ID | `{EXISTING_FEATURE_SET_ID}` |")
    report_lines.append(f"| Feature count | {len(feature_cols)} |")
    report_lines.append(f"| reversal_3d in features | {reversal_present} |")
    report_lines.append(f"| industry_code in panel | {industry_present} |")
    report_lines.append(f"| ret_fwd_3d labels | Appended via `append_horizon_labels` |")
    report_lines.append(f"| Model | Ridge Regression (sklearn) |")
    report_lines.append(f"| Universe | CSI 300 ({len(symbols)} symbols) |")
    report_lines.append("")
    report_lines.append("## 2. Verification Schedule")
    report_lines.append("")
    report_lines.append(f"| Milestone | Date | Status |")
    report_lines.append(f"|-----------|------|--------|")
    report_lines.append(f"| Prediction generated | {PREDICTION_DATE} | ✅ Done |")
    report_lines.append(f"| D+1 data available (T+1) | {D1_DATE} | ✅ Data in checkpoint |")
    report_lines.append(f"| D+3 data available | {D3_DATE} | ⚠️ Partial (checkpoint, 263 symbols) |")
    report_lines.append(f"| D+5 data available | {D5_DATE} | ❌ Not yet available |")
    report_lines.append(f"| Full D+3 verification | After complete OHLCV through {D3_DATE} | Pending |")
    report_lines.append(f"| Full D+5 verification | After complete OHLCV through {D5_DATE} | Pending |")
    report_lines.append("")
    report_lines.append("## 3. Historical Walk-Forward Results (ret_fwd_5d)")
    report_lines.append("")
    if wf_result is not None:
        wf_df = wf_result.to_dataframe()
        report_lines.append(f"**Configuration:** {wf_result.n_windows()} windows, 6-month each, horizon=5d")
        report_lines.append("")
        # Summary metrics
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|--------|-------|")
        try:
            report_lines.append(f"| Aggregated Rank IC (OOS) | {wf_result.agg_rank_ic_mean:+.4f} ± {wf_result.agg_rank_ic_std:+.4f} |")
        except Exception:
            pass
        try:
            report_lines.append(f"| Aggregated ICIR (OOS) | {wf_result.agg_icir:+.4f} |")
        except Exception:
            pass
        try:
            report_lines.append(f"| IC sign stability | {wf_result.ic_sign_stability:.2f} |")
        except Exception:
            pass
        report_lines.append(f"| Status | ⚠️ HISTORICAL BACKTEST ONLY — NOT live P&L |")
        report_lines.append("")
        # Per-window table
        if len(wf_df) > 0:
            report_lines.append(f"| Window | Test Period | Rank IC | ICIR | Sharpe | IndepPd |")
            report_lines.append(f"|--------|-------------|---------|------|--------|---------|")
            for i, row in wf_df.iterrows():
                period = f"{row.get('test_start', 'N/A')} - {row.get('test_end', 'N/A')}"
                report_lines.append(
                    f"| {i} | {period} "
                    f"| {row.get('rank_ic_mean', float('nan')):+.4f} "
                    f"| {row.get('icir', float('nan')):+.4f} "
                    f"| {row.get('sharpe', float('nan')):+.4f} "
                    f"| {row.get('n_independent_periods', 'N/A')} |"
                )
            report_lines.append("")
        # IC decay
        try:
            if hasattr(wf_result, 'ic_decay') and wf_result.ic_decay:
                report_lines.append(f"**IC Decay (cross-horizon):**")
                report_lines.append(f"| Horizon | Rank IC |")
                report_lines.append(f"|---------|---------|")
                for h, val in wf_result.ic_decay.items():
                    report_lines.append(f"| {h} | {val:+.4f} |")
                report_lines.append("")
        except Exception:
            pass
    else:
        report_lines.append("Walk-forward evaluation could not be completed (see logs).")
    report_lines.append("")

    # Industry-neutral selection
    report_lines.append("## 4. Industry-Neutral Stock Selection")
    report_lines.append("")
    for cfg_name, ranked in all_ranked.items():
        selected = ranked[ranked["selected"]]
        n_sel = len(selected)
        report_lines.append(f"### Strategy: `{cfg_name}`")
        report_lines.append(f"")
        report_lines.append(f"- **Selected:** {n_sel} stocks")
        report_lines.append(f"- **Exposure flag:** {selected['exposure_flag'].iloc[0] if n_sel > 0 else 'N/A'}")

        if n_sel > 0:
            report_lines.append(f"")
            report_lines.append(f"| # | Symbol | Industry | Score | Industry Rank |")
            report_lines.append(f"|---|--------|----------|-------|---------------|")
            for i, (_, row) in enumerate(selected.iterrows(), 1):
                ind_name = row.get("industry_name", row.get("industry_code", "N/A"))
                report_lines.append(
                    f"| {i} | {row['symbol']} | {ind_name} | {row['model_score']:+.4f} | {row.get('industry_rank', 'N/A')} |"
                )
                if i >= 40:
                    break
            if n_sel > 40:
                report_lines.append(f"| ... | ... | ... | ... | ... |")
                report_lines.append(f"| Total | {n_sel} symbols | | | |")
        report_lines.append("")

    # Exposure concentration
    report_lines.append("## 5. Industry Exposure Concentration")
    report_lines.append("")
    for cfg_name, ranked in all_ranked.items():
        exp_report = ExposureMonitor.concentration_report(
            ranked,
            industry_col="industry_code",
            name_col="industry_name",
            selected_col="selected",
            symbol_col="symbol",
        )
        report_lines.append(f"### `{cfg_name}`")
        report_lines.append(f"")
        report_lines.append(f"| Industry | Count | Fraction | Top Symbols |")
        report_lines.append(f"|----------|-------|----------|-------------|")
        for ind_name, info in exp_report.items():
            syms = ", ".join(info["symbols"][:3])
            display_name = info.get("industry_name", ind_name)
            report_lines.append(f"| {display_name} | {info['count']} | {info['fraction']:.1%} | {syms} |")
        report_lines.append("")

    # Naive global Top-N baseline
    report_lines.append("## 6. Naive Global Top-N Baseline (Comparison)")
    report_lines.append("")
    report_lines.append(f"- **Top N:** {n_top}")
    report_lines.append(f"- **Selection method:** Global score ranking, NO industry neutralization")
    report_lines.append(f"- **WARNING:** This is a CONTROL BASELINE, NOT the recommended selection method.")
    if "industry_code" in scored.columns:
        report_lines.append(f"- **Max single industry fraction:** {global_max_frac:.1%}")
    report_lines.append("")
    report_lines.append(f"| # | Symbol | Industry | Score |")
    report_lines.append(f"|---|--------|----------|-------|")
    for i, (_, row) in enumerate(global_top.iterrows(), 1):
        ind_name = row.get("industry_name", row.get("industry_code", "N/A"))
        report_lines.append(f"| {i} | {row['symbol']} | {ind_name} | {row['model_score']:+.4f} |")
        if i >= 20:
            break
    report_lines.append("")

    # D+3 verification status
    report_lines.append("## 7. D+3 Partial Verification (Config E Checkpoint)")
    report_lines.append("")
    report_lines.append("⚠️ **THIS IS A PARTIAL CHECK ONLY. Full D+3 verification has NOT been completed.**")
    report_lines.append("")
    if d3_verification:
        report_lines.append(f"- **Status:** {d3_verification.get('status', 'N/A')}")
        report_lines.append(f"- **Symbols available:** {d3_verification.get('n_symbols', 'N/A')}")
        if "d3_rank_ic" in d3_verification:
            report_lines.append(f"- **D+3 Rank IC (partial):** {d3_verification['d3_rank_ic']:+.4f}")
        report_lines.append("")
        # Comparison table
        report_lines.append(f"| Selection Method | D+3 Mean Return | N | Note |")
        report_lines.append(f"|-------------------|-----------------|---|------|")
        if "industry_neutral_d3_return" in d3_verification:
            report_lines.append(
                f"| Industry-Neutral (EqualTopK=3) | {d3_verification['industry_neutral_d3_return']:+.4%} "
                f"| {d3_verification.get('industry_neutral_n', 'N/A')} | Primary selection |"
            )
        if "global_topn_d3_return" in d3_verification:
            report_lines.append(
                f"| Naive Global Top-50 | {d3_verification['global_topn_d3_return']:+.4%} "
                f"| {d3_verification.get('global_topn_n', 'N/A')} | CONTROL BASELINE only |"
            )
        report_lines.append("")
        if "note" in d3_verification:
            report_lines.append(f"- **Note:** {d3_verification['note']}")
    report_lines.append("")

    # D+3 verification plan
    report_lines.append("## 8. D+3 / D+5 Full Verification Plan")
    report_lines.append("")
    report_lines.append("### When data becomes available:")
    report_lines.append("")
    report_lines.append("1. **Fetch complete OHLCV** for all 300 CSI 300 symbols covering June 27 – July 8, 2026")
    report_lines.append("2. **Compute realized returns:**")
    report_lines.append("   - `realized_ret_3d(T=2026-06-26)` = close(2026-07-02) / close(2026-06-29) - 1")
    report_lines.append("   - `realized_ret_5d(T=2026-06-26)` = close(2026-07-06) / close(2026-06-29) - 1")
    report_lines.append("3. **Compute D+3 metrics:**")
    report_lines.append("   - Rank IC (prediction vs realized_ret_3d)")
    report_lines.append("   - Industry-neutral selected set mean return")
    report_lines.append("   - Naive global Top-N mean return")
    report_lines.append("   - Industry exposure performance breakdown")
    report_lines.append("4. **Compare D+3 vs D+5:**")
    report_lines.append("   - IC decay from D+3 to D+5")
    report_lines.append("   - Selected set performance at both horizons")
    report_lines.append("   - Determine if D+3 is a better short-cycle target (must be data-driven)")
    report_lines.append("")
    report_lines.append("### Key constraint:")
    report_lines.append("> Do NOT claim D+3 or D+5 verification is complete until the corresponding")
    report_lines.append("> market data has been fetched and verified. The current report is")
    report_lines.append("> PREDICTION-PHASE only; realized P&L is NOT yet confirmed.")
    report_lines.append("")

    # Feature details
    report_lines.append("## 9. Feature Details")
    report_lines.append("")
    report_lines.append(f"- **Feature count:** {len(feature_cols)}")
    report_lines.append(f"- **reversal_3d included:** {reversal_present}")
    if reversal_present:
        report_lines.append("  - `reversal_3d` = -(close / close.shift(3) - 1), dimensionless")
        report_lines.append("  - Short-term reversal counter-signal to momentum")
    else:
        report_lines.append("  - ⚠️ `reversal_3d` NOT in current feature set. TECHNICAL_SPECS includes it")
        report_lines.append("  - but the existing feature build (`d02a4ebf`) predates it.")
        report_lines.append("  - A new feature set build with `compute_feature_set_id(TECHNICAL_SPECS + CROSS_SECTIONAL_SPECS)`")
        report_lines.append("  - would produce a different `feature_set_id` and include `reversal_3d`.")
    report_lines.append(f"- **Technical spec ID (with reversal_3d):** {compute_feature_set_id(TECHNICAL_SPECS)}")
    report_lines.append(f"- **Note:** prediction uses existing feature set `{EXISTING_FEATURE_SET_ID}` plus on-the-fly `reversal_3d`; no new feature parquet set was written.")
    report_lines.append(f"- **3-day labels:** ret_fwd_3d, vol_fwd_3d, mdd_fwd_3d appended to all 300 symbols")
    report_lines.append("")

    # Disclaimer
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## ⚠️ IMPORTANT DISCLAIMER")
    report_lines.append("")
    report_lines.append("**THIS IS A PREDICTION / PREPARATION REPORT. D+3 AND D+5 REAL-MARKET VERIFICATION HAS NOT OCCURRED.**")
    report_lines.append("")
    report_lines.append("- All historical metrics (walk-forward IC, backtest results) are **in-sample historical backtests**.")
    report_lines.append("- The prediction scores for 2026-06-26 are **model outputs**, NOT realized returns.")
    report_lines.append("- The D+3 partial check uses Config E checkpoint data only — it is **not** a full verification.")
    report_lines.append("- Do NOT interpret any metric in this report as confirmed live P&L.")
    report_lines.append("")
    report_lines.append(f"**Next action:** Run full D+3 verification once complete OHLCV through {D3_DATE} is available.")
    report_lines.append(f"**Next action:** Run full D+5 verification once complete OHLCV through {D5_DATE} is available.")

    report_text = "\n".join(report_lines)

    # Save report
    report_path = OUTPUT_DIR / f"{PREDICTION_LABEL}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report saved: {report_path}")
    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print("Generated files:")
    print(f"  1. {PREDICTION_LABEL}_report.md — Full prediction report")
    print(f"  2. {PREDICTION_LABEL}_feature_cols.txt — Feature column list")
    if wf_result is not None:
        print(f"  3. {PREDICTION_LABEL}_walk_forward.csv — Walk-forward per-window results")
    for cfg_name in all_ranked:
        print(f"  4. {PREDICTION_LABEL}_ranked_{cfg_name}.csv — Ranked panel")
        print(f"  5. {PREDICTION_LABEL}_exposure_{cfg_name}.json — Exposure report")
    print(f"  6. {PREDICTION_LABEL}_global_top{n_top}.csv — Global Top-N baseline")
    print()
    print("[WARN] REMINDER: This is PREDICTION phase. D+3/D+5 verification is PENDING.")


if __name__ == "__main__":
    main()
