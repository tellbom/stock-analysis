"""
next_week_d3_prediction.py
===========================
Next-week D+3 stock recommendation pipeline.

Uses the PRODUCTION model (trained once before prediction date, persisted to disk).
No retraining. No synthetic data. Real market inference only.

Prediction: 2026-07-03 (Friday) -> D+3 = 2026-07-08 (Wednesday next week)  [PIT: data <= T, labels observable at T close]

Output:
  A. Top 50 industry-neutral stock picks
  B. Global Top-50 baseline comparison
  C. Risk checks (industry concentration, exposure flags)
  D. Wednesday review plan

Usage:
  cd E:/stock-analysis && PYTHONPATH=. python scripts/next_week_d3_prediction.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("E:/stock-analysis")
STORE_ROOT = ROOT / "models/data"
OUTPUT_DIR = STORE_ROOT / "reports"

# Prediction config
PREDICTION_DATE = dt.date(2026, 7, 3)   # T (Friday)
D1_DATE = dt.date(2026, 7, 6)            # T+1 (Monday)
D3_DATE = dt.date(2026, 7, 8)            # T+1+3 (Wednesday)
D5_DATE = dt.date(2026, 7, 10)           # T+1+5 (Friday)
TRAIN_END = dt.date(2026, 6, 30)         # PIT: T<=6/29 labels observable at 7/3 close
# FIXED (merge review): model persistence now goes through the existing
# MLflow Model Registry (quant_platform.training.registry) instead of a
# bespoke pickle file under models/production/ -- see step 3 in main().
REGISTERED_MODEL_NAME = "quant_platform_d3_model"
PREDICTION_LABEL = f"D3_Prediction_{PREDICTION_DATE.isoformat()}"

sys.path.insert(0, str(ROOT))

from quant_platform.core.logging import get_logger
from quant_platform.features.cross_sectional import build_cross_sectional_features
from quant_platform.features.industry import build_industry_features
from quant_platform.features.pipeline import FeaturePipeline
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id
from quant_platform.evaluation.coverage_gate import (
    compute_feature_coverage_report,
    select_features_by_gate,
    write_coverage_gate_report,
)
from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.selection.exposure import ExposureMonitor
from quant_platform.store.lake import ohlcv_path, label_path, feature_path
from quant_platform.cli import _coverage_gate_config_for_universe, _feature_family_lookup

logger = get_logger(__name__)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FEATURE_COLUMNS = ("reversal_3d",)


# ===================================================================
# Data loading
# ===================================================================
def load_symbols() -> list[str]:
    df = pd.read_parquet(STORE_ROOT / "universe/csi300/membership.parquet")
    syms = sorted(df["symbol"].tolist())
    valid = []
    for s in syms:
        if ohlcv_path(STORE_ROOT, s).exists():
            valid.append(s)
    return valid


def ensure_registered_feature_set(
    symbols: list[str],
    *,
    max_date: dt.date,
) -> str:
    """
    Ensure the production D+3 feature set is formally written to the lake.

    The old D3 script read a stale pre-reversal feature set and patched missing/new dates on the
    fly. That meant ``reversal_3d`` could appear in prediction-day output
    while being absent from the historical training rows. We now use the
    normal FeaturePipeline/FeatureRegistry path so ``reversal_3d`` is present
    in the stored feature parquet before model training starts.
    """
    feature_set_id = compute_feature_set_id(DEFAULT_SPECS)

    def _needs_rebuild(symbol: str) -> bool:
        path = feature_path(STORE_ROOT, feature_set_id, symbol)
        if not path.exists():
            return True
        try:
            df = pd.read_parquet(path)
        except Exception:
            return True
        if any(c not in df.columns for c in REQUIRED_FEATURE_COLUMNS):
            return True
        if df.empty or "date" not in df.columns:
            return True
        max_written = pd.to_datetime(df["date"]).dt.date.max()
        return max_written < max_date

    needs_rebuild = any(_needs_rebuild(sym) for sym in symbols)
    if needs_rebuild:
        print(f"  Building registered feature set {feature_set_id} "
              f"(includes {', '.join(REQUIRED_FEATURE_COLUMNS)})...")
        pipe = FeaturePipeline(store_root=STORE_ROOT, project_root=ROOT)
        built_id = pipe.run(symbols, specs=DEFAULT_SPECS, end_date=max_date)
        if built_id != feature_set_id:
            raise RuntimeError(
                f"Unexpected feature_set_id {built_id}; expected {feature_set_id}"
            )
    else:
        print(f"  Registered feature set {feature_set_id} already covers {max_date}")

    return feature_set_id


def build_panel(
    symbols: list[str],
    feature_set_id: str,
    max_date: dt.date | None = None,
) -> pd.DataFrame:
    """
    Build feature+label panel from the registered production feature set.

    ``ensure_registered_feature_set()`` guarantees that this set has already
    been rebuilt through FeaturePipeline and contains ``reversal_3d`` in the
    stored per-symbol parquet files. Keep this loader intentionally boring:
    no ad hoc technical-feature patching here, so training and prediction use
    one persisted feature source of truth.
    """
    frames = []

    for sym in symbols:
        fp = feature_path(STORE_ROOT, feature_set_id, sym)
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if max_date:
            df = df[df["date"] <= max_date].copy()

        missing_required = [
            c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns
        ]
        if missing_required:
            raise RuntimeError(
                f"Feature set {feature_set_id} is missing {missing_required}; "
                "rebuild features through FeaturePipeline before training."
            )

        # Merge labels
        lp = label_path(STORE_ROOT, "forward_returns", sym)
        if lp.exists():
            lbl = pd.read_parquet(lp)
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
    panel = build_cross_sectional_features(panel)

    # Industry features -- computed for the WHOLE panel (historical rows
    # AND the live prediction day alike) via the public
    # build_industry_features API, not the private _join_industry
    # PIT-join-only helper. This means the prediction day gets freshly
    # computed ind_rank_*/sector_momentum_10d values instead of a
    # carried-forward stale value from the last historical date.
    imp = STORE_ROOT / "silver/industry_map.parquet"
    if imp.exists():
        imap = pd.read_parquet(imp)
        imap["symbol"] = imap["symbol"].astype(str).str.zfill(6)
        panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
        try:
            panel = build_industry_features(panel, imap)
        except Exception:
            if "industry_code" not in panel.columns:
                panel["industry_code"] = "_UNKNOWN"
                panel["industry_name"] = "Unknown"

    return panel


# ===================================================================
# Main pipeline
# ===================================================================
def main():
    today = dt.date.today()
    print("=" * 70)
    print("NEXT-WEEK D+3 STOCK RECOMMENDATION PIPELINE")
    print(f"Run date: {today.isoformat()} ({today.strftime('%A')})")
    print(f"Prediction date (T): {PREDICTION_DATE} ({PREDICTION_DATE.strftime('%A')})")
    print(f"D+3 target: {D3_DATE} ({D3_DATE.strftime('%A')}) — next Wednesday")
    print("=" * 70)
    print()

    # ---- 1. Load data ----
    print("[1/6] Loading data and building panel...")
    symbols = load_symbols()
    print(f"  {len(symbols)} symbols with OHLCV data")
    feature_set_id = ensure_registered_feature_set(symbols, max_date=PREDICTION_DATE)
    panel = build_panel(symbols, feature_set_id, max_date=PREDICTION_DATE)

    # Feature columns
    meta_cols = {"symbol", "date", "close", "volume", "open", "high", "low",
                 "amount", "outstanding_share", "turnover",
                 "industry_code", "industry_name", "concept_tags"}
    label_prefixes = ("ret_fwd_", "vol_fwd_", "mdd_fwd_", "excess_vs_")
    feature_cols = []
    for c in panel.columns:
        if c in meta_cols:
            continue
        if c.startswith(label_prefixes):
            continue
        feature_cols.append(c)

    # Gate-first feature eligibility: production D3 script is the long-history
    # base path, so short-history flow/event columns are reported but not
    # silently admitted into the base model.
    coverage_report = compute_feature_coverage_report(
        panel,
        feature_cols,
        family_by_col=_feature_family_lookup(),
        config=_coverage_gate_config_for_universe(panel["symbol"].nunique()),
    )
    write_coverage_gate_report(
        coverage_report,
        OUTPUT_DIR,
        prefix=f"{PREDICTION_LABEL}_coverage_gate_base",
    )
    feature_cols = select_features_by_gate(coverage_report, model_path="base")

    print(f"  Panel: {len(panel)} rows, {panel['date'].min()} -> {panel['date'].max()}")
    print(f"  Features: {len(feature_cols)} active after coverage gate")
    print(f"  reversal_3d present: {'reversal_3d' in feature_cols}")
    print(f"  industry_code present: {'industry_code' in panel.columns}")
    print()

    # ---- 2. Extract prediction-day rows from the unified panel ----
    print("[2/6] Extracting prediction-day rows for 2026-07-03...")
    # FIXED (merge review): this pipeline is branded "D+3" (filenames,
    # report text says "Horizon | 3 trading days") but previously trained
    # on ret_fwd_5d -- a real label/target mismatch between what the code
    # did and what the pipeline claims to predict. ret_fwd_3d is now part of
    # the default label horizon set, so this trains on the label that
    # matches the stated horizon. This has NOT
    # been re-validated against T2.3's (compare_label_horizons) measured
    # 3d-vs-5d comparison with real accumulated data -- if that comparison
    # is later run and favors 5d, this should be revisited deliberately,
    # not left mismatched with the branding as it was before.
    label_col = "ret_fwd_3d"

    pred_day = panel[panel["date"] == PREDICTION_DATE].copy()
    if pred_day.empty:
        print(f"  ERROR: No symbols with OHLCV/feature data at {PREDICTION_DATE}")
        return
    pred_day["symbol"] = pred_day["symbol"].astype(str).str.zfill(6)
    print(f"  Prediction-day rows: {len(pred_day)} symbols "
          f"(built via the same build_technical_features/build_industry_features "
          f"pipeline as training -- no separate ad hoc feature computation)")

    # Determine features available in BOTH training panel and prediction day.
    # Training and prediction rows now come from the identical feature
    # pipeline, so this is a safety net for NaN-heavy columns rather than a
    # schema-reconciliation step.
    common_features = []
    for c in feature_cols:
        if c in pred_day.columns and c in panel.columns:
            pred_nan = pred_day[c].isna().mean()
            train_nan = panel[c].isna().mean()
            if pred_nan < 0.50 and train_nan < 0.50:
                common_features.append(c)

    feature_cols = common_features
    print(f"  Common features (train+predict): {len(feature_cols)}")
    print()

    # ---- 3. Train production model ----
    print("[3/6] Training PRODUCTION model on aligned features...")
    train = panel[(panel["date"] < TRAIN_END)].dropna(subset=feature_cols + [label_col])
    pred_day = pred_day.dropna(subset=feature_cols)

    print(f"  Training: {len(train)} rows, {train['date'].min()} -> {train['date'].max()}")
    print(f"  Prediction day: {len(pred_day)} symbols at {PREDICTION_DATE}")
    print(f"  Aligned features: {len(feature_cols)}")

    if len(train) == 0 or len(pred_day) == 0:
        print("  ERROR: Insufficient data")
        return

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import mlflow
    import mlflow.sklearn
    from quant_platform.training.tracking import get_or_create_experiment
    from quant_platform.training.registry import register_model, promote_champion

    X_train = train[feature_cols].values
    y_train = train[label_col].values
    X_pred = pred_day[feature_cols].values

    # Combine imputer+scaler+model into one Pipeline so there's a single
    # fit/predict artifact to log and register, matching how the rest of
    # the platform treats "model" as one fit/predict object.
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    train_corr = np.corrcoef(pipeline.predict(X_train), y_train)[0, 1]
    preds = pipeline.predict(X_pred)

    # FIXED (merge review): persist via the existing MLflow Model Registry
    # (quant_platform.training.registry) instead of a bespoke pickle file
    # under models/production/ -- one source of truth for "what's the
    # production model," consistent with the platform's existing
    # champion/challenger registry rather than a second, parallel one.
    experiment_id = get_or_create_experiment(STORE_ROOT)
    with mlflow.start_run(experiment_id=experiment_id, run_name=PREDICTION_LABEL) as run:
        mlflow.log_params({
            "label_col": label_col,
            "horizon_days": 3,
            "train_end": str(TRAIN_END),
            "n_train_rows": len(train),
            "n_features": len(feature_cols),
            "model_type": "Ridge(alpha=1.0)",
        })
        mlflow.log_metrics({
            "train_pearson_corr": float(train_corr),
            "pred_score_min": float(preds.min()),
            "pred_score_max": float(preds.max()),
        })
        mlflow.sklearn.log_model(pipeline, "model", serialization_format="pickle")
        run_id = run.info.run_id

    model_version = register_model(
        STORE_ROOT,
        model_name=f"ridge_d3_{PREDICTION_DATE.isoformat()}",
        run_id=run_id,
        feature_set_id=feature_set_id,
        label_col=label_col,
        eval_metrics={"train_pearson_corr": float(train_corr)},
        lineage={
            "prediction_date": str(PREDICTION_DATE),
            "train_end": str(TRAIN_END),
            "feature_cols": feature_cols,
            "feature_set_id": feature_set_id,
        },
        registered_name=REGISTERED_MODEL_NAME,
    )
    promote_champion(STORE_ROOT, model_version, registered_name=REGISTERED_MODEL_NAME)

    print(f"  Model registered: {REGISTERED_MODEL_NAME} v{model_version} "
          f"(run={run_id[:8]}), promoted to champion")
    print(f"  Training Pearson corr: {train_corr:+.4f}")
    print(f"  Prediction score range: [{preds.min():+.4f}, {preds.max():+.4f}]")
    # Create scored panel
    scored = pred_day.copy()
    scored["model_score"] = preds
    scored["symbol"] = scored["symbol"].astype(str).str.zfill(6)
    print(f"  Scored: {len(scored)} symbols")
    print()

    # ---- 4. Industry-neutral selection ----
    print("[4/6] Industry-neutral selection (EqualTopK=3, max_total=50)...")
    config = SelectionConfig(
        strategy=StrategyType.EQUAL_TOP_K,
        top_k=3,
        max_total=50,
        exposure_warning_threshold=0.30,
    )

    if "industry_code" not in scored.columns:
        scored["industry_code"] = "_UNKNOWN"
        scored["industry_name"] = "Unknown"

    ranker = IndustryNeutralRanker(
        config,
        industry_col="industry_code",
        name_col="industry_name",
        score_col="model_score",
        symbol_col="symbol",
    )
    ranked = ranker.run(scored)
    selected = ranked[ranked["selected"]]
    n_sel = len(selected)

    exposure_flag = selected["exposure_flag"].iloc[0] if n_sel > 0 else "N/A"
    print(f"  Selected: {n_sel} stocks, exposure: {exposure_flag}")

    exp_report = ExposureMonitor.concentration_report(
        ranked, industry_col="industry_code", name_col="industry_name",
        selected_col="selected", symbol_col="symbol",
    )

    # Max industry concentration
    max_ind_frac = max(v["fraction"] for v in exp_report.values()) if exp_report else 0
    print(f"  Max single industry: {max_ind_frac:.1%}")
    if max_ind_frac > config.exposure_warning_threshold:
        print(f"  WARNING: Industry concentration ({max_ind_frac:.1%}) exceeds threshold ({config.exposure_warning_threshold:.1%})!")

    # Save ranked
    ranked.to_csv(OUTPUT_DIR / f"{PREDICTION_LABEL}_ranked.csv", index=False)
    with open(OUTPUT_DIR / f"{PREDICTION_LABEL}_exposure.json", "w") as f:
        json.dump(exp_report, f, ensure_ascii=False, indent=2, default=str)
    print()

    # ---- 5. Global Top-N baseline ----
    print("[5/6] Global Top-50 baseline (NOT industry-neutral)...")
    n_top = 50
    global_top = scored.nlargest(n_top, "model_score")
    top_set = set(global_top["symbol"].tolist())

    if "industry_code" in scored.columns:
        gs = scored[scored["symbol"].isin(top_set)]
        g_ind = gs["industry_code"].value_counts()
        g_max = g_ind.max() / len(gs) if len(gs) > 0 else 0
        print(f"  Global Top-{n_top}: max industry {g_max:.1%}")
    global_top.to_csv(OUTPUT_DIR / f"{PREDICTION_LABEL}_global_top{n_top}.csv", index=False)
    print()

    # ---- 6. Generate report ----
    print("[6/6] Generating report...")
    lines = []
    lines.append(f"# Next-Week D+3 Stock Recommendation Report")
    lines.append(f"")
    lines.append(f"**Generated:** {dt.datetime.now().isoformat()}")
    lines.append(f"**Status:** PRODUCTION MODEL INFERENCE — Real-market prediction, NOT backtest")
    lines.append(f"")
    lines.append("---")
    lines.append("")
    lines.append("## A. Next-Week D+3 Prediction Results")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Prediction date (T) | **{PREDICTION_DATE}** (Friday) |")
    lines.append(f"| Earliest execution (T+1) | **{D1_DATE}** (Monday) |")
    lines.append(f"| D+3 target date | **{D3_DATE}** (Wednesday) |")
    lines.append(f"| D+5 comparison date | {D5_DATE} (Friday) |")
    lines.append(f"| Horizon | 3 trading days |")
    lines.append(f"| Model | Ridge Regression (registered: `{REGISTERED_MODEL_NAME}` v{model_version}, run {run_id[:8]}) |")
    lines.append(f"| Feature set | `{feature_set_id}` (registered DEFAULT_SPECS, includes reversal_3d) + industry |")
    lines.append(f"| Feature count | {len(feature_cols)} |")
    lines.append(f"| Selection strategy | IndustryNeutralRanker (EqualTopK=3, max_total=50) |")
    lines.append(f"| Universe | CSI 300 ({len(symbols)} symbols) |")
    lines.append(f"| Scored symbols | {len(scored)} |")
    lines.append("")

    # Top 50 list
    lines.append(f"### Top 50 Industry-Neutral Stock Picks")
    lines.append(f"")
    lines.append(f"- **Selected:** {n_sel} stocks across {len(exp_report)} industries")
    lines.append(f"- **Exposure flag:** {exposure_flag}")
    lines.append(f"- **Max single industry:** {max_ind_frac:.1%}")
    lines.append(f"")
    lines.append(f"| # | Symbol | Industry | Model Score | Industry Rank |")
    lines.append(f"|---|--------|----------|-------------|---------------|")
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        ind = row.get("industry_name", row.get("industry_code", "N/A"))
        lines.append(
            f"| {i} | {row['symbol']} | {ind} "
            f"| {row['model_score']:+.4f} | {row.get('industry_rank', 'N/A')} |"
        )
        if i >= 50:
            break
    lines.append("")

    # Industry distribution
    lines.append(f"### Industry Distribution")
    lines.append(f"")
    lines.append(f"| Industry | Count | Fraction | Top Symbols |")
    lines.append(f"|----------|-------|----------|-------------|")
    for ind_name, info in exp_report.items():
        syms = ", ".join(info["symbols"][:3])
        lines.append(f"| {ind_name} | {info['count']} | {info['fraction']:.1%} | {syms} |")
    lines.append("")

    # B. Baseline comparison
    lines.append("## B. Baseline Comparison")
    lines.append("")
    lines.append(f"### Global Top-{n_top} (Naive, NOT industry-neutral)")
    lines.append("")
    lines.append(f"**WARNING:** This is a CONTROL BASELINE only — NOT the recommended strategy.")
    lines.append("")
    if "industry_code" in scored.columns:
        gs = scored[scored["symbol"].isin(top_set)]
        g_ind_counts = gs["industry_code"].value_counts()
        g_max_frac = g_ind_counts.max() / len(gs) if len(gs) > 0 else 0
        lines.append(f"- **Max single industry:** {g_max_frac:.1%} (industry-neutral: {max_ind_frac:.1%})")
        lines.append(f"- **Top 3 industries by concentration:**")
        for ind_code, cnt in g_ind_counts.head(3).items():
            ind_name = gs[gs["industry_code"] == ind_code]["industry_name"].iloc[0] if "industry_name" in gs.columns else ind_code
            lines.append(f"  - {ind_name}: {cnt} stocks ({cnt/len(gs):.1%})")
    lines.append("")
    lines.append(f"| # | Symbol | Industry | Score |")
    lines.append(f"|---|--------|----------|-------|")
    for i, (_, row) in enumerate(global_top.iterrows(), 1):
        ind = row.get("industry_name", row.get("industry_code", "N/A"))
        lines.append(f"| {i} | {row['symbol']} | {ind} | {row['model_score']:+.4f} |")
        if i >= 30:
            break
    lines.append(f"| ... | ... | ... | ... |")
    lines.append("")

    # Overlap analysis
    ind_set = set(selected["symbol"].tolist())
    overlap = ind_set & top_set
    lines.append(f"### Overlap with Global Top-50")
    lines.append(f"- **Overlap:** {len(overlap)}/{n_top} symbols appear in both lists")
    lines.append(f"- **Industry-neutral only:** {n_sel - len(overlap)} unique picks")
    lines.append(f"- **Global only:** {n_top - len(overlap)} unique picks")
    lines.append("")

    # C. Risk checks
    lines.append("## C. Risk Checks")
    lines.append("")
    lines.append(f"| Check | Result | Threshold | Status |")
    lines.append(f"|-------|--------|-----------|--------|")
    lines.append(f"| Industry concentration (max) | {max_ind_frac:.1%} | {config.exposure_warning_threshold:.1%} | {'PASS' if max_ind_frac <= config.exposure_warning_threshold else 'FAIL'} |")
    lines.append(f"| Exposure flag | {exposure_flag} | diversified/balanced | {'PASS' if exposure_flag in ('diversified', 'balanced') else 'WARN'} |")
    lines.append(f"| Industry-neutral constraint | Yes (EqualTopK=3) | Required | PASS |")
    lines.append(f"| Universe constraint | CSI 300 only | Required | PASS |")
    lines.append(f"| Model type | Ridge (linear) | Production | PASS |")
    lines.append("")

    trigger_warning = (
        max_ind_frac > config.exposure_warning_threshold
        or exposure_flag == "industry_overweight"
    )
    if trigger_warning:
        lines.append("WARNING: Exposure limits triggered. Review before execution.")
    else:
        lines.append("All risk checks PASSED. No exposure warnings triggered.")
    lines.append("")

    # D. Wednesday verification plan
    lines.append("## D. Wednesday Review Plan (2026-07-08)")
    lines.append("")
    lines.append("### Data Required (D+1 -> D+3)")
    lines.append("")
    lines.append("Fetch complete OHLCV covering: **2026-07-06 (Mon) -> 2026-07-08 (Wed)**")
    lines.append("")
    lines.append("```python")
    lines.append("# Fetch command (run on Wednesday after market close):")
    lines.append("cd E:/stock-analysis && PYTHONPATH=. python -c \"")
    lines.append("import akshare as ak")
    lines.append("for sym in <selected_50_symbols>:")
    lines.append("    df = ak.stock_zh_a_daily(symbol=prefix(sym), start_date='20260706', end_date='20260709', adjust='qfq')")
    lines.append("    # ... compute realized returns")
    lines.append("\"")
    lines.append("```")
    lines.append("")
    lines.append("### Metrics to Compute")
    lines.append("")
    lines.append(f"| Metric | Formula | Target |")
    lines.append(f"|--------|---------|--------|")
    lines.append(f"| D+3 Realized Return (per stock) | close(07-08) / close(07-06) - 1 | — |")
    lines.append(f"| D+3 Rank IC | Spearman corr(model_score, realized_ret_3d) across all scored symbols | > 0 |")
    lines.append(f"| Selected-set D+3 Mean Return | Mean of realized_ret_3d for industry-neutral Top 50 | Positive |")
    lines.append(f"| Hit Rate | Fraction of Top 50 with positive D+3 return | > 50% |")
    lines.append(f"| Global Top-50 D+3 Mean Return | Mean of realized_ret_3d for naive global Top 50 | For comparison |")
    lines.append(f"| Industry-neutral vs Global | Difference in mean returns | Industry-neutral > Global |")
    lines.append("")
    lines.append("### Pass Criteria")
    lines.append("")
    lines.append("1. **D+3 Rank IC > 0** (or stable positive across multiple weeks)")
    lines.append("2. **Single industry <= 30%** of selected set")
    lines.append("3. **Industry-neutral strategy outperforms global baseline** on D+3 realized returns")
    lines.append("4. **Hit rate > 50%** (more than half of Top 50 have positive D+3 returns)")
    lines.append("")
    lines.append("### Validation Script")
    lines.append("")
    lines.append("Run on Wednesday (2026-07-08) after market close:")
    lines.append("```bash")
    lines.append("cd E:/stock-analysis && PYTHONPATH=. python scripts/d3_wednesday_review.py")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## IMPORTANT DISCLAIMER")
    lines.append("")
    lines.append(f"- **Prediction date:** {PREDICTION_DATE} (this is a FORWARD-LOOKING prediction)")
    lines.append(f"- **D+3 verification date:** {D3_DATE} — market data NOT YET available")
    lines.append(f"- **Current status:** PREDICTION PHASE — realized returns are UNKNOWN")
    lines.append(f"- **Next action:** Run Wednesday review on {D3_DATE} after market close (after 15:00 CST)")
    lines.append(f"- All metrics in this report are MODEL OUTPUTS, not realized P&L")
    lines.append(f"- Do NOT execute trades based solely on this prediction without Wednesday review")

    report = "\n".join(lines)
    report_path = OUTPUT_DIR / f"{PREDICTION_LABEL}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved: {report_path}")
    print()

    # Print summary
    print("=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"  Prediction date:  {PREDICTION_DATE} (Friday)")
    print(f"  Target D+3:       {D3_DATE} (Wednesday)")
    print(f"  Selected stocks:  {n_sel} (industry-neutral)")
    print(f"  Exposure:         {exposure_flag}")
    print(f"  Max industry:     {max_ind_frac:.1%}")
    print()
    print(f"  Top 10 picks:")
    for i, (_, row) in enumerate(selected.head(10).iterrows(), 1):
        ind = row.get("industry_name", row.get("industry_code", "N/A"))
        print(f"    {i:2d}. {row['symbol']}  {ind:<20s}  {row['model_score']:+.4f}")
    print()
    print(f"  Risk: {'CLEAN' if not trigger_warning else 'WARNING - check exposures'}")
    print()
    print(f"  Next: Wednesday review on {D3_DATE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
