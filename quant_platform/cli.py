"""
quant_platform.cli
==================
Unified command-line entry point for the quant research platform.

Usage
-----
# Full pipeline: collect → features → labels → model → verdict
python -m quant_platform.cli run --store-root /data/lake --universe csi300

# Run individual stages
python -m quant_platform.cli collect  --store-root /data/lake --universe csi300
python -m quant_platform.cli features --store-root /data/lake --label ret_fwd_20d
python -m quant_platform.cli model    --store-root /data/lake --label ret_fwd_20d --horizon 20
python -m quant_platform.cli status   --store-root /data/lake

# Override universe symbols from CSV (when AKShare is network-blocked)
python -m quant_platform.cli collect --store-root /data/lake --universe csi300 \
    --symbols-csv /path/to/symbols.csv

Subcommands
-----------
  run       Full pipeline end-to-end (P0 → P1 → P2)
  collect   P0: universe + calendar + OHLCV ingestion
  features  P1: feature engineering + label construction + leakage check
  model     P2: LightGBM OOF + evaluation + backtest + alpha verdict
  status    Print current lake contents and quality report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


def _step(text: str) -> None:
    print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] {text}", flush=True)


def _done(text: str) -> None:
    print(f"  ✓ {text}", flush=True)


def _warn(text: str) -> None:
    print(f"  ⚠ {text}", flush=True)


# ---------------------------------------------------------------------------
# Subcommand: collect  (P0)
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> int:
    """P0: universe + calendar + OHLCV ingestion."""
    from quant_platform.store.lake import init_lake
    from quant_platform.ingest.universe_service import UniverseService
    from quant_platform.ingest.calendar_service import CalendarService
    from quant_platform.ingest.catalog import CatalogDrivenCollector

    store_root = Path(args.store_root)
    _banner(f"P0 COLLECT  |  universe={args.universe}  |  store={store_root}")

    init_lake(store_root)

    # --- Universe ---
    _step("Building universe membership table")
    svc = UniverseService(args.universe, store_root)
    if args.symbols_csv:
        df = svc.load_from_csv(args.symbols_csv, has_effective_dates=False)
        _done(f"Loaded {len(df)} symbols from {args.symbols_csv}")
    else:
        try:
            df = svc.fetch_and_save()
            _done(f"Fetched {len(df)} symbols via AKShare")
        except Exception as exc:
            _warn(f"AKShare universe fetch failed: {exc}")
            _warn("Use --symbols-csv to provide a local CSV when AKShare is blocked.")
            return 1

    symbols = svc.get_symbols_as_of()
    _done(f"Universe: {len(symbols)} current members")

    # --- Calendar ---
    _step("Building trading calendar")
    cal = CalendarService(store_root)
    cal.build_and_save()
    cov = cal.coverage()
    _done(
        f"Calendar: {cov['total_trading_days']} trading days  "
        f"({cov['first_date']} → {cov['last_date']})  source={cov['source']}"
    )

    # --- OHLCV ---
    _step(f"Collecting OHLCV for {len(symbols)} symbols  (max_workers={args.workers})")
    collector = CatalogDrivenCollector(
        store_root=store_root,
        universe_key=args.universe,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.workers,
    )
    summary = collector.run(symbols=symbols)
    _done(
        f"OHLCV: {summary.succeeded} succeeded, {summary.failed} failed, "
        f"{summary.skipped_by_catalog} already up-to-date"
    )
    if summary.failed_symbols:
        _warn(f"Failed symbols: {summary.failed_symbols[:10]}")

    # --- Quality report ---
    _step("Running data quality report")
    from quant_platform.store.quality_report import run_quality_report
    report = run_quality_report(
        store_root, universe_key=args.universe, write_file=True
    )
    print()
    report.print_summary()

    return 0 if not report.has_errors else 1


# ---------------------------------------------------------------------------
# Subcommand: features  (P1)
# ---------------------------------------------------------------------------

def cmd_features(args: argparse.Namespace) -> int:
    """P1: feature engineering + label construction + leakage check."""
    from quant_platform.ingest.universe_service import UniverseService
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id
    from quant_platform.labels.builder import build_labels, build_label_panel
    from quant_platform.labels.leakage_harness import run_leakage_harness
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path
    import pandas as pd

    store_root = Path(args.store_root)
    _banner(f"P1 FEATURES  |  label={args.label}  |  horizons={args.horizons}")

    # Resolve symbols
    svc = UniverseService(args.universe, store_root)
    try:
        symbols = svc.get_symbols_as_of()
    except FileNotFoundError:
        _warn("Universe membership not found. Run 'collect' first.")
        return 1
    _done(f"Universe: {len(symbols)} symbols")

    # --- Feature pipeline ---
    _step("Computing technical features")
    pipe = FeaturePipeline(
        store_root=store_root,
        project_root=args.project_root,
        include_fundamentals=args.include_fundamentals,
    )
    fset_id = pipe.run(symbols, specs=DEFAULT_SPECS)
    _done(f"Features written  feature_set_id={fset_id}")

    # --- Labels ---
    _step(f"Building forward-return labels  horizons={args.horizons}")
    results = build_labels(store_root, symbols, horizons=args.horizons)
    n_ok = sum(1 for v in results.values() if v > 0)
    _done(f"Labels: {n_ok}/{len(symbols)} symbols written")

    # --- Leakage harness ---
    _step("Running leakage harness")
    panel = pipe.build_panel(symbols, fset_id, add_cross_sectional=True)
    label_panel = build_label_panel(store_root, symbols, horizons=args.horizons)

    ohlcv_frames = []
    for sym in symbols[:50]:   # sample for canary test (50 is enough)
        df = read_ohlcv(ohlcv_path(store_root, sym))
        if not df.empty:
            ohlcv_frames.append(df)
    ohlcv_df = pd.concat(ohlcv_frames, ignore_index=True) if ohlcv_frames else pd.DataFrame()

    harness = run_leakage_harness(panel, label_panel, ohlcv_df, horizons=args.horizons)
    harness.print_summary()

    if not harness.passed:
        _warn("Leakage harness FAILED — do not proceed to model training.")
        return 1

    _done(f"Leakage harness passed  feature_set_id={fset_id}")
    print(f"\n  feature_set_id = {fset_id}  (pass this to 'model' subcommand)")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: model  (P2)
# ---------------------------------------------------------------------------

def cmd_model(args: argparse.Namespace) -> int:
    """P2: LightGBM OOF + evaluation + backtest + alpha verdict + MLflow."""
    import pandas as pd
    import numpy as np

    from quant_platform.ingest.universe_service import UniverseService
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.labels.builder import build_label_panel
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path, feature_path, label_path
    from quant_platform.training.lgbm_model import fit_oof, fit_final_model
    from quant_platform.training.splitter import make_lockbox_split
    from quant_platform.training.tracking import (
        get_or_create_experiment, RunLogger, make_manifest
    )
    from quant_platform.evaluation.metrics import evaluate
    from quant_platform.evaluation.baselines import run_baseline_gauntlet
    from quant_platform.evaluation.backtest import run_backtest
    from quant_platform.evaluation.robustness import run_robustness_tests
    from quant_platform.evaluation.alpha_verdict import render_verdict
    from quant_platform.features.registry import DEFAULT_SPECS

    store_root  = Path(args.store_root)
    label_col   = args.label
    horizon     = args.horizon
    lockbox_mo  = args.lockbox_months
    n_splits    = args.n_splits

    _banner(
        f"P2 MODEL  |  label={label_col}  horizon={horizon}  "
        f"n_splits={n_splits}  lockbox={lockbox_mo}mo"
    )

    # --- Load symbols ---
    svc = UniverseService(args.universe, store_root)
    try:
        symbols = svc.get_symbols_as_of()
    except FileNotFoundError:
        _warn("Universe not found. Run 'collect' first.")
        return 1

    # --- Assemble panel ---
    _step("Assembling feature + label panel")
    feat_set_id = args.feature_set_id
    if not feat_set_id:
        # Try to auto-detect from lake
        from quant_platform.store.lake import features_root
        feat_dirs = [d for d in features_root(store_root).iterdir() if d.is_dir()] \
                    if features_root(store_root).exists() else []
        if not feat_dirs:
            _warn("No feature sets found. Run 'features' first.")
            return 1
        feat_set_id = feat_dirs[-1].name
        _warn(f"No --feature-set-id given, using latest: {feat_set_id}")

    pipe  = FeaturePipeline(store_root=store_root, project_root=args.project_root)
    panel = pipe.build_panel(symbols, feat_set_id, add_cross_sectional=True)
    if panel.empty:
        _warn("Panel is empty — no feature files found for the given feature_set_id.")
        return 1

    label_panel = build_label_panel(store_root, symbols, horizons=[horizon])
    if label_col not in label_panel.columns:
        _warn(f"Label column '{label_col}' not found. Run 'features' with matching horizon.")
        return 1

    panel = panel.merge(
        label_panel[["symbol", "date", label_col]],
        on=["symbol", "date"], how="inner"
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Feature columns = all numeric cols except meta + labels
    meta = {"symbol", "date"}
    label_like = {c for c in panel.columns if c.startswith(("ret_fwd_","vol_fwd_","mdd_fwd_"))}
    fund_str    = {"fund_period_end","fund_period_type","fund_announce_date"}
    exclude = meta | label_like | fund_str | {"close"}
    feature_cols = [
        c for c in panel.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(panel[c])
        and panel[c].notna().any()
    ]
    _done(
        f"Panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, "
        f"{len(feature_cols)} features, dates {panel['date'].min().date()} → {panel['date'].max().date()}"
    )

    # --- Lockbox split ---
    _step(f"Carving lockbox split ({lockbox_mo} months)")
    train_val, lockbox = make_lockbox_split(panel, lockbox_months=lockbox_mo)
    train_val = train_val.sort_values(["date","symbol"]).reset_index(drop=True)
    lockbox   = lockbox.sort_values(["date","symbol"]).reset_index(drop=True)
    _done(
        f"train_val={len(train_val):,} rows  "
        f"(max {pd.to_datetime(train_val['date']).max().date()})   "
        f"lockbox={len(lockbox):,} rows  "
        f"(min {pd.to_datetime(lockbox['date']).min().date()})"
    )

    # --- MLflow setup ---
    exp_id = get_or_create_experiment(store_root, "quant_platform_p2")

    # --- OOF training ---
    _step(f"Fitting LightGBM OOF  (n_splits={n_splits}, horizon={horizon}, embargo={horizon})")
    lgbm_params = {
        "objective": "regression", "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
        "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42,
        "verbose": -1, "n_jobs": -1,
    }

    with RunLogger(store_root, exp_id, run_name=f"oof_{label_col}") as run_log:
        run_log.log_params({
            "feature_set_id": feat_set_id, "label_col": label_col,
            "horizon": horizon, "n_splits": n_splits, "lockbox_months": lockbox_mo,
            **lgbm_params,
        })

        oof = fit_oof(
            train_val, feature_cols, label_col,
            n_splits=n_splits, horizon=horizon,
            lgbm_params=lgbm_params, seed=42,
        )
        _done(f"OOF: {oof.oof_predictions.notna().sum():,} predictions across {oof.n_folds} folds")

        # --- OOF evaluation ---
        _step("Evaluating OOF predictions")
        oof_eval = evaluate(
            oof.oof_predictions,
            train_val[label_col],
            pd.to_datetime(train_val["date"]),
            label_col=label_col,
        )
        oof_eval.print_summary()
        run_log.log_metrics(oof_eval.summary_dict())

        # --- Baseline gauntlet ---
        _step("Running baseline gauntlet")
        oof_aligned = oof.oof_predictions.reset_index(drop=True)
        baseline_table = run_baseline_gauntlet(
            train_val, label_col, oof_aligned,
            model_name="LightGBM_OOF", seed=42,
        )
        print("\n" + baseline_table.to_string())

        # --- OOF backtest ---
        _step("Running cost-aware signal backtest (train_val)")
        tv_bt_panel = train_val.copy()
        tv_bt_panel["pred"] = oof.oof_predictions
        oof_bt = run_backtest(
            tv_bt_panel, pred_col="pred", return_col=label_col,
            cost_bps=args.cost_bps, horizon=horizon,
        )
        oof_bt.print_summary()
        run_log.log_metrics({f"oof_bt_{k}": v for k, v in oof_bt.summary_dict().items()})

        # --- Robustness + null tests ---
        _step("Running robustness and null tests")
        rob = run_robustness_tests(
            train_val, feature_cols, label_col,
            baseline_oof=oof.oof_predictions,
            n_splits=n_splits, horizon=horizon,
            shuffle_threshold=0.05,
        )
        rob.print_summary()
        run_log.log_metrics(rob.summary_dict())

        # --- Final model + lockbox ---
        _step("Fitting final model on train_val")
        final_model = fit_final_model(train_val, feature_cols, label_col, lgbm_params)
        _done("Final model fitted")

        _step("Evaluating on lockbox (ONE-TIME ONLY)")
        lb_pred = pd.Series(
            final_model.predict(lockbox[feature_cols]),
            index=lockbox.index,
        )
        lb_eval = evaluate(
            lb_pred, lockbox[label_col],
            pd.to_datetime(lockbox["date"]),
            label_col=f"{label_col}_lockbox",
        )
        lb_eval.print_summary()
        run_log.log_metrics({f"lb_{k}": v for k, v in lb_eval.summary_dict().items()})

        lb_bt_panel = lockbox.copy()
        lb_bt_panel["pred"] = lb_pred
        lb_bt = run_backtest(
            lb_bt_panel, pred_col="pred", return_col=label_col,
            cost_bps=args.cost_bps, horizon=horizon,
        )
        lb_bt.print_summary()
        run_log.log_metrics({f"lb_bt_{k}": v for k, v in lb_bt.summary_dict().items()})

        # --- Alpha verdict ---
        _step("Rendering alpha verdict")
        verdict = render_verdict(
            store_root, oof_eval, baseline_table, oof_bt, rob,
            lockbox_eval=lb_eval, lockbox_backtest=lb_bt,
        )
        run_log.log_metrics({
            "verdict_lockbox_rank_ic": verdict.lockbox_rank_ic,
            "verdict_lockbox_sharpe":  verdict.lockbox_sharpe,
        })
        run_log.log_params({"verdict": verdict.verdict, "confidence": verdict.confidence})

        # --- Reproducibility manifest ---
        manifest = make_manifest(
            store_root, feat_set_id, feature_cols, label_col, lgbm_params, seed=42,
            extra={"verdict": verdict.verdict, "run_id": run_log.run_id},
        )
        run_log.log_artifact_json(manifest, f"manifest_{run_log.run_id[:8]}.json")

        print(f"\n  run_id = {run_log.run_id}")

    # --- Text report ---
    _step("Writing text report")
    report_lines = _build_text_report(
        feat_set_id, label_col, horizon, len(symbols), panel,
        train_val, lockbox, feature_cols, oof, oof_eval, oof_bt,
        lb_eval, lb_bt, baseline_table, verdict,
    )
    report_path = store_root / f"p2_report_{label_col}.txt"
    json_path   = store_root / f"p2_report_{label_col}.json"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    _done(f"Report → {report_path}")

    # JSON summary
    import numpy as np
    def _default(v):
        if isinstance(v, (np.bool_,)):     return bool(v)
        if isinstance(v, (np.integer,)):   return int(v)
        if isinstance(v, (np.floating,)):  return float(v)
        raise TypeError(type(v))

    json_path.write_text(
        json.dumps({
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "feature_set_id": feat_set_id, "label_col": label_col,
            "horizon": horizon, "lockbox_months": lockbox_mo,
            "symbols": len(symbols),
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "oof_eval": oof_eval.summary_dict(),
            "oof_backtest": oof_bt.summary_dict(),
            "lockbox_eval": lb_eval.summary_dict(),
            "lockbox_backtest": lb_bt.summary_dict(),
            "verdict": verdict.to_dict(),
            "robustness": rob.summary_dict(),
        }, indent=2, default=_default),
        encoding="utf-8",
    )
    _done(f"JSON → {json_path}")

    print(f"\n{'='*68}")
    print(f"  VERDICT: {verdict.verdict}  (confidence: {verdict.confidence})")
    print(f"{'='*68}\n")

    return 0


def _build_text_report(
    feat_set_id, label_col, horizon, n_symbols, panel,
    train_val, lockbox, feature_cols, oof, oof_eval, oof_bt,
    lb_eval, lb_bt, baseline_table, verdict,
) -> list[str]:
    import pandas as pd
    lines = [
        "=" * 72,
        f"P2 LIGHTGBM BASELINE  {dt.datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        f"Feature set  : {feat_set_id}",
        f"Label        : {label_col}  |  horizon={horizon}",
        f"Symbols      : {n_symbols}  |  panel rows={len(panel):,}  |  "
        f"train_val={len(train_val):,}  |  lockbox={len(lockbox):,}",
        f"Dates        : {pd.to_datetime(panel['date']).min().date()} → "
        f"{pd.to_datetime(panel['date']).max().date()}",
        f"Features     : {len(feature_cols)}",
        "",
        "OOF Evaluation:",
    ]
    for k, v in oof_eval.summary_dict().items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Train/Val Backtest:"]
    for k, v in oof_bt.summary_dict().items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Lockbox Evaluation:"]
    for k, v in lb_eval.summary_dict().items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Lockbox Backtest:"]
    for k, v in lb_bt.summary_dict().items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Baselines:", baseline_table.to_string(), ""]
    lines += ["Fold Metrics:"]
    for fm in oof.fold_metrics:
        lines.append(
            f"  fold={fm['fold']} n_train={fm['n_train']} "
            f"n_val={fm['n_val']} ic={fm['ic_pearson']:.6f}"
        )
    lines += [
        "",
        f"VERDICT: {verdict.verdict}  (confidence={verdict.confidence})",
        "Evidence:",
        *[f"  {e}" for e in verdict.evidence],
        "Caveats:",
        *[f"  {c}" for c in verdict.caveats],
        "",
        "=" * 72,
    ]
    return lines


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    """Print current lake contents and quality report."""
    from quant_platform.store.lake import (
        ohlcv_dir, features_root, labels_root, universe_root,
        calendar_path, catalog_path,
    )
    from quant_platform.ingest.catalog import CollectorCatalog

    store_root = Path(args.store_root)
    _banner(f"LAKE STATUS  |  {store_root}")

    def _count(d: Path, pattern: str = "*.parquet") -> int:
        return len(list(d.glob(pattern))) if d.exists() else 0

    # OHLCV
    n_ohlcv = _count(ohlcv_dir(store_root))
    print(f"\n  OHLCV symbols       : {n_ohlcv}")

    # Calendar
    has_cal = calendar_path(store_root).exists()
    print(f"  Calendar            : {'✓' if has_cal else '✗ (run collect)'}")

    # Universe
    u_root = universe_root(store_root)
    if u_root.exists():
        for d in sorted(u_root.iterdir()):
            mp = d / "membership.parquet"
            if mp.exists():
                import pandas as pd
                df = pd.read_parquet(mp)
                open_count = df["out_date"].isna().sum()
                print(f"  Universe [{d.name}]   : {open_count} current members ({len(df)} total rows)")
    else:
        print("  Universe            : ✗ (run collect)")

    # Feature sets
    f_root = features_root(store_root)
    if f_root.exists():
        for fdir in sorted(f_root.iterdir()):
            if fdir.is_dir():
                n = _count(fdir)
                print(f"  Features [{fdir.name}]: {n} symbol files")
    else:
        print("  Features            : ✗ (run features)")

    # Labels
    l_root = labels_root(store_root)
    if l_root.exists():
        for ldir in sorted(l_root.iterdir()):
            n = _count(ldir)
            print(f"  Labels [{ldir.name}]: {n} symbol files")
    else:
        print("  Labels              : ✗ (run features)")

    # Catalog
    cat_p = catalog_path(store_root)
    if cat_p.exists():
        cat = CollectorCatalog(store_root)
        s = cat.summary()
        print(f"  Catalog             : {s.get('total', 0)} entries  {s.get('by_status', {})}")
    else:
        print("  Catalog             : ✗ (no collection runs yet)")

    # Reports
    for fname in ("quality_report.txt", "alpha_verdict.txt"):
        p = store_root / fname
        print(f"  {fname:24s}: {'✓' if p.exists() else '✗'}")

    # Quality report summary
    qr = store_root / "quality_report.txt"
    if qr.exists():
        print("\n--- Quality report ---")
        for line in qr.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(("[ERROR]", "[WARN]", "PASS", "FAIL", "SURVIVORSHIP")):
                print(f"  {line.strip()}")

    # Verdict
    av = store_root / "alpha_verdict.txt"
    if av.exists():
        print("\n--- Alpha verdict ---")
        for line in av.read_text(encoding="utf-8").splitlines():
            if "VERDICT" in line or "confidence" in line.lower():
                print(f"  {line.strip()}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: run  (full pipeline)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline: collect → features → model."""
    _banner("FULL PIPELINE  P0 → P1 → P2")

    rc = cmd_collect(args)
    if rc != 0:
        print("\n[ABORT] collect stage failed — stopping.")
        return rc

    rc = cmd_features(args)
    if rc != 0:
        print("\n[ABORT] features stage failed — stopping.")
        return rc

    rc = cmd_model(args)
    return rc


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store-root",  required=True,
                   help="Root directory of the Parquet data lake")
    p.add_argument("--universe",    default="csi300",
                   help="Universe key: csi300 | csi500 | csi1000 (default: csi300)")
    p.add_argument("--project-root", default=None,
                   help="Directory containing technical_indicators.py (default: auto-detect)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_platform.cli",
        description="Quant research platform — unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Full pipeline: collect → features → model")
    _add_common_args(p_run)
    _add_collect_args(p_run)
    # features + model share --label; define once here for the 'run' subcommand
    p_run.add_argument("--label",       default="ret_fwd_20d",
                        help="Label column and model target (default: ret_fwd_20d)")
    p_run.add_argument("--horizons",    type=int, nargs="+", default=[1, 5, 20])
    p_run.add_argument("--horizon",     type=int,   default=20)
    p_run.add_argument("--include-fundamentals", action="store_true")
    p_run.add_argument("--feature-set-id",  default=None)
    p_run.add_argument("--n-splits",        type=int,   default=5)
    p_run.add_argument("--lockbox-months",  type=int,   default=12)
    p_run.add_argument("--cost-bps",        type=float, default=10.0)
    p_run.add_argument("--n-estimators",    type=int,   default=200)
    p_run.add_argument("--learning-rate",   type=float, default=0.05)
    p_run.add_argument("--num-leaves",      type=int,   default=31)

    # --- collect ---
    p_col = sub.add_parser("collect", help="P0: universe + calendar + OHLCV ingestion")
    _add_common_args(p_col)
    _add_collect_args(p_col)

    # --- features ---
    p_feat = sub.add_parser("features", help="P1: features + labels + leakage check")
    _add_common_args(p_feat)
    _add_features_args(p_feat)

    # --- model ---
    p_model = sub.add_parser("model", help="P2: LightGBM + evaluation + verdict")
    _add_common_args(p_model)
    _add_model_args(p_model)

    # --- status ---
    p_status = sub.add_parser("status", help="Show lake contents and reports")
    _add_common_args(p_status)

    return parser


def _add_collect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbols-csv", default=None,
                   help="Path to CSV with 'symbol' column (bypasses AKShare)")
    p.add_argument("--start-date",  default=None,
                   help="OHLCV history start date YYYY-MM-DD (default: 3 years ago)")
    p.add_argument("--end-date",    default=None,
                   help="OHLCV history end date YYYY-MM-DD (default: today)")
    p.add_argument("--workers",     type=int, default=1,
                   help="Thread workers for OHLCV collection (default: 1, keep low to avoid IP ban)")


def _add_features_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--label",       default="ret_fwd_20d",
                   help="Primary label column (default: ret_fwd_20d)")
    p.add_argument("--horizons",    type=int, nargs="+", default=[1, 5, 20],
                   help="Label horizons in trading days (default: 1 5 20)")
    p.add_argument("--include-fundamentals", action="store_true",
                   help="Include PIT fundamental features (requires T0.7 data)")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--label",           default="ret_fwd_20d")
    p.add_argument("--horizon",         type=int,   default=20,
                   help="Label horizon matching --label (default: 20)")
    p.add_argument("--feature-set-id",  default=None,
                   help="8-char feature set ID (auto-detected if omitted)")
    p.add_argument("--n-splits",        type=int,   default=5)
    p.add_argument("--lockbox-months",  type=int,   default=12)
    p.add_argument("--cost-bps",        type=float, default=10.0,
                   help="One-way transaction cost in basis points (default: 10)")
    p.add_argument("--n-estimators",    type=int,   default=200)
    p.add_argument("--learning-rate",   type=float, default=0.05)
    p.add_argument("--num-leaves",      type=int,   default=31)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # Route to subcommand
    dispatch = {
        "run":      cmd_run,
        "collect":  cmd_collect,
        "features": cmd_features,
        "model":    cmd_model,
        "status":   cmd_status,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
