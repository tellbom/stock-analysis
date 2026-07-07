"""
quant_platform.cli
==================
Unified command-line entry point for the quant research platform.

Usage
-----
# Full pipeline: P0 collect → P4B enrich → P1 features → P2 model
python -m quant_platform.cli run --store-root /data/lake --universe csi300

# Run individual stages
python -m quant_platform.cli collect   --store-root /data/lake --universe csi300
python -m quant_platform.cli enrich    --store-root /data/lake --universe csi300
python -m quant_platform.cli features  --store-root /data/lake
python -m quant_platform.cli model     --store-root /data/lake
python -m quant_platform.cli diagnose  --store-root /data/lake
python -m quant_platform.cli status    --store-root /data/lake

# Override universe symbols from CSV (when AKShare is network-blocked)
python -m quant_platform.cli collect --store-root /data/lake --universe csi300 \
    --symbols-csv /path/to/symbols.csv

Subcommands
-----------
  run       Full pipeline end-to-end (collect → enrich → features → model)
  collect   P0: universe + calendar + OHLCV + index ingestion
  enrich    P4B: valuation / industry / capital flow / margin / lockup ingest
  features  P1+P4: feature engineering + labels + leakage check
  model     P2+P4A: walk-forward OOS + Ridge baseline + alpha verdict
  diagnose  P4C: single-factor IC diagnostic + collinearity pruning
  status    Print current lake contents and quality report

Phase 4 changes (relative to original P0–P3 CLI)
-------------------------------------------------
  collect   - added --with-index flag to fetch CSI 300 index OHLCV (P4A-05)
  enrich    - NEW: runs all P4B data source collectors serially
  features  - primary label defaults to ret_fwd_5d (was ret_fwd_20d)
            - horizons default to [1, 3, 5, 10, 20] (added 3d/10d, P4A-04)
            - flags: --include-valuation / --include-industry /
                     --include-flow / --include-margin / --include-lockup
            - builds excess_vs_csi300 labels when index OHLCV available
  model     - walk-forward OOS replaces static lockbox (--walk-forward)
            - adds Ridge linear baseline run (P4A-06)
            - shows subperiod IC stability index (P4A-02)
            - accepts --n-windows / --window-months for walk-forward config
            - legacy lockbox preserved with --use-lockbox flag
  diagnose  - NEW: runs FeatureICReport + FeaturePruner + optional regime
  status    - shows P4B silver directories coverage
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
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
    print(f"  [OK] {text}", flush=True)


def _warn(text: str) -> None:
    print(f"  [WARN] {text}", flush=True)


def _info(text: str) -> None:
    print(f"  -> {text}", flush=True)


def _resolve_symbols(args: argparse.Namespace, store_root: Path) -> list[str]:
    """Shared helper: resolve current universe symbols or raise."""
    from quant_platform.ingest.universe_service import UniverseService
    svc = UniverseService(args.universe, store_root)
    try:
        return svc.get_symbols_as_of()
    except FileNotFoundError:
        _warn("Universe membership not found. Run 'collect' first.")
        return []


def _auto_detect_feature_set(store_root: Path) -> str | None:
    """Return the most recently created feature set id, or None."""
    from quant_platform.store.lake import features_root
    fr = features_root(store_root)
    if not fr.exists():
        return None
    dirs = sorted([d for d in fr.iterdir() if d.is_dir()])
    return dirs[-1].name if dirs else None


def _build_feature_cols(panel) -> list[str]:
    """Standard feature column filter: numeric, non-meta, non-label."""
    candidates, _ = _feature_candidates(panel)
    return [c for c in candidates if panel[c].notna().any()]


def _feature_candidates(panel) -> tuple[list[str], dict[str, str]]:
    """Return numeric feature candidates and excluded all-NaN reasons."""
    import pandas as pd
    meta      = {"symbol", "date", "close"}
    lbl_like  = {c for c in panel.columns if c.startswith(
        ("ret_fwd_", "vol_fwd_", "mdd_fwd_", "excess_vs_", "residual_ret_")
    )}
    str_cols  = {"fund_period_end", "fund_period_type", "fund_announce_date",
                 "industry_code", "industry_name", "concept_tags", "lock_type"}
    exclude = meta | lbl_like | str_cols
    candidates = [
        c for c in panel.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(panel[c])
    ]
    excluded = {c: "all-NaN" for c in candidates if not panel[c].notna().any()}
    return candidates, excluded


def _feature_family_lookup() -> dict[str, str]:
    """Map feature names to audit families used before training."""
    from quant_platform.features.registry import TECHNICAL_SPECS, CROSS_SECTIONAL_SPECS
    from quant_platform.features.valuation import VALUATION_SPECS
    from quant_platform.features.industry import INDUSTRY_SPECS

    family_by_col = {"volume": "raw_aux", "pe_ttm": "raw_aux", "pb": "raw_aux",
                     "turnover_pct": "raw_aux"}
    for spec in TECHNICAL_SPECS:
        family_by_col[spec.name] = "technical"
    for spec in CROSS_SECTIONAL_SPECS:
        family_by_col[spec.name] = "cross_sectional"
    for spec in VALUATION_SPECS:
        family_by_col[spec.name] = "valuation"
    for spec in INDUSTRY_SPECS:
        family_by_col[spec.name] = "industry"
    for mod_name, attr, family in [
        ("quant_platform.features.flow", "FLOW_SPECS", "flow"),
        ("quant_platform.features.sector_flow", "SECTOR_FLOW_SPECS", "sector_flow"),
        ("quant_platform.features.margin", "MARGIN_SPECS", "margin"),
        ("quant_platform.features.event", "LOCKUP_SPECS", "event"),
        ("quant_platform.features.event", "ANNOUNCEMENT_EVENT_SPECS", "announcement_events"),
        ("quant_platform.features.event", "DRAGON_TIGER_SPECS", "dragon_tiger"),
        ("quant_platform.features.event", "BLOCK_TRADE_SPECS", "block_trade"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            for spec in getattr(mod, attr, []):
                family_by_col[spec.name] = family
        except ImportError:
            pass
    for col in (
        "fund_revenue", "fund_net_profit", "fund_eps", "fund_roe",
        "fund_lag_days",
    ):
        family_by_col[col] = "fundamental"
    return family_by_col


def _coverage_gate_config_for_universe(n_symbols: int):
    """Scale the 250/300 coverage rule for small tests or custom universes."""
    import math
    from quant_platform.evaluation.coverage_gate import CoverageGateConfig

    threshold = min(250, max(1, math.ceil(n_symbols * 0.80)))
    return CoverageGateConfig(
        recent_symbol_threshold=threshold,
        recent_20d_symbol_threshold=threshold,
    )


def _apply_coverage_gate(
    panel,
    feature_cols: list[str],
    *,
    store_root: Path,
    model_path: str,
    prefix: str,
) -> tuple[list[str], object]:
    """Run coverage gate, persist report, and return allowed features."""
    from quant_platform.evaluation.coverage_gate import (
        compute_feature_coverage_report,
        select_features_by_gate,
        write_coverage_gate_report,
    )
    from quant_platform.features.registry import feature_metadata_lookup

    family_by_col = _feature_family_lookup()
    cfg = _coverage_gate_config_for_universe(panel["symbol"].nunique())
    report = compute_feature_coverage_report(
        panel,
        feature_cols,
        family_by_col=family_by_col,
        feature_metadata=feature_metadata_lookup(),
        config=cfg,
    )
    out_dir = store_root / "reports"
    csv_path, md_path = write_coverage_gate_report(report, out_dir, prefix=prefix)
    allowed = select_features_by_gate(report, model_path=model_path)
    _info(
        f"Coverage gate ({model_path}): {len(feature_cols)} -> {len(allowed)} "
        f"features  report={csv_path.name}"
    )
    _info(f"Coverage gate markdown: {md_path}")
    return allowed, report


def _audit_training_features(
    panel,
    feature_cols: list[str],
    include_valuation: bool = False,
    include_industry: bool = False,
) -> dict:
    """Print and validate the exact feature list that will enter training."""
    from quant_platform.features.valuation import VALUATION_SPECS
    from quant_platform.features.industry import INDUSTRY_SPECS

    _, excluded = _feature_candidates(panel)
    family_by_col = _feature_family_lookup()
    feature_families = {c: family_by_col.get(c, "raw_aux") for c in feature_cols}

    print("\nFeature audit:")
    print(f"  final feature_count = {len(feature_cols)}")
    counts: dict[str, int] = {}
    for fam in feature_families.values():
        counts[fam] = counts.get(fam, 0) + 1
    for fam in (
        "technical", "cross_sectional", "raw_aux", "valuation", "industry",
        "flow", "sector_flow", "margin", "event", "fundamental",
        "announcement_events", "dragon_tiger", "block_trade",
    ):
        print(f"  {fam:16s}: {counts.get(fam, 0)}")
    print("  final LightGBM feature columns:")
    for col in feature_cols:
        print(f"    - {col} [{feature_families[col]}]")
    if excluded:
        print("  excluded feature candidates:")
        for col, reason in sorted(excluded.items()):
            print(f"    - {col}: {reason}")

    valuation_cols = {s.name for s in VALUATION_SPECS}
    industry_cols = {s.name for s in INDUSTRY_SPECS}
    if include_valuation and not (valuation_cols & set(feature_cols)):
        raise RuntimeError(
            "--include-valuation was set, but no valuation features entered the training list"
        )
    if include_industry and not (industry_cols & set(feature_cols)):
        raise RuntimeError(
            "--include-industry was set, but no industry features entered the training list"
        )

    return {
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "families": feature_families,
        "family_counts": counts,
        "excluded": excluded,
    }


# ---------------------------------------------------------------------------
# Subcommand: collect  (P0 + P4A index)
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> int:
    """P0: universe + calendar + OHLCV ingestion; optionally fetch index OHLCV."""
    from quant_platform.store.lake import init_lake
    from quant_platform.ingest.universe_service import UniverseService
    from quant_platform.ingest.calendar_service import CalendarService
    from quant_platform.ingest.catalog import CatalogDrivenCollector

    store_root = Path(args.store_root)
    _banner(f"P0 COLLECT  |  universe={args.universe}  |  store={store_root}")

    init_lake(store_root)

    if getattr(args, "index_only", False):
        _step("Fetching CSI 300 index OHLCV only (P4A-05)")
        try:
            from quant_platform.ingest.index_collector import IndexCollector
            ic = IndexCollector(store_root)
            result = ic.run(start_date=args.start_date or "2015-01-01")
            _done(f"Index OHLCV: 000300 rows_new={result.get('000300', 0)}")
            return 0
        except Exception as exc:
            _warn(f"Index OHLCV collection failed: {exc}")
            return 1

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

    # --- P4A-05 + F4: CSI 300 index OHLCV, with offline proxy fallback ---
    if getattr(args, "with_index", True):
        _step("Fetching CSI 300 index OHLCV (P4A-05)")
        from quant_platform.store.lake import index_ohlcv_path
        index_path = index_ohlcv_path(store_root, "000300")
        try:
            from quant_platform.ingest.index_collector import IndexCollector
            ic = IndexCollector(store_root)
            ic.run(start_date=args.start_date or "2015-01-01")
            if index_path.exists():
                _done("Index OHLCV: 000300 up to date")
            else:
                raise RuntimeError("index collector returned without writing 000300")
        except Exception as exc:
            _warn(f"Index OHLCV collection failed: {exc}")
            if not index_path.exists():
                _step("Building offline CSI 300 proxy from constituent OHLCV (F4)")
                try:
                    from quant_platform.ingest.index_proxy import build_index_proxy
                    build_index_proxy(store_root, symbols=symbols)
                    _warn(
                        "Using equal-weighted constituent proxy for CSI 300; "
                        "collect the real 000300 index when network is available."
                    )
                except Exception as proxy_exc:
                    _warn(
                        "Proxy build also failed; excess_vs_csi300 labels will be skipped: "
                        f"{proxy_exc}"
                    )

    # --- Quality report ---
    _step("Running data quality report")
    from quant_platform.store.quality_report import run_quality_report
    report = run_quality_report(store_root, universe_key=args.universe, write_file=True)
    print()
    report.print_summary()

    return 0 if not report.has_errors else 1


# ---------------------------------------------------------------------------
# Subcommand: enrich  (P4B)
# ---------------------------------------------------------------------------

def cmd_enrich(args: argparse.Namespace) -> int:
    """P4B: collect valuation, industry, capital flow, margin, lockup data."""
    import datetime as dt

    store_root = Path(args.store_root)
    _banner(f"P4B ENRICH  |  universe={args.universe}  |  store={store_root}")

    symbols = _resolve_symbols(args, store_root)
    if not symbols:
        return 1
    _done(f"Universe: {len(symbols)} symbols")

    errors: list[str] = []

    # --- Valuation (Tencent Finance, batch) ---
    if not getattr(args, "skip_valuation", False):
        _step(f"Collecting valuation data (PE/PB/市值/换手率) for {len(symbols)} symbols")
        try:
            from quant_platform.ingest.valuation_collector import ValuationCollector
            vc = ValuationCollector(store_root)
            res = vc.run(symbols)
            n_ok = sum(1 for v in res.values() if v)
            _done(f"Valuation: {n_ok}/{len(symbols)} symbols written")
        except Exception as exc:
            _warn(f"Valuation collection failed: {exc}")
            errors.append("valuation")

    # --- Industry (Eastmoney, serial, ~5 min) ---
    if not getattr(args, "skip_industry", False):
        _step(f"Collecting industry classification for {len(symbols)} symbols  (~5 min)")
        try:
            from quant_platform.ingest.industry_collector import IndustryCollector
            ic = IndustryCollector(store_root, fetch_concepts=True)
            imap = ic.run(symbols)
            _done(f"Industry map: {len(imap)} SCD rows")
        except Exception as exc:
            _warn(f"Industry collection failed: {exc}")
            errors.append("industry")

    # --- Capital flow (Eastmoney push2his, ~5 min) ---
    if not getattr(args, "skip_flow", False):
        _step(f"Collecting capital flow (资金流) for {len(symbols)} symbols  (~5 min)")
        try:
            from quant_platform.ingest.flow_collector import FundFlowCollector
            fc = FundFlowCollector(store_root)
            res = fc.run(symbols)
            n_ok = sum(1 for v in res.values() if v > 0)
            _done(f"Fund flow: {n_ok}/{len(symbols)} symbols had new data")
        except Exception as exc:
            _warn(f"Fund flow collection failed: {exc}")
            errors.append("flow")

    # --- Margin trading (Eastmoney datacenter, ~5 min) ---
    if not getattr(args, "skip_margin", False):
        _step(f"Collecting margin trading (融资融券) for {len(symbols)} symbols  (~5 min)")
        try:
            from quant_platform.ingest.margin_collector import MarginCollector
            mc = MarginCollector(store_root)
            res = mc.run(symbols)
            n_ok = sum(1 for v in res.values() if v > 0)
            _done(f"Margin: {n_ok} symbols had data (rest not margin-eligible)")
        except Exception as exc:
            _warn(f"Margin collection failed: {exc}")
            errors.append("margin")

    # --- Lockup expiry (Eastmoney datacenter, ~5 min) ---
    if not getattr(args, "skip_lockup", False):
        _step(f"Collecting lockup expiry (限售解禁) for {len(symbols)} symbols  (~5 min)")
        try:
            from quant_platform.ingest.lockup_collector import LockupCollector
            lc = LockupCollector(store_root)
            res = lc.run(symbols)
            n_ok = sum(1 for v in res.values() if v > 0)
            _done(f"Lockup: {n_ok} symbols have upcoming expiry events")
        except Exception as exc:
            _warn(f"Lockup collection failed: {exc}")
            errors.append("lockup")

    if getattr(args, "with_sector_flow", False):
        _step("Collecting sector fund-flow proxy data (行业资金流; explicit opt-in)")
        try:
            import pandas as pd
            from quant_platform.ingest.sector_fund_flow_collector import SectorFundFlowCollector
            from quant_platform.store.lake import industry_map_path

            imap_path = industry_map_path(store_root)
            if not imap_path.exists():
                raise RuntimeError("industry_map.parquet missing; run industry collector first")
            imap = pd.read_parquet(imap_path)
            sector_names = sorted(
                str(x) for x in imap.get("industry_name", pd.Series(dtype=str)).dropna().unique()
                if str(x).strip()
            )
            sfc = SectorFundFlowCollector(store_root)
            res = sfc.run_sectors(sector_names)
            n_ok = sum(1 for v in res.values() if v > 0)
            _done(f"Sector fund-flow proxy: {n_ok}/{len(sector_names)} sectors written")
        except Exception as exc:
            _warn(f"Sector fund-flow proxy collection failed: {exc}")
            errors.append("sector_flow")

    if errors:
        _warn(f"Enrich completed with failures: {errors}")
        _warn("Failed sources will be silently omitted from feature building.")
    else:
        _done("All P4B data sources collected successfully")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: features  (P1 + P4A/B)
# ---------------------------------------------------------------------------

def cmd_features(args: argparse.Namespace) -> int:
    """P1+P4: feature engineering + multi-horizon labels + leakage check."""
    import pandas as pd
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.features.registry import DEFAULT_SPECS
    from quant_platform.labels.builder import (
        build_labels, build_label_panel,
        DEFAULT_HORIZONS, PRIMARY_LABEL_COL,
    )
    from quant_platform.labels.leakage_harness import run_leakage_harness
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path

    store_root  = Path(args.store_root)
    label_col   = args.label
    horizons    = args.horizons

    _banner(
        f"P1+P4 FEATURES  |  label={label_col}  "
        f"horizons={horizons}  store={store_root}"
    )

    symbols = _resolve_symbols(args, store_root)
    if not symbols:
        return 1
    _done(f"Universe: {len(symbols)} symbols")

    # --- Feature pipeline ---
    _step("Computing technical + P4B features")
    include_val  = getattr(args, "include_valuation", False)
    include_ind  = getattr(args, "include_industry", False)
    include_flow = getattr(args, "include_flow", False)
    include_sector_flow = getattr(args, "include_sector_flow", False)
    include_marg = getattr(args, "include_margin", False)
    include_lock = getattr(args, "include_lockup", False)
    include_fund = getattr(args, "include_fundamentals", False)

    _info(
        f"valuation={include_val}  industry={include_ind}  "
        f"flow={include_flow}  margin={include_marg}  "
        f"lockup={include_lock}  fundamentals={include_fund}  "
        f"sector_flow={include_sector_flow}"
    )

    pipe = FeaturePipeline(
        store_root=store_root,
        project_root=getattr(args, "project_root", None),
        include_fundamentals=include_fund,
        include_valuation=include_val,
        include_industry=include_ind,
        include_flow=include_flow,
        include_margin=include_marg,
        include_sector_flow=include_sector_flow,
    )
    fset_id = pipe.run(symbols, specs=DEFAULT_SPECS)
    _done(f"Technical features written  feature_set_id={fset_id}")

    # --- Multi-horizon labels (P4A-04) ---
    _step(f"Building forward-return labels  horizons={horizons}")
    results = build_labels(store_root, symbols, horizons=horizons)
    n_ok = sum(1 for v in results.values() if v > 0)
    _done(f"Labels: {n_ok}/{len(symbols)} symbols written")

    # --- Assemble panel (triggers P4B panel-level builders) ---
    _step("Assembling panel with P4B feature builders")
    panel = pipe.build_panel(symbols, fset_id, add_cross_sectional=True)
    feature_cols = _build_feature_cols(panel)
    try:
        _audit_training_features(
            panel,
            feature_cols,
            include_valuation=include_val,
            include_industry=include_ind,
        )
    except RuntimeError as exc:
        _warn(str(exc))
        return 1
    _done(f"Panel: {len(panel):,} rows × {panel.shape[1]} columns")

    # --- Lockup features (separate panel builder since it needs lockup data) ---
    if include_lock:
        _step("Building lockup expiry features (P4C-03)")
        try:
            from quant_platform.features.event import (
                build_lockup_features, load_lockup_panel,
            )
            from quant_platform.features.valuation import load_valuation_panel
            lockup_panel = load_lockup_panel(store_root, symbols)
            val_panel = load_valuation_panel(store_root, symbols) if include_val else None
            if not lockup_panel.empty:
                panel = build_lockup_features(panel, lockup_panel, val_panel)
                _done("Lockup features added")
            else:
                _warn("No lockup data found — run 'enrich --skip-* except lockup' first")
        except ImportError as exc:
            _warn(f"Lockup features unavailable: {exc}")

    # --- Build label panel (includes excess_vs_csi300 if index available) ---
    _step("Building label panel")
    label_panel = build_label_panel(
        store_root, symbols, horizons=horizons, add_excess_csi300=True
    )
    _done(f"Label panel: {len(label_panel):,} rows, columns: "
          f"{[c for c in label_panel.columns if c not in ('symbol','date')]}")

    # --- Excess-vs-industry labels (P4B-05) ---
    if include_ind and "industry_code" in panel.columns:
        _step("Building excess-vs-industry labels (P4B-05)")
        try:
            from quant_platform.features.industry import build_excess_vs_industry_labels
            from quant_platform.ingest.industry_collector import load_industry_map
            imap = load_industry_map(store_root)
            if not imap.empty:
                label_panel = build_excess_vs_industry_labels(
                    label_panel, imap, horizons=horizons
                )
                _done("Excess-vs-industry labels added to label panel")
        except ImportError as exc:
            _warn(f"Industry excess labels unavailable: {exc}")

    # --- Leakage harness ---
    _step("Running leakage harness")
    ohlcv_frames = []
    for sym in symbols[:50]:
        df = read_ohlcv(ohlcv_path(store_root, sym))
        if not df.empty:
            ohlcv_frames.append(df)
    ohlcv_df = pd.concat(ohlcv_frames, ignore_index=True) if ohlcv_frames else pd.DataFrame()

    harness = run_leakage_harness(panel, label_panel, ohlcv_df, horizons=horizons)
    harness.print_summary()

    if not harness.passed:
        _warn("Leakage harness FAILED — do not proceed to model training.")
        return 1

    _done(f"Leakage harness passed  feature_set_id={fset_id}")
    print(f"\n  feature_set_id = {fset_id}")
    print(f"  primary label  = {label_col}")
    print(f"  horizons       = {horizons}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: model  (P2 + P4A walk-forward + Ridge)
# ---------------------------------------------------------------------------

def cmd_model(args: argparse.Namespace) -> int:
    """P2+P4A: Walk-forward OOS + Ridge baseline + LightGBM + alpha verdict."""
    import pandas as pd
    import numpy as np

    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.labels.builder import build_label_panel
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path
    from quant_platform.training.lgbm_model import fit_oof, fit_final_model
    from quant_platform.training.tracking import get_or_create_experiment, RunLogger, make_manifest
    from quant_platform.evaluation.metrics import evaluate
    from quant_platform.evaluation.baselines import run_baseline_gauntlet
    from quant_platform.evaluation.backtest import run_backtest
    from quant_platform.evaluation.robustness import run_robustness_tests
    from quant_platform.evaluation.alpha_verdict import render_verdict
    from quant_platform.features.registry import DEFAULT_SPECS

    store_root = Path(args.store_root)
    label_col  = args.label
    horizon    = args.horizon
    n_splits   = args.n_splits
    use_wf     = not getattr(args, "use_lockbox", False)

    _banner(
        f"P2+P4A MODEL  |  label={label_col}  horizon={horizon}d  "
        f"eval={'walk-forward' if use_wf else 'lockbox'}"
    )

    # --- Load symbols + panel ---
    symbols = _resolve_symbols(args, store_root)
    if not symbols:
        return 1

    feat_set_id = args.feature_set_id or _auto_detect_feature_set(store_root)
    if not feat_set_id:
        _warn("No feature sets found. Run 'features' first.")
        return 1
    if not args.feature_set_id:
        _warn(f"No --feature-set-id given, using latest: {feat_set_id}")

    include_val = getattr(args, "include_valuation", False)
    include_ind = getattr(args, "include_industry", False)
    include_flow = getattr(args, "include_flow", False)
    include_sector_flow = getattr(args, "include_sector_flow", False)
    include_marg = getattr(args, "include_margin", False)
    gate_model_path = getattr(args, "gate_model_path", "base")
    _info(
        f"feature flags: valuation={include_val}  industry={include_ind}  "
        f"flow={include_flow}  sector_flow={include_sector_flow}  "
        f"margin={include_marg}  gate={gate_model_path}"
    )

    _step("Assembling feature + label panel")
    import inspect

    pipeline_kwargs = {
        "store_root": store_root,
        "project_root": getattr(args, "project_root", None),
        "include_valuation": include_val,
        "include_industry": include_ind,
    }
    if "include_flow" in inspect.signature(FeaturePipeline).parameters:
        pipeline_kwargs["include_flow"] = include_flow
    if "include_sector_flow" in inspect.signature(FeaturePipeline).parameters:
        pipeline_kwargs["include_sector_flow"] = include_sector_flow
    if "include_margin" in inspect.signature(FeaturePipeline).parameters:
        pipeline_kwargs["include_margin"] = include_marg
    pipe = FeaturePipeline(**pipeline_kwargs)
    panel = pipe.build_panel(symbols, feat_set_id, add_cross_sectional=True)
    if panel.empty:
        _warn("Panel is empty — no feature files found.")
        return 1

    # Merge all available label horizons
    all_horizons = getattr(args, "horizons", [horizon])
    label_panel  = build_label_panel(
        store_root, symbols, horizons=all_horizons, add_excess_csi300=True
    )
    label_cols_to_merge = [
        c for c in label_panel.columns
        if c not in ("symbol", "date") and c not in panel.columns
    ]
    panel = panel.merge(
        label_panel[["symbol", "date"] + label_cols_to_merge],
        on=["symbol", "date"], how="inner",
    )
    if label_col not in panel.columns:
        _warn(f"Label '{label_col}' not found. Run 'features' with matching horizon.")
        return 1

    # F1: attach canonical 'close' for baselines without producing close_x/close_y.
    # This column is excluded from feature_cols by _build_feature_cols(), so it
    # does not enter the model feature matrix.
    close_frames = []
    for sym in symbols:
        df_c = read_ohlcv(ohlcv_path(store_root, sym))
        if not df_c.empty and "close" in df_c.columns:
            df_c["date"] = pd.to_datetime(df_c["date"]).dt.date
            close_frames.append(df_c[["symbol", "date", "close"]])
    if close_frames:
        drop_close = [c for c in panel.columns if c in ("close", "close_x", "close_y")]
        if drop_close:
            panel = panel.drop(columns=drop_close)
        panel = panel.merge(
            pd.concat(close_frames, ignore_index=True),
            on=["symbol", "date"], how="left",
        )
    if "close" not in panel.columns:
        _warn("Could not attach 'close'; momentum/mean-rev baselines will return NaN")

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

    feature_cols = _build_feature_cols(panel)

    # Apply pruning (P4C-02) if a pruning log exists
    try:
        from quant_platform.features.pruning import FeaturePruner
        pruner = FeaturePruner(store_root)
        active_cols = pruner.get_active_feature_cols(feature_cols)
        if len(active_cols) < len(feature_cols):
            _info(
                f"Pruning applied: {len(feature_cols)} -> {len(active_cols)} features"
            )
            feature_cols = active_cols
    except ImportError:
        pass

    if not getattr(args, "skip_coverage_gate", False):
        feature_cols, coverage_report = _apply_coverage_gate(
            panel,
            feature_cols,
            store_root=store_root,
            model_path=gate_model_path,
            prefix=f"coverage_gate_{label_col}_{gate_model_path}",
        )
        if not feature_cols:
            _warn("Coverage gate rejected all features; inspect the coverage report.")
            return 1
    else:
        coverage_report = None

    try:
        feature_audit = _audit_training_features(
            panel,
            feature_cols,
            include_valuation=include_val,
            include_industry=include_ind,
        )
    except RuntimeError as exc:
        _warn(str(exc))
        return 1

    _done(
        f"Panel: {len(panel):,} rows  {panel['symbol'].nunique()} symbols  "
        f"{len(feature_cols)} features  "
        f"{panel['date'].min().date()} -> {panel['date'].max().date()}"
    )

    lgbm_params = {
        "objective": "regression", "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
        "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42,
        "verbose": -1, "n_jobs": -1,
    }

    exp_id = get_or_create_experiment(store_root, "quant_platform_p4")

    with RunLogger(store_root, exp_id, run_name=f"model_{label_col}") as run_log:
        run_log.log_params({
            "feature_set_id": feat_set_id, "label_col": label_col,
            "horizon": horizon, "n_splits": n_splits,
            "eval_method": "walk_forward" if use_wf else "lockbox",
            **lgbm_params,
        })

        # ----------------------------------------------------------------
        # Walk-forward OOS evaluation (P4A-03) - primary path
        # ----------------------------------------------------------------
        wf_result = None
        if use_wf:
            _step(
                f"Walk-forward OOS evaluation  "
                f"(n_windows={args.n_windows}, window_months={args.window_months}d, "
                f"horizon={horizon})"
            )
            from quant_platform.evaluation.walk_forward import WalkForwardEvaluator
            from quant_platform.training.lgbm_model import build_lgbm_pipeline

            wf = WalkForwardEvaluator(
                n_windows=args.n_windows,
                window_months=args.window_months,
                horizon=horizon,
                cost_bps=args.cost_bps,
            )
            wf_result = wf.run(
                panel, feature_cols, label_col,
                model_factory=lambda: build_lgbm_pipeline(lgbm_params),
            )
            wf_result.print_summary()
            run_log.log_metrics({
                "wf_agg_rank_ic":       wf_result.agg_rank_ic_mean,
                "wf_agg_icir":          wf_result.agg_icir,
                "wf_agg_sharpe":        wf_result.agg_sharpe,
                "wf_ic_sign_stability": wf_result.ic_sign_stability,
                "wf_indep_periods":     wf_result.total_independent_periods,
            })

        # ----------------------------------------------------------------
        # LightGBM OOF on full train panel (for baselines + robustness)
        # ----------------------------------------------------------------
        _step(f"Fitting LightGBM OOF  (n_splits={n_splits}, horizon={horizon})")
        oof = fit_oof(
            panel, feature_cols, label_col,
            n_splits=n_splits, horizon=horizon,
            lgbm_params=lgbm_params, seed=42,
        )
        _done(f"OOF: {oof.oof_predictions.notna().sum():,} predictions, {oof.n_folds} folds")

        oof_eval = evaluate(
            oof.oof_predictions, panel[label_col],
            pd.to_datetime(panel["date"]), label_col=label_col,
        )
        oof_eval.print_summary()
        run_log.log_metrics(oof_eval.summary_dict())

        # ----------------------------------------------------------------
        # P4A-06: Ridge linear baseline diagnostic
        # ----------------------------------------------------------------
        _step("Fitting Ridge linear baseline (P4A-06)")
        try:
            from quant_platform.training.model_zoo import RidgeModel, fit_zoo_model_oof
            ridge = RidgeModel()
            ridge_preds, _ = fit_zoo_model_oof(
                ridge, panel, feature_cols, label_col,
                n_splits=n_splits, horizon=horizon,
            )
            ridge_eval = evaluate(
                ridge_preds, panel[label_col],
                pd.to_datetime(panel["date"]), label_col=f"{label_col}_ridge",
            )
            ridge_ic   = ridge_eval.rank_ic_mean
            lgbm_ic    = oof_eval.rank_ic_mean
            ratio      = lgbm_ic / ridge_ic if abs(ridge_ic) > 1e-6 else float("nan")
            _done(
                f"Ridge OOF Rank IC = {ridge_ic:+.4f}  |  "
                f"LightGBM / Ridge ratio = {ratio:.2f}"
            )
            if not np.isnan(ratio) and ratio > 1.3:
                _warn(
                    f"GBM/Ridge ratio {ratio:.2f} > 1.3 - GBM may be fitting "
                    "nonlinear noise.  Consider reducing num_leaves."
                )
            run_log.log_metrics({
                "ridge_rank_ic": ridge_ic, "ridge_icir": ridge_eval.icir,
                "lgbm_ridge_ratio": ratio,
            })
        except ImportError:
            _warn("RidgeModel not available - skipping baseline (merge model_zoo.py)")

        # ----------------------------------------------------------------
        # Baseline gauntlet
        # ----------------------------------------------------------------
        _step("Running baseline gauntlet")
        oof_aligned = oof.oof_predictions.reset_index(drop=True)
        baseline_table = run_baseline_gauntlet(
            panel, label_col, oof_aligned, model_name="LightGBM_OOF", seed=42,
        )
        print("\n" + baseline_table.to_string())

        # ----------------------------------------------------------------
        # OOF backtest (on full panel - for sanity check)
        # ----------------------------------------------------------------
        _step("Running OOF cost-aware backtest")
        bt_panel = panel.copy()
        bt_panel["pred"] = oof.oof_predictions
        oof_bt = run_backtest(
            bt_panel, pred_col="pred", return_col=label_col,
            cost_bps=args.cost_bps, horizon=horizon,
        )
        oof_bt.print_summary()
        run_log.log_metrics({f"oof_bt_{k}": v for k, v in oof_bt.summary_dict().items()})

        # ----------------------------------------------------------------
        # Robustness + null tests (P4A-01 embargo fix already in module)
        # ----------------------------------------------------------------
        _step("Running robustness and null tests")
        rob = run_robustness_tests(
            panel, feature_cols, label_col,
            baseline_oof=oof.oof_predictions,
            n_splits=n_splits, horizon=horizon,
            shuffle_threshold=0.05,
        )
        rob.print_summary()
        run_log.log_metrics(rob.summary_dict())

        # P4A-02: surface subperiod stability
        if not np.isnan(rob.subperiod_ic_ratio):
            _info(
                f"Subperiod IC ratio = {rob.subperiod_ic_ratio:.3f}  "
                f"- {rob.subperiod_interpretation if hasattr(rob, 'subperiod_interpretation') else ''}"
            )

        # ----------------------------------------------------------------
        # Legacy lockbox path (--use-lockbox flag)
        # ----------------------------------------------------------------
        lb_eval = lb_bt = None
        if not use_wf:
            from quant_platform.training.splitter import make_lockbox_split
            lockbox_mo = getattr(args, "lockbox_months", 12)
            _step(f"Carving lockbox split ({lockbox_mo} months)")
            train_val, lockbox = make_lockbox_split(
                panel, lockbox_months=lockbox_mo, horizon=horizon
            )
            _done(
                f"train_val={len(train_val):,}  lockbox={len(lockbox):,} rows  "
                f"(lockbox from {pd.to_datetime(lockbox['date']).min().date()})"
            )
            _step("Fitting final model + lockbox evaluation (ONE-TIME ONLY)")
            final_model = fit_final_model(train_val, feature_cols, label_col, lgbm_params)
            lb_pred = pd.Series(
                final_model.predict(lockbox[feature_cols]), index=lockbox.index
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

        # ----------------------------------------------------------------
        # Alpha verdict (P4A walk-forward integrated)
        # ----------------------------------------------------------------
        _step("Rendering alpha verdict")
        verdict = render_verdict(
            store_root, oof_eval, baseline_table, oof_bt, rob,
            lockbox_eval=lb_eval,
            lockbox_backtest=lb_bt,
            walk_forward_result=wf_result,
        )
        run_log.log_params({"verdict": verdict.verdict, "confidence": verdict.confidence})
        if use_wf:
            run_log.log_metrics({
                "verdict_wf_rank_ic":   verdict.wf_agg_rank_ic,
                "verdict_wf_icir":      verdict.wf_agg_icir,
                "verdict_wf_sharpe":    verdict.wf_agg_sharpe,
            })
        else:
            run_log.log_metrics({
                "verdict_lockbox_rank_ic": verdict.lockbox_rank_ic,
                "verdict_lockbox_sharpe":  verdict.lockbox_sharpe,
            })

        # ----------------------------------------------------------------
        # Reproducibility manifest
        # ----------------------------------------------------------------
        manifest = make_manifest(
            store_root, feat_set_id, feature_cols, label_col, lgbm_params, seed=42,
            extra={"verdict": verdict.verdict, "run_id": run_log.run_id},
        )
        run_log.log_artifact_json(manifest, f"manifest_{run_log.run_id[:8]}.json")
        print(f"\n  run_id = {run_log.run_id}")

    # --- Persist JSON report ---
    _step("Writing model report")
    _write_model_report(
        store_root, feat_set_id, label_col, horizon,
        len(symbols), panel, feature_cols, oof, oof_eval, oof_bt,
        lb_eval, lb_bt, wf_result, baseline_table, rob, verdict,
    )

    print(f"\n{'='*68}")
    print(f"  VERDICT: {verdict.verdict}  (confidence: {verdict.confidence})")
    if wf_result and not np.isnan(wf_result.agg_rank_ic_mean):
        print(f"  Walk-forward Rank IC : {wf_result.agg_rank_ic_mean:+.4f}")
        print(f"  Walk-forward ICIR    : {wf_result.agg_icir:+.4f}")
        print(f"  IC sign stability    : {wf_result.ic_sign_stability:.2f}")
    print(f"{'='*68}\n")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: diagnose  (P4C)
# ---------------------------------------------------------------------------

def cmd_diagnose(args: argparse.Namespace) -> int:
    """P4C: single-factor IC diagnostic + collinearity pruning + optional regime analysis."""
    import pandas as pd

    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.labels.builder import build_label_panel, DEFAULT_HORIZONS

    store_root = Path(args.store_root)
    _banner(f"P4C DIAGNOSE  |  store={store_root}")

    symbols = _resolve_symbols(args, store_root)
    if not symbols:
        return 1

    feat_set_id = args.feature_set_id or _auto_detect_feature_set(store_root)
    if not feat_set_id:
        _warn("No feature sets found. Run 'features' first.")
        return 1

    # --- Assemble panel ---
    _step("Assembling panel for IC diagnostics")
    pipe  = FeaturePipeline(store_root=store_root,
                            project_root=getattr(args, "project_root", None))
    panel = pipe.build_panel(symbols, feat_set_id, add_cross_sectional=True)
    if panel.empty:
        _warn("Panel is empty.")
        return 1

    label_panel = build_label_panel(
        store_root, symbols, horizons=DEFAULT_HORIZONS, add_excess_csi300=True
    )
    label_cols_to_merge = [
        c for c in label_panel.columns if c not in ("symbol", "date") and c not in panel.columns
    ]
    panel = panel.merge(
        label_panel[["symbol", "date"] + label_cols_to_merge],
        on=["symbol", "date"], how="left",
    )

    # Attach close prices for IC decay computation
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path
    close_frames = []
    for sym in symbols:
        df_c = read_ohlcv(ohlcv_path(store_root, sym))
        if not df_c.empty and "close" in df_c.columns:
            df_c["date"] = pd.to_datetime(df_c["date"]).dt.date
            close_frames.append(df_c[["symbol", "date", "close"]])
    if close_frames:
        panel = panel.merge(
            pd.concat(close_frames, ignore_index=True),
            on=["symbol", "date"], how="left",
        )

    feature_cols = _build_feature_cols(panel)
    _done(f"Panel: {len(panel):,} rows, {len(feature_cols)} features")

    # --- P4C-01: Single-factor IC diagnostic ---
    _step(f"Computing per-feature IC diagnostic ({len(feature_cols)} features)")
    try:
        from quant_platform.evaluation.feature_ic import compute_feature_ic_report

        label_cols = [
            c for c in [
                "ret_fwd_1d", "ret_fwd_5d", "ret_fwd_10d", "ret_fwd_20d",
                "excess_vs_csi300_5d",
            ] if c in panel.columns
        ]
        ic_report = compute_feature_ic_report(
            panel, feature_cols, label_cols=label_cols, store_root=store_root
        )
        ic_report.print_summary(top_n=20)

        # Surface pruning candidates
        candidates = ic_report.pruning_candidates(ic_threshold=0.01, tstat_threshold=1.5)
        if candidates:
            _info(f"Weak IC features (candidates for pruning): {candidates}")

    except ImportError as exc:
        _warn(f"feature_ic module not available: {exc}")
        ic_report = None

    # --- P4C-02: Collinearity pruning ---
    if ic_report is not None and not getattr(args, "skip_pruning", False):
        _step(f"Running collinearity pruning  (threshold={args.corr_threshold:.0%})")
        try:
            from quant_platform.features.pruning import FeaturePruner
            pruner = FeaturePruner(store_root)
            prune_result = pruner.run(
                ic_report,
                primary_label=args.label,
                corr_threshold=args.corr_threshold,
            )
            prune_result.print_summary()

            active_cols = pruner.get_active_feature_cols(
                feature_cols, pruning_result=prune_result
            )
            _done(
                f"Pruning: {len(feature_cols)} -> {len(active_cols)} active features  "
                f"({prune_result.n_pruned} pruned)"
            )

            # Record in research ledger
            try:
                from quant_platform.evaluation.research_ledger import ResearchLedger
                ledger = ResearchLedger(store_root)
                ledger.record(
                    model_name="FeaturePruning",
                    feature_set_id=feat_set_id,
                    label_col=args.label,
                    fold_seed="pruning",
                    raw_icir=0.0,
                    n_dates=panel["date"].nunique(),
                    notes=prune_result.to_ledger_note(),
                )
            except Exception:
                pass
        except ImportError as exc:
            _warn(f"Pruning module not available: {exc}")

    # --- P4C-05: Regime analysis (optional) ---
    if getattr(args, "with_regime", False):
        _step("Running walk-forward regime analysis (P4C-05)")
        try:
            from quant_platform.evaluation.regime_analysis import RegimeAnalyser
            from quant_platform.features.registry import (
                TECHNICAL_SPECS, CROSS_SECTIONAL_SPECS,
            )

            tech_names = {s.name for s in TECHNICAL_SPECS + CROSS_SECTIONAL_SPECS}
            val_names  = {c for c in feature_cols if c.startswith("cs_") and
                          c not in tech_names and "flow" not in c and "margin" not in c}
            flow_names  = {c for c in feature_cols if "flow" in c}
            marg_names  = {c for c in feature_cols if "margin" in c or "rzrq" in c}
            event_names = {c for c in feature_cols if "unlock" in c}

            feature_groups = {}
            if any(c in feature_cols for c in tech_names):
                feature_groups["technical"] = [c for c in feature_cols if c in tech_names]
            if val_names:
                feature_groups["valuation"] = [c for c in feature_cols if c in val_names]
            if flow_names:
                feature_groups["flow"] = [c for c in feature_cols if c in flow_names]
            if marg_names:
                feature_groups["margin"] = [c for c in feature_cols if c in marg_names]
            if event_names:
                feature_groups["event"] = [c for c in feature_cols if c in event_names]

            label_col = getattr(args, "label", "ret_fwd_5d")
            if label_col not in panel.columns:
                label_col = "ret_fwd_5d" if "ret_fwd_5d" in panel.columns else None

            if label_col and feature_groups:
                analyser = RegimeAnalyser(
                    store_root=store_root,
                    n_windows=getattr(args, "n_windows", 5),
                    window_months=getattr(args, "window_months", 12),
                    horizon=getattr(args, "horizon", 5),
                )
                regime_report = analyser.run(
                    panel, feature_groups, label_col,
                    save_csv=True, record_to_ledger=True,
                )
                regime_report.print_summary()
            else:
                _warn("Regime analysis skipped: insufficient feature groups or label missing")
        except ImportError as exc:
            _warn(f"Regime analysis not available: {exc}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    """Print current lake contents, P4B silver coverage, and quality report."""
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
    print(f"  OHLCV symbols       : {n_ohlcv}")

    # Calendar
    has_cal = calendar_path(store_root).exists()
    print(f"  Calendar            : {'[OK]' if has_cal else '[missing] run collect'}")

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
        print("  Universe            : [missing] run collect")

    # P4B silver tables
    silver = store_root / "silver"
    p4b_tables = [
        ("index_ohlcv", "Index OHLCV (P4A-05)"),
        ("valuation",   "Valuation PE/PB/mcap (P4B-01)"),
        ("fund_flow",   "Capital flow (P4B-06)"),
        ("margin",      "Margin trading (P4B-08)"),
        ("lockup",      "Lockup expiry (P4C-03)"),
    ]
    for sub, label in p4b_tables:
        n = _count(silver / sub) if (silver / sub).exists() else 0
        status = f"{n} symbol files" if n > 0 else "[missing] run enrich"
        print(f"  {label:28s}: {status}")

    # Industry map (single file)
    imap_path = silver / "industry_map.parquet"
    if imap_path.exists():
        import pandas as pd
        idf = pd.read_parquet(imap_path)
        print(f"  Industry map (P4B-03)         : {len(idf)} SCD rows, "
              f"{idf['symbol'].nunique()} symbols")
    else:
        print(f"  Industry map (P4B-03)         : [missing] run enrich")

    # Feature sets
    f_root = features_root(store_root)
    if f_root.exists():
        for fdir in sorted(f_root.iterdir()):
            if fdir.is_dir():
                n = _count(fdir)
                print(f"  Features [{fdir.name}]: {n} symbol files")
    else:
        print("  Features            : [missing] run features")

    # Labels
    l_root = labels_root(store_root)
    if l_root.exists():
        for ldir in sorted(l_root.iterdir()):
            n = _count(ldir)
            print(f"  Labels [{ldir.name}]: {n} symbol files")
    else:
        print("  Labels              : [missing] run features")

    # Evaluation outputs
    eval_dir = store_root / "evaluation"
    if eval_dir.exists():
        ic_reports = sorted(eval_dir.glob("feature_ic_report_*.csv"))
        regime_reports = sorted(eval_dir.glob("regime_analysis_*.csv"))
        print(f"  IC diagnostic reports : {len(ic_reports)} CSV(s)")
        print(f"  Regime reports        : {len(regime_reports)} CSV(s)")
        pruning_log = eval_dir / "feature_pruning_log.parquet"
        if pruning_log.exists():
            import pandas as pd
            pl = pd.read_parquet(pruning_log)
            print(f"  Pruning log           : {len(pl)} pruning decisions")

    # Catalog
    cat_p = catalog_path(store_root)
    if cat_p.exists():
        cat = CollectorCatalog(store_root)
        s = cat.summary()
        print(f"  Catalog             : {s.get('total', 0)} entries  {s.get('by_status', {})}")
    else:
        print("  Catalog             : [missing] no collection runs yet")

    # Reports
    for fname in ("quality_report.txt", "alpha_verdict.txt"):
        p = store_root / fname
        print(f"  {fname:24s}: {'[OK]' if p.exists() else '[missing]'}")

    # Quality report summary
    qr = store_root / "quality_report.txt"
    if qr.exists():
        print("\n--- Quality report ---")
        for line in qr.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(("[ERROR]", "[WARN]", "PASS", "FAIL", "SURVIVORSHIP")):
                print(f"  {line.strip()}")

    # Verdict summary
    av = store_root / "alpha_verdict.txt"
    if av.exists():
        print("\n--- Alpha verdict ---")
        for line in av.read_text(encoding="utf-8").splitlines():
            if "VERDICT" in line or "walk-forward" in line.lower() or "confidence" in line.lower():
                print(f"  {line.strip()}")

    return 0


# ---------------------------------------------------------------------------
# Subcommand: fuse  (Gate-first base/recent fusion)
# ---------------------------------------------------------------------------

def _normalise_ranked_input(df, prefix: str):
    """Normalize score/rank/pct columns for gate fusion."""
    import pandas as pd

    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    if "trade_date" not in out.columns and "date" in out.columns:
        out["trade_date"] = out["date"]
    if f"{prefix}_score" not in out.columns:
        for candidate in ("model_score", "score", "pred", "prediction"):
            if candidate in out.columns:
                out[f"{prefix}_score"] = out[candidate]
                break
    if f"{prefix}_score" not in out.columns:
        raise ValueError(f"{prefix} input needs {prefix}_score or model_score")
    out[f"{prefix}_score"] = pd.to_numeric(out[f"{prefix}_score"], errors="coerce")
    if f"{prefix}_rank" not in out.columns:
        out[f"{prefix}_rank"] = out[f"{prefix}_score"].rank(
            ascending=False, method="first"
        ).astype(int)
    if f"{prefix}_pct" not in out.columns:
        out[f"{prefix}_pct"] = out[f"{prefix}_score"].rank(
            ascending=True, pct=True, method="average"
        )
    keep = ["symbol", "trade_date", f"{prefix}_score", f"{prefix}_rank", f"{prefix}_pct"]
    for extra in ("risk_flags", "event_flags"):
        if extra in out.columns:
            keep.append(extra)
    return out[keep]


def cmd_fuse(args: argparse.Namespace) -> int:
    """Gate-first fusion of base and recent ranked outputs."""
    import pandas as pd
    from quant_platform.selection.gate_fusion import (
        gate_first_fusion,
        write_gate_fusion_outputs,
    )

    _banner("D3 GATE-FIRST FUSION")
    base = _normalise_ranked_input(pd.read_csv(args.base_ranked, dtype={"symbol": str}), "base")
    recent = _normalise_ranked_input(pd.read_csv(args.recent_ranked, dtype={"symbol": str}), "recent")

    merged = base.merge(
        recent,
        on=["symbol", "trade_date"],
        how="inner",
        suffixes=("_base", "_recent"),
    )
    if merged.empty:
        _warn("No overlapping (symbol, trade_date) rows between base and recent ranked files.")
        return 1

    for col in ("risk_flags", "event_flags"):
        left = f"{col}_base"
        right = f"{col}_recent"
        if left in merged.columns or right in merged.columns:
            left_values = (
                merged[left].fillna("").astype(str)
                if left in merged.columns else ""
            )
            right_values = (
                merged[right].fillna("").astype(str)
                if right in merged.columns else ""
            )
            merged[col] = (
                left_values
                + ";"
                + right_values
            ).str.strip(";")
        elif col not in merged.columns:
            merged[col] = ""

    fused = gate_first_fusion(merged)
    out_dir = Path(args.output_dir)
    csv_path, md_path = write_gate_fusion_outputs(
        fused,
        out_dir,
        prefix=args.prefix,
        top_n=args.top_n,
    )
    _done(f"Fused ranked -> {csv_path}")
    _done(f"Fusion report -> {md_path}")
    print(f"  rows={len(fused)}  top_tier={fused['gate_tier'].iloc[0]}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run  (full pipeline)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    """Full pipeline: collect -> enrich -> features -> model."""
    _banner("FULL PIPELINE  P0 -> P4B -> P1+P4 -> P2+P4A")

    rc = cmd_collect(args)
    if rc != 0:
        print("\n[ABORT] collect stage failed - stopping.")
        return rc

    if not getattr(args, "skip_enrich", False):
        rc = cmd_enrich(args)
        if rc != 0:
            _warn("Enrich stage had failures - continuing (P4B data will be partial)")

    rc = cmd_features(args)
    if rc != 0:
        print("\n[ABORT] features stage failed - stopping.")
        return rc

    return cmd_model(args)


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------

def _write_model_report(
    store_root, feat_set_id, label_col, horizon, n_symbols, panel,
    feature_cols, oof, oof_eval, oof_bt, lb_eval, lb_bt,
    wf_result, baseline_table, rob, verdict,
) -> None:
    """Write human-readable text + JSON model report."""
    import numpy as np, pandas as pd

    source_fp = _source_fingerprint(Path.cwd())
    lines = [
        "=" * 72,
        f"P2+P4A MODEL REPORT  {dt.datetime.now().isoformat(timespec='seconds')}",
        "=" * 72,
        f"Feature set  : {feat_set_id}",
        f"Label        : {label_col}  |  horizon={horizon}d",
        f"Symbols      : {n_symbols}  |  panel rows={len(panel):,}",
        f"Dates        : {pd.to_datetime(panel['date']).min().date()} -> "
        f"{pd.to_datetime(panel['date']).max().date()}",
        f"Features     : {len(feature_cols)}",
        "",
        "OOF Evaluation:",
        *[f"  {k}: {v}" for k, v in oof_eval.summary_dict().items()],
        "", "OOF Backtest:",
        *[f"  {k}: {v}" for k, v in oof_bt.summary_dict().items()],
    ]

    if wf_result is not None:
        lines += ["", "Walk-Forward OOS Evaluation:"]
        df_wf = wf_result.to_dataframe()
        if not df_wf.empty:
            lines.append(df_wf.to_string(index=False))
        lines += [
            f"  Agg Rank IC : {wf_result.agg_rank_ic_mean:+.4f}",
            f"  Agg ICIR    : {wf_result.agg_icir:+.4f}",
            f"  Agg Sharpe  : {wf_result.agg_sharpe:+.4f}",
            f"  Per-window Sharpe mean: {getattr(wf_result, 'per_window_sharpe_mean', float('nan')):+.4f}",
            f"  IC stability: {wf_result.ic_sign_stability:.2f}",
        ]

    if lb_eval is not None:
        lines += ["", "Lockbox Evaluation:"]
        lines += [f"  {k}: {v}" for k, v in lb_eval.summary_dict().items()]
    if lb_bt is not None:
        lines += ["", "Lockbox Backtest:"]
        lines += [f"  {k}: {v}" for k, v in lb_bt.summary_dict().items()]

    lines += [
        "", "Baselines:", baseline_table.to_string(), "",
        "Robustness:",
        f"  Subperiod IC first={rob.first_half_ric:+.4f}  "
        f"second={rob.second_half_ric:+.4f}  "
        f"ratio={rob.subperiod_ic_ratio:.3f}",
        "", "Fold Metrics:",
        *[
            f"  fold={fm['fold']} n_train={fm['n_train']} "
            f"n_val={fm['n_val']} ic={fm['ic_pearson']:.6f}"
            for fm in oof.fold_metrics
        ],
        "",
        f"VERDICT: {verdict.verdict}  (confidence={verdict.confidence})",
        "Evidence:",
        *[f"  {e}" for e in verdict.evidence],
        "Caveats:",
        *[f"  {c}" for c in verdict.caveats],
        "", "=" * 72,
    ]

    report_path = store_root / f"p2_report_{label_col}.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _done(f"Report -> {report_path}")

    def _default(v):
        if isinstance(v, (np.bool_,)):   return bool(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        raise TypeError(type(v))

    json_doc = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "feature_set_id": feat_set_id, "label_col": label_col,
        "horizon": horizon, "symbols": n_symbols,
        "feature_count": len(feature_cols), "features": feature_cols,
        "oof_eval": oof_eval.summary_dict(),
        "oof_backtest": oof_bt.summary_dict(),
        "robustness": rob.summary_dict(),
        "verdict": verdict.to_dict(),
    }
    if wf_result is not None:
        json_doc["walk_forward"] = {
            "agg_rank_ic":       wf_result.agg_rank_ic_mean,
            "agg_icir":          wf_result.agg_icir,
            "agg_sharpe":        wf_result.agg_sharpe,
            "per_window_sharpe_mean": getattr(wf_result, "per_window_sharpe_mean", float("nan")),
            "ic_sign_stability": wf_result.ic_sign_stability,
            "total_indep_periods": wf_result.total_independent_periods,
        }
    if lb_eval is not None:
        json_doc["lockbox_eval"]    = lb_eval.summary_dict()
        json_doc["lockbox_backtest"] = lb_bt.summary_dict() if lb_bt else {}

    json_path = store_root / f"p2_report_{label_col}.json"
    json_path.write_text(json.dumps(json_doc, indent=2, default=_default), encoding="utf-8")
    _done(f"JSON  -> {json_path}")


def _source_fingerprint(project_root: Path) -> dict:
    """Return hashes for key source files."""
    files = [
        "quant_platform/cli.py",
        "quant_platform/training/splitter.py",
        "quant_platform/training/lgbm_model.py",
        "quant_platform/evaluation/metrics.py",
        "quant_platform/evaluation/robustness.py",
        "quant_platform/evaluation/walk_forward.py",
        "quant_platform/evaluation/alpha_verdict.py",
        "quant_platform/features/technical.py",
    ]
    file_hashes = {}
    for rel in files:
        p = project_root / rel
        file_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

    def _git(git_args):
        try:
            proc = subprocess.run(
                ["git", "-c", f"safe.directory={project_root.as_posix()}", *git_args],
                cwd=project_root, check=False, text=True, capture_output=True,
                encoding="utf-8", errors="replace", timeout=10,
            )
            return proc.stdout.strip() if proc.returncode == 0 else None
        except Exception:
            return None

    git_head   = _git(["rev-parse", "HEAD"])
    git_status = _git(["status", "--short"])
    git_diff   = _git(["diff", "--", "quant_platform", "tests"])
    return {
        "generated_at":    dt.datetime.now().isoformat(timespec="seconds"),
        "git_head":        git_head,
        "git_dirty":       bool(git_status),
        "git_status_short": git_status,
        "git_diff_sha256": hashlib.sha256((git_diff or "").encode()).hexdigest(),
        "files":           file_hashes,
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store-root",   required=True,
                   help="Root directory of the Parquet data lake")
    p.add_argument("--universe",     default="csi300",
                   help="Universe key: csi300 | csi500 | csi1000 (default: csi300)")
    p.add_argument("--project-root", default=None,
                   help="Directory containing technical_indicators.py (default: auto-detect)")


def _add_collect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbols-csv",  default=None,
                   help="Path to CSV with 'symbol' column (bypasses AKShare)")
    p.add_argument("--start-date",   default=None,
                   help="OHLCV history start date YYYY-MM-DD (default: 3 years ago)")
    p.add_argument("--end-date",     default=None,
                   help="OHLCV history end date YYYY-MM-DD (default: today)")
    p.add_argument("--workers",      type=int, default=1,
                   help="Thread workers for OHLCV collection (default: 1)")
    p.add_argument("--no-index",     action="store_true",
                   help="Skip CSI 300 index OHLCV collection")
    p.add_argument("--index-only",   action="store_true",
                   help="Only collect CSI 300 index OHLCV; skip universe, calendar, and stock OHLCV")


def _add_enrich_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--skip-valuation", action="store_true", help="Skip valuation collector")
    p.add_argument("--skip-industry",  action="store_true", help="Skip industry collector")
    p.add_argument("--skip-flow",      action="store_true", help="Skip capital flow collector")
    p.add_argument("--skip-margin",    action="store_true", help="Skip margin trading collector")
    p.add_argument("--skip-lockup",    action="store_true", help="Skip lockup expiry collector")
    p.add_argument("--with-sector-flow", action="store_true",
                   help="Collect sector/industry fund-flow proxy data (explicit opt-in)")


def _add_features_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--label",     default="ret_fwd_5d",       # P4A-04: default changed to 5d
                   help="Primary label column (default: ret_fwd_5d)")
    p.add_argument("--horizons",  type=int, nargs="+", default=[1, 3, 5, 10, 20],  # P4A-04
                   help="Label horizons in trading days (default: 1 3 5 10 20)")
    p.add_argument("--include-fundamentals", action="store_true")
    p.add_argument("--include-valuation",    action="store_true",
                   help="Add P4B valuation/size features (requires enrich)")
    p.add_argument("--include-industry",     action="store_true",
                   help="Add P4B industry-relative features (requires enrich)")
    p.add_argument("--include-flow",         action="store_true",
                   help="Add P4B capital flow features (requires enrich)")
    p.add_argument("--include-sector-flow",  action="store_true",
                   help="Add sector fund-flow proxy features (not stock-level fund_flow)")
    p.add_argument("--include-margin",       action="store_true",
                   help="Add P4B margin trading features (requires enrich)")
    p.add_argument("--include-lockup",       action="store_true",
                   help="Add P4C lockup expiry features (requires enrich)")


def _add_model_args(
    p: argparse.ArgumentParser,
    include_label: bool = True,
    include_horizons: bool = True,
) -> None:
    if include_label:
        p.add_argument("--label",      default="ret_fwd_5d",   # P4A-04: 5d default
                       help="Label column (default: ret_fwd_5d)")
    p.add_argument("--horizon",        type=int, default=5,    # P4A-04: 5d default
                   help="Label horizon matching --label (default: 5)")
    if include_horizons:
        p.add_argument("--horizons",   type=int, nargs="+", default=[1, 3, 5, 10, 20])
    p.add_argument("--feature-set-id", default=None,
                   help="8-char feature set ID (auto-detected if omitted)")
    p.add_argument("--n-splits",       type=int, default=5)
    p.add_argument("--cost-bps",       type=float, default=10.0,
                   help="One-way transaction cost in bps (default: 10)")
    p.add_argument("--n-estimators",   type=int, default=200)
    p.add_argument("--learning-rate",  type=float, default=0.05)
    p.add_argument("--num-leaves",     type=int, default=31)
    if "--include-valuation" not in p._option_string_actions:
        p.add_argument("--include-valuation", action="store_true",
                       help="Add P4B valuation/size features during model panel assembly")
    if "--include-industry" not in p._option_string_actions:
        p.add_argument("--include-industry",  action="store_true",
                       help="Add P4B industry-relative features during model panel assembly")
    if "--include-flow" not in p._option_string_actions:
        p.add_argument("--include-flow",      action="store_true",
                       help="Add short-history capital flow features during model panel assembly")
    if "--include-sector-flow" not in p._option_string_actions:
        p.add_argument("--include-sector-flow", action="store_true",
                       help="Add sector fund-flow proxy features during model panel assembly")
    if "--include-margin" not in p._option_string_actions:
        p.add_argument("--include-margin",    action="store_true",
                       help="Add P4B margin trading features during model panel assembly")
    p.add_argument("--gate-model-path", choices=["base", "recent"], default="base",
                   help="Coverage gate path: base long-history or recent enhanced (default: base)")
    p.add_argument("--skip-coverage-gate", action="store_true",
                   help="Skip feature coverage gate and use the legacy feature filter")
    # Walk-forward (P4A-03) — on by default; --use-lockbox for legacy path
    p.add_argument("--walk-forward",   action="store_true", default=True,
                   help="Use walk-forward OOS evaluation (default: True)")
    p.add_argument("--use-lockbox",    action="store_true",
                   help="Use legacy static lockbox instead of walk-forward")
    p.add_argument("--n-windows",      type=int, default=5,
                   help="Number of walk-forward windows (default: 5)")
    p.add_argument("--window-months",  type=int, default=12,
                   help="Length of each walk-forward window in months (default: 12)")
    p.add_argument("--lockbox-months", type=int, default=12,
                   help="Lockbox length in months (legacy --use-lockbox only)")


def _add_diagnose_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--label",          default="ret_fwd_5d")
    p.add_argument("--horizon",        type=int, default=5)
    p.add_argument("--feature-set-id", default=None)
    p.add_argument("--corr-threshold", type=float, default=0.85,
                   help="Spearman correlation threshold for pruning (default: 0.85)")
    p.add_argument("--skip-pruning",   action="store_true",
                   help="Run IC diagnostic only, skip collinearity pruning")
    p.add_argument("--with-regime",    action="store_true",
                   help="Also run walk-forward regime analysis (P4C-05; slow)")
    p.add_argument("--n-windows",      type=int, default=5)
    p.add_argument("--window-months",  type=int, default=12)


def _add_fuse_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-ranked", required=True,
                   help="CSV with base_score/base_pct or model_score columns")
    p.add_argument("--recent-ranked", required=True,
                   help="CSV with recent_score/recent_pct or model_score columns")
    p.add_argument("--output-dir", required=True,
                   help="Directory for D3_fused_ranked.csv and fusion report")
    p.add_argument("--prefix", default="D3_fused",
                   help="Output prefix (default: D3_fused)")
    p.add_argument("--top-n", type=int, default=50,
                   help="Rows to show in markdown report (default: 50)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_platform.cli",
        description="Quant research platform — unified CLI  (Phase 4A/B/C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Full pipeline: collect → enrich → features → model")
    _add_common_args(p_run)
    _add_collect_args(p_run)
    _add_enrich_args(p_run)
    _add_features_args(p_run)
    _add_model_args(p_run, include_label=False, include_horizons=False)
    p_run.add_argument("--skip-enrich", action="store_true",
                       help="Skip P4B data enrichment in full run")

    # --- collect ---
    p_col = sub.add_parser("collect", help="P0: universe + calendar + OHLCV + index")
    _add_common_args(p_col)
    _add_collect_args(p_col)

    # --- enrich ---
    p_enr = sub.add_parser("enrich", help="P4B: valuation/industry/flow/margin/lockup ingest")
    _add_common_args(p_enr)
    _add_enrich_args(p_enr)

    # --- features ---
    p_feat = sub.add_parser("features", help="P1+P4: features + labels + leakage check")
    _add_common_args(p_feat)
    _add_features_args(p_feat)

    # --- model ---
    p_model = sub.add_parser("model", help="P2+P4A: walk-forward OOS + Ridge + verdict")
    _add_common_args(p_model)
    _add_model_args(p_model)

    # --- diagnose ---
    p_diag = sub.add_parser("diagnose", help="P4C: IC diagnostic + collinearity pruning")
    _add_common_args(p_diag)
    _add_diagnose_args(p_diag)

    # --- status ---
    p_status = sub.add_parser("status", help="Show lake contents and reports")
    _add_common_args(p_status)

    # --- fuse ---
    p_fuse = sub.add_parser("fuse", help="Gate-first fusion for base/recent ranked CSVs")
    _add_fuse_args(p_fuse)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args   = parser.parse_args(argv)

    # --no-index maps to with_index=False
    if hasattr(args, "no_index"):
        args.with_index = not args.no_index

    dispatch = {
        "run":      cmd_run,
        "collect":  cmd_collect,
        "enrich":   cmd_enrich,
        "features": cmd_features,
        "model":    cmd_model,
        "diagnose": cmd_diagnose,
        "status":   cmd_status,
        "fuse":     cmd_fuse,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
