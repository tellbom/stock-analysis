"""
Run one D3 base/recent/Gate-first recommendation pass.

This script is intentionally date-stamped and one-shot oriented.  It truncates
all feature construction and prediction to actual_as_of_date, writes outputs
with that date in the filename, and does not run yesterday's comparison flow.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE_ROOT = ROOT / "models/data"
REPORT_DIR = STORE_ROOT / "reports"
REQUESTED_AS_OF_DATE = dt.date(2026, 7, 6)

sys.path.insert(0, str(ROOT))

from quant_platform.cli import (  # noqa: E402
    _build_feature_cols,
    _coverage_gate_config_for_universe,
    _feature_family_lookup,
)
from quant_platform.evaluation.coverage_gate import (  # noqa: E402
    compute_feature_coverage_report,
    select_features_by_gate,
    write_coverage_gate_report,
)
from quant_platform.evaluation.explainability import build_explainability_report  # noqa: E402
from quant_platform.features.event import build_lockup_features, load_lockup_panel  # noqa: E402
from quant_platform.features.pipeline import FeaturePipeline  # noqa: E402
from quant_platform.features.registry import DEFAULT_SPECS, feature_metadata_lookup  # noqa: E402
from quant_platform.labels.builder import build_label_panel  # noqa: E402
from quant_platform.selection.fusion_score import actionable_pool  # noqa: E402
from quant_platform.selection.gate_fusion import gate_first_fusion, write_gate_fusion_outputs  # noqa: E402
from quant_platform.selection.reco_cards import build_reco_cards  # noqa: E402
from quant_platform.store.lake import ohlcv_path  # noqa: E402
from quant_platform.store.parquet_store import read_ohlcv  # noqa: E402
from quant_platform.training.lgbm_model import fit_final_model  # noqa: E402


LABEL_COL = "ret_fwd_3d"
HORIZON = 3
BASE_PARAMS = {
    "objective": "regression",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}
RECENT_PARAMS = {
    **BASE_PARAMS,
    "n_estimators": 120,
    "num_leaves": 15,
    "min_child_samples": 20,
}


def _load_symbols() -> list[str]:
    membership = pd.read_parquet(STORE_ROOT / "universe/csi300/membership.parquet")
    symbols = sorted(membership["symbol"].astype(str).str.zfill(6).unique().tolist())
    return [s for s in symbols if ohlcv_path(STORE_ROOT, s).exists()]


def _ohlcv_counts(symbols: list[str]) -> dict[dt.date, int]:
    counts: dict[dt.date, int] = {}
    for sym in symbols:
        df = read_ohlcv(ohlcv_path(STORE_ROOT, sym))
        if df.empty:
            continue
        dates = pd.to_datetime(df["date"], errors="coerce").dt.date.dropna().unique()
        for date in dates:
            if date <= REQUESTED_AS_OF_DATE:
                counts[date] = counts.get(date, 0) + 1
    return counts


def _resolve_actual_as_of(symbols: list[str]) -> tuple[dt.date, str, dict[dt.date, int]]:
    counts = _ohlcv_counts(symbols)
    threshold = min(250, max(1, int(len(symbols) * 0.8)))
    requested_count = counts.get(REQUESTED_AS_OF_DATE, 0)
    if requested_count >= threshold:
        return REQUESTED_AS_OF_DATE, "requested date has sufficient OHLCV coverage", counts
    candidates = [d for d, c in counts.items() if d <= REQUESTED_AS_OF_DATE and c >= threshold]
    if not candidates:
        raise RuntimeError("No complete-enough OHLCV date found at or before requested_as_of_date")
    actual = max(candidates)
    reason = (
        f"requested_as_of_date {REQUESTED_AS_OF_DATE} has OHLCV coverage "
        f"{requested_count}/{len(symbols)}; using latest complete date {actual} "
        f"with coverage {counts.get(actual, 0)}/{len(symbols)}"
    )
    return actual, reason, counts


def _coverage_for_dir(subdir: str, symbols: list[str], as_of: dt.date) -> dict:
    path = STORE_ROOT / subdir
    files = list(path.glob("*.parquet")) if path.exists() else []
    latest_dates: list[dt.date] = []
    as_of_files = 0
    for file in files:
        try:
            df = pd.read_parquet(file)
        except Exception:
            continue
        date_col = next(
            (c for c in ("date", "trade_date", "announce_date", "unlock_date", "event_date", "period_end")
             if c in df.columns),
            None,
        )
        if date_col is None or df.empty:
            continue
        dates = pd.to_datetime(df[date_col], errors="coerce").dt.date.dropna()
        dates = dates[dates <= as_of]
        if dates.empty:
            continue
        latest_dates.append(max(dates))
        if (dates == as_of).any():
            as_of_files += 1
    return {
        "path": subdir,
        "files": len(files),
        "symbol_file_coverage": f"{len(files)}/{len(symbols)}",
        "as_of_files": as_of_files,
        "latest_min": min(latest_dates).isoformat() if latest_dates else None,
        "latest_max": max(latest_dates).isoformat() if latest_dates else None,
    }


def _build_panel(
    symbols: list[str],
    actual_as_of: dt.date,
    *,
    include_flow: bool,
    include_lockup: bool,
) -> tuple[pd.DataFrame, str]:
    pipe = FeaturePipeline(
        store_root=STORE_ROOT,
        project_root=ROOT,
        include_fundamentals=True,
        include_valuation=True,
        include_industry=True,
        include_flow=include_flow,
        include_margin=True,
    )
    feature_set_id = pipe.run(symbols, specs=DEFAULT_SPECS, end_date=actual_as_of)
    panel = pipe.build_panel(symbols, feature_set_id, add_cross_sectional=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel = panel[panel["date"] <= actual_as_of].copy()

    if include_lockup:
        lockup_panel = load_lockup_panel(STORE_ROOT, symbols)
        if not lockup_panel.empty:
            try:
                from quant_platform.features.valuation import load_valuation_panel

                valuation_panel = load_valuation_panel(STORE_ROOT, symbols)
            except Exception:
                valuation_panel = None
            panel = build_lockup_features(panel, lockup_panel, valuation_panel)

    labels = build_label_panel(STORE_ROOT, symbols, horizons=[HORIZON], add_excess_csi300=False)
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    labels = labels[labels["date"] <= actual_as_of].copy()
    label_cols = [c for c in labels.columns if c not in ("symbol", "date") and c not in panel.columns]
    panel = panel.merge(labels[["symbol", "date"] + label_cols], on=["symbol", "date"], how="left")
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel, feature_set_id


def _feature_importance_top20(model, feature_cols: list[str]) -> pd.DataFrame:
    imp = _feature_importance_all(model, feature_cols)
    return imp.head(20).reset_index(drop=True)


def _feature_importance_all(model, feature_cols: list[str]) -> pd.DataFrame:
    lgbm = model.named_steps.get("lgbm")
    if lgbm is None or not hasattr(lgbm, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": lgbm.feature_importances_,
    })
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def _score_model(
    panel: pd.DataFrame,
    actual_as_of: dt.date,
    feature_cols: list[str],
    *,
    train_start: dt.date | None,
    params: dict,
    score_prefix: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, object]:
    train = panel[(panel["date"] < actual_as_of) & panel[LABEL_COL].notna()].copy()
    if train_start is not None:
        train = train[train["date"] >= train_start].copy()
    pred = panel[panel["date"] == actual_as_of].copy()
    pred = pred[pred[feature_cols].notna().mean(axis=1) > 0.0].copy()

    if train.empty:
        raise RuntimeError(f"{score_prefix} train set is empty")
    if pred.empty:
        raise RuntimeError(f"{score_prefix} prediction day has no usable rows")

    model = fit_final_model(train, feature_cols, LABEL_COL, params)
    preds = model.predict(pred[feature_cols])
    scored = pred[["symbol", "date"]].copy()
    scored["trade_date"] = scored["date"]
    scored[f"{score_prefix}_score"] = preds
    scored[f"{score_prefix}_rank"] = scored[f"{score_prefix}_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    scored[f"{score_prefix}_pct"] = scored[f"{score_prefix}_score"].rank(
        ascending=True, pct=True
    )
    scored = scored.sort_values(f"{score_prefix}_rank").reset_index(drop=True)

    meta = {
        "train_start": train["date"].min().isoformat(),
        "train_end": train["date"].max().isoformat(),
        "prediction_date": actual_as_of.isoformat(),
        "x_train_rows": int(len(train)),
        "prediction_rows": int(len(pred)),
        "feature_count": int(len(feature_cols)),
    }
    importance_all = _feature_importance_all(model, feature_cols)
    return scored, meta, importance_all.head(20).reset_index(drop=True), importance_all, model


def _risk_flags(panel: pd.DataFrame, actual_as_of: dt.date) -> pd.DataFrame:
    pred = panel[panel["date"] == actual_as_of][["symbol", "date"]].copy()
    flags = []
    for _, row in panel[panel["date"] == actual_as_of].iterrows():
        risk = []
        event = []
        if "days_to_next_unlock" in row.index and pd.notna(row["days_to_next_unlock"]):
            days = float(row["days_to_next_unlock"])
            ratio = float(row.get("unlock_size_ratio", 0) or 0)
            # structured flags carry an explicit @known_at (the as-of date) so
            # the gate admits them to the risk channel (GF-04b req #3).
            asof = actual_as_of.isoformat()
            if days <= 5 and ratio >= 0.05:
                risk.append(f"high_unlock:{int(days)}d_ratio_{ratio:.2%}@{asof}")
            elif days <= 10:
                event.append(f"unlock:{int(days)}d@{asof}")
        flags.append({
            "symbol": row["symbol"],
            "trade_date": row["date"],
            "risk_flags": ";".join(risk),
            "event_flags": ";".join(event),
        })
    return pd.DataFrame(flags) if flags else pred.assign(trade_date=actual_as_of, risk_flags="", event_flags="")


def _top_symbols(df: pd.DataFrame, score_col: str, n: int = 20) -> list[str]:
    return df.sort_values(score_col, ascending=False)["symbol"].head(n).astype(str).tolist()


def _emdatah5_fund_flow_gate_from_this_run(recent_gate: pd.DataFrame, family_by_col: dict) -> dict:
    """
    Build the fund-flow gate summary from THIS RUN's recent_gate coverage
    report, not a persisted CSV from some earlier, unrelated run.

    FIXED (review finding #4): the previous `_load_emdatah5_fund_flow_gate`
    read `REPORT_DIR / "fund_flow_emdatah5_coverage_gate.csv"` -- whatever
    happened to be on disk from a prior run/step -- and embedded it in
    this run's report/JSON as though it reflected the current fetch. That
    field could silently disagree with `flow_in_recent`/`recent_gate`
    (which ARE computed live, correctly, a few lines below in main()),
    giving a false impression of freshness. This function derives the
    same summary shape directly from the live `recent_gate` DataFrame
    already computed this run.
    """
    if recent_gate.empty:
        return {"status": "no_candidates_this_run"}
    flow_rows = recent_gate[recent_gate["feature_name"].map(lambda c: family_by_col.get(c) == "flow")]
    if flow_rows.empty:
        return {"status": "no_flow_candidates_this_run"}
    row = flow_rows.iloc[0]
    return {
        "status": "computed_this_run",
        "recent_symbol_coverage": int(row.get("recent_symbol_coverage", 0)),
        "latest_available_date": str(row.get("latest_available_date", "")),
        "recent_20d_avg_symbol_coverage": float(row.get("recent_20d_avg_symbol_coverage", 0.0)),
        "available_trading_days": int(row.get("available_trading_days", 0)),
        "field_missing_rate": float(flow_rows["overall_missing_rate"].mean()),
        "is_allowed_for_recent_model": bool(row.get("is_allowed_for_recent_model", False)),
        "rejection_reason": str(row.get("rejection_reason", "")),
    }


def _write_list_section(lines: list[str], title: str, df: pd.DataFrame, score_col: str, n: int = 20) -> None:
    lines += [f"## {title}", "", "| Rank | Symbol | Score |", "|---:|---|---:|"]
    for i, (_, row) in enumerate(df.sort_values(score_col, ascending=False).head(n).iterrows(), 1):
        lines.append(f"| {i} | {row['symbol']} | {float(row[score_col]):+.6f} |")
    lines.append("")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = _load_symbols()
    actual_as_of, actual_reason, ohlcv_counts = _resolve_actual_as_of(symbols)
    date_tag = actual_as_of.isoformat()

    output_paths = {
        "base_ranked": REPORT_DIR / f"D3_base_ranked_{date_tag}.csv",
        "recent_ranked": REPORT_DIR / f"D3_recent_ranked_{date_tag}.csv",
        "gate_ranked": REPORT_DIR / f"D3_gate_fused_ranked_{date_tag}.csv",
        "hybrid_ranked": REPORT_DIR / f"D3_hybrid_ranked_{date_tag}.csv",
        "reco_cards": REPORT_DIR / f"D3_hybrid_reco_cards_{date_tag}.json",
        "report": REPORT_DIR / f"D3_gate_emdatah5_fund_flow_run_report_{date_tag}.md",
        "json": REPORT_DIR / f"D3_gate_emdatah5_fund_flow_run_summary_{date_tag}.json",
    }
    coverage_summary = {
        "ohlcv": {
            "requested_coverage": f"{ohlcv_counts.get(REQUESTED_AS_OF_DATE, 0)}/{len(symbols)}",
            "actual_coverage": f"{ohlcv_counts.get(actual_as_of, 0)}/{len(symbols)}",
        },
        "features": {
            "1474f2c4": _coverage_for_dir("features/1474f2c4", symbols, actual_as_of),
            "d02a4ebf": _coverage_for_dir("features/d02a4ebf", symbols, actual_as_of),
        },
        "labels": _coverage_for_dir("labels/forward_returns", symbols, actual_as_of),
        "fund_flow": _coverage_for_dir("silver/fund_flow", symbols, actual_as_of),
        "fundamentals": _coverage_for_dir("silver/fundamentals", symbols, actual_as_of),
        "lockup": _coverage_for_dir("silver/lockup", symbols, actual_as_of),
        "announcement_events": _coverage_for_dir("silver/announcement_events", symbols, actual_as_of),
        "valuation": _coverage_for_dir("silver/valuation", symbols, actual_as_of),
        "margin": _coverage_for_dir("silver/margin", symbols, actual_as_of),
    }

    base_panel, base_feature_set_id = _build_panel(
        symbols, actual_as_of, include_flow=False, include_lockup=False
    )
    recent_panel, recent_feature_set_id = _build_panel(
        symbols, actual_as_of, include_flow=True, include_lockup=True
    )

    family_by_col = _feature_family_lookup()
    feature_metadata = feature_metadata_lookup()
    cfg = _coverage_gate_config_for_universe(len(symbols))

    base_candidates = _build_feature_cols(base_panel)
    base_gate = compute_feature_coverage_report(
        base_panel, base_candidates, family_by_col=family_by_col,
        feature_metadata=feature_metadata, as_of_date=actual_as_of, config=cfg
    )
    base_gate_csv, base_gate_md = write_coverage_gate_report(
        base_gate, REPORT_DIR, prefix=f"D3_base_coverage_gate_{date_tag}"
    )
    base_features = select_features_by_gate(base_gate, model_path="base")
    base_features = [c for c in base_features if family_by_col.get(c) != "flow"]

    recent_candidates = _build_feature_cols(recent_panel)
    recent_gate = compute_feature_coverage_report(
        recent_panel, recent_candidates, family_by_col=family_by_col,
        feature_metadata=feature_metadata, as_of_date=actual_as_of, config=cfg
    )
    recent_gate_csv, recent_gate_md = write_coverage_gate_report(
        recent_gate, REPORT_DIR, prefix=f"D3_recent_coverage_gate_{date_tag}"
    )
    recent_features = select_features_by_gate(recent_gate, model_path="recent")

    flow_cols = [c for c in recent_candidates if family_by_col.get(c) == "flow" or "flow" in c]
    flow_in_recent = [c for c in recent_features if c in flow_cols]
    flow_in_base = [c for c in base_features if c in flow_cols]
    fund_flow_gate = _emdatah5_fund_flow_gate_from_this_run(recent_gate, family_by_col)
    gate_mode = "fund_flow_enhanced_gate" if flow_in_recent else "standard_gate"

    base_feature_cols_path = REPORT_DIR / f"D3_base_X_train_columns_{date_tag}.csv"
    recent_feature_cols_path = REPORT_DIR / f"D3_recent_emdatah5_X_train_columns_{date_tag}.csv"
    pd.DataFrame({"feature": base_features}).to_csv(base_feature_cols_path, index=False)
    pd.DataFrame({"feature": recent_features}).to_csv(recent_feature_cols_path, index=False)

    base_scored, base_meta, base_importance, _base_importance_all, _base_model = _score_model(
        base_panel, actual_as_of, base_features, train_start=None, params=BASE_PARAMS, score_prefix="base"
    )
    base_importance_path = REPORT_DIR / f"D3_base_feature_importance_top20_{date_tag}.csv"
    base_importance.to_csv(base_importance_path, index=False)
    base_scored.to_csv(output_paths["base_ranked"], index=False)

    recent_status = "trained"
    recent_failure_reason = ""
    recent_scored = pd.DataFrame()
    recent_meta = {}
    recent_importance = pd.DataFrame()
    try:
        unique_train_dates = sorted(
            d for d in recent_panel.loc[
                (recent_panel["date"] < actual_as_of) & recent_panel[LABEL_COL].notna(), "date"
            ].unique()
        )
        if not unique_train_dates:
            raise RuntimeError("no recent training dates with observable ret_fwd_3d")
        train_start = unique_train_dates[max(0, len(unique_train_dates) - 120)]
        if len(recent_features) == 0:
            raise RuntimeError("coverage gate rejected all recent features")
        recent_scored, recent_meta, recent_importance, recent_importance_all, recent_model = _score_model(
            recent_panel,
            actual_as_of,
            recent_features,
            train_start=train_start,
            params=RECENT_PARAMS,
            score_prefix="recent",
        )
        recent_importance_path = REPORT_DIR / f"D3_recent_feature_importance_top20_{date_tag}.csv"
        recent_importance.to_csv(recent_importance_path, index=False)
        fund_flow_importance = recent_importance_all[recent_importance_all["feature"].isin(flow_cols)].copy()
        fund_flow_importance_path = REPORT_DIR / f"D3_recent_fund_flow_feature_importance_{date_tag}.csv"
        fund_flow_importance.to_csv(fund_flow_importance_path, index=False)
        recent_scored.to_csv(output_paths["recent_ranked"], index=False)
    except Exception as exc:
        recent_status = "failed"
        recent_failure_reason = str(exc)
        recent_scored = pd.DataFrame({
            "symbol": base_scored["symbol"],
            "date": base_scored["date"],
            "trade_date": base_scored["trade_date"],
            "recent_score": np.nan,
            "recent_rank": np.nan,
            "recent_pct": np.nan,
        })
        recent_scored.to_csv(output_paths["recent_ranked"], index=False)
        fund_flow_importance_path = REPORT_DIR / f"D3_recent_fund_flow_feature_importance_{date_tag}.csv"
        pd.DataFrame(columns=["feature", "importance"]).to_csv(fund_flow_importance_path, index=False)

    if recent_status == "trained":
        fused_input = base_scored.merge(
            recent_scored[["symbol", "trade_date", "recent_score", "recent_rank", "recent_pct"]],
            on=["symbol", "trade_date"],
            how="inner",
        )
        fused_input = fused_input.merge(_risk_flags(recent_panel, actual_as_of), on=["symbol", "trade_date"], how="left")
        fused_input[["risk_flags", "event_flags"]] = fused_input[["risk_flags", "event_flags"]].fillna("")
        fused = gate_first_fusion(fused_input)
        fused.to_csv(output_paths["gate_ranked"], index=False)
        # Also emit the standard helper report with a date-tagged prefix.
        write_gate_fusion_outputs(fused, REPORT_DIR, prefix=f"D3_gate_fused_{date_tag}", top_n=50)
        # SR-01: hybrid score contract -- gate tiers act as a hard actionable
        # filter, smooth fusion (default rrf) ranks within the A/B pool.
        hybrid = actionable_pool(fused)
        hybrid.to_csv(output_paths["hybrid_ranked"], index=False)
        # SR-04: recommendation cards -- reuse the already-fitted recent
        # model's SHAP drivers rather than fitting a second explainer.
        recent_asof = recent_panel[recent_panel["date"] == actual_as_of]
        # SHAP fix: TreeExplainer cannot consume a sklearn Pipeline -- passing the
        # full pipeline made every SHAP driver silently fall back to 0.0.  Unwrap
        # the final LGBM estimator (build_lgbm_pipeline names it "lgbm").
        recent_estimator = recent_model.named_steps["lgbm"]
        explain_report = build_explainability_report(
            recent_estimator,
            recent_asof,
            recent_features,
            panel_meta=recent_asof[["symbol"]],
        )
        reco_cards = build_reco_cards(hybrid, explain_report.per_symbol_drivers, family_by_col)
        output_paths["reco_cards"].write_text(
            json.dumps(reco_cards, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        fused = base_scored.copy()
        fused["recent_score"] = np.nan
        fused["recent_rank"] = np.nan
        fused["recent_pct"] = np.nan
        fused["short_boost"] = np.nan
        fused["risk_flags"] = ""
        fused["event_flags"] = ""
        fused["gate_tier"] = "BASE_ONLY"
        fused["gate_reason"] = f"recent model unavailable: {recent_failure_reason}"
        fused["final_rank"] = fused["base_rank"]
        fused.to_csv(output_paths["gate_ranked"], index=False)

    base_top20 = _top_symbols(base_scored, "base_score", 20)
    recent_top20 = _top_symbols(recent_scored.dropna(subset=["recent_score"]), "recent_score", 20) if recent_status == "trained" else []
    fused_top20 = fused.sort_values("final_rank")["symbol"].head(20).astype(str).tolist()
    intersections = {
        "base_recent_top20": len(set(base_top20) & set(recent_top20)),
        "base_gate_top20": len(set(base_top20) & set(fused_top20)),
        "recent_gate_top20": len(set(recent_top20) & set(fused_top20)),
        "all_three_top20": len(set(base_top20) & set(recent_top20) & set(fused_top20)),
    }
    tier_counts = fused["gate_tier"].value_counts().to_dict()
    boosted = fused[fused["gate_tier"] == "B_SHORT_BOOST"]["symbol"].head(50).astype(str).tolist()
    downgraded = fused[fused["gate_tier"].isin(["C_DOWNGRADE_OBSERVE", "RISK_DOWNGRADE"])]["symbol"].head(50).astype(str).tolist()
    vetoed = fused[fused["gate_tier"] == "RISK_VETO"]["symbol"].head(50).astype(str).tolist()

    lines = [
        f"# D3 Gate-First Run Report {date_tag}",
        "",
        f"- requested_as_of_date: {REQUESTED_AS_OF_DATE}",
        f"- actual_as_of_date: {actual_as_of}",
        f"- actual date reason: {actual_reason}",
        f"- label: {LABEL_COL}",
        "",
        "## Data Coverage Summary",
        "",
        "```json",
        json.dumps(coverage_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Feature Gate Summary",
        "",
        f"- base_feature_set_id: `{base_feature_set_id}`",
        f"- recent_enhanced_feature_set_id: `{recent_feature_set_id}`",
        f"- base candidates: {len(base_candidates)}; base admitted: {len(base_features)}",
        f"- recent candidates: {len(recent_candidates)}; recent admitted: {len(recent_features)}",
        f"- gate_mode: {gate_mode}",
        f"- fund flow entered recent model: {'yes' if flow_in_recent else 'no'} ({', '.join(flow_in_recent) if flow_in_recent else 'none'})",
        f"- fund_flow coverage gate result: {fund_flow_gate}",
        f"- base final feature list: `{base_feature_cols_path}`",
        f"- recent final feature list: `{recent_feature_cols_path}`",
        f"- fund flow forbidden from base model: {'yes' if not flow_in_base else 'no'}",
        f"- base coverage report: `{base_gate_csv}`",
        f"- recent coverage report: `{recent_gate_csv}`",
        "",
        "## Model Summary",
        "",
        f"- base_model_d3: {base_meta}",
        f"- recent_enhanced_model_d3 status: {recent_status}",
        f"- recent_enhanced_model_d3: {recent_meta if recent_meta else recent_failure_reason}",
        f"- base feature importance Top20: `{base_importance_path}`",
        f"- fund_flow feature importance: `{fund_flow_importance_path}`",
    ]
    if recent_status == "trained":
        lines.append(f"- recent feature importance Top20: `{recent_importance_path}`")
    lines += [
        "",
        "## Gate Summary",
        "",
        f"- tier counts: {tier_counts}",
        f"- risk veto count: {tier_counts.get('RISK_VETO', 0)}",
        f"- top20 intersections: {intersections}",
        f"- recent boosted symbols: {boosted}",
        f"- recent downgraded symbols: {downgraded}",
        f"- event risk vetoed symbols: {vetoed}",
        "",
    ]
    _write_list_section(lines, "Base Top20", base_scored, "base_score")
    if recent_status == "trained":
        _write_list_section(lines, "Recent Top20", recent_scored, "recent_score")
    else:
        lines += ["## Recent Top20", "", f"Recent model failed: {recent_failure_reason}", ""]
    lines += ["## Gate Fused Top20", "", "| Rank | Symbol | Tier | Reason |", "|---:|---|---|---|"]
    for _, row in fused.sort_values("final_rank").head(20).iterrows():
        lines.append(f"| {int(row['final_rank'])} | {row['symbol']} | {row['gate_tier']} | {row['gate_reason']} |")
    lines += [
        "",
        "## Verification Artifacts",
        "",
        f"- `{output_paths['base_ranked']}`",
        f"- `{output_paths['recent_ranked']}`",
        f"- `{output_paths['gate_ranked']}`",
    ]
    if output_paths["hybrid_ranked"].exists():
        lines.append(f"- `{output_paths['hybrid_ranked']}`")
    if output_paths["reco_cards"].exists():
        lines.append(f"- `{output_paths['reco_cards']}`")
    lines.append(f"- `{output_paths['report']}`")
    output_paths["report"].write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "requested_as_of_date": REQUESTED_AS_OF_DATE.isoformat(),
        "actual_as_of_date": actual_as_of.isoformat(),
        "actual_reason": actual_reason,
        "coverage_summary": coverage_summary,
        "base_meta": base_meta,
        "recent_status": recent_status,
        "recent_meta": recent_meta,
        "recent_failure_reason": recent_failure_reason,
        "flow_in_recent": flow_in_recent,
        "flow_in_base": flow_in_base,
        "fund_flow_gate": fund_flow_gate,
        "gate_mode": gate_mode,
        "tier_counts": tier_counts,
        "intersections": intersections,
        "boosted": boosted,
        "downgraded": downgraded,
        "vetoed": vetoed,
        "outputs": {k: str(v) for k, v in output_paths.items()},
        "base_feature_cols": str(base_feature_cols_path),
        "recent_feature_cols": str(recent_feature_cols_path),
        "fund_flow_feature_importance": str(fund_flow_importance_path),
        "base_top20": base_top20,
        "recent_top20": recent_top20,
        "fused_top20": fused_top20,
    }
    output_paths["json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
