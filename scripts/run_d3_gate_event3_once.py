"""
Run one D3 base/recent-event3/Gate-first recommendation pass.

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
EVENT_START_DATE = dt.date(2026, 1, 1)

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
from quant_platform.features.event import (  # noqa: E402
    EVENT3_SPECS,
    build_event3_features,
    load_announcement_events_panel,
    load_block_trade_panel,
    load_dragon_tiger_panel,
)
from quant_platform.features.pipeline import FeaturePipeline  # noqa: E402
from quant_platform.features.registry import DEFAULT_SPECS, feature_metadata_lookup  # noqa: E402
from quant_platform.ingest.announcement_events_collector import AnnouncementEventsCollector  # noqa: E402
from quant_platform.ingest.block_trade_collector import BlockTradeCollector  # noqa: E402
from quant_platform.ingest.dragon_tiger_collector import DragonTigerCollector  # noqa: E402
from quant_platform.labels.builder import build_label_panel  # noqa: E402
from quant_platform.selection.gate_fusion import gate_first_fusion, write_gate_fusion_outputs  # noqa: E402
from quant_platform.store.lake import ohlcv_path  # noqa: E402
from quant_platform.store.parquet_store import read_ohlcv  # noqa: E402
from quant_platform.training.lgbm_model import fit_final_model  # noqa: E402


LABEL_COL = "ret_fwd_3d"
HORIZON = 3
EVENT3_SPECS_FULL = DEFAULT_SPECS + EVENT3_SPECS
EVENT3_FEATURES = [s.name for s in EVENT3_SPECS]
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


def _run_event3_collectors(symbols: list[str], actual_as_of: dt.date, trading_dates: list[dt.date]) -> dict:
    start = EVENT_START_DATE.isoformat()
    end = actual_as_of.isoformat()
    summary: dict[str, dict] = {}

    ann = AnnouncementEventsCollector(STORE_ROOT)
    ann_res = ann.run(symbols, start_date=start, end_date=end, trading_dates=trading_dates, overwrite=False)
    summary["announcement_events"] = {
        "ok_files_or_empty": int(sum(1 for v in ann_res.values() if v >= 0)),
        "failed": int(sum(1 for v in ann_res.values() if v < 0)),
        "rows_fetched": int(sum(v for v in ann_res.values() if v > 0)),
    }

    dtc = DragonTigerCollector(STORE_ROOT)
    dt_res = dtc.run(symbols, start_date=start, end_date=end, trading_dates=trading_dates, overwrite=False)
    summary["dragon_tiger"] = {
        "ok_files_or_empty": int(sum(1 for v in dt_res.values() if v >= 0)),
        "failed": int(sum(1 for v in dt_res.values() if v < 0)),
        "rows_fetched": int(sum(v for v in dt_res.values() if v > 0)),
    }

    btc = BlockTradeCollector(STORE_ROOT)
    bt_res = btc.run(symbols, start_date=start, end_date=end, trading_dates=trading_dates, overwrite=False)
    summary["block_trade"] = {
        "ok_files_or_empty": int(sum(1 for v in bt_res.values() if v >= 0)),
        "failed": int(sum(1 for v in bt_res.values() if v < 0)),
        "rows_fetched": int(sum(v for v in bt_res.values() if v > 0)),
    }
    return summary


def _event_feature_stats(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    family_by_col = _feature_family_lookup()
    for col in feature_cols:
        if col not in panel.columns:
            continue
        s = pd.to_numeric(panel[col], errors="coerce")
        rows.append({
            "feature_name": col,
            "feature_family": family_by_col.get(col, "raw_aux"),
            "source": {
                "announcement_events": "cninfo",
                "dragon_tiger": "datacenter-web",
                "block_trade": "datacenter-web",
            }.get(family_by_col.get(col, ""), "feature_panel"),
            "missing_rate": float(s.isna().mean()),
            "non_zero_rate": float((s.fillna(0) != 0).mean()),
        })
    return pd.DataFrame(rows)


def _event_coverage_gate(panel: pd.DataFrame, gate: pd.DataFrame, actual_as_of: dt.date, symbols: list[str]) -> pd.DataFrame:
    family_by_col = _feature_family_lookup()
    rows = []
    event_families = {"announcement_events", "dragon_tiger", "block_trade"}
    for family in ("announcement_events", "dragon_tiger", "block_trade"):
        cols = [c for c in EVENT3_FEATURES if family_by_col.get(c) == family and c in panel.columns]
        source = "cninfo" if family == "announcement_events" else "datacenter-web"
        if cols:
            subset = panel[cols]
            missing_rate = float(subset.isna().mean().mean())
            non_zero_rate = float((subset.fillna(0) != 0).mean().mean())
            event_sparsity = float(1.0 - non_zero_rate)
            admitted = gate[gate["feature_name"].isin(cols)] if not gate.empty else pd.DataFrame()
            allowed_recent = bool(not admitted.empty and admitted["is_allowed_for_recent_model"].all())
            allowed_base = bool(not admitted.empty and admitted["is_allowed_for_base_model"].any())
            rejection = "; ".join(sorted(set(admitted.get("rejection_reason", pd.Series(dtype=str)).astype(str)))) if not admitted.empty else "not-evaluated"
        else:
            missing_rate = 1.0
            non_zero_rate = 0.0
            event_sparsity = 1.0
            allowed_recent = False
            allowed_base = False
            rejection = "no-feature-columns"

        cov = _coverage_for_dir(f"silver/{family}", symbols, actual_as_of)
        rows.append({
            "feature_family": family,
            "source": source,
            "covered_symbols": cov["symbol_file_coverage"],
            "date_range": f"{cov['latest_min']} -> {cov['latest_max']}",
            "latest_available_date": cov["latest_max"],
            "event_sparsity": event_sparsity,
            "pit_safe": True,
            "non_zero_rate": non_zero_rate,
            "missing_rate": missing_rate,
            "is_allowed_for_recent_model": allowed_recent,
            "is_allowed_for_base_model": allowed_base,
            "rejection_reason": rejection,
        })
    return pd.DataFrame(rows)


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
            (c for c in ("date", "available_date", "trade_date", "announce_date", "unlock_date", "event_date", "period_end")
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
    specs: list,
    include_event3: bool,
) -> tuple[pd.DataFrame, str]:
    pipe = FeaturePipeline(
        store_root=STORE_ROOT,
        project_root=ROOT,
        include_fundamentals=True,
        include_valuation=True,
        include_industry=True,
        include_flow=False,
        include_margin=False,
    )
    feature_set_id = pipe.run(symbols, specs=specs, end_date=actual_as_of)
    panel = pipe.build_panel(symbols, feature_set_id, add_cross_sectional=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel = panel[panel["date"] <= actual_as_of].copy()

    if include_event3:
        panel = build_event3_features(
            panel,
            load_announcement_events_panel(STORE_ROOT, symbols),
            load_dragon_tiger_panel(STORE_ROOT, symbols),
            load_block_trade_panel(STORE_ROOT, symbols),
        )

    labels = build_label_panel(STORE_ROOT, symbols, horizons=[HORIZON], add_excess_csi300=False)
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    labels = labels[labels["date"] <= actual_as_of].copy()
    label_cols = [c for c in labels.columns if c not in ("symbol", "date") and c not in panel.columns]
    panel = panel.merge(labels[["symbol", "date"] + label_cols], on=["symbol", "date"], how="left")
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel, feature_set_id


def _feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
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
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
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
    return scored, meta, _feature_importance(model, feature_cols)


def _risk_flags(panel: pd.DataFrame, actual_as_of: dt.date) -> pd.DataFrame:
    pred = panel[panel["date"] == actual_as_of][["symbol", "date"]].copy()
    flags = []
    for _, row in panel[panel["date"] == actual_as_of].iterrows():
        risk = []
        event = []
        # structured flags carry an explicit @known_at (the as-of date) so the
        # gate admits them to the risk/veto/downgrade channel (GF-04b req #3).
        asof = actual_as_of.isoformat()
        if float(row.get("has_risk_announcement_5d", 0) or 0) > 0:
            risk.append(f"risk_warning@{asof}")
        if float(row.get("has_major_event_10d", 0) or 0) > 0:
            event.append(f"major_announcement@{asof}")
        if float(row.get("has_dragon_tiger_5d", 0) or 0) > 0:
            event.append(f"dragon_tiger@{asof}")
        if float(row.get("has_large_discount_block_trade_20d", 0) or 0) > 0:
            event.append(f"large_discount_block_trade@{asof}")
        flags.append({
            "symbol": row["symbol"],
            "trade_date": row["date"],
            "risk_flags": ";".join(risk),
            "event_flags": ";".join(event),
        })
    return pd.DataFrame(flags) if flags else pred.assign(trade_date=actual_as_of, risk_flags="", event_flags="")


def _top_symbols(df: pd.DataFrame, score_col: str, n: int = 20) -> list[str]:
    return df.sort_values(score_col, ascending=False)["symbol"].head(n).astype(str).tolist()


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
    trading_dates = sorted(d for d, c in ohlcv_counts.items() if d <= actual_as_of and c >= min(250, max(1, int(len(symbols) * 0.8))))
    collector_summary = _run_event3_collectors(symbols, actual_as_of, trading_dates)

    output_paths = {
        "base_ranked": REPORT_DIR / f"D3_base_ranked_{date_tag}.csv",
        "recent_ranked": REPORT_DIR / f"D3_recent_ranked_{date_tag}.csv",
        "gate_ranked": REPORT_DIR / f"D3_gate_fused_ranked_{date_tag}.csv",
        "report": REPORT_DIR / f"D3_gate_event3_run_report_{date_tag}.md",
        "json": REPORT_DIR / f"D3_gate_event3_run_summary_{date_tag}.json",
        "integration_report": ROOT / "a_stock_data_event3_integration_report.md",
    }
    for path in output_paths.values():
        if path.exists():
            path.unlink()

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
        "announcement_events": _coverage_for_dir("silver/announcement_events", symbols, actual_as_of),
        "dragon_tiger": _coverage_for_dir("silver/dragon_tiger", symbols, actual_as_of),
        "block_trade": _coverage_for_dir("silver/block_trade", symbols, actual_as_of),
        "valuation": _coverage_for_dir("silver/valuation", symbols, actual_as_of),
        "margin": _coverage_for_dir("silver/margin", symbols, actual_as_of),
    }

    base_panel, base_feature_set_id = _build_panel(
        symbols, actual_as_of, specs=DEFAULT_SPECS, include_event3=False
    )
    recent_panel, recent_feature_set_id = _build_panel(
        symbols, actual_as_of, specs=EVENT3_SPECS_FULL, include_event3=True
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

    event_feature_stats = _event_feature_stats(recent_panel, EVENT3_FEATURES)
    event_feature_stats_path = REPORT_DIR / f"D3_event3_feature_missing_nonzero_{date_tag}.csv"
    event_feature_stats.to_csv(event_feature_stats_path, index=False)
    event_gate = _event_coverage_gate(recent_panel, recent_gate, actual_as_of, symbols)
    event_gate_path = REPORT_DIR / f"D3_event3_coverage_gate_{date_tag}.csv"
    event_gate.to_csv(event_gate_path, index=False)

    event_features_in_recent = [c for c in recent_features if c in EVENT3_FEATURES]
    event_features_in_base = [c for c in base_features if c in EVENT3_FEATURES]
    margin_in_recent = [c for c in recent_features if family_by_col.get(c) == "margin"]
    margin_in_base = [c for c in base_features if family_by_col.get(c) == "margin"]
    base_feature_cols_path = REPORT_DIR / f"D3_base_X_train_columns_{date_tag}.csv"
    recent_feature_cols_path = REPORT_DIR / f"D3_recent_event3_X_train_columns_{date_tag}.csv"
    pd.DataFrame({"feature": base_features, "family": [family_by_col.get(c, "raw_aux") for c in base_features]}).to_csv(
        base_feature_cols_path, index=False
    )
    pd.DataFrame({"feature": recent_features, "family": [family_by_col.get(c, "raw_aux") for c in recent_features]}).to_csv(
        recent_feature_cols_path, index=False
    )

    base_scored, base_meta, base_importance = _score_model(
        base_panel, actual_as_of, base_features, train_start=None, params=BASE_PARAMS, score_prefix="base"
    )
    base_importance_path = REPORT_DIR / f"D3_base_feature_importance_top20_{date_tag}.csv"
    base_importance.head(20).to_csv(base_importance_path, index=False)
    base_scored.to_csv(output_paths["base_ranked"], index=False)

    recent_status = "trained"
    recent_failure_reason = ""
    recent_scored = pd.DataFrame()
    recent_meta = {}
    recent_importance = pd.DataFrame()
    event_importance_path = REPORT_DIR / f"D3_event3_feature_importance_{date_tag}.csv"
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
        recent_scored, recent_meta, recent_importance = _score_model(
            recent_panel,
            actual_as_of,
            recent_features,
            train_start=train_start,
            params=RECENT_PARAMS,
            score_prefix="recent",
        )
        recent_importance_path = REPORT_DIR / f"D3_recent_feature_importance_top20_{date_tag}.csv"
        recent_importance.head(20).to_csv(recent_importance_path, index=False)
        event_importance_path = REPORT_DIR / f"D3_event3_feature_importance_{date_tag}.csv"
        recent_importance[recent_importance["feature"].isin(EVENT3_FEATURES)].to_csv(event_importance_path, index=False)
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

    ann_in_recent = any(family_by_col.get(c) == "announcement_events" for c in event_features_in_recent)
    dt_in_recent = any(family_by_col.get(c) == "dragon_tiger" for c in event_features_in_recent)
    bt_in_recent = any(family_by_col.get(c) == "block_trade" for c in event_features_in_recent)

    lines = [
        f"# D3 Gate Event3 Run Report {date_tag}",
        "",
        "- gate_mode: `event3_enhanced_gate`",
        f"- requested_as_of_date: {REQUESTED_AS_OF_DATE}",
        f"- actual_as_of_date: {actual_as_of}",
        f"- actual date reason: {actual_reason}",
        f"- label: {LABEL_COL}",
        f"- collector_summary: {collector_summary}",
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
        f"- recent_event_feature_set_id: `{recent_feature_set_id}`",
        f"- base candidates: {len(base_candidates)}; base admitted: {len(base_features)}",
        f"- recent candidates: {len(recent_candidates)}; recent admitted: {len(recent_features)}",
        f"- announcement_events entered recent model: {'yes' if ann_in_recent else 'no'}",
        f"- dragon_tiger entered recent model: {'yes' if dt_in_recent else 'no'}",
        f"- block_trade entered recent model: {'yes' if bt_in_recent else 'no'}",
        f"- event3 features entered recent model: {event_features_in_recent}",
        f"- event3 features entered base model: {event_features_in_base if event_features_in_base else 'none'}",
        f"- margin supplement cross-validation only: yes; margin features in recent={margin_in_recent or 'none'}, base={margin_in_base or 'none'}",
        f"- base X_train.columns: `{base_feature_cols_path}`",
        f"- recent X_train.columns: `{recent_feature_cols_path}`",
        f"- base coverage report: `{base_gate_csv}`",
        f"- recent coverage report: `{recent_gate_csv}`",
        f"- event3 coverage gate: `{event_gate_path}`",
        f"- event3 missing/non-zero rates: `{event_feature_stats_path}`",
        "",
        "## Model Summary",
        "",
        f"- base_model_d3: {base_meta}",
        f"- recent_enhanced_model_d3 status: {recent_status}",
        f"- recent_enhanced_model_d3: {recent_meta if recent_meta else recent_failure_reason}",
        f"- base feature importance Top20: `{base_importance_path}`",
    ]
    if recent_status == "trained":
        lines.append(f"- recent feature importance Top20: `{recent_importance_path}`")
        lines.append(f"- event3 feature importance: `{event_importance_path}`")
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
        f"- `{output_paths['report']}`",
    ]
    output_paths["report"].write_text("\n".join(lines), encoding="utf-8")

    def _top20_table(symbols_list: list[str]) -> list[str]:
        return [f"{i}. {sym}" for i, sym in enumerate(symbols_list, 1)]

    integration_lines = [
        "# A-Stock Data Event3 Integration Report",
        "",
        f"- actual_as_of_date: {actual_as_of}",
        f"- gate_mode: event3_enhanced_gate",
        "",
        "## 1. cninfo 公告覆盖情况",
        "",
        f"- coverage: {coverage_summary['announcement_events']}",
        f"- entered_recent_model: {'yes' if ann_in_recent else 'no'}",
        "",
        "## 2. 龙虎榜覆盖情况",
        "",
        f"- coverage: {coverage_summary['dragon_tiger']}",
        f"- entered_recent_model: {'yes' if dt_in_recent else 'no'}",
        "",
        "## 3. 大宗交易覆盖情况",
        "",
        f"- coverage: {coverage_summary['block_trade']}",
        f"- entered_recent_model: {'yes' if bt_in_recent else 'no'}",
        "",
        "## 4. 三类事件因子是否进入 recent model",
        "",
        f"- announcement_events: {'yes' if ann_in_recent else 'no'}",
        f"- dragon_tiger: {'yes' if dt_in_recent else 'no'}",
        f"- block_trade: {'yes' if bt_in_recent else 'no'}",
        "",
        "## 5. 融资融券补充是否仅用于交叉验证",
        "",
        f"- yes; margin features in model: recent={margin_in_recent or 'none'}, base={margin_in_base or 'none'}",
        "",
        "## 6. 是否重跑 D3 Gate",
        "",
        f"- yes; requested_as_of_date={REQUESTED_AS_OF_DATE}; actual_as_of_date={actual_as_of}",
        "",
        "## 7. Base Top20",
        "",
        *_top20_table(base_top20),
        "",
        "## 8. Recent Top20",
        "",
        *(_top20_table(recent_top20) if recent_top20 else [f"Recent model failed: {recent_failure_reason}"]),
        "",
        "## 9. Gate Fused Top20",
        "",
        *_top20_table(fused_top20),
        "",
        "## 10. 生成文件路径",
        "",
        *[f"- {k}: `{v}`" for k, v in output_paths.items()],
        f"- base_X_train_columns: `{base_feature_cols_path}`",
        f"- recent_X_train_columns: `{recent_feature_cols_path}`",
        f"- event3_coverage_gate: `{event_gate_path}`",
        f"- event3_missing_nonzero: `{event_feature_stats_path}`",
        f"- event3_feature_importance: `{event_importance_path}`",
        "",
        "## 11. 是否建议 Claude 评审",
        "",
        "- yes; 建议重点评审 PIT 日期、事件稀疏 coverage gate、以及 event3 特征进入 recent-only 的边界。",
    ]
    output_paths["integration_report"].write_text("\n".join(integration_lines), encoding="utf-8")

    summary = {
        "requested_as_of_date": REQUESTED_AS_OF_DATE.isoformat(),
        "actual_as_of_date": actual_as_of.isoformat(),
        "actual_reason": actual_reason,
        "coverage_summary": coverage_summary,
        "base_meta": base_meta,
        "recent_status": recent_status,
        "recent_meta": recent_meta,
        "recent_failure_reason": recent_failure_reason,
        "gate_mode": "event3_enhanced_gate",
        "collector_summary": collector_summary,
        "event_features_in_recent": event_features_in_recent,
        "event_features_in_base": event_features_in_base,
        "margin_cross_validation_only": True,
        "margin_in_recent": margin_in_recent,
        "margin_in_base": margin_in_base,
        "tier_counts": tier_counts,
        "intersections": intersections,
        "boosted": boosted,
        "downgraded": downgraded,
        "vetoed": vetoed,
        "outputs": {k: str(v) for k, v in output_paths.items()},
        "base_feature_cols": str(base_feature_cols_path),
        "recent_feature_cols": str(recent_feature_cols_path),
        "event_gate": str(event_gate_path),
        "event_feature_stats": str(event_feature_stats_path),
        "event_feature_importance": str(event_importance_path),
        "base_top20": base_top20,
        "recent_top20": recent_top20,
        "fused_top20": fused_top20,
    }
    output_paths["json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
