"""Run the existing long/recent LightGBM + gate + RRF strategy for D+3/D+5.

Example: python scripts/run_dual_horizon_rrf.py --as-of 2026-10-09
Writes research rankings and the exact input-feature audit; no registry promotion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_d3_gate_once as legacy
from quant_platform.cli import _build_feature_cols, _coverage_gate_config_for_universe, _feature_family_lookup
from quant_platform.evaluation.coverage_gate import compute_feature_coverage_report, select_features_by_gate
from quant_platform.features.flow import FLOW_SPECS
from quant_platform.features.registry import feature_metadata_lookup
from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.fusion_score import actionable_pool
from quant_platform.selection.gate_fusion import gate_first_fusion
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.store.lake import ohlcv_path
from quant_platform.store.parquet_store import read_ohlcv


def observable_target(prices: pd.DataFrame, as_of: dt.date, horizon: int) -> pd.DataFrame:
    """Use the project's T+1 close convention, excluding prices after as_of."""
    if horizon not in (3, 5):
        raise ValueError("Only the requested raw D+3/D+5 targets are supported")
    frame = prices[["symbol", "date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame[frame["date"] <= as_of].sort_values(["symbol", "date"])
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate price dates")
    groups = frame.groupby("symbol", sort=False)
    frame[f"ret_fwd_{horizon}d"] = groups["close"].shift(-(horizon + 1)) / groups["close"].shift(-1) - 1
    frame[f"label_end_{horizon}d"] = groups["date"].shift(-(horizon + 1))
    return frame.drop(columns="close")


def recent_train_start(panel: pd.DataFrame, as_of: dt.date, label: str) -> dt.date:
    dates = sorted(panel.loc[(panel.date < as_of) & panel[label].notna(), "date"].unique())
    if len(dates) < 80:
        raise RuntimeError(f"{label}: fewer than 80 observable training dates")
    return dates[max(0, len(dates) - 120)]


def select_rrf(fused: pd.DataFrame, attributes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = actionable_pool(fused, method="rrf", k=60)
    if pool.empty:
        raise RuntimeError("No actionable stocks passed the gate")
    pool = pool.merge(attributes, on="symbol", how="left", validate="one_to_one")
    ranked = IndustryNeutralRanker(
        SelectionConfig(strategy=StrategyType.EQUAL_TOP_K, top_k=3, max_total=50),
        industry_col="industry_code", name_col="industry_name",
        score_col="fusion_score_col", symbol_col="symbol",
    ).run(pool)
    picks = ranked[ranked.selected].sort_values(
        ["fusion_score_col", "symbol"], ascending=[False, True]
    ).reset_index(drop=True)
    picks.insert(0, "recommendation_rank", np.arange(1, len(picks) + 1))
    return ranked, picks


def run(as_of: dt.date, horizons: list[int], out_dir: Path, n_jobs: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = legacy._load_symbols()
    prices = pd.concat([read_ohlcv(ohlcv_path(legacy.STORE_ROOT, s)) for s in symbols], ignore_index=True)
    prices["symbol"] = prices.symbol.astype(str).str.zfill(6)
    prices["date"] = pd.to_datetime(prices.date).dt.date
    covered = set(prices.loc[prices.date == as_of, "symbol"])
    if len(covered) < 250:
        raise RuntimeError(f"{as_of}: insufficient OHLCV coverage {len(covered)}/300")
    print(f"Building enriched panel for {len(symbols)} stocks through {as_of}", flush=True)
    # Flow/event builders only add columns; base candidates exclude these
    # families below. Build once so both horizons consume identical inputs.
    panel, feature_set_id = legacy._build_panel(
        symbols, as_of, include_flow=True, include_lockup=True, strict_features=True
    )
    panel["symbol"] = panel.symbol.astype(str).str.zfill(6)
    if panel.duplicated(["symbol", "date"]).any():
        raise RuntimeError("Duplicate rows in feature panel")
    feature_candidates = _build_feature_cols(panel)
    # Stored labels may include outcomes after the requested date. Replace
    # targets in memory from prices truncated at as_of; never rewrite labels.
    for h in horizons:
        target = observable_target(prices, as_of, h)
        panel = panel.drop(columns=[f"ret_fwd_{h}d"], errors="ignore").merge(
            target, on=["symbol", "date"], how="left", validate="one_to_one"
        )
    families = _feature_family_lookup()
    gate = compute_feature_coverage_report(
        panel, feature_candidates, family_by_col=families,
        feature_metadata=feature_metadata_lookup(), as_of_date=as_of,
        config=_coverage_gate_config_for_universe(len(symbols)),
    )
    gate.to_csv(out_dir / "feature_coverage.csv", index=False)
    base_features = [c for c in select_features_by_gate(gate, model_path="base") if families.get(c) != "flow"]
    recent_features = select_features_by_gate(gate, model_path="recent")
    expected_flow = [s.name for s in FLOW_SPECS]
    flow_features = [c for c in recent_features if c in expected_flow]
    if not flow_features:
        raise RuntimeError("Coverage gate rejected every fund-flow feature; refusing silent base-only output")
    if set(base_features) & set(expected_flow):
        raise RuntimeError("Short-history flow entered the long-history base model")
    pd.DataFrame({"feature": base_features}).to_csv(out_dir / "base_features.csv", index=False)
    pd.DataFrame({"feature": recent_features}).to_csv(out_dir / "recent_features.csv", index=False)
    pred = panel[panel.date == as_of].copy()
    if set(pred.symbol) != covered:
        raise RuntimeError("Feature-day universe differs from the available OHLCV universe")
    names = pd.read_parquet(legacy.STORE_ROOT / "universe/csi300/membership.parquet")
    name_map = dict(zip(names.symbol.astype(str).str.zfill(6), names.name))
    attributes = pred[["symbol", "close", "industry_code", "industry_name"]].copy()
    attributes["name"] = attributes.symbol.map(name_map)
    risk = legacy._risk_flags(panel, as_of)
    calendar = pd.read_parquet(legacy.STORE_ROOT / "calendar/trading_calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar.date).dt.date
    future = sorted(calendar.loc[calendar.is_trading & (calendar.date > as_of), "date"])
    if len(future) <= max(horizons):
        raise RuntimeError("Trading calendar does not cover the target observation dates")
    summary = {
        "as_of": as_of.isoformat(), "feature_set_id": feature_set_id,
        "universe_count": len(symbols), "scored_count": len(pred),
        "excluded_symbols": sorted(set(symbols) - covered),
        "base_features": base_features, "recent_features": recent_features,
        "flow_features": flow_features, "fusion_method": "rrf", "rrf_k": 60,
        "selection": "A/B gate; equal_top_k=3; max_total=50; rank by RRF",
        "entry_date": future[0].isoformat(), "horizons": {},
    }
    print(f"Features: base={len(base_features)}, recent={len(recent_features)}, flow={flow_features}", flush=True)
    for h in horizons:
        label = f"ret_fwd_{h}d"
        start = recent_train_start(panel, as_of, label)
        print(f"D+{h}: fitting {label}; recent start {start}", flush=True)
        base, base_meta, _, base_imp, _ = legacy._score_model(
            panel, as_of, base_features, train_start=None,
            params={**legacy.BASE_PARAMS, "n_jobs": n_jobs}, score_prefix="base", label_col=label,
        )
        recent, recent_meta, _, recent_imp, _ = legacy._score_model(
            panel, as_of, recent_features, train_start=start,
            params={**legacy.RECENT_PARAMS, "n_jobs": n_jobs}, score_prefix="recent", label_col=label,
        )
        for scored, prefix in ((base, "base"), (recent, "recent")):
            if set(scored.symbol) != covered or not np.isfinite(scored[f"{prefix}_score"]).all():
                raise RuntimeError(f"D+{h}: missing or nonfinite {prefix} scores")
        joined = base.merge(recent[["symbol", "trade_date", "recent_score", "recent_rank", "recent_pct"]],
                            on=["symbol", "trade_date"], validate="one_to_one")
        joined = joined.merge(risk, on=["symbol", "trade_date"], how="left", validate="one_to_one")
        joined[["risk_flags", "event_flags"]] = joined[["risk_flags", "event_flags"]].fillna("")
        fused = gate_first_fusion(joined)
        ranked, picks = select_rrf(fused, attributes)
        flow_imp = recent_imp[recent_imp.feature.isin(flow_features)]
        train = panel[(panel.date >= start) & (panel.date < as_of) & panel[label].notna()]
        flow_audit = {
            c: {
                "train_nonnull_rows": int(train[c].notna().sum()),
                "train_nonnull_dates": int(train.loc[train[c].notna(), "date"].nunique()),
                "prediction_nonnull_rows": int(pred[c].notna().sum()),
                "importance": int(flow_imp.loc[flow_imp.feature == c, "importance"].iloc[0]),
            } for c in flow_features
        }
        if min(v["train_nonnull_dates"] for v in flow_audit.values()) < 80:
            raise RuntimeError(f"D+{h}: fewer than 80 training dates with usable flow")
        for name, frame in (("base", base), ("recent", recent), ("gate", fused),
                            ("rrf_pool", ranked), ("picks", picks),
                            ("base_importance", base_imp), ("recent_importance", recent_imp)):
            frame.to_csv(out_dir / f"D{h}_{name}.csv", index=False)
        summary["horizons"][str(h)] = {
            "label_col": label, "target_date": future[h].isoformat(),
            "base": base_meta, "recent": recent_meta,
            "max_observed_label_end": max(panel.loc[panel[label].notna(), f"label_end_{h}d"]).isoformat(),
            "recent_train_dates": int(train.date.nunique()),
            "flow_audit": flow_audit, "actionable_count": len(ranked), "selected_count": len(picks),
            "tier_counts": fused.gate_tier.value_counts().to_dict(),
            "selected": json.loads(picks.to_json(orient="records", date_format="iso")),
        }
        print(f"D+{h}: {len(picks)} selected; flow split importances {flow_imp.to_dict('records')}", flush=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {out_dir / 'run_summary.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", choices=(3, 5), default=[3, 5])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    output = args.out_dir or legacy.REPORT_DIR / f"dual_horizon_rrf_{args.as_of}"
    run(args.as_of, sorted(set(args.horizons)), output, args.n_jobs)


if __name__ == "__main__":
    main()
