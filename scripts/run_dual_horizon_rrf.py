"""Run the weekly RRF strategy for D+3/D+5.

Base model (default): shared Ridge32 baseline
(quant_platform.selection.ridge_baseline) — the SAME prediction the
ordinary arm of compare_original_hybrid.py uses.  The long-history
LightGBM base remains available as --base-model lgbm-long, explicitly
marked as an independent strategy, not the weekly baseline.

Score gate: closed (2026-10-14 verdict); selection ranks the full RRF pool.

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
from quant_platform.selection.config import weekly_selection_config
from quant_platform.selection.fusion_score import fusion_score
from quant_platform.selection.gate_fusion import gate_first_fusion
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.selection.ridge_baseline import ORIGINAL_FEATURES, fit_predict_ridge32
from quant_platform.store.lake import ohlcv_path
from quant_platform.store.parquet_store import read_ohlcv

#: Base-model choices. "ridge32" reuses the shared weekly baseline;
#: "lgbm-long" keeps the legacy long-history LightGBM as an explicitly
#: marked INDEPENDENT strategy (not the weekly baseline).
BASE_MODEL_CHOICES = ("ridge32", "lgbm-long")


def observable_target(prices: pd.DataFrame, as_of: dt.date, horizon: int,
                      calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Use the project's T+1 close convention, excluding prices after as_of."""
    if horizon not in (3, 5):
        raise ValueError("Only the requested raw D+3/D+5 targets are supported")
    frame = prices[["symbol", "date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame[frame["date"] <= as_of].sort_values(["symbol", "date"])
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate price dates")
    if frame.empty:
        frame[f"ret_fwd_{horizon}d"] = np.nan
        frame[f"label_end_{horizon}d"] = None
        return frame.drop(columns="close")
    if calendar is None:
        dates = sorted(frame.date.unique())
    else:
        dates = sorted(pd.to_datetime(calendar.loc[calendar.is_trading, "date"]).dt.date.unique())
        dates = [d for d in dates if frame.date.min() <= d <= as_of]
        if not set(frame.date).issubset(dates):
            raise ValueError("Prices contain dates outside the trading calendar")
    closes = frame.pivot(index="date", columns="symbol", values="close").reindex(dates)
    values = closes.shift(-(horizon + 1)) / closes.shift(-1) - 1
    complete = closes.notna() & (closes > 0)
    for offset in range(1, horizon + 2):
        complete &= closes.shift(-offset).notna() & (closes.shift(-offset) > 0)
    values = values.where(complete)
    rows = pd.Index(dates).get_indexer(frame.date)
    cols = closes.columns.get_indexer(frame.symbol)
    frame[f"ret_fwd_{horizon}d"] = values.to_numpy()[rows, cols]
    end_map = pd.Series(dates, index=dates).shift(-(horizon + 1))
    frame[f"label_end_{horizon}d"] = frame.date.map(end_map)
    return frame.drop(columns="close")


def recent_train_start(panel: pd.DataFrame, as_of: dt.date, label: str) -> dt.date:
    dates = sorted(panel.loc[(panel.date < as_of) & panel[label].notna(), "date"].unique())
    if len(dates) < 80:
        raise RuntimeError(f"{label}: fewer than 80 observable training dates")
    return dates[max(0, len(dates) - 120)]


def select_rrf(
    fused: pd.DataFrame,
    attributes: pd.DataFrame,
    sector_cap_tech: int | None = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """RRF 全量候选选股（Gate 关闭），返回 (ranked, picks, cap_stats)。

    配置走 weekly_selection_config()，与 compare_original_hybrid.py 同口径。
    """
    # 2026-10-14 战略裁决：Gate 已关闭。actionable_pool 的 ACTIONABLE_TIERS
    # 过滤在 D3/D5 两组独立实验中稳定伤害收益（D3 -0.14pp，D5 -0.10pp），
    # 系统性剔除逆势上涨的防御股（如中信银行+1.32%、交通银行+1.92%、宁波港+1.67%）。
    # 改为直接对全量候选做 RRF 融合排序；fused 仍经过 gate_first_fusion，
    # tier 列会写入 D{h}_gate.csv 和 run_summary.json 以便复核。
    pool = fusion_score(fused, method="rrf", k=60)
    if pool.empty:
        raise RuntimeError("No stocks to rank")
    pool = pool.merge(attributes, on="symbol", how="left", validate="one_to_one")
    pool = pool.sort_values(["fusion_score_col", "symbol"], ascending=[False, True], kind="stable")
    ranker = IndustryNeutralRanker(
        weekly_selection_config(sector_cap_tech=sector_cap_tech),
        industry_col="industry_code", name_col="industry_name",
        score_col="fusion_score_col", symbol_col="symbol",
    )
    ranked = ranker.run(pool)
    picks = ranked[ranked.selected].sort_values(
        ["fusion_score_col", "symbol"], ascending=[False, True]
    ).reset_index(drop=True)
    picks.insert(0, "recommendation_rank", np.arange(1, len(picks) + 1))
    return ranked, picks, (ranker.last_sector_cap_stats or {})


def _run_shared_weekly(
    as_of: dt.date, horizons: list[int], out_dir: Path, n_jobs: int,
    sector_cap_tech: int | None, overwrite: bool,
) -> dict:
    """Use the exact comparison pipeline, including its recent rows and 37 columns.

    Sharing only the Ridge fit function is insufficient: the previous entry
    still fit a 40-column recent model on a different row set. Both entry
    points now execute one implementation and export the same scores/picks.
    """
    from scripts.compare_original_hybrid import run as compare_run

    controlled = compare_run(
        as_of, out_dir / "comparison", None, n_jobs,
        sector_cap_tech=sector_cap_tech, overwrite=overwrite,
        horizons=tuple(horizons),
    )
    attributes = pd.read_csv(out_dir / "comparison/prediction_attributes.csv", dtype={"symbol": str})
    summary = {
        "as_of": controlled["as_of"], "entry_date": controlled["entry_date"],
        "base_model": "ridge32_shared_baseline", "base_features": ORIGINAL_FEATURES,
        "recent_features": ORIGINAL_FEATURES + controlled["flow_features"],
        "flow_features": controlled["flow_features"], "fusion_method": "rrf", "rrf_k": 60,
        "scored_count": controlled["scored_count"], "excluded_symbols": controlled["excluded_symbols"],
        "selection": controlled["selection"], "score_gate_enabled": False,
        "retrospective_reconstruction": controlled["retrospective_reconstruction"],
        "prediction_feature_hash": controlled["prediction_feature_hash"],
        "canonical_training_summary": "comparison/training_summary.json", "horizons": {},
    }
    pd.DataFrame({"feature": summary["base_features"]}).to_csv(out_dir / "base_features.csv", index=False)
    pd.DataFrame({"feature": summary["recent_features"]}).to_csv(out_dir / "recent_features.csv", index=False)
    for h in horizons:
        folder = out_dir / "comparison"
        base = pd.read_csv(folder / f"D{h}_original_scores.csv", dtype={"symbol": str})
        recent = pd.read_csv(folder / f"D{h}_recent_flow_scores.csv", dtype={"symbol": str})
        gate = pd.read_csv(folder / f"D{h}_score_gate.csv", dtype={"symbol": str})
        ranked, picks, stats = select_rrf(gate, attributes, sector_cap_tech)
        expected = pd.read_csv(folder / f"D{h}_hybrid_flow_picks.csv", dtype={"symbol": str})
        if picks.symbol.tolist() != expected.symbol.tolist() or not np.allclose(
            picks.fusion_score_col, expected.fusion_score_col, rtol=0, atol=1e-15
        ):
            raise RuntimeError("Weekly and comparison hybrid selections differ")
        importance = pd.read_csv(folder / f"D{h}_recent_flow_importance.csv")
        for name, frame in (("base", base), ("recent", recent), ("gate", gate),
                            ("rrf_pool", ranked), ("picks", picks), ("recent_importance", importance)):
            frame.to_csv(out_dir / f"D{h}_{name}.csv", index=False)
        detail = controlled["horizons"][str(h)]
        summary["horizons"][str(h)] = {
            "label_col": detail["label_col"], "target_date": detail["target_date"],
            "base": detail["baseline_meta"],
            "recent": {"train_start": detail["recent_train_start"], "train_end": detail["recent_train_end"],
                       "x_train_rows": detail["shared_recent_train_rows"], "feature_count": 37,
                       "shared_row_label_hash": detail["shared_recent_row_label_hash"]},
            "recent_train_dates": detail["shared_recent_dates"],
            "original_scores_hash": detail["original_scores_hash"],
            "flow_audit": {c: {"train_nonnull_rows": detail["shared_recent_train_rows"],
                               "train_nonnull_dates": detail["flow_train_nonnull_dates"][c],
                               "prediction_nonnull_rows": controlled["scored_count"],
                               "importance": detail["flow_importances"][c]}
                           for c in controlled["flow_features"]},
            "selected_count": len(picks), "tier_counts": gate.gate_tier.value_counts().to_dict(),
            "selection_audit": stats, "matches_comparison_hybrid": True,
            "selected": json.loads(picks.to_json(orient="records", date_format="iso")),
        }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved shared weekly strategy {out_dir / 'run_summary.json'}", flush=True)
    return summary


def run(
    as_of: dt.date,
    horizons: list[int],
    out_dir: Path,
    n_jobs: int,
    base_model: str = "ridge32",
    sector_cap_tech: int | None = 10,
    overwrite: bool = False,
) -> dict:
    """Weekly RRF strategy entry.

    base_model: "ridge32" (default) reuses the shared Ridge32 baseline —
        identical to the ordinary arm of compare_original_hybrid.py.
        "lgbm-long" keeps the legacy long-history LightGBM base, marked
        in the summary as an independent strategy.
    sector_cap_tech: tech-sector cap; None/0 disables (historical repro).
    overwrite: False refuses to write into a non-empty out_dir.
    """
    if base_model not in BASE_MODEL_CHOICES:
        raise ValueError(f"base_model must be one of {BASE_MODEL_CHOICES}, got {base_model!r}")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory {out_dir} already contains files; refusing to overwrite "
            "historical reports. Use a new --out-dir or pass --overwrite."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if base_model == "ridge32":
        return _run_shared_weekly(as_of, horizons, out_dir, n_jobs, sector_cap_tech, overwrite)
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
    selection_cfg = weekly_selection_config(sector_cap_tech=sector_cap_tech)
    summary = {
        "as_of": as_of.isoformat(), "feature_set_id": feature_set_id,
        "universe_count": len(symbols), "scored_count": len(pred),
        "excluded_symbols": sorted(set(symbols) - covered),
        "base_model": (
            "ridge32_shared_baseline"
            if base_model == "ridge32"
            else "lgbm_long_independent_strategy"
        ),
        "base_features": base_features, "recent_features": recent_features,
        "flow_features": flow_features, "fusion_method": "rrf", "rrf_k": 60,
        "selection": (
            "RRF fusion, no gate (gate closed 2026-10-14); "
            f"equal_top_k=3; max_total=50; sector_cap_tech={sector_cap_tech}; "
            f"strategy_version={selection_cfg.version}; rank by RRF"
        ),
        "entry_date": future[0].isoformat(), "horizons": {},
    }
    print(f"Features: base={len(base_features)}, recent={len(recent_features)}, flow={flow_features}", flush=True)
    for h in horizons:
        label = f"ret_fwd_{h}d"
        start = recent_train_start(panel, as_of, label)
        print(f"D+{h}: fitting {label}; recent start {start}; base_model={base_model}", flush=True)
        if base_model == "ridge32":
            # 共享基线：与 compare_original_hybrid.py 的 ordinary 臂同一份 Ridge32 预测。
            base, base_meta, _ = fit_predict_ridge32(panel, as_of, label, ORIGINAL_FEATURES)
            base_imp = pd.DataFrame(columns=["feature", "importance"])
        else:
            # 独立策略：长期 LightGBM 基线（35/40 因子），明确标记，不混入周基线。
            base, base_meta, _, base_imp, _ = legacy._score_model(
                panel, as_of, base_features, train_start=None,
                params={**legacy.BASE_PARAMS, "n_jobs": n_jobs}, score_prefix="base", label_col=label,
            )
            base_meta["independent_strategy"] = True
            base_meta["note"] = (
                "lgbm-long is an INDEPENDENT strategy, not the shared weekly baseline"
            )
        recent, recent_meta, _, recent_imp, _ = legacy._score_model(
            panel, as_of, recent_features, train_start=start,
            params={**legacy.RECENT_PARAMS, "n_jobs": n_jobs}, score_prefix="recent", label_col=label,
        )
        for scored, prefix in ((base, "base"), (recent, "recent")):
            if prefix == "base" and base_model == "ridge32":
                # 共享基线按 32 特征完备性过滤（与 compare_original_hybrid.py 同口径），
                # 允许略小于 full universe；recent 仍要求全覆盖。
                if len(scored) < 250 or not np.isfinite(scored["base_score"]).all():
                    raise RuntimeError(f"D+{h}: insufficient Ridge32 base coverage {len(scored)}")
            elif set(scored.symbol) != covered or not np.isfinite(scored[f"{prefix}_score"]).all():
                raise RuntimeError(f"D+{h}: missing or nonfinite {prefix} scores")
        joined = base.merge(recent[["symbol", "trade_date", "recent_score", "recent_rank", "recent_pct"]],
                            on=["symbol", "trade_date"], validate="one_to_one")
        joined = joined.merge(risk, on=["symbol", "trade_date"], how="left", validate="one_to_one")
        joined[["risk_flags", "event_flags"]] = joined[["risk_flags", "event_flags"]].fillna("")
        fused = gate_first_fusion(joined)
        ranked, picks, cap_stats = select_rrf(fused, attributes, sector_cap_tech)
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
            "selection_audit": {
                "pre_cap_count": cap_stats.get("pre_count"),
                "post_cap_count": cap_stats.get("post_count"),
                "sector_cap_enabled": cap_stats.get("enabled"),
                "dropped_by_sector_cap": cap_stats.get("dropped_symbols", []),
            },
            "selected": json.loads(picks.to_json(orient="records", date_format="iso")),
        }
        print(f"D+{h}: {len(picks)} selected; flow split importances {flow_imp.to_dict('records')}", flush=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {out_dir / 'run_summary.json'}", flush=True)
    return summary


def _parse_sector_cap(value: str) -> int | None:
    """--sector-cap-tech 解析：'off'/'none' -> None（历史复现口径），否则为非负整数。"""
    if value.lower() in ("off", "none"):
        return None
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("--sector-cap-tech must be >= 0 or 'off'")
    return ivalue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", choices=(3, 5), default=[3, 5])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--base-model", choices=BASE_MODEL_CHOICES, default="ridge32",
        help="ridge32=共享周基线（默认）；lgbm-long=长期LightGBM独立策略。",
    )
    parser.add_argument(
        "--sector-cap-tech", type=_parse_sector_cap, default=10,
        help="科技大类合计上限；'off' 关闭约束（历史复现口径）。默认 10。",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="允许覆盖已有内容的输出目录；默认拒绝，保护历史报告。",
    )
    args = parser.parse_args()
    output = args.out_dir or legacy.REPORT_DIR / f"dual_horizon_rrf_{args.as_of}"
    run(args.as_of, sorted(set(args.horizons)), output, args.n_jobs,
        base_model=args.base_model, sector_cap_tech=args.sector_cap_tech,
        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
