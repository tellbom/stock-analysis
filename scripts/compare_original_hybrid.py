"""Compare original Ridge D3/D5 with hybrids that reuse its exact scores.

Research only: cached features are read, targets are censored at --as-of,
and no lake files or registered production models are changed. The recent
no-flow/flow arms share training rows, dates, parameters and scoring universe.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_platform.features import industry as industry_module
from quant_platform.features.flow import FLOW_SPECS, build_flow_features, load_flow_panel
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id
from quant_platform.features.valuation import load_valuation_panel
from quant_platform.selection.config import weekly_selection_config
from quant_platform.selection.fusion_score import actionable_pool, fusion_score
from quant_platform.selection.gate_fusion import gate_first_fusion
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.selection.ridge_baseline import (
    ORIGINAL_FEATURES,
    fit_predict_ridge32,
    score_frame,
)
from quant_platform.training.lgbm_model import fit_final_model
from scripts.next_week_d3_prediction import build_panel, load_symbols
from scripts.run_d3_gate_once import RECENT_PARAMS, _feature_importance_all
from scripts.run_dual_horizon_rrf import observable_target
from temp.latest_dual_horizon import _fast_join_industry

STORE = ROOT / "models/data"
FLOW_FEATURES = [s.name for s in FLOW_SPECS]
# 诊断组后缀：分数 Gate 诊断臂的输出文件/策略名标记，不作为默认选股结果。
DIAGNOSTIC_SUFFIX = "_diagnostic"
ARM_NAMES = {
    "ordinary": "原版Ridge32",
    "hybrid_no_flow": "原版+近期模型（无资金流）",
    "hybrid_flow": "原版+近期模型（有资金流）",
    "hybrid_flow_score_gate_diagnostic": "有资金流混合+分数Gate（诊断，不作为选股结果）",
    "hybrid_timeweighted": "混合_时间权重+短窗口（P0/P1 实验臂）",
}
# 诊断臂的策略键（带后缀），与主策略臂区分。
DIAGNOSTIC_ARM = "hybrid_flow_score_gate_diagnostic"


def fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns], index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


# 保持旧测试的导入兼容：scores 即共享基线模块的 score_frame。
scores = score_frame


def combine(base: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    if set(base.symbol) != set(recent.symbol):
        raise ValueError("The baseline and recent scoring universes differ")
    out = base.merge(recent.drop(columns="date"), on=["symbol", "trade_date"], validate="one_to_one")
    out["risk_flags"] = ""
    out["event_flags"] = ""
    return out


def select(
    frame: pd.DataFrame,
    attributes: pd.DataFrame,
    score_col: str,
    sector_cap_tech: int | None = 10,
) -> tuple[pd.DataFrame, dict]:
    """选股并返回 (picks, cap_stats)。

    普通与混合必须同口径：配置统一走 weekly_selection_config()，
    sector_cap_tech 由调用方传入（None/0 = 历史复现口径，关闭约束）。
    """
    frame = frame.merge(attributes, on="symbol", how="left", validate="one_to_one")
    frame = frame.sort_values([score_col, "symbol"], ascending=[False, True], kind="stable")
    ranker = IndustryNeutralRanker(
        weekly_selection_config(sector_cap_tech=sector_cap_tech),
        industry_col="industry_code", name_col="industry_name", score_col=score_col,
    )
    ranked = ranker.run(frame)
    picks = ranked[ranked.selected].sort_values([score_col, "symbol"], ascending=[False, True]).reset_index(drop=True)
    picks.insert(0, "recommendation_rank", np.arange(1, len(picks) + 1))
    return picks, (ranker.last_sector_cap_stats or {})


def verify_reference(
    picks: pd.DataFrame,
    train: pd.DataFrame,
    reference: dict,
    horizon: int,
    check_picks: bool = True,
) -> dict:
    """核对历史参考。

    check_picks=True（默认）：完整核对，要求选股名单与参考完全一致。
    check_picks=False：板块约束开启时期望名单本来就会变，只核对训练窗口
    和基线分数（分数漂移仍是硬错误），名单核对跳过并记录。
    """
    expected = reference["horizons"][str(horizon)]
    old = expected["selected"]
    if len(train) != expected["train_rows"] or train.date.min().isoformat() != expected["train_first_date"] or train.date.max().isoformat() != expected["train_last_date"]:
        raise RuntimeError("Original training window/row count cannot be reproduced")
    result = {"train_rows_match": True, "ordered_selection_matches": None}
    if check_picks:
        if picks.symbol.tolist() != [r["symbol"] for r in old]:
            raise RuntimeError("Original ordered selection cannot be reproduced")
        result["ordered_selection_matches"] = True
    else:
        result["ordered_selection_skipped"] = "sector cap enabled; picks expected to differ"
    # 分数核对：只对参考中仍存在的 symbol 比对（约束只做减法，不改变分数）。
    ref_scores = {r["symbol"]: r["model_score"] for r in old}
    common = picks[picks.symbol.isin(ref_scores)]
    if len(common) == 0:
        raise RuntimeError("No overlapping symbols with reference for score check")
    difference = float(np.max(np.abs(
        common.base_score.to_numpy() - np.array([ref_scores[s] for s in common.symbol])
    )))
    if difference > 1e-12:
        raise RuntimeError(f"Original scores changed: {difference}")
    result["max_score_difference"] = difference
    result["score_check_symbols"] = int(len(common))
    return result


def run(
    as_of: dt.date,
    out: Path,
    reference_path: Path | None,
    n_jobs: int,
    sector_cap_tech: int | None = 10,
    overwrite: bool = False,
    horizons: tuple[int, ...] = (3, 5),
    use_time_weight: bool = False,
    recent_window: int = 120,
) -> dict:
    """每周对照实验入口。

    sector_cap_tech: 科技大类合计上限；None/0 = 历史复现口径（关闭约束）。
    overwrite: False 时若 out 已有内容则拒绝写入，保护历史报告不被覆盖。
    use_time_weight: P0 实验开关。True 时新增 hybrid_timeweighted 臂，
        其 base 用时间衰减权重 Ridge；冻结的 4 个臂不受影响。
    recent_window: P1 实验参数。recent 模型的 label 日期窗口；
        120 = 冻结基线。实验臂传入 60（约 3 个月，D+3 短线 regime 响应）。
    """
    horizons = tuple(sorted(set(horizons)))
    if not horizons or any(h not in (3, 5) for h in horizons):
        raise ValueError("Only raw D3/D5 horizons are supported")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Output directory {out} already contains files; refusing to overwrite "
            "historical reports. Use a new --out-dir or pass --overwrite."
        )
    out.mkdir(parents=True, exist_ok=True)
    symbols = load_symbols()
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path else None
    if reference and reference["as_of"] != as_of.isoformat():
        raise ValueError("Reference batch does not match --as-of")
    joiner = industry_module._join_industry
    industry_module._join_industry = _fast_join_industry
    try:
        panel = build_panel(symbols, compute_feature_set_id(DEFAULT_SPECS), max_date=as_of)
    finally:
        industry_module._join_industry = joiner
    panel["symbol"] = panel.symbol.astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel.date).dt.date
    prices = pd.concat([
        pd.read_parquet(STORE / f"silver/ohlcv/{s}.parquet", columns=["symbol", "date", "close"])
        for s in symbols
    ], ignore_index=True)
    prices["symbol"] = prices.symbol.astype(str).str.zfill(6)
    calendar = pd.read_parquet(STORE / "calendar/trading_calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar.date).dt.date
    for h in horizons:
        panel = panel.drop(columns=[f"ret_fwd_{h}d"], errors="ignore").merge(
            observable_target(prices, as_of, h, calendar=calendar), on=["symbol", "date"], how="left", validate="one_to_one",
        )
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    flow = load_flow_panel(STORE, symbols)
    flow = flow[flow.date <= as_of].copy()
    valuation = load_valuation_panel(STORE, symbols)
    valuation["date"] = pd.to_datetime(valuation.date).dt.date
    valuation = valuation[valuation.date <= as_of].copy()
    # Only append the five flow columns; preserve every original input value
    # and the original row order, including the Ridge matrix order.
    flow_features = build_flow_features(panel[["symbol", "date"]], flow, valuation)
    panel = panel.drop(columns=FLOW_FEATURES, errors="ignore").merge(
        flow_features[["symbol", "date"] + FLOW_FEATURES],
        on=["symbol", "date"], how="left", validate="one_to_one",
    )
    pred = panel[panel.date == as_of].dropna(subset=ORIGINAL_FEATURES).copy()
    if len(pred) < 250 or pred[FLOW_FEATURES].isna().any().any():
        raise RuntimeError("Prediction universe has insufficient original/flow coverage")
    names = pd.read_parquet(STORE / "universe/csi300/membership.parquet")
    name_map = dict(zip(names.symbol.astype(str).str.zfill(6), names.name))
    attributes = pred[["symbol", "industry_code", "industry_name", "close"]].copy()
    attributes["name"] = attributes.symbol.map(name_map)
    calendar = pd.read_parquet(STORE / "calendar/trading_calendar.parquet")
    calendar["date"] = pd.to_datetime(calendar.date).dt.date
    future = sorted(calendar.loc[calendar.is_trading & (calendar.date > as_of), "date"])
    if len(future) <= max(horizons):
        raise RuntimeError("Insufficient trading calendar")
    selection_cfg = weekly_selection_config(sector_cap_tech=sector_cap_tech)
    summary = {
        "as_of": as_of.isoformat(), "entry_date": future[0].isoformat(),
        "research_only": True, "retrospective_reconstruction": as_of < dt.date.today(),
        "original_features": ORIGINAL_FEATURES, "flow_features": FLOW_FEATURES,
        "original_algorithm": "Ridge(alpha=1.0)",
        "baseline_strategy_id": "ridge32_weekly_baseline",
        "recent_algorithm": "LightGBM", "recent_params": {**RECENT_PARAMS, "n_jobs": n_jobs},
        "rrf_k": 60, "recent_window_dates": 120,  # 冻结基线口径；实验臂见 horizons 下 experiment 段
        "experiment": {
            "enabled": bool(use_time_weight or recent_window != 120),
            "use_time_weight": bool(use_time_weight),
            "time_weight_half_life_days": 180 if use_time_weight else None,
            "recent_window_dates": recent_window,
            # P1：60 ≈ 3 个月 label 日期。D+3 短线 regime 变化快，
            # 120 天窗口混入过多旧 regime 信号；60 天让 recent 模型更快响应。
            "arm": "hybrid_timeweighted",
        },
        "data_contract_version": "qfq-full-history-calendar-v1",
        "label_date_basis": "exchange_calendar_complete_forward_window",
        "selection": {
            "top_k_per_industry": 3, "max_total": 50,
            "sector_cap_tech": sector_cap_tech,
            "strategy_version": selection_cfg.version,
        },
        "gate_scope": "score tiers only; no additional event/risk factors",
        "scored_count": len(pred), "excluded_symbols": sorted(set(symbols) - set(pred.symbol)),
        "prediction_feature_hash": fingerprint(pred, ["symbol", "date"] + ORIGINAL_FEATURES),
        "flow_source_counts": flow.source.value_counts().to_dict() if "source" in flow else {},
        "flow_latest_date": flow.date.max().isoformat(), "horizons": {},
    }
    all_picks = []
    attributes.to_csv(out / "prediction_attributes.csv", index=False)
    for h in horizons:
        label = f"ret_fwd_{h}d"
        # 共享 Ridge32 基线：与 run_dual_horizon_rrf.py 同一份实现。
        base, base_meta, model = fit_predict_ridge32(panel, as_of, label)
        train = panel[panel.date < as_of].dropna(subset=ORIGINAL_FEATURES + [label]).copy()
        ordinary, ordinary_cap = select(base, attributes, "base_score", sector_cap_tech)
        # 历史复现口径（约束关闭）才要求名单完全一致；约束开启时只核对分数。
        check_picks = sector_cap_tech is None or sector_cap_tech == 0
        reference_audit = (
            verify_reference(ordinary, train, reference, h, check_picks=check_picks)
            if reference else {"reference_supplied": False}
        )
        dates = sorted(train.date.unique())[-120:]
        recent_available = train[train.date.isin(dates)]
        recent_train = recent_available.dropna(subset=FLOW_FEATURES).copy()
        if recent_train.date.nunique() < 80:
            raise RuntimeError(f"D{h}: insufficient shared recent training dates")
        recent_scores = {}
        importances = {}
        for arm, features in (("no_flow", ORIGINAL_FEATURES), ("flow", ORIGINAL_FEATURES + FLOW_FEATURES)):
            fitted = fit_final_model(recent_train, features, label, {**RECENT_PARAMS, "n_jobs": n_jobs})
            recent_scores[arm] = scores(pred, fitted.predict(pred[features]), "recent")
            imp = _feature_importance_all(fitted, features)
            imp.to_csv(out / f"D{h}_recent_{arm}_importance.csv", index=False)
            importances[arm] = dict(zip(imp.feature, imp.importance.astype(int)))
            recent_scores[arm].to_csv(out / f"D{h}_recent_{arm}_scores.csv", index=False)
            joblib.dump(fitted, out / f"D{h}_recent_{arm}.joblib")
        no_flow = fusion_score(combine(base, recent_scores["no_flow"]))
        with_flow = fusion_score(combine(base, recent_scores["flow"]))
        gated = gate_first_fusion(combine(base, recent_scores["flow"]))
        gated.to_csv(out / f"D{h}_score_gate.csv", index=False)
        # 主策略臂：分数 Gate 保持关闭（2026-10-14 战略裁决）。
        hybrid_no_flow, no_flow_cap = select(no_flow, attributes, "fusion_score_col", sector_cap_tech)
        hybrid_flow, flow_cap = select(with_flow, attributes, "fusion_score_col", sector_cap_tech)
        # 诊断臂：恢复真正的分数 Gate 过滤，仅用于诊断 tier 行为，
        # 明确不作为默认选股结果（文件名/strategy 列带 _diagnostic 后缀）。
        diag_pool = actionable_pool(gated)
        diag_picks, diag_cap = select(diag_pool, attributes, "fusion_score_col", sector_cap_tech)
        arms = {
            "ordinary": (ordinary, ordinary_cap, False),
            "hybrid_no_flow": (hybrid_no_flow, no_flow_cap, False),
            "hybrid_flow": (hybrid_flow, flow_cap, False),
            DIAGNOSTIC_ARM: (diag_picks, diag_cap, True),
        }
        # 等数对照只在主策略臂之间取最小；诊断臂不参与、不拉低对照数。
        main_arms = {k: v for k, v in arms.items() if not v[2]}
        equal_n = min(len(p) for p, _, _ in main_arms.values())
        if equal_n == 0:
            raise RuntimeError("No selections in a comparison arm")
        selection_audit = {}
        for key, (picks, cap_stats, is_diagnostic) in arms.items():
            picks = picks.copy()
            picks["horizon"] = h
            picks["strategy"] = key
            picks["strategy_name"] = ARM_NAMES[key]
            picks["is_diagnostic"] = is_diagnostic
            picks["in_equal_count_comparison"] = (not is_diagnostic) & (picks.recommendation_rank <= equal_n)
            picks.to_csv(out / f"D{h}_{key}_picks.csv", index=False, encoding="utf-8-sig")
            all_picks.append(picks)
            selection_audit[key] = {
                "is_diagnostic": is_diagnostic,
                "pre_cap_count": cap_stats.get("pre_count"),
                "post_cap_count": cap_stats.get("post_count"),
                "sector_cap_enabled": cap_stats.get("enabled"),
                "dropped_by_sector_cap": cap_stats.get("dropped_symbols", []),
            }
        base.to_csv(out / f"D{h}_original_scores.csv", index=False)
        joblib.dump(model, out / f"D{h}_original_ridge.joblib")
        # ---- 实验臂：hybrid_timeweighted（P0 时间权重 + P1 短窗口）----
        # 在冻结臂全部处理完成后独立计算：不参与 equal_n、不进 selection_audit、
        # 不影响冻结臂的任何输出。默认关闭（use_time_weight=False 且 recent_window=120
        # 时整个块跳过），冻结基线零改动。
        experiment_audit: dict = {"enabled": False}
        if use_time_weight or recent_window != 120:
            # P0：同一份 Ridge32 实现，仅打开时间权重开关。
            base_tw, base_tw_meta, _ = fit_predict_ridge32(
                panel, as_of, label, use_time_weight=True,
            )
            # P1：recent 窗口按参数取值（实验默认 60 ≈ 3 个月）。
            dates_tw = sorted(train.date.unique())[-recent_window:]
            recent_tw_avail = train[train.date.isin(dates_tw)]
            recent_tw = recent_tw_avail.dropna(subset=FLOW_FEATURES).copy()
            if recent_tw.date.nunique() < 80:
                raise RuntimeError(f"D{h}: insufficient experiment recent training dates")
            fitted_tw = fit_final_model(
                recent_tw, ORIGINAL_FEATURES + FLOW_FEATURES, label,
                {**RECENT_PARAMS, "n_jobs": n_jobs},
            )
            recent_tw_scores = scores(
                pred, fitted_tw.predict(pred[ORIGINAL_FEATURES + FLOW_FEATURES]), "recent")
            fused_tw = fusion_score(combine(base_tw, recent_tw_scores))
            tw_picks, tw_cap = select(
                fused_tw, attributes, "fusion_score_col", sector_cap_tech)
            tw_picks = tw_picks.copy()
            tw_picks["horizon"] = h
            tw_picks["strategy"] = "hybrid_timeweighted"
            tw_picks["strategy_name"] = "混合_时间权重+短窗口"
            tw_picks["is_diagnostic"] = False
            # 实验臂不参与等数对照（避免拉低冻结臂的 equal_n）。
            tw_picks["in_equal_count_comparison"] = False
            tw_picks.to_csv(out / f"D{h}_hybrid_timeweighted_picks.csv",
                            index=False, encoding="utf-8-sig")
            all_picks.append(tw_picks)
            joblib.dump(fitted_tw, out / f"D{h}_recent_timeweighted.joblib")
            experiment_audit = {
                "enabled": True,
                "arm": "hybrid_timeweighted",
                "base_time_weighted": base_tw_meta.get("time_weighted"),
                "base_train_rows": base_tw_meta.get("train_rows"),
                "recent_window_dates": recent_window,
                "recent_train_dates": int(recent_tw.date.nunique()),
                "recent_train_start": recent_tw.date.min().isoformat(),
                "recent_train_end": recent_tw.date.max().isoformat(),
                "pick_count": len(tw_picks),
                "pre_cap_count": tw_cap.get("pre_count"),
                "post_cap_count": tw_cap.get("post_count"),
            }
            print(f"D{h}: experiment hybrid_timeweighted: {len(tw_picks)} picks, "
                  f"recent {recent_tw.date.nunique()} dates", flush=True)
        summary["horizons"][str(h)] = {
            "label_col": label, "target_date": future[h].isoformat(), "reference_audit": reference_audit,
            "baseline_meta": base_meta,
            "original_train_rows": len(train), "original_train_start": train.date.min().isoformat(),
            "original_train_end": train.date.max().isoformat(),
            "original_training_input_hash": fingerprint(train, ["symbol", "date"] + ORIGINAL_FEATURES + [label]),
            "original_scores_hash": fingerprint(base, ["symbol", "base_score", "base_rank"]),
            "shared_recent_train_rows": len(recent_train), "shared_recent_dates": int(recent_train.date.nunique()),
            "recent_train_start": recent_train.date.min().isoformat(), "recent_train_end": recent_train.date.max().isoformat(),
            "recent_rows_excluded_for_missing_flow": len(recent_available) - len(recent_train),
            "shared_recent_row_label_hash": fingerprint(recent_train, ["symbol", "date", label]),
            "recent_feature_counts": {"no_flow": 32, "flow": 37},
            "flow_importances": {c: importances["flow"][c] for c in FLOW_FEATURES},
            "flow_train_nonnull_dates": {c: int(recent_train.loc[recent_train[c].notna(), "date"].nunique()) for c in FLOW_FEATURES},
            "equal_count_n": equal_n, "full_selection_counts": {key: len(p) for key, (p, _, _) in arms.items()},
            "selection_audit": selection_audit,
            "experiment": experiment_audit,
        }
        print(f"D{h}: reference={reference_audit}; shared recent={len(recent_train)} rows/{recent_train.date.nunique()} dates; counts={summary['horizons'][str(h)]['full_selection_counts']}", flush=True)
    combined = pd.concat(all_picks, ignore_index=True)
    combined.to_csv(out / "all_picks.csv", index=False, encoding="utf-8-sig")
    (out / "requested_symbols.json").write_text(json.dumps(sorted(combined.symbol.unique()), indent=2), encoding="utf-8")
    (out / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--original-reference", type=Path)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--sector-cap-tech", type=_parse_sector_cap, default=10,
        help="科技大类合计上限；'off' 关闭约束（历史复现口径）。默认 10。",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="允许覆盖已有内容的输出目录；默认拒绝，保护历史报告。",
    )
    parser.add_argument(
        "--use-time-weight", action="store_true",
        help="P0 实验：新增 hybrid_timeweighted 臂，其 Ridge32 base 用指数时间衰减权重 "
             "(半衰期 180 天)。冻结的 4 个臂不受影响。默认关闭。",
    )
    parser.add_argument(
        "--recent-window", type=int, default=120,
        help="P1 实验：hybrid_timeweighted 臂的 recent 模型 label 日期窗口。120=冻结基线口径； "
             "实验建议 60（约 3 个月，D+3 短线 regime 响应）。默认 120。",
    )
    args = parser.parse_args()
    run(args.as_of, args.out_dir, args.original_reference, args.n_jobs,
        sector_cap_tech=args.sector_cap_tech, overwrite=args.overwrite,
        use_time_weight=args.use_time_weight, recent_window=args.recent_window)


if __name__ == "__main__":
    main()
