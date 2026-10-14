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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_platform.features import industry as industry_module
from quant_platform.features.flow import FLOW_SPECS, build_flow_features, load_flow_panel
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id
from quant_platform.features.valuation import load_valuation_panel
from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.fusion_score import actionable_pool, fusion_score
from quant_platform.selection.gate_fusion import gate_first_fusion
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.training.lgbm_model import fit_final_model
from scripts.next_week_d3_prediction import build_panel, load_symbols
from scripts.run_d3_gate_once import RECENT_PARAMS, _feature_importance_all
from scripts.run_dual_horizon_rrf import observable_target
from temp.latest_dual_horizon import _fast_join_industry

STORE = ROOT / "models/data"
# The original 32-column Ridge contract, verified against the delivered file.
ORIGINAL_FEATURES = [
    "ma_5", "ma_10", "ma_20", "ma_60", "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j", "rsi_6", "rsi_12", "rsi_24", "boll_upper",
    "boll_mid", "atr_14", "adx_14", "obv", "cci_14", "roc_10", "willr_14",
    "stoch_k", "stoch_d", "reversal_3d", "cs_rank_close", "cs_rank_volume",
    "cs_rank_rsi_6", "cs_rank_roc_10", "cs_rank_ma_5", "cs_zscore_close",
    "cs_zscore_volume", "cs_zscore_rsi_6",
]
FLOW_FEATURES = [s.name for s in FLOW_SPECS]
ARM_NAMES = {
    "ordinary": "原版Ridge32",
    "hybrid_no_flow": "原版+近期模型（无资金流）",
    "hybrid_flow": "原版+近期模型（有资金流）",
    "hybrid_flow_score_gate": "有资金流混合+分数Gate",
}


def fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns], index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def scores(pred: pd.DataFrame, values: np.ndarray, prefix: str) -> pd.DataFrame:
    if len(values) != len(pred) or not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite predictions")
    result = pred[["symbol", "date"]].copy()
    result["trade_date"] = result.date
    result[f"{prefix}_score"] = values
    result[f"{prefix}_rank"] = result[f"{prefix}_score"].rank(ascending=False, method="first").astype(int)
    result[f"{prefix}_pct"] = result[f"{prefix}_score"].rank(ascending=True, pct=True)
    return result


def combine(base: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    if set(base.symbol) != set(recent.symbol):
        raise ValueError("The baseline and recent scoring universes differ")
    out = base.merge(recent.drop(columns="date"), on=["symbol", "trade_date"], validate="one_to_one")
    out["risk_flags"] = ""
    out["event_flags"] = ""
    return out


def select(frame: pd.DataFrame, attributes: pd.DataFrame, score_col: str) -> pd.DataFrame:
    frame = frame.merge(attributes, on="symbol", how="left", validate="one_to_one")
    ranked = IndustryNeutralRanker(
        SelectionConfig(strategy=StrategyType.EQUAL_TOP_K, top_k=3, max_total=50),
        industry_col="industry_code", name_col="industry_name", score_col=score_col,
    ).run(frame)
    picks = ranked[ranked.selected].sort_values([score_col, "symbol"], ascending=[False, True]).reset_index(drop=True)
    picks.insert(0, "recommendation_rank", np.arange(1, len(picks) + 1))
    return picks


def verify_reference(picks: pd.DataFrame, train: pd.DataFrame, reference: dict, horizon: int) -> dict:
    expected = reference["horizons"][str(horizon)]
    old = expected["selected"]
    if len(train) != expected["train_rows"] or train.date.min().isoformat() != expected["train_first_date"] or train.date.max().isoformat() != expected["train_last_date"]:
        raise RuntimeError("Original training window/row count cannot be reproduced")
    if picks.symbol.tolist() != [r["symbol"] for r in old]:
        raise RuntimeError("Original ordered selection cannot be reproduced")
    difference = float(np.max(np.abs(picks.base_score.to_numpy() - np.array([r["model_score"] for r in old]))))
    if difference > 1e-12:
        raise RuntimeError(f"Original scores changed: {difference}")
    return {"ordered_selection_matches": True, "train_rows_match": True, "max_score_difference": difference}


def run(as_of: dt.date, out: Path, reference_path: Path | None, n_jobs: int) -> dict:
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
    for h in (3, 5):
        panel = panel.drop(columns=[f"ret_fwd_{h}d"], errors="ignore").merge(
            observable_target(prices, as_of, h), on=["symbol", "date"], how="left", validate="one_to_one",
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
    if len(future) < 6:
        raise RuntimeError("Insufficient trading calendar")
    summary = {
        "as_of": as_of.isoformat(), "entry_date": future[0].isoformat(),
        "research_only": True, "retrospective_reconstruction": True,
        "original_features": ORIGINAL_FEATURES, "flow_features": FLOW_FEATURES,
        "original_algorithm": "Ridge(alpha=1.0)",
        "recent_algorithm": "LightGBM", "recent_params": {**RECENT_PARAMS, "n_jobs": n_jobs},
        "rrf_k": 60, "recent_window_dates": 120,
        "selection": {"top_k_per_industry": 3, "max_total": 50},
        "gate_scope": "score tiers only; no additional event/risk factors",
        "scored_count": len(pred), "excluded_symbols": sorted(set(symbols) - set(pred.symbol)),
        "prediction_feature_hash": fingerprint(pred, ["symbol", "date"] + ORIGINAL_FEATURES),
        "flow_source_counts": flow.source.value_counts().to_dict() if "source" in flow else {},
        "flow_latest_date": flow.date.max().isoformat(), "horizons": {},
    }
    all_picks = []
    for h in (3, 5):
        label = f"ret_fwd_{h}d"
        train = panel[panel.date < as_of].dropna(subset=ORIGINAL_FEATURES + [label]).copy()
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        model.fit(train[ORIGINAL_FEATURES].values, train[label].values)
        base = scores(pred, model.predict(pred[ORIGINAL_FEATURES].values), "base")
        ordinary = select(base, attributes, "base_score")
        reference_audit = verify_reference(ordinary, train, reference, h) if reference else {"reference_supplied": False}
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
        arms = {
            "ordinary": ordinary,
            "hybrid_no_flow": select(no_flow, attributes, "fusion_score_col"),
            "hybrid_flow": select(with_flow, attributes, "fusion_score_col"),
            "hybrid_flow_score_gate": select(actionable_pool(gated), attributes, "fusion_score_col"),
        }
        equal_n = min(len(p) for p in arms.values())
        if equal_n == 0:
            raise RuntimeError("No selections in a comparison arm")
        for key, picks in arms.items():
            picks = picks.copy()
            picks["horizon"] = h
            picks["strategy"] = key
            picks["strategy_name"] = ARM_NAMES[key]
            picks["in_equal_count_comparison"] = picks.recommendation_rank <= equal_n
            picks.to_csv(out / f"D{h}_{key}_picks.csv", index=False, encoding="utf-8-sig")
            all_picks.append(picks)
        base.to_csv(out / f"D{h}_original_scores.csv", index=False)
        joblib.dump(model, out / f"D{h}_original_ridge.joblib")
        summary["horizons"][str(h)] = {
            "label_col": label, "target_date": future[h].isoformat(), "reference_audit": reference_audit,
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
            "equal_count_n": equal_n, "full_selection_counts": {key: len(p) for key, p in arms.items()},
        }
        print(f"D{h}: reference={reference_audit}; shared recent={len(recent_train)} rows/{recent_train.date.nunique()} dates; counts={summary['horizons'][str(h)]['full_selection_counts']}", flush=True)
    combined = pd.concat(all_picks, ignore_index=True)
    combined.to_csv(out / "all_picks.csv", index=False, encoding="utf-8-sig")
    (out / "requested_symbols.json").write_text(json.dumps(sorted(combined.symbol.unique()), indent=2), encoding="utf-8")
    (out / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--original-reference", type=Path)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    run(args.as_of, args.out_dir, args.original_reference, args.n_jobs)


if __name__ == "__main__":
    main()
