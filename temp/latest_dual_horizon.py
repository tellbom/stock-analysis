"""One-off, as-of 2026-09-30 D+3/D+5 research export.

Uses the existing feature, coverage-gate, Ridge, and selection components.
Does not change the registered production champion.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_platform.cli import _coverage_gate_config_for_universe, _feature_family_lookup
from quant_platform.evaluation.coverage_gate import (
    compute_feature_coverage_report,
    select_features_by_gate,
)
from quant_platform.features import industry as industry_module
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id, feature_metadata_lookup
from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.ranker import IndustryNeutralRanker
from scripts.next_week_d3_prediction import (
    build_panel,
    load_symbols,
)

AS_OF = dt.date(2026, 9, 30)
STORE = ROOT / "models" / "data"
OUT = ROOT / "temp" / "latest_dual_horizon.json"


def _fast_join_industry(panel: pd.DataFrame, industry_map: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of the project's non-overlapping SCD PIT join."""
    if industry_map.empty:
        out = panel.copy()
        out["industry_code"] = "_UNKNOWN"
        out["industry_name"] = ""
        return out
    left = panel.copy()
    left["date"] = pd.to_datetime(left["date"])
    left["symbol"] = left["symbol"].astype(str).str.zfill(6)
    right = industry_map[["symbol", "effective_date", "out_date", "industry_code", "industry_name"]].copy()
    right["symbol"] = right["symbol"].astype(str).str.zfill(6)
    right["effective_date"] = pd.to_datetime(right["effective_date"])
    right["out_date"] = pd.to_datetime(right["out_date"])
    left = left.sort_values(["date", "symbol"])
    right = right.sort_values(["effective_date", "symbol"])
    out = pd.merge_asof(
        left, right, left_on="date", right_on="effective_date",
        by="symbol", direction="backward",
    )
    expired = out["out_date"].notna() & (out["out_date"] <= out["date"])
    out.loc[expired, "industry_code"] = "_UNKNOWN"
    out.loc[expired, "industry_name"] = ""
    out["industry_code"] = out["industry_code"].fillna("_UNKNOWN")
    out["industry_name"] = out["industry_name"].fillna("")
    out["date"] = out["date"].dt.date
    return out.drop(columns=["effective_date", "out_date"])


def main() -> None:
    symbols = load_symbols()
    print(f"Universe: {len(symbols)}", flush=True)

    stale_labels = []
    for symbol in symbols:
        prices = pd.read_parquet(STORE / "silver" / "ohlcv" / f"{symbol}.parquet", columns=["date"])
        labels = pd.read_parquet(STORE / "labels" / "forward_returns" / f"{symbol}.parquet", columns=["date"])
        if str(labels["date"].max())[:10] < str(prices["date"].max())[:10]:
            stale_labels.append(symbol)
    if stale_labels:
        raise RuntimeError(f"Stale labels: {stale_labels[:10]} ({len(stale_labels)} total)")
    print("All labels match their per-symbol OHLCV dates", flush=True)

    feature_set_id = compute_feature_set_id(DEFAULT_SPECS)
    stale_features = []
    for symbol in symbols:
        prices = pd.read_parquet(STORE / "silver" / "ohlcv" / f"{symbol}.parquet", columns=["date"])
        features = pd.read_parquet(STORE / "features" / feature_set_id / f"{symbol}.parquet", columns=["date"])
        if str(features["date"].max())[:10] < str(prices["date"].max())[:10]:
            stale_features.append(symbol)
    if stale_features:
        raise RuntimeError(f"Stale features: {stale_features[:10]} ({len(stale_features)} total)")
    industry_module._join_industry = _fast_join_industry
    panel = build_panel(symbols, feature_set_id, max_date=AS_OF)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    pred_day = panel.loc[panel["date"] == AS_OF].copy()
    if len(pred_day) < 250:
        raise RuntimeError(f"Prediction-day coverage too low: {len(pred_day)}/300")

    meta_cols = {
        "symbol", "date", "close", "volume", "open", "high", "low",
        "amount", "outstanding_share", "turnover", "industry_code",
        "industry_name", "concept_tags",
    }
    label_prefixes = ("ret_fwd_", "vol_fwd_", "mdd_fwd_", "excess_vs_")
    feature_cols = [
        c for c in panel.columns
        if c not in meta_cols and not c.startswith(label_prefixes)
    ]
    gate = compute_feature_coverage_report(
        panel,
        feature_cols,
        family_by_col=_feature_family_lookup(),
        feature_metadata=feature_metadata_lookup(),
        config=_coverage_gate_config_for_universe(panel["symbol"].nunique()),
    )
    feature_cols = select_features_by_gate(gate, model_path="base")
    feature_cols = [
        c for c in feature_cols
        if c in pred_day.columns
        and pred_day[c].isna().mean() < 0.50
        and panel[c].isna().mean() < 0.50
    ]
    if not feature_cols:
        raise RuntimeError("No eligible features")
    print(f"Panel: {len(panel)} rows; {len(feature_cols)} eligible features; prediction rows: {len(pred_day)}", flush=True)

    names = pd.read_parquet(STORE / "universe" / "csi300" / "membership.parquet")
    name_by_symbol = dict(zip(names["symbol"].astype(str).str.zfill(6), names["name"]))
    result = {
        "as_of": AS_OF.isoformat(),
        "feature_set_id": feature_set_id,
        "feature_count": len(feature_cols),
        "universe_count": len(symbols),
        "prediction_day_rows": len(pred_day),
        "horizons": {},
    }

    for horizon in (3, 5):
        label_col = f"ret_fwd_{horizon}d"
        if label_col not in panel.columns:
            raise RuntimeError(f"Missing exact target {label_col}")
        train = panel.loc[panel["date"] < AS_OF].dropna(subset=feature_cols + [label_col])
        score_day = pred_day.dropna(subset=feature_cols).copy()
        if train.empty or score_day.empty:
            raise RuntimeError(f"Insufficient data for {label_col}")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        model.fit(train[feature_cols].values, train[label_col].values)
        score_day["model_score"] = model.predict(score_day[feature_cols].values)
        if not np.isfinite(score_day["model_score"]).all():
            raise RuntimeError(f"Nonfinite prediction for {label_col}")
        config = SelectionConfig(
            strategy=StrategyType.EQUAL_TOP_K,
            top_k=3,
            max_total=50,
            exposure_warning_threshold=0.30,
        )
        if "industry_code" not in score_day.columns:
            score_day["industry_code"] = "_UNKNOWN"
            score_day["industry_name"] = "未知"
        ranked = IndustryNeutralRanker(
            config,
            industry_col="industry_code",
            name_col="industry_name",
            score_col="model_score",
            symbol_col="symbol",
        ).run(score_day)
        selected = ranked.loc[ranked["selected"]].sort_values(
            ["model_score", "symbol"], ascending=[False, True]
        )
        rows = []
        for rank, (_, row) in enumerate(selected.iterrows(), 1):
            symbol = str(row["symbol"]).zfill(6)
            rows.append({
                "rank": rank,
                "symbol": symbol,
                "name": name_by_symbol.get(symbol, ""),
                "industry_code": str(row["industry_code"]),
                "industry_name": str(row["industry_name"]),
                "industry_rank": int(row["industry_rank"]),
                "model_score": float(row["model_score"]),
                "close": float(row["close"]),
                "exposure_flag": str(row["exposure_flag"]),
            })
        result["horizons"][str(horizon)] = {
            "label_col": label_col,
            "train_rows": len(train),
            "train_last_date": max(train["date"]).isoformat(),
            "train_first_date": min(train["date"]).isoformat(),
            "scored_count": len(score_day),
            "selected_count": len(rows),
            "selected": rows,
        }
        print(f"{label_col}: train={len(train)} through {max(train['date'])}, scored={len(score_day)}, selected={len(rows)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
