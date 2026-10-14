"""Offline inference replay for the corrected 2026-10-14 two-stock review.

Uses committed CSV inputs, Ridge parameters and LightGBM text models.
No network, parquet data lake, sklearn joblib loading, training or file writes.
Default: NumPy/Pandas/LightGBM. --ridge-only: NumPy/Pandas only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_platform.selection.config import weekly_selection_config
from quant_platform.selection.fusion_score import actionable_pool, fusion_score
from quant_platform.selection.gate_fusion import gate_first_fusion
from quant_platform.selection.ranker import IndustryNeutralRanker

FIXTURE = ROOT / "docs/reviews/2026-10-14-two-stock-review"
FROZEN = ROOT / "docs/reviews/2026-10-14-qfq-corrected"
TARGETS = ("688521", "301308")


def verify_manifest(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Fixture hash mismatch: {directory.name}/{name}")


def check_close(actual, expected, context: str) -> float:
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise RuntimeError(f"Invalid shape or nonfinite values: {context}")
    error = float(np.max(np.abs(a - e)))
    if error > 1e-12:
        raise RuntimeError(f"Replay mismatch: {context}, max absolute error={error}")
    return error


def select_symbols(frame: pd.DataFrame, attributes: pd.DataFrame, col: str) -> list[str]:
    frame = frame.merge(attributes, on="symbol", validate="one_to_one")
    frame = frame.sort_values([col, "symbol"], ascending=[False, True], kind="stable")
    ranker = IndustryNeutralRanker(
        weekly_selection_config(sector_cap_tech=10),
        industry_col="industry_code", name_col="industry_name", score_col=col,
    )
    ranked = ranker.run(frame)
    return ranked[ranked.selected].sort_values([col, "symbol"], ascending=[False, True]).symbol.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-only", action="store_true")
    args = parser.parse_args()
    verify_manifest(FIXTURE)
    verify_manifest(FROZEN)
    parameters = json.loads((FIXTURE / "model_parameters.json").read_text(encoding="utf-8"))
    # Exact float round trips preserve branches at split thresholds, including
    # paths whose leaves have equal scores but different SHAP allocations.
    features = pd.read_csv(FIXTURE / "prediction_features_299.csv", dtype={"symbol": str}, float_precision="round_trip")
    attributes = pd.read_csv(FROZEN / "prediction_attributes.csv", dtype={"symbol": str})
    picks = pd.read_csv(FROZEN / "all_picks.csv", dtype={"symbol": str})
    ridge_terms = pd.read_csv(FIXTURE / "ridge_contributions.csv", dtype={"symbol": str}, float_precision="round_trip")
    tree_terms = pd.read_csv(FIXTURE / "tree_contributions.csv", dtype={"symbol": str}, float_precision="round_trip")
    if len(features) != 299 or features.symbol.duplicated().any():
        raise RuntimeError("Invalid frozen scoring universe")
    reports, selections, target_rows = [], [], []
    for h in (3, 5):
        score_frames = {}
        stems = ("original_ridge",) if args.ridge_only else ("original_ridge", "recent_no_flow", "recent_flow")
        for stem in stems:
            par = parameters[f"D{h}_{stem}"]
            cols = par["features"]
            x = features[cols].to_numpy(dtype=float)
            x = np.where(np.isnan(x), np.asarray(par["imputer_statistics"]), x)
            z = (x - np.asarray(par["scaler_mean"])) / np.asarray(par["scaler_scale"])
            if stem == "original_ridge":
                values = z @ np.asarray(par["coefficients"]) + par["intercept"]
            else:
                import lightgbm as lgb
                booster = lgb.Booster(model_file=str(FIXTURE / par["booster_file"]))
                values = booster.predict(z)
                contributions = booster.predict(z, pred_contrib=True)
                check_close(contributions.sum(axis=1), values, f"D{h} {stem} SHAP sum")
            prefix = "base" if stem == "original_ridge" else "recent"
            frame = features[["symbol", "date"]].copy()
            frame["trade_date"] = frame.date
            frame[f"{prefix}_score"] = values
            frame[f"{prefix}_rank"] = frame[f"{prefix}_score"].rank(ascending=False, method="first").astype(int)
            frame[f"{prefix}_pct"] = frame[f"{prefix}_score"].rank(pct=True)
            name = "original" if stem == "original_ridge" else stem
            expected = pd.read_csv(FROZEN / f"D{h}_{name}_scores.csv", dtype={"symbol": str}).set_index("symbol").loc[features.symbol]
            error = check_close(values, expected[f"{prefix}_score"], f"D{h} {stem} scores")
            if frame[f"{prefix}_rank"].tolist() != expected[f"{prefix}_rank"].tolist():
                raise RuntimeError(f"Rank mismatch: D{h} {stem}")
            for sym in TARGETS:
                i = int(np.flatnonzero(features.symbol.to_numpy() == sym)[0])
                if stem == "original_ridge":
                    terms = z[i] * np.asarray(par["coefficients"])
                    saved = ridge_terms[(ridge_terms.horizon == h) & (ridge_terms.symbol == sym)].set_index("feature").loc[cols]
                    check_close(terms * 100, saved.contribution_pp, f"D{h} {sym} Ridge contributions")
                else:
                    saved = tree_terms[(tree_terms.horizon == h) & (tree_terms.model == stem) & (tree_terms.symbol == sym)].set_index("feature").loc[cols + ["__expected_value__"]]
                    check_close(contributions[i] * 100, saved.contribution_pp, f"D{h} {sym} tree contributions")
                target_rows.append({"horizon": h, "model": stem, "symbol": sym, "score_pct": float(values[i] * 100), "global_rank": int(frame.iloc[i][f"{prefix}_rank"])})
            score_frames[stem] = frame
            reports.append({"horizon": h, "model": stem, "rows": len(frame), "scores_and_ranks_match": True, "max_abs_error": error})
        arms = ["ordinary"] if args.ridge_only else ["ordinary", "hybrid_no_flow", "hybrid_flow", "hybrid_flow_score_gate_diagnostic"]
        for arm in arms:
            if arm == "ordinary":
                frame, col = score_frames["original_ridge"], "base_score"
            else:
                recent_stem = "recent_no_flow" if arm == "hybrid_no_flow" else "recent_flow"
                combo = score_frames["original_ridge"].merge(score_frames[recent_stem].drop(columns="date"), on=["symbol", "trade_date"], validate="one_to_one")
                combo["risk_flags"], combo["event_flags"] = "", ""
                frame = actionable_pool(gate_first_fusion(combo)) if arm.endswith("diagnostic") else fusion_score(combo, k=60)
                col = "fusion_score_col"
            selected = select_symbols(frame, attributes, col)
            expected = picks[(picks.horizon == h) & (picks.strategy == arm)].sort_values("recommendation_rank").symbol.tolist()
            if selected != expected:
                raise RuntimeError(f"Ordered selection mismatch: D{h} {arm}")
            selections.append({"horizon": h, "arm": arm, "count": len(selected), "ordered_picks_match": True,
                               "target_ranks": {s: selected.index(s) + 1 if s in selected else None for s in TARGETS}})
    print(json.dumps({"mode": "Ridge only" if args.ridge_only else "all six models", "score_checks": reports,
                      "selection_checks": selections, "target_scores": target_rows,
                      "scope": "frozen inference and attribution; no training or realized-return validation"}, indent=2))


if __name__ == "__main__":
    main()
