"""Replay October 14 Ridge input sensitivity using committed offline fixtures.

This is a frozen-model audit, not retraining or a new recommendation.
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
from quant_platform.features.cross_sectional import build_cross_sectional_features

DEFAULT_EVIDENCE = ROOT / "docs/reviews/2026-10-14-weekly/qfq_audit"


def predict_snapshot(frame: pd.DataFrame, features: list[str], model: dict) -> pd.Series:
    matrix = frame[features].to_numpy(dtype=float)
    parameters = [np.asarray(model[k], dtype=float) for k in
                  ("imputer_statistics", "scaler_mean", "scaler_scale", "ridge_coef")]
    statistics, means, scales, coefs = parameters
    if any(len(v) != len(features) for v in parameters):
        raise ValueError("Snapshot feature/parameter lengths differ")
    if not all(np.isfinite(v).all() for v in parameters) or (scales <= 0).any():
        raise ValueError("Invalid frozen preprocessing parameters")
    matrix = np.where(np.isnan(matrix), statistics, matrix)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite fixture inputs")
    values = ((matrix - means) / scales) @ coefs + float(model["ridge_intercept"])
    return pd.Series(values, index=frame.symbol)


def replay(evidence: Path, rebuild_technical: bool = False) -> dict:
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected_hash in manifest["files"].items():
        path = (evidence / relative).resolve()
        if not path.is_relative_to(evidence.resolve()):
            raise ValueError("Fixture path escapes evidence directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Evidence hash mismatch: {relative}")
    fixture = evidence / "fixtures"
    expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    snapshot = json.loads((fixture / "ridge_snapshot.json").read_text(encoding="utf-8"))
    features = snapshot["feature_order"]
    csv_options = {"dtype": {"symbol": str}, "float_precision": "round_trip"}
    pred = pd.read_csv(fixture / "prediction_features_299.csv", **csv_options)
    corrected = pd.read_csv(fixture / "300803_corrected_technical_row.csv", **csv_options)
    fresh = pd.read_csv(fixture / "300803_fresh_sina_qfq_full.csv", **csv_options)
    old = pd.read_csv(fixture / "300803_old_ohlcv.csv", **csv_options)
    for frame in (pred, corrected, fresh, old):
        frame["date"] = pd.to_datetime(frame.date).dt.date
    as_of = pd.Timestamp(expected["signal_date"]).date()
    if len(pred) != expected["prediction_rows"] or pred.symbol.duplicated().any() or not pred.date.eq(as_of).all():
        raise ValueError("Wrong prediction date or universe in fixture")
    if len(corrected) != 1 or corrected.symbol.iloc[0] != "300803" or corrected.date.iloc[0] != as_of:
        raise ValueError("Wrong corrected stock/date in fixture")
    if len(fresh) != expected["full_fresh_rows"] or fresh.date.max() != as_of or old.date.max() != as_of:
        raise ValueError("Wrong full price history in fixture")
    if rebuild_technical:
        from quant_platform.features.technical import build_technical_features
        rebuilt = build_technical_features(fresh.sort_values("date"), project_root=ROOT)
        technical_cols = [c for c in features if not c.startswith("cs_")]
        actual = rebuilt.loc[rebuilt.date.eq(as_of), technical_cols].to_numpy(dtype=float)
        reference = corrected[technical_cols].to_numpy(dtype=float)
        if not np.isfinite(actual).all() or not np.allclose(actual, reference, atol=1e-10, rtol=1e-10):
            raise ValueError("Regenerated technical inputs differ from the captured fixture")
    corrected_panel = pred.copy()
    for col in corrected.columns:
        if col in corrected_panel.columns:
            corrected_panel.loc[corrected_panel.symbol.eq("300803"), col] = corrected.iloc[0][col]
    corrected_panel = build_cross_sectional_features(corrected_panel)
    results = []
    for target in expected["sensitivity"]:
        h = str(target["horizon"])
        frozen_scores = predict_snapshot(pred, features, snapshot["models"][h])
        corrected_scores = predict_snapshot(corrected_panel, features, snapshot["models"][h])
        values = {
            "horizon": int(h),
            "original_score_pct": float(frozen_scores["300803"] * 100),
            "recomputed_input_score_pct": float(corrected_scores["300803"] * 100),
            "original_rank": int(frozen_scores.rank(ascending=False, method="first")["300803"]),
            "recomputed_input_rank": int(corrected_scores.rank(ascending=False, method="first")["300803"]),
        }
        for col in ("original_score_pct", "recomputed_input_score_pct"):
            if abs(values[col] - target[col]) > 1e-10:
                raise ValueError(f"D{h} {col}: replay differs from expected evidence")
        for col in ("original_rank", "recomputed_input_rank"):
            if values[col] != target[col]:
                raise ValueError(f"D{h} {col}: replay ranking differs")
        results.append(values)
    merged = old[["date", "close"]].merge(fresh[["date", "close"]], on="date", suffixes=("_old", "_fresh"))
    changed = int(((merged.close_old - merged.close_fresh).abs() > 0.020001).sum())
    if changed != 896:
        raise ValueError("Captured guide history mismatch count differs")
    history = expected["history_comparison"]
    missing_dates = sorted(str(d) for d in set(fresh.date) - set(old.date))
    if (len(old) != history["old_rows"] or len(merged) != history["common_rows"]
            or missing_dates != history["missing_old_dates"]):
        raise ValueError("Captured historical date coverage differs")
    return {"status": "passed", "signal_date": str(as_of), "prediction_rows": len(pred),
            "guide_price_mismatches": changed, "technical_rebuilt": rebuild_technical,
            "old_missing_dates": missing_dates,
            "sensitivity": results, "scope": "fixed original models; one stock corrected inputs; not retrained recommendations"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--rebuild-technical", action="store_true", help="Regenerate guide indicators from the full fresh OHLCV fixture")
    args = parser.parse_args()
    print(json.dumps(replay(args.evidence_dir, args.rebuild_technical), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
