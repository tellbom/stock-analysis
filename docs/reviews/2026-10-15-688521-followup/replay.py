"""Read-only replay of the four experimental models and follow-up evidence.

Run from any directory. Requires NumPy, Pandas and LightGBM; no local data lake,
network, joblib loading, retraining or output-file writes are used.
"""
from pathlib import Path
import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EVIDENCE = json.loads((HERE / "evidence.json").read_text(encoding="utf-8"))
ACCEPTANCE = ROOT / "docs/reviews/2026-10-15-ffae653-acceptance"
PREVIOUS = ROOT / "docs/reviews/2026-10-14-two-stock-review"


def close(actual, expected, context):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise RuntimeError(f"Invalid values or shape: {context}")
    error = float(np.max(np.abs(a - e)))
    if error > 1e-12:
        raise RuntimeError(f"Replay mismatch: {context}; max absolute error {error}")
    return error


def main():
    for relative, expected in EVIDENCE["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")

    parameters = json.loads(
        (ACCEPTANCE / "experimental_model_parameters.json").read_text(encoding="utf-8")
    )
    features = pd.read_csv(
        PREVIOUS / "prediction_features_299.csv",
        dtype={"symbol": str}, float_precision="round_trip",
    )
    if len(features) != 299 or features.symbol.duplicated().any():
        raise RuntimeError("Frozen scoring universe must contain 299 unique symbols")
    for record in EVIDENCE["experimental_replay"]:
        par = parameters[record["model"]]
        columns = par["features"]
        x = features[columns].to_numpy(dtype=float)
        x = np.where(np.isnan(x), np.asarray(par["imputer_statistics"]), x)
        z = (x - np.asarray(par["scaler_mean"])) / np.asarray(par["scaler_scale"])
        is_base = "coefficients" in par
        if is_base:
            values = z @ np.asarray(par["coefficients"]) + par["intercept"]
            terms = np.column_stack((
                z * np.asarray(par["coefficients"]),
                np.full(len(z), par["intercept"]),
            ))
            names = columns + ["__intercept__"]
        else:
            booster = lgb.Booster(model_file=str(ACCEPTANCE / par["booster_file"]))
            values = booster.predict(z)
            terms = booster.predict(z, pred_contrib=True)
            names = columns + ["__expected_value__"]
        kind = "base" if is_base else "recent"
        scores = pd.read_csv(
            ACCEPTANCE / f"D{record['horizon']}_experiment_{kind}_scores.csv",
            dtype={"symbol": str}, float_precision="round_trip",
        ).set_index("symbol").loc[features.symbol]
        error = close(values, scores[f"{kind}_score"], record["model"])
        close(terms.sum(axis=1), values, f"{record['model']} contribution sum")
        for target in record["terms"]:
            i = features.index[features.symbol == target["symbol"]][0]
            close(values[i], target["prediction"], "target prediction")
            close(
                terms[i] * 100,
                [target["contribution_pp"][name] for name in names],
                f"{record['model']} {target['symbol']} full contributions in pp",
            )
            if not is_base:
                flow_total = sum(
                    terms[i, j] * 100 for j, name in enumerate(names)
                    if name.startswith("cs_") and "flow" in name
                )
                close(flow_total, target["flow_shap_sum_pp"], "flow SHAP sum in pp")
        print(f"{record['model']}: 299 scores and full target contributions OK; error={error}")

    for record in EVIDENCE["selection_records"]:
        path = ACCEPTANCE / f"D{record['horizon']}_{record['arm']}_picks.csv"
        picks = pd.read_csv(path, dtype={"symbol": str}, float_precision="round_trip").fillna("")
        row = picks[picks.symbol == record["symbol"]].iloc[0]
        for name, expected in record.items():
            if name in ("horizon", "arm"):
                continue
            if isinstance(expected, (int, float)):
                close(row[name], expected, f"{path.name} {record['symbol']} {name}")
            elif row[name] != expected:
                raise RuntimeError(f"Selection field mismatch: {path.name} {name}")
        if "fusion_score_col" in record:
            rrf = 1 / (60 + record["base_rank"]) + 1 / (60 + record["recent_rank"])
            close(rrf, record["fusion_score_col"], "RRF formula")
    print("16 selection records, RRF calculations and frozen source hashes OK")
    print("Scope: inference and explanation only; no retraining or realized-return validation")


if __name__ == "__main__":
    main()
