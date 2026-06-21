"""
evaluation.explainability
=========================
SHAP explainability suite (T3.6).

Produces:
  1. Global feature importance — mean |SHAP| across the panel.
  2. Per-symbol top drivers — the top-k SHAP features for each stock's
     most recent prediction.
  3. Fold stability — how consistent are the top-5 features across CV folds?
     A model whose important features change radically fold-to-fold is
     suspicious.
  4. Economic-sensibility review template — a structured dict that a human
     reviewer can fill in to gate promotion (T3.4 gating).

All methods require a fitted native model (e.g. LightGBM booster) and the
raw feature DataFrame.  The Pipeline scaler is bypassed for SHAP because
TreeExplainer works on raw tree leaf values, not the scaled input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExplainabilityReport:
    """Results of the SHAP explainability suite."""
    global_importance:   pd.Series       = field(default_factory=pd.Series)
    top_features:        list[str]        = field(default_factory=list)
    fold_stability_score: float           = float("nan")  # Jaccard similarity of top-5 across folds
    per_symbol_drivers:  dict[str, list]  = field(default_factory=dict)  # symbol → [(feature, shap_val)]
    review_template:     dict             = field(default_factory=dict)

    def print_summary(self, n: int = 10) -> None:
        print("\n" + "=" * 55)
        print("EXPLAINABILITY REPORT")
        print("=" * 55)
        print(f"  Top {n} features by mean |SHAP|:")
        for i, (feat, val) in enumerate(
            self.global_importance.nlargest(n).items(), 1
        ):
            print(f"    {i:2d}. {feat:30s}  {val:.6f}")
        print(f"  Fold stability (Jaccard top-5): {self.fold_stability_score:.3f}")
        print("=" * 55 + "\n")


def compute_shap_importance(
    model,                 # native model (unwrapped from Pipeline)
    X: pd.DataFrame,
    max_rows: int = 2000,
) -> tuple[np.ndarray, pd.Index]:
    """
    Compute SHAP values for X using TreeExplainer.

    Returns (shap_values_array, feature_names).
    Subsamples to max_rows for speed.
    """
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)

    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]   # classification: use first class
        return shap_vals, X.columns
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        return np.zeros((len(X), X.shape[1])), X.columns


def build_explainability_report(
    model,                          # native model (e.g. lgbm booster)
    X_panel: pd.DataFrame,          # full feature panel (no label)
    feature_cols: list[str],
    panel_meta: pd.DataFrame | None = None,   # optional: contains 'symbol', 'date'
    fold_models: list | None = None,          # list of per-fold native models for stability
    top_k: int = 5,
) -> ExplainabilityReport:
    """
    Run the full SHAP explainability suite.

    Parameters
    ----------
    model : native model (unwrapped from Pipeline via get_native_model())
    X_panel : pd.DataFrame with feature_cols columns
    feature_cols : list of feature column names
    panel_meta : optional DataFrame with 'symbol' column for per-symbol breakdown
    fold_models : list of per-fold native models for stability scoring
    top_k : number of features for stability comparison
    """
    report = ExplainabilityReport()
    X = X_panel[feature_cols].fillna(0)

    # 1. Global importance
    shap_vals, feat_names = compute_shap_importance(model, X)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
    report.global_importance = importance
    report.top_features = importance.head(top_k).index.tolist()
    logger.info("SHAP global top-%d: %s", top_k, report.top_features)

    # 2. Per-symbol top drivers (last row per symbol)
    if panel_meta is not None and "symbol" in panel_meta.columns:
        X_with_meta = X.copy()
        X_with_meta["symbol"] = panel_meta["symbol"].values
        for symbol, grp in X_with_meta.groupby("symbol"):
            last_row = grp.drop(columns=["symbol"]).iloc[[-1]]
            sv, fn   = compute_shap_importance(model, last_row, max_rows=1)
            if sv.size > 0:
                drivers = sorted(
                    zip(fn, sv[0]), key=lambda x: abs(x[1]), reverse=True
                )[:top_k]
                report.per_symbol_drivers[str(symbol)] = [
                    {"feature": f, "shap": round(float(v), 6)} for f, v in drivers
                ]

    # 3. Fold stability (Jaccard similarity of top-k across fold models)
    if fold_models and len(fold_models) >= 2:
        top_sets = []
        for fm in fold_models:
            sv_f, fn_f = compute_shap_importance(fm, X)
            imp_f = pd.Series(np.abs(sv_f).mean(axis=0), index=fn_f)
            top_sets.append(set(imp_f.nlargest(top_k).index))

        # Pairwise Jaccard, then average
        n = len(top_sets)
        jaccard_sum = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                inter = len(top_sets[i] & top_sets[j])
                union = len(top_sets[i] | top_sets[j])
                jaccard_sum += inter / union if union > 0 else 1.0
                pairs += 1
        report.fold_stability_score = jaccard_sum / pairs if pairs > 0 else float("nan")
        logger.info("SHAP fold stability (Jaccard top-%d): %.3f", top_k,
                    report.fold_stability_score)

    # 4. Economic-sensibility review template
    report.review_template = {
        "top_features":       report.top_features,
        "fold_stability":     round(report.fold_stability_score, 3),
        "reviewer_sign_off":  None,    # human fills this in
        "plausibility_notes": "",      # human fills this in
        "suspicious_features": [],     # human fills this in
        "promotion_recommended": None, # True / False / None (pending)
    }

    return report


def save_explainability_report(
    report:     ExplainabilityReport,
    store_root: Path | str,
    model_name: str,
    run_id:     str,
) -> None:
    """Save the explainability report to <store_root>/explainability/."""
    import json
    out_dir = Path(store_root) / "explainability" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global importance CSV
    report.global_importance.to_csv(out_dir / "global_importance.csv")

    # Per-symbol drivers JSON
    (out_dir / "per_symbol_drivers.json").write_text(
        json.dumps(report.per_symbol_drivers, indent=2)
    )

    # Review template JSON
    (out_dir / "review_template.json").write_text(
        json.dumps(report.review_template, indent=2)
    )

    logger.info(
        "Explainability report saved → %s (%d symbols, stability=%.3f)",
        out_dir, len(report.per_symbol_drivers), report.fold_stability_score,
    )
