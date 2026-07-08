"""
quant_platform.selection.reco_cards
====================================
SR-04: per-symbol recommendation cards.

Joins the actionable pool (post SR-01 hybrid filter/rank) with an
already-computed SHAP explainability report's ``per_symbol_drivers`` (see
``quant_platform.evaluation.explainability.build_explainability_report``)
and a feature -> family lookup (see ``quant_platform.cli._feature_family_lookup``).

This module never fits a model or computes SHAP values itself -- it is a
pure join/formatting layer over results the caller already has in scope
(``run_d3_gate_once.py`` already fits base/recent LightGBM models and can
build one explainability report to source cards from, rather than paying
for a second explainer pass).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_TOP_N = 5
UNKNOWN_FAMILY = "other"


def build_reco_cards(
    actionable_df: pd.DataFrame,
    per_symbol_drivers: dict[str, list],
    family_by_col: dict[str, str],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, dict]:
    """
    Build a recommendation card per symbol in ``actionable_df``.

    Parameters
    ----------
    actionable_df : must contain a ``symbol`` column (typically the SR-01
        ``actionable_pool()`` output). ``gate_tier`` / ``industry_rank``
        are included in the card when present.
    per_symbol_drivers : symbol -> [{"feature": str, "shap": float}, ...],
        as produced by ``build_explainability_report`` (already sorted by
        |shap| descending).
    family_by_col : feature name -> family label. Features absent from the
        lookup are tagged ``"other"``.
    top_n : number of top drivers to keep per card.

    Returns
    -------
    dict[symbol, {"drivers": [...], "family_totals": {...},
                  "gate_tier": ..., "industry_rank": ...}]

    Symbols with no entry in ``per_symbol_drivers`` still get a card with
    an empty ``drivers`` list (e.g. SHAP wasn't computed for that symbol).
    """
    if "symbol" not in actionable_df.columns:
        raise ValueError("build_reco_cards requires a 'symbol' column in actionable_df")

    has_gate_tier = "gate_tier" in actionable_df.columns
    has_industry_rank = "industry_rank" in actionable_df.columns

    cards: dict[str, dict] = {}
    for _, row in actionable_df.iterrows():
        symbol = str(row["symbol"])
        raw_drivers = per_symbol_drivers.get(symbol, [])
        top_drivers = sorted(raw_drivers, key=lambda d: abs(d["shap"]), reverse=True)[:top_n]

        drivers = [
            {
                "feature": d["feature"],
                "shap": d["shap"],
                "family": family_by_col.get(d["feature"], UNKNOWN_FAMILY),
            }
            for d in top_drivers
        ]

        family_totals: dict[str, float] = {}
        for d in drivers:
            family_totals[d["family"]] = family_totals.get(d["family"], 0.0) + abs(d["shap"])

        cards[symbol] = {
            "drivers": drivers,
            "family_totals": family_totals,
            "gate_tier": row["gate_tier"] if has_gate_tier else None,
            "industry_rank": row["industry_rank"] if has_industry_rank else None,
        }

    return cards
