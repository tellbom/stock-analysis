"""Reconstruct the coverage-gated factor lists for two recommendation dates."""

import datetime as dt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_platform.cli import _coverage_gate_config_for_universe, _feature_family_lookup
from quant_platform.evaluation.coverage_gate import (
    compute_feature_coverage_report,
    select_features_by_gate,
)
from quant_platform.features import industry as industry_module
from quant_platform.features.registry import DEFAULT_SPECS, compute_feature_set_id, feature_metadata_lookup
from scripts.next_week_d3_prediction import build_panel, load_symbols
from temp.latest_dual_horizon import _fast_join_industry

industry_module._join_industry = _fast_join_industry
symbols = load_symbols()
panel = build_panel(symbols, compute_feature_set_id(DEFAULT_SPECS), max_date=dt.date(2026, 10, 9))
meta_cols = {
    "symbol", "date", "close", "volume", "open", "high", "low", "amount",
    "outstanding_share", "turnover", "industry_code", "industry_name", "concept_tags",
}
label_prefixes = ("ret_fwd_", "vol_fwd_", "mdd_fwd_", "excess_vs_")


def selected(as_of: dt.date) -> list[str]:
    subset = panel.loc[panel["date"] <= as_of]
    pred_day = subset.loc[subset["date"] == as_of]
    candidates = [
        c for c in subset.columns
        if c not in meta_cols and not c.startswith(label_prefixes)
    ]
    gate = compute_feature_coverage_report(
        subset,
        candidates,
        family_by_col=_feature_family_lookup(),
        feature_metadata=feature_metadata_lookup(),
        config=_coverage_gate_config_for_universe(subset["symbol"].nunique()),
    )
    return [
        c for c in select_features_by_gate(gate, model_path="base")
        if c in pred_day.columns
        and pred_day[c].isna().mean() < 0.50
        and subset[c].isna().mean() < 0.50
    ]


old = selected(dt.date(2026, 9, 30))
new = selected(dt.date(2026, 10, 9))
print("2026-09-30", len(old), old)
print("2026-10-09", len(new), new)
print("same_order", old == new)
print("removed", sorted(set(old) - set(new)))
print("added", sorted(set(new) - set(old)))
