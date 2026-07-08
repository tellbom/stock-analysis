from __future__ import annotations

import pandas as pd

from quant_platform.selection.reco_cards import build_reco_cards


def _actionable_df():
    return pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "gate_tier": ["A_MAIN", "B_SHORT_BOOST", "A_MAIN"],
        "industry_rank": [1, 2, 1],
    })


def _drivers():
    return {
        "000001": [
            {"feature": "rsi_14", "shap": 0.05},
            {"feature": "pe_ttm", "shap": -0.03},
            {"feature": "mom_20d", "shap": 0.02},
        ],
        "000002": [
            {"feature": "flow_net_5d", "shap": 0.08},
            {"feature": "unknown_feature_x", "shap": -0.01},
        ],
        # 000003 intentionally missing -- no SHAP entry for it
    }


def _family_lookup():
    return {
        "rsi_14": "technical",
        "mom_20d": "technical",
        "pe_ttm": "valuation",
        "flow_net_5d": "flow",
    }


def test_every_symbol_gets_a_card():
    cards = build_reco_cards(_actionable_df(), _drivers(), _family_lookup())
    assert set(cards.keys()) == {"000001", "000002", "000003"}


def test_symbol_missing_from_drivers_gets_empty_driver_list():
    cards = build_reco_cards(_actionable_df(), _drivers(), _family_lookup())
    assert cards["000003"]["drivers"] == []
    assert cards["000003"]["family_totals"] == {}
    assert cards["000003"]["gate_tier"] == "A_MAIN"
    assert cards["000003"]["industry_rank"] == 1


def test_unknown_feature_tagged_other_family():
    cards = build_reco_cards(_actionable_df(), _drivers(), _family_lookup())
    driver_by_feature = {d["feature"]: d for d in cards["000002"]["drivers"]}
    assert driver_by_feature["unknown_feature_x"]["family"] == "other"
    assert driver_by_feature["flow_net_5d"]["family"] == "flow"


def test_family_totals_sum_to_driver_set():
    cards = build_reco_cards(_actionable_df(), _drivers(), _family_lookup())
    for symbol, card in cards.items():
        expected_total = sum(abs(d["shap"]) for d in card["drivers"])
        assert abs(sum(card["family_totals"].values()) - expected_total) < 1e-9


def test_top_n_limits_driver_count():
    cards = build_reco_cards(_actionable_df(), _drivers(), _family_lookup(), top_n=2)
    assert len(cards["000001"]["drivers"]) == 2
    # highest |shap| kept: rsi_14 (0.05), pe_ttm (-0.03) over mom_20d (0.02)
    kept_features = {d["feature"] for d in cards["000001"]["drivers"]}
    assert kept_features == {"rsi_14", "pe_ttm"}
