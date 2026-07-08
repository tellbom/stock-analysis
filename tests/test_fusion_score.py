from __future__ import annotations

import pandas as pd

from quant_platform.selection.fusion_score import actionable_pool, fusion_score
from quant_platform.selection.gate_fusion import ACTIONABLE_TIERS, gate_first_fusion


def _synthetic_scored_frame() -> pd.DataFrame:
    # 8 symbols spanning the gate tiers: A_MAIN, B_SHORT_BOOST, C_DOWNGRADE,
    # D_OBSERVE, E_REJECT, and one RISK_VETO -- shaped like the frame
    # gate_first_fusion expects (base_pct/recent_pct + base_rank/recent_rank).
    rows = [
        # symbol, base_score, base_pct, recent_score, recent_pct, risk_flags
        ("000001", 0.90, 0.90, 0.70, 0.70, ""),                              # A_MAIN
        ("000002", 0.85, 0.85, 0.65, 0.65, ""),                              # A_MAIN
        ("000003", 0.65, 0.65, 0.90, 0.90, ""),                              # B_SHORT_BOOST
        ("000004", 0.82, 0.82, 0.20, 0.20, ""),                              # C_DOWNGRADE_OBSERVE
        ("000005", 0.30, 0.30, 0.92, 0.92, ""),                              # D_OBSERVE
        ("000006", 0.10, 0.10, 0.10, 0.10, ""),                              # E_REJECT
        ("000007", 0.95, 0.95, 0.95, 0.95, f"high_unlock:3d_ratio_5%@2026-07-08"),  # RISK_VETO
        ("000008", 0.88, 0.88, 0.62, 0.62, ""),                              # A_MAIN
    ]
    df = pd.DataFrame(
        rows,
        columns=["symbol", "base_score", "base_pct", "recent_score", "recent_pct", "risk_flags"],
    )
    df["trade_date"] = "2026-07-08"
    return df


def test_actionable_pool_excludes_non_actionable_tiers():
    scored = _synthetic_scored_frame()
    gated = gate_first_fusion(scored)
    pool = actionable_pool(gated)

    assert set(pool["gate_tier"].unique()) <= set(ACTIONABLE_TIERS)
    excluded_tiers = {"C_DOWNGRADE_OBSERVE", "D_OBSERVE", "E_REJECT", "RISK_VETO"}
    assert not set(pool["symbol"]) & set(
        gated[gated["gate_tier"].isin(excluded_tiers)]["symbol"]
    )
    # the vetoed/downgraded/rejected/observed symbols by construction
    assert "000007" not in set(pool["symbol"])  # RISK_VETO
    assert "000006" not in set(pool["symbol"])  # E_REJECT
    assert "000004" not in set(pool["symbol"])  # C_DOWNGRADE_OBSERVE
    assert "000005" not in set(pool["symbol"])  # D_OBSERVE


def test_actionable_pool_order_matches_smooth_score_order():
    scored = _synthetic_scored_frame()
    gated = gate_first_fusion(scored)
    pool = actionable_pool(gated, method="rrf")

    expected_order = pool.sort_values("fusion_score_col", ascending=False)["symbol"].tolist()
    assert pool["symbol"].tolist() == expected_order
    assert pool["fusion_rank"].tolist() == list(range(1, len(pool) + 1))


def test_actionable_pool_real_date_parity_subset_of_gate_ranked():
    scored = _synthetic_scored_frame()
    gated = gate_first_fusion(scored)
    pool = actionable_pool(gated)

    gate_actionable_set = set(gated[gated["gate_tier"].isin(ACTIONABLE_TIERS)]["symbol"])
    assert set(pool["symbol"]) == gate_actionable_set


def test_fusion_score_rrf_requires_ranks():
    df = pd.DataFrame({"symbol": ["a", "b"], "base_pct": [0.5, 0.6], "recent_pct": [0.4, 0.3]})
    try:
        fusion_score(df, method="rrf")
        assert False, "expected ValueError for missing rank columns"
    except ValueError as exc:
        assert "base_rank" in str(exc) or "recent_rank" in str(exc)


def test_fusion_score_recent_only():
    df = pd.DataFrame({"symbol": ["a", "b", "c"], "recent_pct": [0.2, 0.9, 0.5]})
    out = fusion_score(df, method="recent_only")
    assert out["symbol"].tolist() == ["b", "c", "a"]


def test_fusion_score_fixed_weight_bounds():
    df = pd.DataFrame({"symbol": ["a"], "base_pct": [0.5], "recent_pct": [0.5]})
    try:
        fusion_score(df, method="fixed_weight", base_weight=1.5)
        assert False, "expected ValueError for out-of-range base_weight"
    except ValueError:
        pass


def test_fusion_score_unknown_method_rejected():
    df = pd.DataFrame({"symbol": ["a"], "base_pct": [0.5], "recent_pct": [0.5]})
    try:
        fusion_score(df, method="bogus")
        assert False, "expected ValueError for unknown method"
    except ValueError:
        pass
