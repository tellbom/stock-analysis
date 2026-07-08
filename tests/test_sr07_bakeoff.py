from __future__ import annotations

import pandas as pd
import pytest

from scripts.sr07_bakeoff import (
    TARGET_SIZE,
    apply_sr_arms,
    build_hybrid_pool,
    decide_sr_promotion,
    _rank_with_selection_priority,
)


def _scored_frame(n: int) -> pd.DataFrame:
    """
    n symbols spanning gate tiers: half strong A_MAIN, a few B_SHORT_BOOST,
    the rest weak (E_REJECT) so build_hybrid_pool has something to filter out.
    """
    rows = []
    for i in range(n):
        symbol = f"{i:06d}"
        if i % 5 == 0:
            base_pct, recent_pct = 0.55, 0.90  # B_SHORT_BOOST
        elif i % 4 == 0:
            base_pct, recent_pct = 0.10, 0.10  # E_REJECT
        else:
            base_pct, recent_pct = 0.85, 0.70  # A_MAIN
        rows.append(
            {
                "symbol": symbol,
                "trade_date": "2026-06-01",
                "base_score": n - i,
                "base_pct": base_pct,
                "recent_score": n - i,
                "recent_pct": recent_pct,
            }
        )
    return pd.DataFrame(rows)


def test_build_hybrid_pool_filters_to_actionable_tiers_only():
    frame = _scored_frame(20)
    pool = build_hybrid_pool(frame)

    assert set(pool["gate_tier"].unique()) <= {"A_MAIN", "B_SHORT_BOOST"}
    assert "fusion_score_col" in pool.columns
    assert "fusion_rank" in pool.columns
    assert len(pool) < len(frame)


def test_rank_with_selection_priority_puts_selected_first_in_fusion_order():
    pool = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "fusion_rank": [1, 2, 3, 4],
            "fusion_score_col": [0.9, 0.8, 0.7, 0.6],
        }
    )
    out = _rank_with_selection_priority(pool, selected={"C", "D"})

    assert out.sort_values("arm_rank")["symbol"].tolist() == ["C", "D", "A", "B"]
    assert out["arm_rank"].tolist() == [1, 2, 3, 4]
    assert (out["arm_score"] == out["fusion_score_col"]).all()


def test_apply_sr_arms_cold_start_matches_baseline_top_by_target_size():
    frame = _scored_frame(200)
    pool = build_hybrid_pool(frame)

    arms, selected = apply_sr_arms(pool, prior_selected=set())

    baseline_top = set(arms["baseline"].sort_values("arm_rank")["symbol"].head(TARGET_SIZE))
    turnover_top = set(arms["turnover_aware"].sort_values("arm_rank")["symbol"].head(TARGET_SIZE))
    assert turnover_top == baseline_top
    assert selected == baseline_top


def test_apply_sr_arms_threads_prior_selected_and_bounds_turnover():
    frame = _scored_frame(200)
    pool = build_hybrid_pool(frame)

    _first_arms, first_selected = apply_sr_arms(pool, prior_selected=set())
    _second_arms, second_selected = apply_sr_arms(pool, prior_selected=first_selected)

    # Same pool, same prior -> nothing should want to change (zero turnover).
    assert second_selected == first_selected


def _agg(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_decide_sr_promotion_promotes_when_all_three_conditions_pass():
    aggregate = _agg(
        [
            {"arm": "baseline", "metric": "top20_ret_fwd_3d_net", "mean": 0.010},
            {"arm": "turnover_aware", "metric": "top20_ret_fwd_3d_net", "mean": 0.011},
            {"arm": "turnover_aware", "metric": "turnover_top20", "mean": 0.30},
        ]
    )
    weak = _agg(
        [
            {"arm": "baseline", "metric": "top20_ret_fwd_3d_net", "mean": 0.005},
            {"arm": "turnover_aware", "metric": "top20_ret_fwd_3d_net", "mean": 0.006},
        ]
    )

    decision = decide_sr_promotion(aggregate, weak)

    assert decision["recommendation"] == "promote"
    assert decision["turnover_pass"] is True
    assert decision["net_pass"] is True
    assert decision["weak_regime_pass"] is True


def test_decide_sr_promotion_defers_when_turnover_ok_but_net_fails():
    aggregate = _agg(
        [
            {"arm": "baseline", "metric": "top20_ret_fwd_3d_net", "mean": 0.020},
            {"arm": "turnover_aware", "metric": "top20_ret_fwd_3d_net", "mean": 0.010},
            {"arm": "turnover_aware", "metric": "turnover_top20", "mean": 0.30},
        ]
    )
    weak = _agg(
        [
            {"arm": "baseline", "metric": "top20_ret_fwd_3d_net", "mean": 0.005},
            {"arm": "turnover_aware", "metric": "top20_ret_fwd_3d_net", "mean": 0.006},
        ]
    )

    decision = decide_sr_promotion(aggregate, weak)

    assert decision["recommendation"] == "defer_to_reviewers"
    assert decision["turnover_pass"] is True
    assert decision["net_pass"] is False


def test_decide_sr_promotion_rejects_when_turnover_too_high():
    aggregate = _agg(
        [
            {"arm": "baseline", "metric": "top20_ret_fwd_3d_net", "mean": 0.010},
            {"arm": "turnover_aware", "metric": "top20_ret_fwd_3d_net", "mean": 0.011},
            {"arm": "turnover_aware", "metric": "turnover_top20", "mean": 0.75},
        ]
    )
    weak = _agg(
        [
            {"arm": "baseline", "metric": "top20_ret_fwd_3d_net", "mean": 0.005},
            {"arm": "turnover_aware", "metric": "top20_ret_fwd_3d_net", "mean": 0.006},
        ]
    )

    decision = decide_sr_promotion(aggregate, weak)

    assert decision["recommendation"] == "reject"
    assert decision["turnover_pass"] is False


def test_decide_sr_promotion_rejects_when_weak_regime_worse():
    aggregate = _agg(
        [
            {"arm": "baseline", "metric": "top20_ret_fwd_3d_net", "mean": 0.010},
            {"arm": "turnover_aware", "metric": "top20_ret_fwd_3d_net", "mean": 0.011},
            {"arm": "turnover_aware", "metric": "turnover_top20", "mean": 0.30},
        ]
    )
    weak = _agg(
        [
            {"arm": "baseline", "metric": "top20_ret_fwd_3d_net", "mean": 0.010},
            {"arm": "turnover_aware", "metric": "top20_ret_fwd_3d_net", "mean": 0.002},
        ]
    )

    decision = decide_sr_promotion(aggregate, weak)

    assert decision["recommendation"] == "reject"
    assert decision["turnover_pass"] is True
    assert decision["net_pass"] is True
    assert decision["weak_regime_pass"] is False
