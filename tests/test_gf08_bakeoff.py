import pandas as pd
import pytest

from scripts.gf08_bakeoff import (
    add_fixed_weight_rank,
    add_rrf_rank,
    compute_arm_date_metrics,
    compute_turnover,
)


def test_rrf_and_fixed_weight_rank_highest_score_first():
    frame = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003"],
            "base_rank": [1, 3, 2],
            "recent_rank": [3, 1, 2],
            "base_pct": [0.9, 0.2, 0.5],
            "recent_pct": [0.1, 0.95, 0.5],
        }
    )

    rrf = add_rrf_rank(frame, k=60)
    fixed = add_fixed_weight_rank(frame, base_weight=0.3)

    assert set(rrf.columns) >= {"arm_score", "arm_rank"}
    assert rrf.sort_values("arm_rank")["symbol"].tolist() == ["000001", "000002", "000003"]
    assert fixed.sort_values("arm_rank")["symbol"].tolist() == ["000002", "000003", "000001"]


def test_compute_arm_date_metrics_uses_positive_excess_and_cost():
    frame = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003", "000004"],
            "trade_date": ["2026-06-01"] * 4,
            "arm_score": [4.0, 3.0, 2.0, 1.0],
            "ret_fwd_3d": [0.04, -0.01, 0.02, 0.00],
            "ret_fwd_3d_cs": [0.0275, -0.0225, 0.0075, -0.0125],
            "ret_fwd_5d": [0.05, -0.02, 0.01, 0.0],
        }
    )

    metrics = compute_arm_date_metrics(frame, "demo", ks=(2,), cost_bps=10)
    row = metrics.set_index("metric").to_dict()["value"]

    assert row["top2_ret_fwd_3d_gross"] == pytest.approx(0.015)
    assert row["top2_ret_fwd_3d_net"] == pytest.approx(0.014)
    assert row["precision_at_2_excess"] == pytest.approx(0.5)
    assert row["top2_ret_fwd_3d_cs_gross"] == pytest.approx(0.0025)


def test_compute_turnover_between_consecutive_dates():
    ranked = {
        "2026-06-01": ["000001", "000002", "000003"],
        "2026-06-04": ["000002", "000004", "000005"],
    }

    assert compute_turnover(ranked, k=3) == pytest.approx(2 / 3)
