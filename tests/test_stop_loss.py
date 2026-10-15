"""Tests for the pre-declared stop-loss counterfactual (stop-loss-v1).

Pure-logic tests (no sklearn, no data files) so they run in the cloud.
OHLCV-backed tests need the local store and are marked accordingly.
"""

import math

import pytest

from quant_platform.evaluation.stop_loss import (
    EXIT_COST_BPS,
    STOP_LOSS_LEVELS,
    STOP_LOSS_PRIMARY,
    STOP_LOSS_RULE_DECLARED,
    STOP_LOSS_RULE_VERSION,
    apply_stop_loss,
    paired_summary,
)


def test_thresholds_are_pre_registered():
    """Guard: thresholds were fixed 2026-10-15 BEFORE any results.

    If you are here because you want to change a threshold: that is a NEW
    rule version, not an edit.  Bump STOP_LOSS_RULE_VERSION, set a new
    declaration date, and document the rationale — then update this test.
    Silent threshold tuning is exactly what this guard exists to prevent.
    """
    assert STOP_LOSS_PRIMARY == 0.08
    assert STOP_LOSS_LEVELS == (0.08, 0.05)
    assert STOP_LOSS_RULE_VERSION == "stop-loss-v1"
    assert STOP_LOSS_RULE_DECLARED == "2026-10-15"


def test_no_trigger_returns_nan_for_caller_to_fill():
    # Lows never breach 8% below entry 100.
    res = apply_stop_loss(100.0, [99.0, 98.5, 97.0], 0.08)
    assert res.stopped is False
    assert res.stop_day_offset is None
    assert math.isnan(res.realized_return)


def test_trigger_on_first_day():
    res = apply_stop_loss(100.0, [91.0, 99.0, 99.0], 0.08)
    assert res.stopped is True
    assert res.stop_day_offset == 1
    assert res.trigger_low == pytest.approx(91.0)
    # -8% minus one-way exit slippage.
    assert res.realized_return == pytest.approx(-0.08 - EXIT_COST_BPS / 10_000.0)


def test_trigger_on_later_day_uses_earliest():
    res = apply_stop_loss(100.0, [99.0, 90.0, 89.0], 0.08)
    assert res.stopped is True
    assert res.stop_day_offset == 2
    assert res.trigger_low == pytest.approx(90.0)


def test_boundary_low_exactly_at_stop_triggers():
    # low == stop price counts as breached (<=).
    res = apply_stop_loss(100.0, [92.0, 99.0], 0.08)
    assert res.stopped is True
    assert res.stop_day_offset == 1


def test_sensitivity_threshold_triggers_more_often():
    lows = [94.5, 96.0, 96.0]  # -5.5% dip: breaches 5% (94.5<=95.0), not 8%.
    assert apply_stop_loss(100.0, lows, 0.05).stopped is True
    assert apply_stop_loss(100.0, lows, 0.08).stopped is False


def test_invalid_inputs_yield_nan_not_fabrication():
    assert math.isnan(apply_stop_loss(0.0, [99.0], 0.08).realized_return)
    assert math.isnan(apply_stop_loss(-5.0, [99.0], 0.08).realized_return)
    assert math.isnan(apply_stop_loss(100.0, [], 0.08).realized_return)
    assert math.isnan(apply_stop_loss(100.0, [float("nan")], 0.08).realized_return)
    assert math.isnan(apply_stop_loss(100.0, [0.0], 0.08).realized_return)
    assert math.isnan(apply_stop_loss(100.0, [99.0], 0.0).realized_return)
    assert math.isnan(apply_stop_loss(100.0, [99.0], 1.5).realized_return)
    assert math.isnan(apply_stop_loss(float("nan"), [99.0], 0.08).realized_return)


def test_paired_summary_counts_and_tail_metrics():
    records = [
        # stopped: fixed +10%, stop realizes -8.075%
        {"ret_fixed": 0.10, "ret_sl": -0.08075, "stopped": True, "stop_pct": 0.08},
        # not stopped: identical legs -> diff 0
        {"ret_fixed": 0.03, "ret_sl": 0.03, "stopped": False, "stop_pct": 0.08},
        # stopped on a loser: fixed -25%, stop caps at -8.075%
        {"ret_fixed": -0.25, "ret_sl": -0.08075, "stopped": True, "stop_pct": 0.08},
    ]
    s = paired_summary(records)
    assert s["n"] == 3
    assert s["n_stopped"] == 2
    assert s["trigger_rate"] == pytest.approx(2 / 3)
    assert s["rule_version"] == STOP_LOSS_RULE_VERSION
    # Paired diffs: (-0.18075, 0, +0.16925) -> mean < 0 here by construction.
    assert s["mean_diff_sl_minus_fixed"] == pytest.approx(
        (-0.18075 + 0.0 + 0.16925) / 3
    )
    assert s["min_ret_sl"] == pytest.approx(-0.08075)
    assert s["min_ret_fixed"] == pytest.approx(-0.25)
    assert "paired_t_stat" in s


def test_paired_summary_empty():
    assert paired_summary([]) == {"n": 0}
    assert paired_summary([{"ret_fixed": float("nan"), "ret_sl": 0.01,
                            "stopped": False, "stop_pct": 0.08}]) == {"n": 0}
