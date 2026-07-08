from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.selection.config import SelectionConfig
from quant_platform.selection.sizing import attach_holding_and_weight


def _panel(symbols, industries, confidences=None):
    data = {"symbol": symbols, "industry_code": industries}
    if confidences is not None:
        data["confidence"] = confidences
    return pd.DataFrame(data)


def test_equal_tilt_weights_sum_to_one_and_are_uniform():
    df = _panel([f"S{i}" for i in range(10)], ["A", "B"] * 5)
    out = attach_holding_and_weight(df, tilt="equal")

    assert abs(out["suggested_weight"].sum() - 1.0) < 1e-9
    assert (out["suggested_weight"] == out["suggested_weight"].iloc[0]).all()
    assert (out["holding_horizon_days"] == 3).all()


def test_confidence_tilt_weights_sum_to_one():
    df = _panel(
        [f"S{i}" for i in range(6)],
        ["A", "A", "B", "B", "C", "C"],
        confidences=[0.9, 0.8, 0.2, 0.2, 0.5, 0.5],
    )
    out = attach_holding_and_weight(df, tilt="confidence")

    assert abs(out["suggested_weight"].sum() - 1.0) < 1e-9


def test_confidence_tilt_respects_industry_cap():
    # Industry "A" dominates confidence -- uncapped it would take ~0.86 of
    # total weight, well above the default 0.30 exposure_warning_threshold.
    # Four industries total (cap capacity 4 * 0.30 = 1.2 >= 1.0) so the cap
    # is actually satisfiable while still summing to 1.
    df = _panel(
        [f"S{i}" for i in range(10)],
        ["A"] * 4 + ["B"] * 2 + ["C"] * 2 + ["D"] * 2,
        confidences=[0.9] * 4 + [0.1] * 6,
    )
    config = SelectionConfig()  # default exposure_warning_threshold = 0.30

    out = attach_holding_and_weight(df, tilt="confidence", config=config)

    industry_totals = out.groupby("industry_code")["suggested_weight"].sum()
    assert (industry_totals <= config.exposure_warning_threshold + 1e-6).all()
    assert abs(out["suggested_weight"].sum() - 1.0) < 1e-6


def test_custom_horizon_days():
    df = _panel(["S0", "S1"], ["A", "A"])
    out = attach_holding_and_weight(df, horizon_days=5, tilt="equal")
    assert (out["holding_horizon_days"] == 5).all()


def test_confidence_tilt_missing_column_raises():
    df = _panel(["S0", "S1"], ["A", "A"])
    with pytest.raises(ValueError):
        attach_holding_and_weight(df, tilt="confidence")


def test_empty_frame_raises():
    df = pd.DataFrame(columns=["symbol", "industry_code"])
    with pytest.raises(ValueError):
        attach_holding_and_weight(df)


def test_unknown_tilt_raises():
    df = _panel(["S0"], ["A"])
    with pytest.raises(ValueError):
        attach_holding_and_weight(df, tilt="bogus")
