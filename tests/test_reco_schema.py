from __future__ import annotations

import pandas as pd

from quant_platform.selection.reco_schema import (
    RECOMMENDATION_SCHEMA,
    recommendation_columns,
    write_recommendation_csv,
)


def _leaky_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["000001", "000002"],
        "date": ["2026-07-08", "2026-07-08"],
        "industry_code": ["801010", "801020"],
        "industry_name": ["A", "B"],
        "model_score": [0.5, 0.4],
        "industry_rank": [1, 1],
        "selected": [True, True],
        "selection_reason": ["industry_top_k", "industry_top_k"],
        "exposure_flag": ["balanced", "balanced"],
        # leakage columns that must never be shipped
        "ret_fwd_1d": [0.01, -0.02],
        "ret_fwd_3d": [0.02, -0.01],
        "ret_fwd_5d": [0.03, 0.00],
        "ret_fwd_3d_cs": [0.01, -0.01],
        "ret_fwd_5d_bin": [1, 0],
        "some_future_fwd_field": [1.0, 2.0],
        "close": [10.0, 20.0],  # not in allow-list either -- also stripped
    })


def test_recommendation_columns_excludes_forward_label_columns():
    df = _leaky_frame()
    cols = recommendation_columns(df)
    for forbidden in ("ret_fwd_1d", "ret_fwd_3d", "ret_fwd_5d", "ret_fwd_3d_cs",
                      "ret_fwd_5d_bin", "some_future_fwd_field"):
        assert forbidden not in cols


def test_recommendation_columns_is_fixed_allow_list():
    df = _leaky_frame()
    cols = recommendation_columns(df)
    assert set(cols) <= set(RECOMMENDATION_SCHEMA)
    # only columns that are both in the schema AND present in df
    expected = [c for c in RECOMMENDATION_SCHEMA if c in df.columns]
    assert cols == expected
    # "close" is present in df but not in the schema -- must be dropped
    assert "close" not in cols


def test_write_recommendation_csv_ships_no_leakage(tmp_path):
    df = _leaky_frame()
    out_path = write_recommendation_csv(df, tmp_path / "reco.csv")
    written = pd.read_csv(out_path)
    for forbidden in ("ret_fwd_1d", "ret_fwd_3d", "ret_fwd_5d", "ret_fwd_3d_cs",
                      "ret_fwd_5d_bin", "some_future_fwd_field", "close"):
        assert forbidden not in written.columns
    assert "symbol" in written.columns
    assert len(written) == 2
