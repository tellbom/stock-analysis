"""Tests for the shared Ridge32 weekly baseline.

Both weekly entries (compare_original_hybrid.py and run_dual_horizon_rrf.py
with --base-model ridge32) must produce IDENTICAL base predictions from this
single implementation.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quant_platform.selection.ridge_baseline import (
    ORIGINAL_FEATURES,
    STRATEGY_ID,
    STRATEGY_VERSION,
    fit_predict_ridge32,
    score_frame,
)

sklearn = pytest.importorskip("sklearn")


def _panel(n_symbols=6, n_dates=10, seed=7):
    rng = np.random.default_rng(seed)
    dates = [dt.date(2026, 9, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    rows = []
    for s in range(n_symbols):
        for d in dates:
            row = {"symbol": f"{s:06d}", "date": d}
            for f in ORIGINAL_FEATURES:
                row[f] = rng.normal()
            # label observable for all but the last 4 dates (D3 maturity)
            row["ret_fwd_3d"] = rng.normal() if d < dates[-4] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def test_original_features_contract_is_32_columns():
    assert len(ORIGINAL_FEATURES) == 32
    assert len(set(ORIGINAL_FEATURES)) == 32


def test_score_frame_columns_and_ranking():
    pred = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "date": [dt.date(2026, 10, 14)] * 3,
    })
    out = score_frame(pred, np.array([0.1, 0.3, 0.2]), "base")
    assert out["base_rank"].tolist() == [3, 1, 2]
    assert out["trade_date"].tolist() == [dt.date(2026, 10, 14)] * 3
    with pytest.raises(ValueError, match="nonfinite"):
        score_frame(pred, np.array([0.1, np.nan, 0.2]), "base")


def test_fit_predict_is_deterministic_and_shared():
    """同一份 panel 两次调用 -> 完全相同的预测（两处入口复用的基础）。"""
    panel = _panel()
    as_of = dt.date(2026, 9, 10)
    scored1, meta1, _ = fit_predict_ridge32(panel, as_of, "ret_fwd_3d")
    scored2, meta2, _ = fit_predict_ridge32(panel, as_of, "ret_fwd_3d")
    pd.testing.assert_frame_equal(scored1, scored2)
    assert meta1["strategy_id"] == STRATEGY_ID == "ridge32_weekly_baseline"
    assert meta1["strategy_version"] == STRATEGY_VERSION
    assert meta1["algorithm"] == "Ridge(alpha=1.0)"
    assert meta1["feature_count"] == 32
    # 训练只用 date < as_of 的行
    assert meta1["train_last_date"] < as_of.isoformat()
    assert meta1["prediction_date"] == as_of.isoformat()


def test_missing_feature_columns_raise_clear_error():
    panel = _panel().drop(columns=["ma_5", "rsi_6"])
    with pytest.raises(ValueError, match="missing 2 feature columns"):
        fit_predict_ridge32(panel, dt.date(2026, 9, 10), "ret_fwd_3d")


def test_empty_train_or_prediction_raises():
    panel = _panel()
    # as_of 早于所有数据 -> 无训练集
    with pytest.raises(RuntimeError, match="empty training set"):
        fit_predict_ridge32(panel, dt.date(2026, 8, 1), "ret_fwd_3d")
