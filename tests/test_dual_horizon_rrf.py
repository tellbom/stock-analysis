import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_d3_gate_once as legacy
from scripts.run_dual_horizon_rrf import observable_target, select_rrf
from quant_platform.features.pipeline import FeaturePipeline


@pytest.mark.parametrize("horizon,last_valid", [(3, 4), (5, 2)])
def test_target_ignores_future_prices_and_uses_t_plus_one(horizon, last_valid):
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(12)]
    prices = pd.DataFrame({"symbol": "000001", "date": dates, "close": np.arange(10.0, 22.0)})
    cutoff = dates[8]
    actual = observable_target(prices, cutoff, horizon)
    label = f"ret_fwd_{horizon}d"
    assert actual.loc[0, label] == pytest.approx((11.0 + horizon) / 11.0 - 1)
    assert actual.loc[actual[label].notna(), "date"].max() == dates[last_valid]
    assert actual.loc[actual[label].notna(), f"label_end_{horizon}d"].max() == cutoff
    prices.loc[prices.date > cutoff, "close"] = 1e9
    pd.testing.assert_frame_equal(actual, observable_target(prices, cutoff, horizon))


def test_d5_model_receives_exact_d5_label(monkeypatch):
    dates = [dt.date(2026, 6, i) for i in (1, 2, 3)]
    frame = pd.DataFrame({"symbol": ["000001"] * 3, "date": dates, "factor": [1, 2, 3],
                          "ret_fwd_3d": [100.0, 200.0, np.nan],
                          "ret_fwd_5d": [0.1, np.nan, np.nan]})
    captured = {}

    def fit(train, features, label, params):
        captured["target"] = label
        captured["values"] = train[label].tolist()
        return SimpleNamespace(named_steps={"lgbm": SimpleNamespace(feature_importances_=[1])},
                               predict=lambda rows: np.full(len(rows), 0.2))

    monkeypatch.setattr(legacy, "fit_final_model", fit)
    _, meta, *_ = legacy._score_model(frame, dates[-1], ["factor"], train_start=None,
                                      params={}, score_prefix="recent", label_col="ret_fwd_5d")
    assert captured == {"target": "ret_fwd_5d", "values": [0.1]}
    assert meta["label_col"] == "ret_fwd_5d"


def test_selection_uses_rrf_over_full_pool_gate_closed():
    """2026-10-14 裁决：分数 Gate 已关闭。select_rrf 对全量候选做 RRF 排序，
    不再按 gate_tier 过滤（RISK_VETO 也参与排名）。"""
    fused = pd.DataFrame({"symbol": ["000001", "000002", "000003"],
                          "gate_tier": ["A_MAIN", "B_SHORT_BOOST", "RISK_VETO"],
                          "base_rank": [2, 10, 1], "recent_rank": [30, 2, 1],
                          "final_rank": [1, 2, 3]})
    attributes = pd.DataFrame({"symbol": fused.symbol, "industry_code": "x", "industry_name": "test"})
    pool, picks, cap_stats = select_rrf(fused, attributes, sector_cap_tech=None)
    # 全量 3 只都在池中（含 RISK_VETO）。RRF: 000003=2/61 > 000002=1/70+1/62 > 000001=1/62+1/90
    assert len(pool) == 3
    assert picks.symbol.tolist() == ["000003", "000002", "000001"]
    assert picks.recommendation_rank.tolist() == [1, 2, 3]
    assert cap_stats["enabled"] is False


def test_strict_feature_build_stops_after_write_failure(tmp_path, monkeypatch):
    pipe = FeaturePipeline(store_root=tmp_path)
    monkeypatch.setattr(pipe.registry, "register", lambda specs: "test-set")

    def failed_write(*args):
        raise OSError("simulated locked feature file")

    monkeypatch.setattr(pipe, "_process_symbol", failed_write)
    with pytest.raises(RuntimeError, match="failed for 1/1"):
        pipe.run(["000001"], strict=True)
