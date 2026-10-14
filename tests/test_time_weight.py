"""P0 时间权重逻辑测试（不需要 sklearn）。

验证 quant_platform.selection.ridge_baseline 的时间衰减权重公式：
weight = 0.5 ** (days_ago / TIME_WEIGHT_HALF_LIFE_DAYS)。
"""
from __future__ import annotations

import inspect

import numpy as np

from quant_platform.selection.ridge_baseline import (
    TIME_WEIGHT_HALF_LIFE_DAYS,
    fit_predict_ridge32,
)


def _weights(days_ago: np.ndarray) -> np.ndarray:
    """复刻 fit_predict_ridge32 内部的权重公式。"""
    return 0.5 ** (days_ago / TIME_WEIGHT_HALF_LIFE_DAYS)


def test_half_life_is_180_days():
    assert TIME_WEIGHT_HALF_LIFE_DAYS == 180


def test_weight_at_half_life_is_half():
    assert _weights(np.array([180.0]))[0] == 0.5


def test_weight_monotonic_decreasing():
    days = np.array([0.0, 1.0, 90.0, 180.0, 365.0, 1095.0])
    w = _weights(days)
    assert w[0] == 1.0  # as_of 前一天权重为 1
    assert bool((w[:-1] > w[1:]).all())  # 严格递减
    assert w[-1] < 0.02  # 3 年前权重可忽略


def test_use_time_weight_defaults_to_false():
    """冻结基线默认行为不变：不传参 = 不加权。"""
    sig = inspect.signature(fit_predict_ridge32)
    param = sig.parameters["use_time_weight"]
    assert param.default is False
