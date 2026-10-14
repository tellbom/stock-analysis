"""Tests for label-maturity time conventions (T+1 close).

Uses a simulated trading calendar (Mon-Fri, no holidays) to verify:
- entry = first trading day after as_of (T+1)
- D+3/D+5 expiry dates
- feature cutoff vs. last mature training sample date are distinct
- future (immature) labels are excluded from training masks
"""
from __future__ import annotations

import datetime as dt

import pytest

from quant_platform.labels.maturity import (
    entry_date,
    expiry_date,
    label_maturity_date,
    last_mature_feature_date,
    training_mask,
)


def _calendar(start=dt.date(2026, 10, 1), end=dt.date(2026, 10, 30)):
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


@pytest.fixture
def cal():
    return _calendar()


def test_entry_is_t_plus_one(cal):
    # 周三收盘 -> 周四入场
    assert entry_date(dt.date(2026, 10, 14), cal) == dt.date(2026, 10, 15)
    # 周五收盘 -> 下周一入场（跳过周末）
    assert entry_date(dt.date(2026, 10, 16), cal) == dt.date(2026, 10, 19)


def test_expiry_dates(cal):
    # future = [10-15, 10-16, 10-19, 10-20, 10-21, 10-22, ...]
    assert expiry_date(dt.date(2026, 10, 14), 3, cal) == dt.date(2026, 10, 20)
    assert expiry_date(dt.date(2026, 10, 14), 5, cal) == dt.date(2026, 10, 22)
    with pytest.raises(ValueError, match="Only D\\+3/D\\+5"):
        expiry_date(dt.date(2026, 10, 14), 10, cal)


def test_label_maturity_is_entry_plus_horizon(cal):
    # 特征日 d 的 D+h 标签在 d + (h+1) 个交易日收盘成熟
    assert label_maturity_date(dt.date(2026, 10, 8), 3, cal) == dt.date(2026, 10, 14)
    assert label_maturity_date(dt.date(2026, 10, 8), 5, cal) == dt.date(2026, 10, 16)
    with pytest.raises(ValueError, match="not a trading day"):
        label_maturity_date(dt.date(2026, 10, 10), 3, cal)  # 周六


def test_last_mature_feature_date_differs_from_cutoff(cal):
    as_of = dt.date(2026, 10, 14)
    # D3: as_of - 4 个交易日；D5: as_of - 6 个交易日
    assert last_mature_feature_date(as_of, 3, cal) == dt.date(2026, 10, 8)
    assert last_mature_feature_date(as_of, 5, cal) == dt.date(2026, 10, 6)
    assert last_mature_feature_date(as_of, 3, cal) < as_of


def test_training_mask_excludes_immature_labels(cal):
    as_of = dt.date(2026, 10, 14)
    dates = [dt.date(2026, 10, 7), dt.date(2026, 10, 8),
             dt.date(2026, 10, 9), dt.date(2026, 10, 14)]
    assert training_mask(dates, as_of, 3, cal) == [True, True, False, False]
    # D5 口径更严：10-07 也不成熟
    assert training_mask(dates, as_of, 5, cal) == [True, False, False, False]


def test_no_trading_day_after_as_of_raises(cal):
    with pytest.raises(ValueError, match="No trading day after"):
        entry_date(dt.date(2026, 10, 30), cal)
