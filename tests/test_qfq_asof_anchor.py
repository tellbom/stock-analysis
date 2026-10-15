"""Regression for a post-cutoff action changing fetched historical QFQ."""
import pandas as pd
import pytest

from quant_platform.ingest.ohlcv_collector import qfq_as_of


def _bars():
    return pd.DataFrame({"symbol": "600219", "date": pd.to_datetime(["2026-10-13", "2026-10-14"]),
                         "open": [4.7, 4.8], "high": [4.9, 5.], "low": [4.6, 4.7],
                         "close": [4.8, 4.86], "volume": [100., 200.]})


def test_future_action_cannot_change_cutoff_prices():
    original = pd.DataFrame({"date": ["2020-01-01"], "qfq_factor": [1.]})
    revised = pd.DataFrame({"date": ["2020-01-01", "2026-10-15"], "qfq_factor": [1.0572112247118, 1.]})
    expected = qfq_as_of(_bars(), original, "20261014")
    actual = qfq_as_of(_bars(), revised, "20261014")
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.close.iloc[-1] == 4.86
    assert actual.volume.tolist() == [100., 200.]


def test_prior_actions_are_applied_without_future_normalization():
    raw = _bars()
    raw.loc[0, "date"] = pd.Timestamp("2026-06-01")
    factor = pd.DataFrame({"date": ["2020-01-01", "2026-07-01", "2026-10-15"],
                           "qfq_factor": [2.2, 1.1, 1.]})
    actual = qfq_as_of(raw, factor, "20261014")
    assert actual.close.tolist() == [2.4, 4.86]


def test_factor_gaps_fail_instead_of_using_unadjusted_prices():
    factors = pd.DataFrame({"date": ["2026-10-14"], "qfq_factor": [1.]})
    with pytest.raises(ValueError, match="cover the complete"):
        qfq_as_of(_bars(), factors, "20261014")


def test_fetch_wrapper_requests_raw_bars_and_anchors_factors(monkeypatch):
    import akshare
    from quant_platform.ingest.ohlcv_collector import _fetch_symbol_raw
    calls = []

    def daily(**kwargs):
        calls.append(kwargs["adjust"])
        if kwargs["adjust"] == "":
            return _bars().drop(columns="symbol")
        assert kwargs["adjust"] == "qfq-factor"
        return pd.DataFrame({"date": ["2020-01-01", "2026-10-15"],
                             "qfq_factor": [1.0572112247118, 1.]})

    monkeypatch.setattr(akshare, "stock_zh_a_daily", daily)
    actual = _fetch_symbol_raw("600219", "20200101", "20261014")
    assert calls == ["", "qfq-factor"]
    assert actual.close.iloc[-1] == 4.86
