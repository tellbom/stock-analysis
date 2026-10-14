import datetime as dt

import pandas as pd
import pytest

from scripts.reconcile_d3_recommendations import reconcile_basket


ENTRY = dt.date(2026, 10, 12)
OBSERVED = dt.date(2026, 10, 13)
TARGET = dt.date(2026, 10, 15)


def fixture():
    rows = pd.DataFrame({"rank": [1, 2], "symbol": ["000001", "000002"]})
    prices = pd.DataFrame({
        "symbol": ["000001", "000001", "000002", "000002", "000300", "000300"],
        "date": [ENTRY, OBSERVED] * 3, "close": [100, 110, 100, 90, 1000, 1010],
    })
    return rows, prices


def test_equal_initial_capital_and_costs():
    rows, prices = fixture()
    result, detail = reconcile_basket(rows, prices, ENTRY, OBSERVED, TARGET, 0.001)
    assert result["gross_return"] == pytest.approx(0)
    assert result["net_if_exit"] == pytest.approx(-0.002)
    assert result["excess_if_exit"] == pytest.approx(-0.012)
    assert result["stock_win_rate"] == 0.5
    assert result["mature"] is False
    assert result["status"] == "interim_D3"
    assert len(detail) == 2


def test_missing_stock_price_never_drops_stock_or_returns_zero():
    rows, prices = fixture()
    prices = prices[~((prices.symbol == "000002") & (prices.date == OBSERVED))]
    result, detail = reconcile_basket(rows, prices, ENTRY, OBSERVED, TARGET, 0.001)
    assert result["priced_count"] == 1
    assert result["selected_count"] == 2
    assert result["gross_return"] is None
    assert result["excess_if_exit"] is None
    assert result["missing_symbols"] == ["000002"]
    assert detail[1]["gross_return"] is None


def test_missing_benchmark_does_not_create_excess():
    rows, prices = fixture()
    result, _ = reconcile_basket(rows, prices[prices.symbol != "000300"], ENTRY, OBSERVED, TARGET, 0.001)
    assert result["gross_return"] is not None
    assert result["benchmark_return"] is None
    assert result["excess_if_exit"] is None


def test_future_prices_are_ignored_and_maturity_is_explicit():
    rows, prices = fixture()
    future = pd.DataFrame({"symbol": ["000001"], "date": [TARGET], "close": [1e9]})
    first, _ = reconcile_basket(rows, prices, ENTRY, OBSERVED, TARGET, 0.001)
    second, _ = reconcile_basket(rows, pd.concat([prices, future]), ENTRY, OBSERVED, TARGET, 0.001)
    assert first == second
    complete = prices.copy()
    complete.loc[complete.date == OBSERVED, "date"] = TARGET
    final, _ = reconcile_basket(rows, complete, ENTRY, TARGET, TARGET, 0.001)
    assert final["mature"] is True
    assert final["status"] == "full_D3"


def test_duplicate_prices_fail():
    rows, prices = fixture()
    with pytest.raises(ValueError, match="Duplicate price"):
        reconcile_basket(rows, pd.concat([prices, prices.iloc[:1]]), ENTRY, OBSERVED, TARGET, 0.001)
