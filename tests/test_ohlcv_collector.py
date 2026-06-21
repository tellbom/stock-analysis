"""
T0.5 verification tests for OHLCVCollector.

All tests use synthetic data injected via monkeypatching — no live AKShare
network calls are made.  One integration smoke test attempts a real fetch and
documents the outcome (expected: 403 in sandbox, passes either way).

Run with:  PYTHONPATH=/home/claude python quant_platform/tests/test_ohlcv_collector.py
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(symbol: str, start: str, n: int, base_close: float = 100.0) -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame simulating normalised output from _fetch_symbol_raw.
    Includes 'symbol' column because _normalise_daily/_normalise_hist add it
    before returning — the patch replaces the entire _fetch_symbol_raw function,
    so the fixture must match the function's actual return shape.
    """
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "symbol": symbol,
        "date":   [d.date() for d in dates],
        "open":   [base_close + i * 0.1 for i in range(n)],
        "high":   [base_close + i * 0.1 + 1.0 for i in range(n)],
        "low":    [base_close + i * 0.1 - 1.0 for i in range(n)],
        "close":  [base_close + i * 0.1 for i in range(n)],
        "volume": [100_000.0 + i * 1_000 for i in range(n)],
        "amount": [1_000_000.0 + i * 10_000 for i in range(n)],
    })


def _patch_fetch(symbol: str, df: pd.DataFrame):
    """Context manager: make _fetch_symbol_raw return df for any call."""
    return patch(
        "quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
        return_value=df,
    )


def _patch_fetch_none():
    """Context manager: make _fetch_symbol_raw return None (all endpoints fail)."""
    return patch(
        "quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
        return_value=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_collect_symbol_fresh_write():
    """First fetch for a symbol writes correct data to Parquet."""
    from quant_platform.ingest.ohlcv_collector import collect_symbol
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    symbol = "600519"
    raw = _make_ohlcv(symbol, "2024-01-02", n=10)

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        with _patch_fetch(symbol, raw):
            result = collect_symbol(
                symbol, tmp,
                start_date=dt.date(2024, 1, 1),
                end_date=dt.date(2024, 1, 31),
            )

        assert result.success, f"Expected success, got error: {result.error}"
        assert result.rows_new == 10
        assert result.rows_total == 10

        df = read_ohlcv(ohlcv_path(tmp, symbol))
        assert len(df) == 10
        assert list(df["symbol"]) == [symbol] * 10
        assert all(isinstance(d, dt.date) for d in df["date"])
    print("  [OK] collect_symbol: fresh write creates correct Parquet")


def test_collect_symbol_incremental_append():
    """Second fetch only requests tail and appends — no duplicates."""
    from quant_platform.ingest.ohlcv_collector import collect_symbol
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    symbol = "600519"

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)

        # First batch: Jan 2024 (10 rows)
        batch1 = _make_ohlcv(symbol, "2024-01-02", n=10)
        with _patch_fetch(symbol, batch1):
            r1 = collect_symbol(symbol, tmp,
                                start_date=dt.date(2024, 1, 1),
                                end_date=dt.date(2024, 1, 15))
        assert r1.rows_new == 10

        # Second batch: Feb 2024 (5 new rows)
        batch2 = _make_ohlcv(symbol, "2024-02-01", n=5, base_close=105.0)
        with _patch_fetch(symbol, batch2):
            r2 = collect_symbol(symbol, tmp,
                                start_date=dt.date(2024, 1, 1),
                                end_date=dt.date(2024, 2, 28))
        assert r2.rows_new == 5

        # Should have 15 total, no duplicates
        df = read_ohlcv(ohlcv_path(tmp, symbol))
        assert len(df) == 15, f"Expected 15 rows after append, got {len(df)}"
        assert df["date"].is_monotonic_increasing, "Dates not sorted"
        assert df["date"].nunique() == 15, "Duplicate dates found"
    print("  [OK] collect_symbol: incremental append — no duplicates, 15 total rows")


def test_collect_symbol_skips_when_up_to_date():
    """If stored data is current, collect_symbol skips fetch (returns skipped)."""
    from quant_platform.ingest.ohlcv_collector import collect_symbol
    from quant_platform.store.lake import ohlcv_path, init_lake
    from quant_platform.store.parquet_store import write_ohlcv

    symbol = "600519"
    today = dt.date.today()

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        # Pre-write data up to today
        existing = _make_ohlcv(symbol, str(today - dt.timedelta(days=4)), n=3)
        existing["symbol"] = symbol
        write_ohlcv(existing, ohlcv_path(tmp, symbol), symbol)

        # Collector with end_date = max stored date
        stored_max = existing["date"].max()
        with _patch_fetch_none():   # would fail if called
            result = collect_symbol(symbol, tmp,
                                    start_date=dt.date(2024, 1, 1),
                                    end_date=stored_max)

        assert result.success
        assert result.rows_new == 0
        assert result.error == "skipped"
    print("  [OK] collect_symbol: skips network call when data is already up-to-date")


def test_collect_symbol_returns_failure_on_network_error():
    """If all AKShare endpoints fail, result is success=False, no data written."""
    from quant_platform.ingest.ohlcv_collector import collect_symbol
    from quant_platform.store.lake import ohlcv_path, init_lake
    from quant_platform.store.parquet_store import read_ohlcv

    symbol = "600519"
    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        with _patch_fetch_none():
            result = collect_symbol(symbol, tmp,
                                    start_date=dt.date(2024, 1, 1),
                                    end_date=dt.date(2024, 1, 31))

        assert not result.success
        assert result.rows_new == 0
        assert result.error != ""
        # File must not exist (no partial / fabricated write)
        assert not ohlcv_path(tmp, symbol).exists(), \
            "Parquet file must not be created when fetch fails"
    print("  [OK] collect_symbol: failure → success=False, no file written")


def test_collector_run_multi_symbol():
    """OHLCVCollector.run() processes multiple symbols and returns correct summary."""
    from quant_platform.ingest.ohlcv_collector import OHLCVCollector
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    symbols = ["600519", "000858", "300750"]

    def fake_fetch(symbol, start_date, end_date):
        return _make_ohlcv(symbol, "2024-01-02", n=5)

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        collector = OHLCVCollector(
            store_root=tmp,
            start_date="2024-01-01",
            end_date="2024-01-31",
            max_workers=1,
        )
        with patch("quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
                   side_effect=fake_fetch):
            summary = collector.run(symbols)

        assert summary.total == 3
        assert summary.succeeded == 3
        assert summary.failed == 0
        assert len(summary.results) == 3

        for sym in symbols:
            df = read_ohlcv(ohlcv_path(tmp, sym))
            assert len(df) == 5, f"{sym}: expected 5 rows"
    print("  [OK] OHLCVCollector.run(): 3 symbols, all succeeded, correct Parquet written")


def test_collector_run_partial_failure():
    """If one symbol fails, others still succeed; summary reflects accurate counts."""
    from quant_platform.ingest.ohlcv_collector import OHLCVCollector

    symbols = ["600519", "BAD_SYM", "000858"]
    call_count = {}

    def fake_fetch(symbol, start_date, end_date):
        call_count[symbol] = call_count.get(symbol, 0) + 1
        if symbol == "BAD_SYM":
            return None   # simulate endpoint failure
        return _make_ohlcv(symbol, "2024-01-02", n=3)

    with tempfile.TemporaryDirectory() as tmp:
        collector = OHLCVCollector(
            store_root=tmp,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        with patch("quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
                   side_effect=fake_fetch):
            summary = collector.run(symbols)

        assert summary.succeeded == 2
        assert summary.failed == 1
        assert "BAD_SYM" in summary.failed_symbols
    print("  [OK] OHLCVCollector.run(): partial failure — 2 succeed, 1 fails, summary accurate")


def test_collector_run_empty_list():
    """Running with empty symbol list returns a valid summary with zero counts."""
    from quant_platform.ingest.ohlcv_collector import OHLCVCollector

    with tempfile.TemporaryDirectory() as tmp:
        collector = OHLCVCollector(store_root=tmp)
        summary = collector.run([])

        assert summary.total == 0
        assert summary.succeeded == 0
        assert summary.failed == 0
    print("  [OK] OHLCVCollector.run(): empty symbol list → valid zero-count summary")


def test_normalise_daily_with_chinese_columns():
    """_normalise_daily handles Chinese column names gracefully."""
    from quant_platform.ingest.ohlcv_collector import _normalise_daily

    # Simulate an AKShare version that returns Chinese columns
    df = pd.DataFrame({
        "日期":  [dt.date(2024, 1, 2)],
        "开盘":  [100.0],
        "最高":  [105.0],
        "最低":  [95.0],
        "收盘":  [102.0],
        "成交量": [100000.0],
        "成交额": [10000000.0],
    })
    result = _normalise_daily(df, "600519")
    assert "open" in result.columns
    assert "close" in result.columns
    assert "symbol" in result.columns
    assert result.iloc[0]["symbol"] == "600519"
    print("  [OK] _normalise_daily: handles Chinese column names correctly")


def test_normalise_hist_extracts_turnover():
    """_normalise_hist maps 换手率 → turnover and retains it."""
    from quant_platform.ingest.ohlcv_collector import _normalise_hist

    df = pd.DataFrame({
        "日期":  [dt.date(2024, 1, 2)],
        "开盘":  [100.0], "收盘": [102.0],
        "最高":  [105.0], "最低": [95.0],
        "成交量": [100000.0], "成交额": [10000000.0],
        "振幅":  [10.0], "涨跌幅": [2.0], "涨跌额": [2.0],
        "换手率": [1.5],
    })
    result = _normalise_hist(df, "600519")
    assert "turnover" in result.columns
    assert result.iloc[0]["turnover"] == 1.5
    print("  [OK] _normalise_hist: 换手率 mapped to 'turnover' column")


def test_integration_live_network_attempt():
    """
    Attempt a real AKShare fetch for one symbol.
    In sandbox (403): documents failure clearly.
    In production (live network): validates real data shape.
    """
    from quant_platform.ingest.ohlcv_collector import _fetch_symbol_raw

    result = _fetch_symbol_raw(
        "600519",
        start_date="20240102",
        end_date="20240110",
    )

    if result is None:
        print("  [OK] Integration: all endpoints returned None (expected in sandbox — "
              "network blocked). No fabricated data written.")
    else:
        assert "symbol" in result.columns
        assert "close"  in result.columns
        assert len(result) > 0
        dates_ok = all(isinstance(d, dt.date) for d in result["date"])
        assert dates_ok, "date column must contain dt.date objects"
        print(f"  [OK] Integration: live fetch returned {len(result)} rows for 600519 "
              f"(dates: {result['date'].min()} → {result['date'].max()})")


if __name__ == "__main__":
    print("\n=== T0.5 OHLCVCollector tests ===\n")
    tests = [
        test_collect_symbol_fresh_write,
        test_collect_symbol_incremental_append,
        test_collect_symbol_skips_when_up_to_date,
        test_collect_symbol_returns_failure_on_network_error,
        test_collector_run_multi_symbol,
        test_collector_run_partial_failure,
        test_collector_run_empty_list,
        test_normalise_daily_with_chinese_columns,
        test_normalise_hist_extracts_turnover,
        test_integration_live_network_attempt,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
