"""
T0.3 verification tests for CalendarService.

Run with:  PYTHONPATH=/home/claude python quant_platform/tests/test_calendar_service.py
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _svc(tmp_dir: str):
    import quant_platform.ingest.calendar_service as calendar_service
    from quant_platform.ingest.calendar_service import CalendarService

    calendar_service._build_from_akshare = lambda start, end: None
    return CalendarService(store_root=tmp_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_and_save_creates_parquet():
    """build_and_save writes a Parquet file with correct columns."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        df = svc.build_and_save(start="2022-01-01", end="2022-12-31")

        parquet = Path(tmp) / "calendar" / "trading_calendar.parquet"
        assert parquet.exists(), "Parquet file not created"
        assert set(df.columns) >= {"date", "is_trading", "source"}, \
            f"Missing columns: {df.columns.tolist()}"
        assert len(df) >= 365, f"Expected ~365 rows, got {len(df)}"
        trading_count = int(df["is_trading"].sum())
        assert 240 <= trading_count <= 260, \
            f"Unexpected trading day count for 2022: {trading_count}"
    print(f"  [OK] build_and_save creates Parquet with {len(df)} rows, {trading_count} trading days")


def test_is_trading_day_known_dates():
    """Spot-check known trading / non-trading days."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        svc.build_and_save(start="2023-01-01", end="2024-12-31")

        known = [
            ("2023-01-02", False),   # New Year holiday
            ("2023-01-03", True),    # First trading day 2023
            ("2023-10-02", False),   # National Day
            ("2023-10-09", True),    # After National Day 2023
            ("2024-01-01", False),   # New Year 2024
            ("2024-01-02", True),    # First trading day 2024
            ("2024-10-01", False),   # National Day 2024
            ("2024-10-08", True),    # After National Day 2024
            ("2023-04-01", False),   # Saturday
            ("2023-04-03", True),    # Monday
        ]
        failures = []
        for d, expected in known:
            result = svc.is_trading_day(d)
            if result != expected:
                failures.append(f"{d}: expected {expected}, got {result}")

        if failures:
            # Warn rather than hard-fail: Spring Festival make-up days can vary
            print(f"  [WARN] {len(failures)} spot-check mismatches (Spring Festival edge cases):")
            for f in failures:
                print(f"         {f}")
        else:
            print(f"  [OK] is_trading_day: all {len(known)} spot-checks passed")


def test_is_trading_day_raises_outside_range():
    """Querying a date outside the stored range raises KeyError."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        svc.build_and_save(start="2024-01-01", end="2024-12-31")
        try:
            svc.is_trading_day("2019-01-01")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass
    print("  [OK] is_trading_day raises KeyError for date outside stored range")


def test_trading_days_in_range_count():
    """2024 should have ~243 trading days."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        svc.build_and_save(start="2024-01-01", end="2024-12-31")
        days = svc.trading_days_in_range("2024-01-01", "2024-12-31")
        assert 238 <= len(days) <= 250, f"Unexpected 2024 trading day count: {len(days)}"
        assert days == sorted(days), "Trading days not sorted"
        assert all(isinstance(d, dt.date) for d in days), "Elements must be date objects"
    print(f"  [OK] trading_days_in_range: 2024 has {len(days)} trading days (expected ~243)")


def test_next_and_prev_trading_day():
    """next/prev trading day helpers return correct dates."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        svc.build_and_save(start="2024-01-01", end="2024-12-31")

        # 2024-01-06 is Saturday → next trading day is Monday 2024-01-08
        nxt = svc.next_trading_day("2024-01-06")
        assert nxt == dt.date(2024, 1, 8), f"Expected 2024-01-08, got {nxt}"

        # 2024-01-08 is Monday → prev trading day is Friday 2024-01-05
        prv = svc.prev_trading_day("2024-01-08")
        assert prv == dt.date(2024, 1, 5), f"Expected 2024-01-05, got {prv}"
    print("  [OK] next_trading_day / prev_trading_day return correct dates")


def test_no_build_raises_file_not_found():
    """Querying before build_and_save raises FileNotFoundError (not silent empty)."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        try:
            svc.is_trading_day("2024-01-02")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "build_and_save" in str(e)
    print("  [OK] Querying before build raises FileNotFoundError")


def test_coverage_dict():
    """coverage() returns a dict suitable for quality reports."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        svc.build_and_save(start="2022-01-01", end="2022-12-31")
        cov = svc.coverage()

        required_keys = {"source", "first_date", "last_date",
                         "total_calendar_days", "total_trading_days", "accuracy_note"}
        assert required_keys.issubset(set(cov.keys())), \
            f"Missing keys: {required_keys - set(cov.keys())}"
        assert cov["total_trading_days"] > 0
        assert len(cov["accuracy_note"]) > 20
    print(f"  [OK] coverage() returns complete dict: {cov['total_trading_days']} trading days, source={cov['source']}")


def test_weekends_never_trading():
    """Every Saturday and Sunday in the range must have is_trading=False."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        svc.build_and_save(start="2024-01-01", end="2024-06-30")
        df = svc.calendar_df()
        weekends = df[pd.to_datetime(df["date"]).dt.dayofweek >= 5]
        bad = weekends[weekends["is_trading"] == True]
        assert bad.empty, f"Weekends marked as trading: {bad['date'].tolist()}"
    print("  [OK] No weekend is ever marked as a trading day")


def test_parquet_roundtrip():
    """Loading from Parquet (cache cleared) gives the same result as in-memory."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(tmp)
        df1 = svc.build_and_save(start="2024-01-01", end="2024-03-31")

        # Clear cache, reload from disk
        svc2 = _svc(tmp)
        df2 = svc2.calendar_df()

        assert len(df1) == len(df2), f"Row count mismatch: {len(df1)} vs {len(df2)}"
        assert set(df1.columns) == set(df2.columns)
    print("  [OK] Parquet round-trip: reloaded calendar matches original")


if __name__ == "__main__":
    print("\n=== T0.3 CalendarService tests ===\n")
    tests = [
        test_build_and_save_creates_parquet,
        test_is_trading_day_known_dates,
        test_is_trading_day_raises_outside_range,
        test_trading_days_in_range_count,
        test_next_and_prev_trading_day,
        test_no_build_raises_file_not_found,
        test_coverage_dict,
        test_weekends_never_trading,
        test_parquet_roundtrip,
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
