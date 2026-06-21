"""
T0.2 verification tests for UniverseService.

Run with:  python -m pytest tests/test_universe_service.py -v
       or:  python tests/test_universe_service.py   (standalone)

These tests use load_from_csv() to bypass network dependency, verifying the
effective-date schema, as-of query logic, and survivorship-bias reporting
without requiring live AKShare access.
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _setup_service(tmp_dir: str):
    """Import and instantiate UniverseService against a temp store root."""
    from quant_platform.ingest.universe_service import UniverseService

    return UniverseService("csi300", store_root=tmp_dir)


def test_unknown_universe_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            from quant_platform.ingest.universe_service import UniverseService

            UniverseService("nonexistent_index", store_root=tmp)
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "nonexistent_index" in str(e)
    print("  [OK] Unknown universe key raises KeyError")


def test_load_from_csv_and_get_symbols():
    """Load a CSV with 5 symbols, verify get_symbols_as_of returns them."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test_cons.csv"
        symbols = ["600519", "000858", "300750", "601318", "000333"]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "name"])
            for s in symbols:
                w.writerow([s, f"Stock_{s}"])

        svc = _setup_service(tmp)
        svc.load_from_csv(csv_path)
        result = svc.get_symbols_as_of()

        assert set(result) == set(symbols), f"Expected {symbols}, got {result}"
        assert result == sorted(result), "Symbols should be sorted"
    print("  [OK] load_from_csv + get_symbols_as_of returns correct symbols")


def test_effective_date_filtering():
    """Symbols added after as_of date should not appear in the result."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "dated_cons.csv"
        today = dt.date.today()
        past = today - dt.timedelta(days=30)
        future = today + dt.timedelta(days=10)

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "name", "in_date", "out_date"])
            w.writerow(["600519", "Moutai", str(past), ""])
            w.writerow(["000858", "Wuliangye", str(future), ""])
            w.writerow(["300750", "CATL", str(past), str(past + dt.timedelta(days=5))])

        svc = _setup_service(tmp)
        svc.load_from_csv(csv_path, has_effective_dates=True)

        result = svc.get_symbols_as_of(today)
        assert "600519" in result
        assert "000858" not in result
        assert "300750" not in result
    print("  [OK] Effective-date filtering excludes future and expired members")


def test_survivorship_status_no_effective_dates():
    """Without effective dates, survivorship_status must flag the bias risk."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "cons.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol"])
            writer.writerow(["600519"])

        svc = _setup_service(tmp)
        svc.load_from_csv(csv_path, has_effective_dates=False)
        status = svc.survivorship_status()

        assert status["has_effective_dates"] is False
        assert status["survivorship_bias_risk"] is True
        assert len(status["note"]) > 20
    print("  [OK] survivorship_status flags bias risk without effective dates")


def test_survivorship_status_with_effective_dates():
    """With effective dates loaded, survivorship_status clears the bias flag."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "cons.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol"])
            writer.writerow(["600519"])

        svc = _setup_service(tmp)
        svc.load_from_csv(csv_path, has_effective_dates=True)
        status = svc.survivorship_status()

        assert status["has_effective_dates"] is True
        assert status["survivorship_bias_risk"] is False
    print("  [OK] survivorship_status clears bias flag with effective dates")


def test_missing_membership_raises():
    """get_symbols_as_of before any fetch raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(tmp)
        try:
            svc.get_symbols_as_of()
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "csi300" in str(e)
    print("  [OK] get_symbols_as_of before fetch raises FileNotFoundError")


def test_fetch_and_save_raises_when_network_blocked():
    """When all endpoints fail, fetch_and_save raises UniverseFetchError."""
    import quant_platform.ingest.universe_service as universe_service
    from quant_platform.ingest.universe_service import UniverseFetchError, UniverseService

    original_csindex = universe_service._fetch_csindex
    original_sina = universe_service._fetch_sina
    universe_service._fetch_csindex = lambda index_code: None
    universe_service._fetch_sina = lambda index_code: None

    with tempfile.TemporaryDirectory() as tmp:
        try:
            svc = UniverseService("csi300", store_root=tmp)
            svc.fetch_and_save()
            assert False, "Should have raised UniverseFetchError"
        except UniverseFetchError as e:
            assert "csi300" in str(e) or "endpoint" in str(e).lower()
            print(f"  [OK] fetch_and_save raises UniverseFetchError: {str(e)[:80]}")
        finally:
            universe_service._fetch_csindex = original_csindex
            universe_service._fetch_sina = original_sina


def test_membership_df_schema():
    """membership_df must have exactly the required columns."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "cons.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "name"])
            w.writerow(["600519", "Moutai"])

        svc = _setup_service(tmp)
        svc.load_from_csv(csv_path)
        df = svc.membership_df()

        required = {"symbol", "in_date", "out_date", "name", "source"}
        assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "600519"
    print("  [OK] membership_df has correct schema")


if __name__ == "__main__":
    print("\n=== T0.2 UniverseService tests ===\n")
    tests = [
        test_unknown_universe_raises,
        test_load_from_csv_and_get_symbols,
        test_effective_date_filtering,
        test_survivorship_status_no_effective_dates,
        test_survivorship_status_with_effective_dates,
        test_missing_membership_raises,
        test_fetch_and_save_raises_when_network_blocked,
        test_membership_df_schema,
    ]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback

            print(f"  [FAIL] {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
