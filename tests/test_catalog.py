"""
T0.6 verification tests for CollectorCatalog and CatalogDrivenCollector.

All tests use synthetic data / monkeypatching — no live network required.

Run with:  PYTHONPATH=/home/claude python quant_platform/tests/test_catalog.py
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
# Helpers
# ---------------------------------------------------------------------------

def _catalog(tmp: str):
    from quant_platform.ingest.catalog import CollectorCatalog
    return CollectorCatalog(store_root=tmp)


def _make_fetch_result(symbol: str, success: bool = True, rows: int = 5,
                       last_date: str = "2024-01-31", error: str = ""):
    from quant_platform.ingest.ohlcv_collector import FetchResult
    return FetchResult(
        symbol=symbol,
        success=success,
        rows_new=rows if success else 0,
        rows_total=rows if success else 0,
        last_date=dt.date.fromisoformat(last_date) if (success and last_date) else None,
        error=error,
    )


def _make_ohlcv(symbol: str, n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "symbol": symbol,
        "date":   [d.date() for d in dates],
        "open":   100.0, "high": 105.0, "low": 95.0,
        "close":  102.0, "volume": 100_000.0,
    })


# ---------------------------------------------------------------------------
# CollectorCatalog tests
# ---------------------------------------------------------------------------

def test_catalog_load_empty_when_no_file():
    """load() returns correct-schema empty DataFrame when catalog absent."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        df = cat.load()
        assert df.empty
        assert "symbol" in df.columns
        assert "status" in df.columns
    print("  [OK] load(): returns empty DataFrame with correct schema when file absent")


def test_catalog_mark_in_progress_and_read_back():
    """mark_in_progress persists to Parquet and is readable."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        cat.mark_in_progress("600519", "ohlcv_daily")

        row = cat.get_row("600519", "ohlcv_daily")
        assert row is not None
        assert row["status"] == "in_progress"
        assert row["last_run_at"] != ""
    print("  [OK] mark_in_progress: persists status and last_run_at")


def test_catalog_mark_success():
    """mark_success records correct last_success_date and rows_total."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        cat.mark_in_progress("600519", "ohlcv_daily")
        result = _make_fetch_result("600519", success=True, rows=10,
                                    last_date="2024-01-31")
        cat.mark_success("600519", "ohlcv_daily", result)

        row = cat.get_row("600519", "ohlcv_daily")
        assert row["status"] == "success"
        assert row["last_success_date"] == "2024-01-31"
        assert int(row["rows_total"]) == 10
        assert row["error_msg"] == ""
    print("  [OK] mark_success: records success status, last_success_date, rows_total")


def test_catalog_mark_failed():
    """mark_failed records failure status and error message."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        cat.mark_in_progress("600519", "ohlcv_daily")
        cat.mark_failed("600519", "ohlcv_daily", "HTTPError 403")

        row = cat.get_row("600519", "ohlcv_daily")
        assert row["status"] == "failed"
        assert "403" in row["error_msg"]
    print("  [OK] mark_failed: records failed status and error message")


def test_crash_recovery_resets_in_progress():
    """reset_in_progress: in_progress rows become pending on startup."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        # Simulate two symbols stuck in_progress (crash scenario)
        for sym in ("600519", "000858"):
            cat.mark_in_progress(sym, "ohlcv_daily")

        # A success row should NOT be reset
        cat.mark_in_progress("300750", "ohlcv_daily")
        result = _make_fetch_result("300750", success=True)
        cat.mark_success("300750", "ohlcv_daily", result)

        reset_count = cat.reset_in_progress()
        assert reset_count == 2, f"Expected 2 reset, got {reset_count}"

        for sym in ("600519", "000858"):
            row = cat.get_row(sym, "ohlcv_daily")
            assert row["status"] == "pending", f"{sym} should be pending after reset"

        row_success = cat.get_row("300750", "ohlcv_daily")
        assert row_success["status"] == "success", "Success row must not be reset"
    print("  [OK] reset_in_progress: resets 2 in_progress→pending, leaves success intact")


def test_needs_fetch_logic():
    """needs_fetch returns True/False correctly for all status combinations."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        end = dt.date(2024, 1, 31)

        # Never fetched → needs fetch
        assert cat.needs_fetch("600519", "ohlcv_daily", end) is True

        # in_progress → needs fetch (crash recovery)
        cat.mark_in_progress("600519", "ohlcv_daily")
        assert cat.needs_fetch("600519", "ohlcv_daily", end) is True

        # failed → needs fetch (retry)
        cat.mark_failed("600519", "ohlcv_daily", "network error")
        assert cat.needs_fetch("600519", "ohlcv_daily", end) is True

        # success with stale date → needs fetch
        result_stale = _make_fetch_result("600519", last_date="2024-01-15")
        cat.mark_success("600519", "ohlcv_daily", result_stale)
        assert cat.needs_fetch("600519", "ohlcv_daily", end) is True

        # success with current date → skip
        result_current = _make_fetch_result("600519", last_date="2024-01-31")
        cat.mark_success("600519", "ohlcv_daily", result_current)
        assert cat.needs_fetch("600519", "ohlcv_daily", end) is False
    print("  [OK] needs_fetch: correct True/False for all status+date combinations")


def test_symbols_needing_fetch_filter():
    """symbols_needing_fetch filters the list correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        end = dt.date(2024, 1, 31)
        symbols = ["600519", "000858", "300750"]

        # Mark 600519 as up-to-date
        cat.mark_in_progress("600519", "ohlcv_daily")
        cat.mark_success("600519", "ohlcv_daily",
                         _make_fetch_result("600519", last_date="2024-01-31"))

        to_fetch = cat.symbols_needing_fetch(symbols, "ohlcv_daily", end)
        assert "600519" not in to_fetch
        assert "000858" in to_fetch
        assert "300750" in to_fetch
    print("  [OK] symbols_needing_fetch: up-to-date symbol excluded, others included")


def test_catalog_upsert_is_idempotent():
    """Calling mark_success twice on the same symbol updates in place (no duplicate rows)."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        for last_date in ("2024-01-15", "2024-01-31"):
            cat.mark_in_progress("600519", "ohlcv_daily")
            cat.mark_success("600519", "ohlcv_daily",
                             _make_fetch_result("600519", last_date=last_date))

        df = cat.load()
        rows_for_symbol = df[
            (df["symbol"] == "600519") & (df["source"] == "ohlcv_daily")
        ]
        assert len(rows_for_symbol) == 1, \
            f"Expected 1 row for 600519, got {len(rows_for_symbol)}"
        assert rows_for_symbol.iloc[0]["last_success_date"] == "2024-01-31"
    print("  [OK] upsert is idempotent: mark_success twice → 1 row with latest date")


def test_catalog_summary_structure():
    """summary() returns a dict with expected keys."""
    with tempfile.TemporaryDirectory() as tmp:
        cat = _catalog(tmp)
        for sym, ok in [("600519", True), ("000858", False), ("300750", True)]:
            cat.mark_in_progress(sym, "ohlcv_daily")
            if ok:
                cat.mark_success(sym, "ohlcv_daily",
                                 _make_fetch_result(sym, success=True))
            else:
                cat.mark_failed(sym, "ohlcv_daily", "error")

        s = cat.summary()
        assert "total" in s
        assert "by_status" in s
        assert s["total"] == 3
        assert s["by_status"].get("success", 0) == 2
        assert s["by_status"].get("failed", 0) == 1
    print("  [OK] summary(): returns correct counts by status")


# ---------------------------------------------------------------------------
# CatalogDrivenCollector tests
# ---------------------------------------------------------------------------

def test_catalog_driven_collector_writes_catalog():
    """CatalogDrivenCollector.run() writes catalog entries for all symbols."""
    from quant_platform.ingest.catalog import CatalogDrivenCollector, CollectorCatalog

    symbols = ["600519", "000858"]

    def fake_fetch(symbol, start_date, end_date):
        return _make_ohlcv(symbol, n=5)

    with tempfile.TemporaryDirectory() as tmp:
        collector = CatalogDrivenCollector(
            store_root=tmp,
            universe_key="csi300",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        with patch("quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
                   side_effect=fake_fetch):
            summary = collector.run(symbols=symbols)

        assert summary.succeeded == 2
        assert summary.failed == 0

        cat = CollectorCatalog(tmp)
        for sym in symbols:
            row = cat.get_row(sym, "ohlcv_daily")
            assert row is not None, f"No catalog entry for {sym}"
            assert row["status"] == "success", f"{sym} status: {row['status']}"
    print("  [OK] CatalogDrivenCollector: run() writes success catalog entries")


def test_catalog_driven_collector_skips_up_to_date():
    """Second run skips symbols already up-to-date in the catalog."""
    from quant_platform.ingest.catalog import CatalogDrivenCollector

    symbols = ["600519"]

    def fake_fetch(symbol, start_date, end_date):
        return _make_ohlcv(symbol, n=5)

    with tempfile.TemporaryDirectory() as tmp:
        collector = CatalogDrivenCollector(
            store_root=tmp,
            universe_key="csi300",
            start_date="2024-01-01",
            end_date="2024-01-05",   # short range
        )
        with patch("quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
                   side_effect=fake_fetch):
            summary1 = collector.run(symbols=symbols)
        assert summary1.fetched == 1

        # Second run with same end_date → catalog says skip
        with patch("quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
                   side_effect=lambda *a, **k: (_ for _ in ()).throw(
                       AssertionError("Should not call fetch — symbol is up-to-date")
                   )):
            summary2 = collector.run(symbols=symbols)

        assert summary2.skipped_by_catalog == 1
        assert summary2.fetched == 0
    print("  [OK] CatalogDrivenCollector: second run skips up-to-date symbols")


def test_catalog_driven_crash_recovery():
    """in_progress rows are reset to pending and re-fetched on next run."""
    from quant_platform.ingest.catalog import CatalogDrivenCollector, CollectorCatalog

    symbols = ["600519"]

    def fake_fetch(symbol, start_date, end_date):
        return _make_ohlcv(symbol, n=5)

    with tempfile.TemporaryDirectory() as tmp:
        # Simulate a crash: manually write an in_progress row
        cat = CollectorCatalog(tmp)
        cat.mark_in_progress("600519", "ohlcv_daily")

        # Now run the collector — it should reset and re-fetch
        collector = CatalogDrivenCollector(
            store_root=tmp,
            universe_key="csi300",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        with patch("quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
                   side_effect=fake_fetch):
            summary = collector.run(symbols=symbols)

        assert summary.succeeded == 1
        row = cat.get_row("600519", "ohlcv_daily")
        assert row["status"] == "success"
    print("  [OK] CatalogDrivenCollector: crash recovery resets in_progress → re-fetches")


def test_catalog_driven_partial_failure_recorded():
    """Failed symbols are recorded in catalog as 'failed', others succeed."""
    from quant_platform.ingest.catalog import CatalogDrivenCollector, CollectorCatalog

    symbols = ["600519", "BAD_SYM"]

    def fake_fetch(symbol, start_date, end_date):
        if symbol == "BAD_SYM":
            return None
        return _make_ohlcv(symbol, n=5)

    with tempfile.TemporaryDirectory() as tmp:
        collector = CatalogDrivenCollector(
            store_root=tmp,
            universe_key="csi300",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        with patch("quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
                   side_effect=fake_fetch):
            summary = collector.run(symbols=symbols)

        assert summary.succeeded == 1
        assert summary.failed == 1
        assert "BAD_SYM" in summary.failed_symbols

        cat = CollectorCatalog(tmp)
        assert cat.get_row("600519", "ohlcv_daily")["status"] == "success"
        assert cat.get_row("BAD_SYM",  "ohlcv_daily")["status"] == "failed"
    print("  [OK] CatalogDrivenCollector: partial failure — catalog reflects per-symbol outcome")


if __name__ == "__main__":
    print("\n=== T0.6 Catalog tests ===\n")
    tests = [
        test_catalog_load_empty_when_no_file,
        test_catalog_mark_in_progress_and_read_back,
        test_catalog_mark_success,
        test_catalog_mark_failed,
        test_crash_recovery_resets_in_progress,
        test_needs_fetch_logic,
        test_symbols_needing_fetch_filter,
        test_catalog_upsert_is_idempotent,
        test_catalog_summary_structure,
        test_catalog_driven_collector_writes_catalog,
        test_catalog_driven_collector_skips_up_to_date,
        test_catalog_driven_crash_recovery,
        test_catalog_driven_partial_failure_recorded,
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
