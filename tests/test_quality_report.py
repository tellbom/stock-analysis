"""
T0.8 verification tests for the quality report.

Tests cover: each check function in isolation, the full aggregator,
and the text-file output.  All use synthetic lake data — no network.

Run with:  PYTHONPATH=/home/claude python quant_platform/tests/test_quality_report.py
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers: build a synthetic lake
# ---------------------------------------------------------------------------

def _build_minimal_lake(
    tmp: str,
    universe_key: str = "csi300",
    symbols: list[str] | None = None,
    ohlcv_rows: int = 10,
    introduce_gap: bool = False,
    introduce_dup: bool = False,
    bad_price: bool = False,
) -> None:
    """
    Populate a temporary store_root with synthetic data covering all layers.
    """
    from quant_platform.store.lake import init_lake, ohlcv_path
    from quant_platform.store.parquet_store import write_ohlcv
    from quant_platform.ingest.calendar_service import CalendarService
    from quant_platform.ingest.universe_service import UniverseService
    import quant_platform.ingest.calendar_service as calendar_service

    symbols = symbols or ["600519", "000858"]
    init_lake(tmp)
    root = Path(tmp)

    # Universe
    csv_path = root / "cons.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "name"])
        for s in symbols:
            w.writerow([s, f"Stock_{s}"])
    svc = UniverseService(universe_key, tmp)
    svc.load_from_csv(csv_path, has_effective_dates=False)

    # Calendar
    original_akshare = calendar_service._build_from_akshare
    calendar_service._build_from_akshare = lambda start, end: None
    try:
        CalendarService(tmp).build_and_save(start="2024-01-01", end="2024-03-31")
    finally:
        calendar_service._build_from_akshare = original_akshare

    # OHLCV
    dates = pd.date_range("2024-01-02", periods=ohlcv_rows, freq="B")
    for sym in symbols:
        df = pd.DataFrame({
            "symbol": sym,
            "date":   [d.date() for d in dates],
            "open":   100.0, "high": 105.0,
            "low":    95.0,  "close": 102.0 if not bad_price else -1.0,
            "volume": 100_000.0,
        })
        if introduce_dup:
            df = pd.concat([df, df.iloc[:1]], ignore_index=True)
            # Write raw to bypass enforce_ohlcv dedup so the check can detect it
            path = ohlcv_path(tmp, sym)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
        else:
            write_ohlcv(df, ohlcv_path(tmp, sym), sym)


def _build_catalog(tmp: str, symbols: list[str],
                   status: str = "success") -> None:
    from quant_platform.ingest.catalog import CollectorCatalog
    from quant_platform.ingest.ohlcv_collector import FetchResult

    cat = CollectorCatalog(tmp)
    for sym in symbols:
        cat.mark_in_progress(sym, "ohlcv_daily")
        if status == "success":
            cat.mark_success(sym, "ohlcv_daily", FetchResult(
                symbol=sym, success=True, rows_new=10, rows_total=10,
                last_date=dt.date(2024, 1, 31),
            ))
        elif status == "failed":
            cat.mark_failed(sym, "ohlcv_daily", "test error")


def _build_fundamentals(tmp: str, symbol: str = "600519",
                        heuristic: bool = False) -> None:
    from quant_platform.store.lake import fundamentals_dir
    fund_dir = Path(tmp) / "silver" / "fundamentals"
    fund_dir.mkdir(parents=True, exist_ok=True)

    rows = [{
        "symbol":        symbol,
        "announce_date": "2023-10-28" if not heuristic else "2023-11-14",
        "period_end":    "2023-09-30",
        "period_type":   "Q3",
        "source":        "yjkb_em" if not heuristic else "financial_abstract_sina_heuristic",
        "revenue":       1_000_000.0,
    }]
    df = pd.DataFrame(rows)
    df.to_parquet(fund_dir / f"{symbol}.parquet", index=False)


# ---------------------------------------------------------------------------
# Individual check tests
# ---------------------------------------------------------------------------

def test_check_universe_missing_file():
    """check_universe raises ERROR when membership file is absent."""
    from quant_platform.store.quality_report import QualityReport, check_universe

    with tempfile.TemporaryDirectory() as tmp:
        report = QualityReport()
        check_universe(report, Path(tmp), "csi300")
        errors = [f for f in report.findings if f.severity == "ERROR"]
        assert errors, "Expected an ERROR for missing membership file"
        assert "csi300" in errors[0].message
    print("  [OK] check_universe: ERROR when membership Parquet absent")


def test_check_universe_survivorship_flag():
    """check_universe emits WARN when has_effective_dates=False."""
    from quant_platform.store.quality_report import QualityReport, check_universe

    with tempfile.TemporaryDirectory() as tmp:
        _build_minimal_lake(tmp)
        report = QualityReport()
        check_universe(report, Path(tmp), "csi300")
        warns = [f for f in report.findings
                 if f.severity == "WARN" and "SURVIVORSHIP" in f.message]
        assert warns, "Expected a WARN for survivorship bias"
        assert "current constituents" in warns[0].message.lower() or \
               "survivorship" in warns[0].message.lower()
    print("  [OK] check_universe: WARN emitted for survivorship bias (no effective dates)")


def test_check_calendar_happy_path():
    """check_calendar passes with INFO findings for a valid calendar."""
    from quant_platform.store.quality_report import QualityReport, check_calendar
    from quant_platform.ingest.calendar_service import CalendarService
    import quant_platform.ingest.calendar_service as calendar_service

    with tempfile.TemporaryDirectory() as tmp:
        original_akshare = calendar_service._build_from_akshare
        calendar_service._build_from_akshare = lambda start, end: None
        try:
            CalendarService(tmp).build_and_save(start="2024-01-01", end="2024-06-30")
        finally:
            calendar_service._build_from_akshare = original_akshare
        report = QualityReport()
        check_calendar(report, Path(tmp))

        errors = [f for f in report.findings if f.severity == "ERROR"]
        assert not errors, f"Unexpected errors: {errors}"
        info_msgs = " ".join(f.message for f in report.findings)
        assert "Weekend invariant" in info_msgs
    print("  [OK] check_calendar: no errors for valid calendar, weekend invariant confirmed")


def test_check_calendar_missing_file():
    """check_calendar emits ERROR when calendar Parquet is absent."""
    from quant_platform.store.quality_report import QualityReport, check_calendar

    with tempfile.TemporaryDirectory() as tmp:
        report = QualityReport()
        check_calendar(report, Path(tmp))
        errors = [f for f in report.findings if f.severity == "ERROR"]
        assert errors
    print("  [OK] check_calendar: ERROR when calendar Parquet absent")


def test_check_ohlcv_no_files():
    """check_ohlcv emits WARN (not ERROR) when no OHLCV files exist."""
    from quant_platform.store.quality_report import QualityReport, check_ohlcv

    with tempfile.TemporaryDirectory() as tmp:
        report = QualityReport()
        check_ohlcv(report, Path(tmp), "csi300")
        warns = [f for f in report.findings if f.severity == "WARN"]
        assert warns
        assert "No OHLCV" in warns[0].message
    print("  [OK] check_ohlcv: WARN (not ERROR) when no OHLCV files found")


def test_check_ohlcv_price_sanity_error():
    """check_ohlcv emits ERROR for rows with close <= 0."""
    from quant_platform.store.quality_report import QualityReport, check_ohlcv

    with tempfile.TemporaryDirectory() as tmp:
        _build_minimal_lake(tmp, bad_price=True)
        report = QualityReport()
        check_ohlcv(report, Path(tmp), "csi300", sample_symbols=["600519"])
        errors = [f for f in report.findings
                  if f.severity == "ERROR" and "close" in f.message.lower()]
        assert errors, f"Expected price ERROR, findings: {report.findings}"
    print("  [OK] check_ohlcv: ERROR for rows with close <= 0")


def test_check_ohlcv_duplicate_detection():
    """check_ohlcv emits ERROR for duplicate (symbol, date) rows."""
    from quant_platform.store.quality_report import QualityReport, check_ohlcv

    with tempfile.TemporaryDirectory() as tmp:
        _build_minimal_lake(tmp, introduce_dup=True)
        report = QualityReport()
        check_ohlcv(report, Path(tmp), "csi300", sample_symbols=["600519"])
        errors = [f for f in report.findings
                  if f.severity == "ERROR" and "duplicate" in f.message.lower()]
        assert errors, f"Expected duplicate ERROR, findings: {report.findings}"
    print("  [OK] check_ohlcv: ERROR for duplicate (symbol, date) pairs")


def test_check_ohlcv_clean_data_no_errors():
    """check_ohlcv passes with no errors for clean synthetic data."""
    from quant_platform.store.quality_report import QualityReport, check_ohlcv

    with tempfile.TemporaryDirectory() as tmp:
        _build_minimal_lake(tmp, ohlcv_rows=15)
        report = QualityReport()
        check_ohlcv(report, Path(tmp), "csi300",
                    sample_symbols=["600519", "000858"])
        errors = [f for f in report.findings if f.severity == "ERROR"]
        assert not errors, f"Unexpected errors on clean data: {errors}"
    print("  [OK] check_ohlcv: no errors for clean synthetic OHLCV data")


def test_check_catalog_success_state():
    """check_catalog reports INFO for all-success catalog."""
    from quant_platform.store.quality_report import QualityReport, check_catalog

    with tempfile.TemporaryDirectory() as tmp:
        _build_catalog(tmp, ["600519", "000858"], status="success")
        report = QualityReport()
        check_catalog(report, Path(tmp))
        errors = [f for f in report.findings if f.severity == "ERROR"]
        warns  = [f for f in report.findings
                  if f.severity == "WARN" and "FAILED" in f.message.upper()]
        assert not errors
        assert not warns
    print("  [OK] check_catalog: INFO-only findings for all-success catalog")


def test_check_catalog_failed_symbols():
    """check_catalog emits WARN listing failed symbols."""
    from quant_platform.store.quality_report import QualityReport, check_catalog

    with tempfile.TemporaryDirectory() as tmp:
        _build_catalog(tmp, ["600519"], status="failed")
        report = QualityReport()
        check_catalog(report, Path(tmp))
        warns = [f for f in report.findings if f.severity == "WARN"]
        assert any("600519" in w.message for w in warns), \
            f"Expected 600519 in WARN, got: {[w.message for w in warns]}"
    print("  [OK] check_catalog: WARN lists failed symbol names")


def test_check_fundamentals_heuristic_warn():
    """check_fundamentals emits WARN for heuristic announce_date rows."""
    from quant_platform.store.quality_report import QualityReport, check_fundamentals

    with tempfile.TemporaryDirectory() as tmp:
        _build_fundamentals(tmp, heuristic=True)
        report = QualityReport()
        check_fundamentals(report, Path(tmp), sample_symbols=["600519"])
        warns = [f for f in report.findings
                 if f.severity == "WARN" and "heuristic" in f.message.lower()]
        assert warns, f"Expected heuristic WARN, findings: {report.findings}"
        assert "45 days" in warns[0].message
    print("  [OK] check_fundamentals: WARN for heuristic announce_date rows")


def test_check_fundamentals_pit_violation():
    """check_fundamentals emits ERROR when announce_date < period_end."""
    from quant_platform.store.quality_report import QualityReport, check_fundamentals
    from quant_platform.store.lake import fundamentals_dir

    with tempfile.TemporaryDirectory() as tmp:
        # Write a row where announce_date is BEFORE period_end — impossible
        fund_dir = Path(tmp) / "silver" / "fundamentals"
        fund_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{
            "symbol":        "600519",
            "announce_date": "2023-09-01",   # before period ends
            "period_end":    "2023-09-30",   # period ends AFTER announce — violation
            "period_type":   "Q3",
            "source":        "test",
        }])
        df.to_parquet(fund_dir / "600519.parquet", index=False)

        report = QualityReport()
        check_fundamentals(report, Path(tmp), sample_symbols=["600519"])
        errors = [f for f in report.findings
                  if f.severity == "ERROR" and "announce_date" in f.message]
        assert errors, f"Expected PIT violation ERROR, findings: {report.findings}"
    print("  [OK] check_fundamentals: ERROR for announce_date < period_end (PIT violation)")


def test_check_fundamentals_no_files():
    """check_fundamentals emits INFO (not ERROR) when no files exist."""
    from quant_platform.store.quality_report import QualityReport, check_fundamentals

    with tempfile.TemporaryDirectory() as tmp:
        report = QualityReport()
        check_fundamentals(report, Path(tmp))
        errors = [f for f in report.findings if f.severity == "ERROR"]
        assert not errors, "Fundamentals absence should not be an ERROR"
        infos = [f for f in report.findings if "fundamental" in f.message.lower()]
        assert infos
    print("  [OK] check_fundamentals: INFO (not ERROR) when no fundamentals files found")


# ---------------------------------------------------------------------------
# Full aggregator test
# ---------------------------------------------------------------------------

def test_run_quality_report_clean_lake():
    """run_quality_report on a clean lake: no errors, report file written."""
    from quant_platform.store.quality_report import run_quality_report

    with tempfile.TemporaryDirectory() as tmp:
        _build_minimal_lake(tmp, ohlcv_rows=15)
        _build_catalog(tmp, ["600519", "000858"], status="success")

        report = run_quality_report(
            store_root=tmp,
            universe_key="csi300",
            sample_symbols=["600519", "000858"],
            write_file=True,
        )

        # Report file must exist
        assert (Path(tmp) / "quality_report.txt").exists()

        # No ERROR-level findings for clean data
        if report.errors:
            print("  [WARN] Unexpected errors on clean lake:")
            for e in report.errors:
                print(f"         {e}")
        assert not report.errors, f"Expected no errors, got: {report.errors}"

        # Summary stats populated
        assert "universe_total_rows" in report.stats
        assert "calendar"            in report.stats
        assert "ohlcv_total_rows"    in report.stats
    print("  [OK] run_quality_report: clean lake → no errors, report file written")


def test_run_quality_report_has_errors_flag():
    """QualityReport.has_errors is True when an ERROR finding exists."""
    from quant_platform.store.quality_report import run_quality_report

    with tempfile.TemporaryDirectory() as tmp:
        # Build lake with bad price data → triggers ERROR
        _build_minimal_lake(tmp, bad_price=True)
        report = run_quality_report(
            store_root=tmp,
            universe_key="csi300",
            sample_symbols=["600519"],
            write_file=False,
        )
        assert report.has_errors, "Expected has_errors=True for bad price data"
    print("  [OK] QualityReport.has_errors=True when bad price data detected")


def test_report_text_file_content():
    """Written quality_report.txt contains key sections."""
    from quant_platform.store.quality_report import run_quality_report

    with tempfile.TemporaryDirectory() as tmp:
        _build_minimal_lake(tmp)
        run_quality_report(store_root=tmp, universe_key="csi300",
                           write_file=True)

        text = (Path(tmp) / "quality_report.txt").read_text(encoding="utf-8")
        for keyword in ("QUALITY REPORT", "universe", "calendar", "ohlcv"):
            assert keyword.lower() in text.lower(), \
                f"Missing section '{keyword}' in report text"
    print("  [OK] quality_report.txt contains all required sections")


if __name__ == "__main__":
    print("\n=== T0.8 Quality Report tests ===\n")
    tests = [
        test_check_universe_missing_file,
        test_check_universe_survivorship_flag,
        test_check_calendar_happy_path,
        test_check_calendar_missing_file,
        test_check_ohlcv_no_files,
        test_check_ohlcv_price_sanity_error,
        test_check_ohlcv_duplicate_detection,
        test_check_ohlcv_clean_data_no_errors,
        test_check_catalog_success_state,
        test_check_catalog_failed_symbols,
        test_check_fundamentals_heuristic_warn,
        test_check_fundamentals_pit_violation,
        test_check_fundamentals_no_files,
        test_run_quality_report_clean_lake,
        test_run_quality_report_has_errors_flag,
        test_report_text_file_content,
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
