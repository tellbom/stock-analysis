"""
T0.4 verification tests for the storage layer.

Covers: lake layout, schema enforcement, Parquet read/write, DuckDB views.

Run with:  PYTHONPATH=/home/claude python quant_platform/tests/test_store.py
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_ohlcv(symbol: str = "600519", n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "symbol": symbol,
        "date":   [d.date() for d in dates],
        "open":   [100.0 + i for i in range(n)],
        "high":   [105.0 + i for i in range(n)],
        "low":    [ 95.0 + i for i in range(n)],
        "close":  [102.0 + i for i in range(n)],
        "volume": [100_000.0 + i * 1000 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Lake layout
# ---------------------------------------------------------------------------

def test_init_lake_creates_directories():
    """init_lake creates all expected directories."""
    from quant_platform.store.lake import init_lake
    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        root = Path(tmp)
        for sub in ("bronze", "silver/ohlcv", "silver/adj_factor",
                    "silver/fundamentals", "universe", "calendar", "catalog"):
            assert (root / sub).is_dir(), f"Missing directory: {sub}"
    print("  [OK] init_lake creates all required directories")


def test_lake_paths_are_consistent():
    """Path helpers return stable, non-overlapping paths."""
    from quant_platform.store.lake import (
        ohlcv_path, adj_factor_path, fundamentals_path,
        bronze_path, catalog_path, calendar_path,
    )
    root = Path("fake_root")
    p_ohlcv  = ohlcv_path(root, "600519")
    p_adj    = adj_factor_path(root, "600519")
    p_fund   = fundamentals_path(root, "600519")
    p_bronze = bronze_path(root, "600519", "akshare_daily", "2024-01-15")
    p_cat    = catalog_path(root)
    p_cal    = calendar_path(root)

    paths = [p_ohlcv, p_adj, p_fund, p_bronze, p_cat, p_cal]
    assert len(set(str(p) for p in paths)) == len(paths), "Path collision detected"
    assert all(p.parts[:len(root.parts)] == root.parts for p in paths), "Paths must be under root"
    assert p_bronze.suffix == ".parquet"
    assert "bronze" in str(p_bronze)
    assert "silver" in str(p_ohlcv)
    print("  [OK] lake path helpers return consistent, non-overlapping paths")


# ---------------------------------------------------------------------------
# Schema enforcement
# ---------------------------------------------------------------------------

def test_enforce_ohlcv_happy_path():
    """enforce_ohlcv accepts a valid DataFrame and normalises it."""
    from quant_platform.store.schemas import enforce_ohlcv
    df = _sample_ohlcv("600519", n=3)
    clean = enforce_ohlcv(df, "600519")
    assert list(clean["symbol"]) == ["600519"] * 3
    assert all(isinstance(d, dt.date) for d in clean["date"])
    assert clean["date"].is_monotonic_increasing
    print("  [OK] enforce_ohlcv: valid DataFrame accepted and normalised")


def test_enforce_ohlcv_missing_column_raises():
    """enforce_ohlcv raises ValueError if a required column is missing."""
    from quant_platform.store.schemas import enforce_ohlcv
    df = _sample_ohlcv().drop(columns=["close"])
    try:
        enforce_ohlcv(df, "600519")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "close" in str(e)
    print("  [OK] enforce_ohlcv: missing required column raises ValueError")


def test_enforce_ohlcv_deduplicates():
    """enforce_ohlcv drops duplicate (symbol, date) rows, keeps first."""
    from quant_platform.store.schemas import enforce_ohlcv
    df = _sample_ohlcv("600519", n=3)
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)  # add duplicate first row
    assert len(df) == 4
    clean = enforce_ohlcv(df, "600519")
    assert len(clean) == 3, f"Expected 3 rows after dedup, got {len(clean)}"
    print("  [OK] enforce_ohlcv: duplicate (symbol, date) rows are deduplicated")


def test_enforce_fundamentals_requires_both_dates():
    """enforce_fundamentals raises if announce_date or period_end is missing."""
    from quant_platform.store.schemas import enforce_fundamentals
    df = pd.DataFrame({
        "symbol": ["600519"],
        "announce_date": ["2024-04-28"],
        # period_end missing
    })
    try:
        enforce_fundamentals(df, "600519")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "period_end" in str(e)
        assert "announce_date" in str(e) or "PIT" in str(e)
    print("  [OK] enforce_fundamentals: missing period_end raises with PIT warning")


# ---------------------------------------------------------------------------
# Parquet read/write
# ---------------------------------------------------------------------------

def test_write_read_ohlcv_roundtrip():
    """write_ohlcv → read_ohlcv gives back identical data."""
    from quant_platform.store.parquet_store import write_ohlcv, read_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        path = ohlcv_path(tmp, "600519")
        df_in = _sample_ohlcv("600519", n=10)

        write_ohlcv(df_in, path, "600519")
        df_out = read_ohlcv(path)

        assert len(df_out) == 10
        assert list(df_out.columns[:7]) == ["symbol","date","open","high","low","close","volume"]
        assert all(isinstance(d, dt.date) for d in df_out["date"])
        assert df_out["close"].tolist() == [102.0 + i for i in range(10)]
    print("  [OK] write_ohlcv / read_ohlcv: round-trip preserves data and types")


def test_read_ohlcv_missing_file_returns_empty():
    """read_ohlcv returns empty DataFrame (not error) when file absent."""
    from quant_platform.store.parquet_store import read_ohlcv

    with tempfile.TemporaryDirectory() as tmp:
        df = read_ohlcv(Path(tmp) / "nonexistent.parquet")
        assert df.empty
        assert "symbol" in df.columns
    print("  [OK] read_ohlcv: absent file returns empty DataFrame with correct columns")


def test_write_ohlcv_is_atomic():
    """Atomic write: target file is either absent or complete, never partial."""
    from quant_platform.store.parquet_store import write_ohlcv, read_ohlcv
    from quant_platform.store.lake import init_lake

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        path = Path(tmp) / "silver" / "ohlcv" / "600519.parquet"

        # Write initial data
        write_ohlcv(_sample_ohlcv("600519", n=5), path, "600519")
        assert path.exists()

        # Write again (overwrite) — target should be valid at all times
        write_ohlcv(_sample_ohlcv("600519", n=8), path, "600519")
        df = read_ohlcv(path)
        assert len(df) == 8, f"Expected 8 rows after overwrite, got {len(df)}"
    print("  [OK] write_ohlcv: atomic overwrite produces valid file")


def test_read_ohlcv_range_filter():
    """read_ohlcv_range filters rows by date range correctly."""
    from quant_platform.store.parquet_store import write_ohlcv, read_ohlcv_range
    from quant_platform.store.lake import init_lake

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        path = Path(tmp) / "silver" / "ohlcv" / "600519.parquet"
        write_ohlcv(_sample_ohlcv("600519", n=10), path, "600519")

        df = read_ohlcv_range(path, start=dt.date(2024, 1, 3), end=dt.date(2024, 1, 5))
        assert not df.empty
        assert df["date"].min() >= dt.date(2024, 1, 3)
        assert df["date"].max() <= dt.date(2024, 1, 5)
    print("  [OK] read_ohlcv_range: date filter applied correctly")


# ---------------------------------------------------------------------------
# DuckDB views
# ---------------------------------------------------------------------------

def test_get_connection_returns_connection():
    """get_connection returns a usable DuckDB connection."""
    from quant_platform.store.duckdb_views import get_connection
    with tempfile.TemporaryDirectory() as tmp:
        con = get_connection(tmp)
        result = con.execute("SELECT 42 AS answer").fetchone()
        assert result[0] == 42
        con.close()
    print("  [OK] get_connection returns usable DuckDB connection")


def test_ohlcv_view_empty_when_no_data():
    """ohlcv view is queryable (returns 0 rows) when no Parquet files exist."""
    from quant_platform.store.duckdb_views import get_connection
    with tempfile.TemporaryDirectory() as tmp:
        con = get_connection(tmp)
        count = con.execute("SELECT count(*) FROM ohlcv").fetchone()[0]
        assert count == 0
        con.close()
    print("  [OK] ohlcv view: empty but queryable before data is ingested")


def test_ohlcv_view_queries_data():
    """ohlcv view returns data after Parquet files are written."""
    from quant_platform.store.duckdb_views import get_connection
    from quant_platform.store.parquet_store import write_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)

        # Write two symbols
        for sym in ("600519", "000858"):
            write_ohlcv(_sample_ohlcv(sym, n=5), ohlcv_path(tmp, sym), sym)

        con = get_connection(tmp)
        count = con.execute("SELECT count(*) FROM ohlcv").fetchone()[0]
        assert count == 10, f"Expected 10 rows, got {count}"

        syms = sorted(con.execute(
            "SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol"
        ).fetchdf()["symbol"].tolist())
        assert syms == ["000858", "600519"]
        con.close()
    print("  [OK] ohlcv view: queries data correctly across multiple symbols")


def test_query_convenience_helper():
    """query() one-shot helper opens, queries, and closes connection."""
    from quant_platform.store.duckdb_views import query
    from quant_platform.store.parquet_store import write_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        write_ohlcv(_sample_ohlcv("600519", n=3), ohlcv_path(tmp, "600519"), "600519")

        df = query(tmp, "SELECT symbol, close FROM ohlcv ORDER BY date")
        assert len(df) == 3
        assert list(df["symbol"]) == ["600519"] * 3
    print("  [OK] query() convenience helper returns correct DataFrame")


def test_duckdb_window_function_on_view():
    """DuckDB window functions work on the ohlcv view (needed for T1.3)."""
    from quant_platform.store.duckdb_views import get_connection
    from quant_platform.store.parquet_store import write_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    def _sample_ohlcv_offset(symbol: str, base_close: float, n: int = 3) -> pd.DataFrame:
        """Variant with distinct close values so ranks are unambiguous."""
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        return pd.DataFrame({
            "symbol": symbol,
            "date":   [d.date() for d in dates],
            "open":   base_close, "high": base_close + 5.0,
            "low":    base_close - 5.0, "close": base_close,
            "volume": 100_000.0,
        })

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        for sym, base in (("600519", 200.0), ("000858", 100.0), ("300750", 150.0)):
            write_ohlcv(_sample_ohlcv_offset(sym, base), ohlcv_path(tmp, sym), sym)

        con = get_connection(tmp)
        df = con.execute("""
            SELECT symbol, date, close,
                   rank() OVER (PARTITION BY date ORDER BY close DESC) AS rank_close
            FROM ohlcv
            ORDER BY date, rank_close
        """).fetchdf()
        # Every date should have 3 rows ranked 1, 2, 3
        for date, grp in df.groupby("date"):
            assert sorted(grp["rank_close"].tolist()) == [1, 2, 3], \
                f"Date {date}: unexpected ranks {grp['rank_close'].tolist()}"
        con.close()
    print("  [OK] DuckDB window functions work correctly on ohlcv view")


if __name__ == "__main__":
    print("\n=== T0.4 Storage Layer tests ===\n")
    tests = [
        test_init_lake_creates_directories,
        test_lake_paths_are_consistent,
        test_enforce_ohlcv_happy_path,
        test_enforce_ohlcv_missing_column_raises,
        test_enforce_ohlcv_deduplicates,
        test_enforce_fundamentals_requires_both_dates,
        test_write_read_ohlcv_roundtrip,
        test_read_ohlcv_missing_file_returns_empty,
        test_write_ohlcv_is_atomic,
        test_read_ohlcv_range_filter,
        test_get_connection_returns_connection,
        test_ohlcv_view_empty_when_no_data,
        test_ohlcv_view_queries_data,
        test_query_convenience_helper,
        test_duckdb_window_function_on_view,
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
