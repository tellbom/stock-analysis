"""
P0 regression tests: QFQ scale-change detection (300803 audit).

The 2026-10-14 audit proved that an incremental tail append cannot detect a
QFQ scale change whose ex-rights date is outside the fetched window:
300803's stored history had 896 of 908 shared dates on the wrong scale,
while the last 12 days matched.

These tests use the committed offline fixtures — no network required.

Run: python -m pytest tests/test_qfq_scale_refresh.py -v
     (or run the file directly; it has a __main__ runner)
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIXTURES = Path(__file__).resolve().parents[1] / "docs/reviews/2026-10-14-weekly/qfq_audit/fixtures"


def _load_fixture(name: str, symbol: str = "300803") -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / name)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = symbol
    return df


def test_verify_qfq_scale_detects_tail_identical_history_different():
    """
    P0 core scenario: tail 12 days identical, 896 earlier dates differ.
    A tail-only overlap check would PASS; the full-overlap check must FAIL.
    """
    from quant_platform.ingest.ohlcv_collector import verify_qfq_scale

    old = _load_fixture("300803_old_ohlcv.csv")
    new = _load_fixture("300803_fresh_sina_qfq_full.csv")

    # Sanity: the audit's numbers
    assert len(old) == 908, f"expected 908 old rows, got {len(old)}"
    assert len(new) == 913, f"expected 913 new rows, got {len(new)}"

    consistent, mismatched, max_diff = verify_qfq_scale(old, new)

    assert not consistent, "MUST detect the QFQ scale change"
    assert len(mismatched) > 800, f"expected ~896 mismatched dates, got {len(mismatched)}"
    assert max_diff > 1.0, f"expected large max diff (old 82.00 vs new 56.55 scale), got {max_diff}"

    # The tail (last 12 days) must be among the MATCHING dates, proving that
    # a tail-only check would have missed this.
    tail_dates = set(pd.to_datetime(new["date"]).dt.date.sort_values().tail(12).tolist())
    assert not (tail_dates & set(mismatched)), "tail dates should match (post-ex-rights)"
    print(f"  [OK] full-overlap check detects scale change: {len(mismatched)} mismatched, max diff {max_diff:.2f}")


def test_verify_qfq_scale_passes_on_identical_data():
    """Same data in, consistent out — no false positives."""
    from quant_platform.ingest.ohlcv_collector import verify_qfq_scale

    new = _load_fixture("300803_fresh_sina_qfq_full.csv")
    consistent, mismatched, max_diff = verify_qfq_scale(new, new.copy())
    assert consistent and not mismatched and max_diff == 0.0
    print("  [OK] identical data passes the scale check")


def test_refresh_symbol_full_history_replaces_on_scale_change():
    """
    End-to-end: stored old-scale history + fresh full fetch with new scale
    → the stored file is REPLACED (not spliced), old file kept on failure.
    """
    from quant_platform.ingest.ohlcv_collector import refresh_symbol_full_history
    from quant_platform.store.parquet_store import read_ohlcv, write_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    symbol = "300803"
    old = _load_fixture("300803_old_ohlcv.csv", symbol)
    new = _load_fixture("300803_fresh_sina_qfq_full.csv", symbol)

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        write_ohlcv(old, ohlcv_path(tmp, symbol), symbol)
        before = read_ohlcv(ohlcv_path(tmp, symbol))
        assert len(before) == 908

        # Mock the network fetch to return the fresh full history
        with patch(
            "quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
            return_value=new,
        ):
            result = refresh_symbol_full_history(
                symbol, tmp,
                start_date=dt.date(2023, 1, 1),
                end_date=dt.date(2026, 10, 14),
            )

        assert result.success, f"expected success, got: {result.error}"
        assert result.error.startswith("scale_refreshed"), f"expected scale_refreshed, got: {result.error}"

        after = read_ohlcv(ohlcv_path(tmp, symbol))
        assert len(after) == 913, f"expected full 913-row replacement, got {len(after)}"
        # Spot-check: the previously-wrong 2026-09-18 close must be on the new scale
        row = after[pd.to_datetime(after["date"]).dt.date == dt.date(2026, 9, 18)]
        assert not row.empty
        assert abs(row.iloc[0]["close"] - 56.55) < 0.5, f"close not on new scale: {row.iloc[0]['close']}"
    print("  [OK] full-history refresh replaces old-scale history atomically")


def test_refresh_symbol_full_history_keeps_old_on_fetch_failure():
    """If the full-history refetch fails, the old file is kept untouched."""
    from quant_platform.ingest.ohlcv_collector import refresh_symbol_full_history
    from quant_platform.store.parquet_store import read_ohlcv, write_ohlcv
    from quant_platform.store.lake import ohlcv_path, init_lake

    symbol = "300803"
    old = _load_fixture("300803_old_ohlcv.csv", symbol)

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        write_ohlcv(old, ohlcv_path(tmp, symbol), symbol)

        with patch(
            "quant_platform.ingest.ohlcv_collector._fetch_symbol_raw",
            return_value=None,  # all endpoints fail
        ):
            result = refresh_symbol_full_history(
                symbol, tmp,
                start_date=dt.date(2023, 1, 1),
                end_date=dt.date(2026, 10, 14),
            )

        assert not result.success
        after = read_ohlcv(ohlcv_path(tmp, symbol))
        assert len(after) == 908, "old file must be kept untouched on failure"
    print("  [OK] refetch failure keeps the old file (no partial write)")


def test_ohlcv_fingerprint_changes_on_price_revision():
    """Feature-cache invalidation: fingerprint must change when prices change."""
    from quant_platform.features.pipeline import ohlcv_fingerprint

    old = _load_fixture("300803_old_ohlcv.csv")
    new = _load_fixture("300803_fresh_sina_qfq_full.csv")

    fp_old = ohlcv_fingerprint(old)
    fp_new = ohlcv_fingerprint(new)
    assert fp_old != fp_new, "fingerprint MUST change on QFQ scale revision"
    assert ohlcv_fingerprint(old) == fp_old, "fingerprint must be deterministic"
    print(f"  [OK] fingerprint changes on revision: {fp_old} -> {fp_new}")


def test_verify_feature_freshness_flags_stale_symbol():
    """verify_feature_freshness flags a symbol whose OHLCV was revised after caching."""
    from quant_platform.features.pipeline import (
        verify_feature_freshness, record_feature_provenance,
    )
    from quant_platform.store.parquet_store import write_ohlcv, write_parquet
    from quant_platform.store.lake import ohlcv_path, init_lake, feature_path

    symbol = "300803"
    old = _load_fixture("300803_old_ohlcv.csv", symbol)
    new = _load_fixture("300803_fresh_sina_qfq_full.csv", symbol)
    feature_set_id = "test1234"

    with tempfile.TemporaryDirectory() as tmp:
        init_lake(tmp)
        # 1. Write OLD ohlcv, build "features", record provenance
        write_ohlcv(old, ohlcv_path(tmp, symbol), symbol)
        fp = feature_path(tmp, feature_set_id, symbol)
        fp.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(old[["symbol", "date"]].copy(), fp)
        record_feature_provenance(tmp, feature_set_id, symbol, old)

        # 2. Fresh features: not stale
        assert verify_feature_freshness(tmp, feature_set_id, [symbol]) == []

        # 3. Revise OHLCV (simulating the QFQ fix landing)
        write_ohlcv(new, ohlcv_path(tmp, symbol), symbol)

        # 4. Now stale
        stale = verify_feature_freshness(tmp, feature_set_id, [symbol])
        assert stale == [symbol], f"expected [{symbol}] stale, got {stale}"
    print("  [OK] verify_feature_freshness flags revised-OHLCV symbols as stale")


if __name__ == "__main__":
    print("\n=== P0 QFQ scale-refresh regression tests (offline fixtures) ===\n")
    tests = [
        test_verify_qfq_scale_detects_tail_identical_history_different,
        test_verify_qfq_scale_passes_on_identical_data,
        test_refresh_symbol_full_history_replaces_on_scale_change,
        test_refresh_symbol_full_history_keeps_old_on_fetch_failure,
        test_ohlcv_fingerprint_changes_on_price_revision,
        test_verify_feature_freshness_flags_stale_symbol,
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
