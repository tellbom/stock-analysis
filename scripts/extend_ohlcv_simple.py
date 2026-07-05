"""
extend_ohlcv_simple.py
======================
DEPRECATED (merge review) — this used to fetch OHLCV via a hand-rolled
akshare call and write straight to Parquet with
``combined.to_parquet(ohlcv_file, index=False)``, bypassing
``store.parquet_store.write_ohlcv`` entirely. That skips
``store.schemas.enforce_ohlcv``'s required-column check, dtype casting,
``(symbol, date)`` dedup, and atomic write -- a bad write here (partial
akshare response, duplicate/unsorted rows, wrong dtypes) would land
directly in the lake with nothing catching it, and every downstream
label/feature/training step assumes that lake data is already clean.
It also duplicated ``ingest.ohlcv_collector.collect_symbol``'s incremental
fetch + market-prefix logic, which already exists and is already correct.

Fix: this is now a thin pointer to ``fetch_latest_ohlcv.py``, which wraps
``ingest.ohlcv_collector.OHLCVCollector`` (safe, schema-enforced,
incremental, atomic write). Kept only so any existing call to
``python scripts/extend_ohlcv_simple.py`` still does the right thing --
prefer calling ``fetch_latest_ohlcv.py`` directly going forward.

Usage:
  cd E:/stock-analysis && PYTHONPATH=. python scripts/extend_ohlcv_simple.py
"""

from __future__ import annotations

import warnings

from fetch_latest_ohlcv import main as _fetch_latest_ohlcv_main


def main() -> None:
    warnings.warn(
        "extend_ohlcv_simple.py is deprecated and now just calls "
        "fetch_latest_ohlcv.py -- call that script directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _fetch_latest_ohlcv_main()


if __name__ == "__main__":
    main()
