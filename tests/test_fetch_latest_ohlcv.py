from __future__ import annotations

import datetime as dt


def test_fetch_latest_ohlcv_accepts_explicit_min_date():
    from scripts.fetch_latest_ohlcv import parse_args, resolve_dates

    args = parse_args(["--min-date", "2026-07-08"])
    min_date, fetch_end_date = resolve_dates(args)

    assert min_date == dt.date(2026, 7, 8)
    assert fetch_end_date == dt.date(2026, 7, 11)


def test_fetch_latest_ohlcv_accepts_explicit_fetch_end_date():
    from scripts.fetch_latest_ohlcv import parse_args, resolve_dates

    args = parse_args(["--min-date", "2026-07-08", "--fetch-end-date", "2026-07-09"])
    min_date, fetch_end_date = resolve_dates(args)

    assert min_date == dt.date(2026, 7, 8)
    assert fetch_end_date == dt.date(2026, 7, 9)
