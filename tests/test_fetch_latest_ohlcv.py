from __future__ import annotations

import datetime as dt


def test_fetch_latest_ohlcv_accepts_as_of():
    from scripts.fetch_latest_ohlcv import parse_args, resolve_dates

    args = parse_args(["--as-of", "2026-10-14"])
    target, fetch_end_date = resolve_dates(args)

    assert target == dt.date(2026, 10, 14)
    assert fetch_end_date == dt.date(2026, 10, 14)


def test_fetch_latest_ohlcv_min_date_backward_compat():
    from scripts.fetch_latest_ohlcv import parse_args, resolve_dates

    args = parse_args(["--min-date", "2026-07-08"])
    target, fetch_end_date = resolve_dates(args)

    assert target == dt.date(2026, 7, 8)
    assert fetch_end_date == dt.date(2026, 7, 8)


def test_fetch_latest_ohlcv_accepts_explicit_fetch_end_date():
    from scripts.fetch_latest_ohlcv import parse_args, resolve_dates

    args = parse_args(["--as-of", "2026-10-14", "--fetch-end-date", "2026-10-15"])
    target, fetch_end_date = resolve_dates(args)

    assert target == dt.date(2026, 10, 14)
    assert fetch_end_date == dt.date(2026, 10, 15)


def test_fetch_latest_ohlcv_defaults_to_today():
    from scripts.fetch_latest_ohlcv import parse_args, resolve_dates

    args = parse_args([])
    target, fetch_end_date = resolve_dates(args)

    assert target == dt.date.today()
    assert fetch_end_date == dt.date.today()


def test_fetch_latest_ohlcv_rejects_end_before_target():
    from scripts.fetch_latest_ohlcv import parse_args, resolve_dates

    import pytest

    args = parse_args(["--as-of", "2026-10-14", "--fetch-end-date", "2026-10-13"])
    with pytest.raises(ValueError, match="must be >="):
        resolve_dates(args)


def test_fetch_latest_ohlcv_force_flag():
    from scripts.fetch_latest_ohlcv import parse_args

    assert parse_args(["--force"]).force is True
    assert parse_args([]).force is False
