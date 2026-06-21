"""
core.market
===========
A-share market utilities: code normalisation, market detection.

Lifted from stock_full_report.py (detect_market) without modification to logic.
The public function returns (prefixed, market) as a named tuple for clarity.
"""

from __future__ import annotations

import datetime as dt
from typing import NamedTuple


class MarketInfo(NamedTuple):
    prefixed: str   # e.g. "sh600519"
    market: str     # "sh" | "sz" | "bj"


def detect_market(code: str) -> MarketInfo:
    """
    Return (prefixed, market) for a 6-digit A-share code.

    Raises ValueError for invalid codes — callers must handle this; no silent
    fallback or default is provided.

    Examples
    --------
    >>> detect_market("600519")
    MarketInfo(prefixed='sh600519', market='sh')
    >>> detect_market("000066")
    MarketInfo(prefixed='sz000066', market='sz')
    """
    code = code.strip()
    # Strip any existing prefix so the function is idempotent on prefixed inputs
    for prefix in ("sh", "sz", "bj"):
        if code.startswith(prefix) and len(code) == 8:
            code = code[2:]
            break

    if not code.isdigit() or len(code) != 6:
        raise ValueError(f"Invalid A-share code: {code!r} — must be 6 digits")

    if code.startswith(("60", "68", "11", "12", "5")):
        return MarketInfo(f"sh{code}", "sh")
    if code.startswith(("00", "30", "20", "15", "16", "18")):
        return MarketInfo(f"sz{code}", "sz")
    if code.startswith(("4", "8", "92")):
        return MarketInfo(f"bj{code}", "bj")
    # Default: treat as Shanghai (matches original logic)
    return MarketInfo(f"sh{code}", "sh")


def last_trade_day(d: dt.date | None = None) -> dt.date:
    """
    Return the most recent weekday on or before *d* (defaults to today).
    Does not account for public holidays — good enough for data-fetch scheduling.
    """
    if d is None:
        d = dt.date.today()
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= dt.timedelta(days=1)
    return d


if __name__ == "__main__":
    # Quick self-test
    cases = [
        ("600519", "sh600519", "sh"),
        ("000066", "sz000066", "sz"),
        ("300750", "sz300750", "sz"),
        ("sh600519", "sh600519", "sh"),   # idempotent on prefixed input
    ]
    all_ok = True
    for raw, want_prefixed, want_market in cases:
        result = detect_market(raw)
        ok = result.prefixed == want_prefixed and result.market == want_market
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] detect_market({raw!r}) → {result}")
        if not ok:
            all_ok = False

    try:
        detect_market("9999999")
        print("  [FAIL] should have raised ValueError for 7-digit code")
        all_ok = False
    except ValueError as e:
        print(f"  [OK] ValueError raised for invalid code: {e}")

    print("\nSelf-test:", "PASSED" if all_ok else "FAILED")
