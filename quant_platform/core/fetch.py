"""
core.fetch
==========
Retry/back-off wrapper for AKShare and other flaky callables.

Promoted from stock_full_report._safe_call. Fails loudly on errors and never
returns fabricated data.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd


_TRANSIENT_KEYWORDS = ("Connection", "Timeout", "Disconnected", "Proxy", "RemoteDisconnected")
_DEFAULT_BACKOFF = (0.5, 1.5, 3.0)


def safe_call(
    fn: Callable,
    *args: Any,
    retries: int = 3,
    label: str = "",
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    Call *fn* with *args*/*kwargs*, retrying transient errors.

    Returns the callable result on success, or None if the call fails. This
    function never returns fabricated or stub data.
    """
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            rows = len(result) if isinstance(result, pd.DataFrame) else "?"
            print(f"  [OK] {label:40s} {rows} rows - {elapsed:.1f}s")
            return result
        except Exception as exc:
            brief = f"{type(exc).__name__}: {exc}"[:90]
            is_transient = any(k in str(exc) for k in _TRANSIENT_KEYWORDS)

            if attempt < retries and is_transient:
                wait = _DEFAULT_BACKOFF[min(attempt, len(_DEFAULT_BACKOFF) - 1)]
                print(
                    f"  [RETRY] {label:40s} transient error, "
                    f"retry {attempt + 1}/{retries} in {wait}s - {brief}"
                )
                time.sleep(wait)
                continue

            print(f"  [FAIL] {label:40s} {brief}")
            return None

    return None
