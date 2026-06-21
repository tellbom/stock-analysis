"""
core.fetch
==========
Retry/back-off wrapper for AKShare (and any flaky callable).

Promoted from stock_full_report._safe_call without logic changes.
Fails loudly on non-transient errors; never returns fabricated data.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd


# Errors that are considered transient and worth retrying
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
    Call *fn* with *args*/*kwargs*, retrying on transient errors up to *retries* times.

    Returns
    -------
    pd.DataFrame | None
        The result of *fn* on success, or None if all attempts fail.
        Never returns fabricated / stub data.

    Behaviour
    ---------
    - Transient errors (connection/timeout): sleep and retry.
    - Non-transient errors: log and return None immediately (no retry).
    - Prints a one-line summary for every attempt (success or failure).
    """
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            rows = len(result) if isinstance(result, pd.DataFrame) else "?"
            print(f"  ✓ {label:40s} {rows} rows · {elapsed:.1f}s")
            return result
        except Exception as exc:
            brief = f"{type(exc).__name__}: {exc}"[:90]
            is_transient = any(k in str(exc) for k in _TRANSIENT_KEYWORDS)

            if attempt < retries and is_transient:
                wait = _DEFAULT_BACKOFF[min(attempt, len(_DEFAULT_BACKOFF) - 1)]
                print(f"  ↺ {label:40s} transient error, retry {attempt+1}/{retries} in {wait}s — {brief}")
                time.sleep(wait)
                continue

            # Non-transient or out of retries — fail loudly, never fabricate
            print(f"  ✗ {label:40s} {brief}")
            return None

    return None  # unreachable but satisfies type checkers
