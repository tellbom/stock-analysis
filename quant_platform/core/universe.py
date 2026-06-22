"""
core.universe
=============
Universe configuration abstraction.

T0.1 scope: define the UniverseConfig dataclass and the registry of known
universe keys.  Actual constituent fetching (with effective-date tables) is
implemented in ingest/universe_service.py (T0.2).

Design decisions
----------------
- Universe is NOT hard-coded to CSI 300; callers pass a UniverseConfig.
- First version may only have *current* constituents; the config carries a flag
  `has_effective_dates` so downstream code and quality reports know whether
  survivorship bias is present.
- Adding a new universe (e.g. CSI 500) requires only a new entry in UNIVERSE_REGISTRY
  — no code changes elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

UniverseKey = Literal["csi300", "csi500", "csi1000", "hs100", "custom", "all_a_share"]


@dataclass(frozen=True)
class UniverseConfig:
    """
    Immutable description of a universe.

    Attributes
    ----------
    key : str
        Canonical identifier, e.g. "csi300".
    display_name : str
        Human-readable label for reports.
    index_code : str | None
        Index code passed to AKShare constituent APIs, or None for custom/all.
    has_effective_dates : bool
        True if the universe service stores historical membership with in/out dates.
        False means current-constituents-only → survivorship bias present.
    survivorship_note : str
        Included verbatim in quality reports when has_effective_dates is False.
    """

    key: str
    display_name: str
    index_code: str | None
    has_effective_dates: bool = False
    survivorship_note: str = (
        "Universe is based on current constituents only. "
        "Historical membership changes are not tracked. "
        "Results may be subject to survivorship bias. "
        "To fix: populate the universe membership table with effective in/out dates."
    )


# ---------------------------------------------------------------------------
# Registry — add new universes here, nowhere else
# ---------------------------------------------------------------------------

UNIVERSE_REGISTRY: dict[str, UniverseConfig] = {
    "csi300": UniverseConfig(
        key="csi300",
        display_name="CSI 300 (沪深300)",
        index_code="000300",
        has_effective_dates=False,  # upgraded to True once T0.2 stores history
    ),
    "csi500": UniverseConfig(
        key="csi500",
        display_name="CSI 500 (中证500)",
        index_code="000905",
        has_effective_dates=False,
    ),
    "csi1000": UniverseConfig(
        key="csi1000",
        display_name="CSI 1000 (中证1000)",
        index_code="000852",
        has_effective_dates=False,
    ),
    "hs100": UniverseConfig(
        key="hs100",
        display_name="HS100",
        index_code=None,
        has_effective_dates=False,
    ),
    "all_a_share": UniverseConfig(
        key="all_a_share",
        display_name="All A-shares",
        index_code=None,
        has_effective_dates=False,
    ),
}


def get_universe(key: str) -> UniverseConfig:
    """
    Return a UniverseConfig by key.

    Raises KeyError with a clear message for unknown keys — never silently
    falls back to a default universe.
    """
    if key not in UNIVERSE_REGISTRY:
        known = ", ".join(sorted(UNIVERSE_REGISTRY))
        raise KeyError(
            f"Unknown universe key {key!r}. Known universes: {known}. "
            "To add a new universe, register it in core/universe.py."
        )
    return UNIVERSE_REGISTRY[key]
