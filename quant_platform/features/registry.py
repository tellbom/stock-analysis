"""
features.registry
=================
Feature-spec registry: declarative feature definitions and version hashing.

Every feature set is described as a list of FeatureSpec objects.
The list is hashed into a short ID (``feature_set_id``) so that re-running
with the same spec reproduces identical output and a different spec produces
a different output directory — no silent overwrites.

Design
------
- Declarative: each feature spec carries {name, family, inputs, window, transform}.
- Versioned:   the feature_set_id is a 8-char hex digest of the sorted spec list.
- Registered:  specs are persisted to ``features/feature_specs.parquet`` so any
  run can look up what columns belong to which feature_set_id.
- The registry does NOT hard-code a list of features; callers build their own
  FeatureSpec lists and register them.  Two built-in spec lists are provided
  as defaults: TECHNICAL_SPECS and CROSS_SECTIONAL_SPECS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from quant_platform.store.lake import feature_spec_path, features_root
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """
    Immutable description of one feature column.

    Attributes
    ----------
    name        : canonical column name in the output Parquet, e.g. "rsi_14"
    family      : "technical" | "cross_sectional" | "fundamental" | "custom"
    inputs      : tuple of raw column names consumed, e.g. ("close",)
    window      : look-back window in trading days (0 for cross-sectional)
    transform   : description of the computation, e.g. "RSI(14)"
    warmup      : number of rows at the start that are unreliable (to be masked)
    """
    name:      str
    family:    str
    inputs:    tuple[str, ...]
    window:    int
    transform: str
    warmup:    int = 0


# ---------------------------------------------------------------------------
# Built-in spec lists
# ---------------------------------------------------------------------------

# Technical features — all sourced from the existing technical_indicators.py
# plus pandas_ta_classic additions.  Warmup periods set conservatively.
TECHNICAL_SPECS: list[FeatureSpec] = [
    # Moving averages
    FeatureSpec("ma_5",   "technical", ("close",), 5,  "MA(5)",   warmup=5),
    FeatureSpec("ma_10",  "technical", ("close",), 10, "MA(10)",  warmup=10),
    FeatureSpec("ma_20",  "technical", ("close",), 20, "MA(20)",  warmup=20),
    FeatureSpec("ma_60",  "technical", ("close",), 60, "MA(60)",  warmup=60),
    # MACD
    FeatureSpec("macd_dif",  "technical", ("close",), 26, "EMA(12)-EMA(26)",     warmup=35),
    FeatureSpec("macd_dea",  "technical", ("close",), 35, "EMA(DIF,9)",          warmup=35),
    FeatureSpec("macd_hist", "technical", ("close",), 35, "(DIF-DEA)*2",         warmup=35),
    # KDJ
    FeatureSpec("kdj_k", "technical", ("high","low","close"), 9, "KDJ-K(9)",  warmup=9),
    FeatureSpec("kdj_d", "technical", ("high","low","close"), 9, "KDJ-D(9)",  warmup=9),
    FeatureSpec("kdj_j", "technical", ("high","low","close"), 9, "KDJ-J(9)",  warmup=9),
    # RSI
    FeatureSpec("rsi_6",  "technical", ("close",), 6,  "RSI(6)",  warmup=6),
    FeatureSpec("rsi_12", "technical", ("close",), 12, "RSI(12)", warmup=12),
    FeatureSpec("rsi_24", "technical", ("close",), 24, "RSI(24)", warmup=24),
    # Bollinger Bands
    FeatureSpec("boll_upper", "technical", ("close",), 20, "BOLL_UPPER(20,2)", warmup=20),
    FeatureSpec("boll_mid",   "technical", ("close",), 20, "BOLL_MID(20)",     warmup=20),
    FeatureSpec("boll_lower", "technical", ("close",), 20, "BOLL_LOWER(20,2)", warmup=20),
    # pandas_ta_classic additions
    FeatureSpec("atr_14",    "technical", ("high","low","close"), 14, "ATR(14)",    warmup=14),
    FeatureSpec("adx_14",    "technical", ("high","low","close"), 14, "ADX(14)",    warmup=28),
    FeatureSpec("obv",       "technical", ("close","volume"),      1, "OBV",        warmup=1),
    FeatureSpec("cci_14",    "technical", ("high","low","close"), 14, "CCI(14)",    warmup=14),
    FeatureSpec("roc_10",    "technical", ("close",),             10, "ROC(10)",    warmup=10),
    FeatureSpec("willr_14",  "technical", ("high","low","close"), 14, "WillR(14)",  warmup=14),
    FeatureSpec("stoch_k",   "technical", ("high","low","close"), 14, "Stoch-K(14)", warmup=14),
    FeatureSpec("stoch_d",   "technical", ("high","low","close"), 14, "Stoch-D(14)", warmup=17),
]

# Cross-sectional features — computed across the universe on each date.
# window=0 signals "no look-back; computed per-date across symbols".
CROSS_SECTIONAL_SPECS: list[FeatureSpec] = [
    FeatureSpec("cs_rank_close",     "cross_sectional", ("close",),  0, "rank(close)/N",     warmup=0),
    FeatureSpec("cs_rank_volume",    "cross_sectional", ("volume",), 0, "rank(volume)/N",    warmup=0),
    FeatureSpec("cs_rank_rsi_6",     "cross_sectional", ("rsi_6",),  0, "rank(rsi_6)/N",     warmup=0),
    FeatureSpec("cs_rank_roc_10",    "cross_sectional", ("roc_10",), 0, "rank(roc_10)/N",    warmup=0),
    FeatureSpec("cs_zscore_close",   "cross_sectional", ("close",),  0, "zscore(close)",     warmup=0),
    FeatureSpec("cs_zscore_volume",  "cross_sectional", ("volume",), 0, "zscore(volume)",    warmup=0),
]

# Default full spec = technical + cross-sectional
DEFAULT_SPECS: list[FeatureSpec] = TECHNICAL_SPECS + CROSS_SECTIONAL_SPECS


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

def compute_feature_set_id(specs: list[FeatureSpec]) -> str:
    """
    Compute a stable 8-char hex ID for a list of FeatureSpec objects.

    The ID is derived from the sorted canonical JSON of all specs so that:
    - Same specs → same ID (reproducible).
    - Different specs → different ID (no silent collision).
    - Order of specs in the list does not matter.
    """
    canonical = sorted(
        json.dumps(asdict(s), sort_keys=True) for s in specs
    )
    digest = hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
    return digest[:8]


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """
    Persist and query feature-set specs.

    Parameters
    ----------
    store_root : Path | str
    """

    def __init__(self, store_root: Path | str) -> None:
        self.store_root = Path(store_root)
        self._path = feature_spec_path(self.store_root)

    def register(self, specs: list[FeatureSpec]) -> str:
        """
        Register a spec list and return its feature_set_id.
        If the same ID already exists, this is a no-op (idempotent).
        """
        fset_id = compute_feature_set_id(specs)
        existing = self._load()

        if not existing.empty and (existing["feature_set_id"] == fset_id).any():
            logger.debug("Feature set %s already registered", fset_id)
            return fset_id

        rows = [
            {
                "feature_set_id": fset_id,
                "name":           s.name,
                "family":         s.family,
                "inputs":         json.dumps(list(s.inputs)),
                "window":         s.window,
                "transform":      s.transform,
                "warmup":         s.warmup,
            }
            for s in specs
        ]
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        self._save(combined)
        logger.info("Registered feature set %s (%d specs)", fset_id, len(specs))
        return fset_id

    def get_specs(self, feature_set_id: str) -> list[FeatureSpec]:
        """Return the FeatureSpec list for a given feature_set_id."""
        df = self._load()
        rows = df[df["feature_set_id"] == feature_set_id]
        if rows.empty:
            raise KeyError(
                f"Feature set '{feature_set_id}' not found in registry. "
                "Call register() first."
            )
        return [
            FeatureSpec(
                name=r["name"], family=r["family"],
                inputs=tuple(json.loads(r["inputs"])),
                window=int(r["window"]), transform=r["transform"],
                warmup=int(r["warmup"]),
            )
            for _, r in rows.iterrows()
        ]

    def list_feature_sets(self) -> list[str]:
        df = self._load()
        if df.empty:
            return []
        return df["feature_set_id"].unique().tolist()

    def _load(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(columns=[
                "feature_set_id","name","family","inputs","window","transform","warmup"
            ])
        return pd.read_parquet(self._path)

    def _save(self, df: pd.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._path, index=False)
