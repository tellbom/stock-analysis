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
#
# NOTE ON NORMALISATION: features that come out of build_technical_features()
# are already normalised to dimensionless ratios (see features/technical.py
# _normalise_price_features).  The transform strings below reflect the *final*
# normalised form, not the raw indicator value.
TECHNICAL_SPECS: list[FeatureSpec] = [
    # Moving average distance ratios  (close/MA - 1, dimensionless)
    FeatureSpec("ma_5",   "technical", ("close",), 5,  "close/MA(5)-1",   warmup=5),
    FeatureSpec("ma_10",  "technical", ("close",), 10, "close/MA(10)-1",  warmup=10),
    FeatureSpec("ma_20",  "technical", ("close",), 20, "close/MA(20)-1",  warmup=20),
    FeatureSpec("ma_60",  "technical", ("close",), 60, "close/MA(60)-1",  warmup=60),
    # MACD / close  (dimensionless)
    FeatureSpec("macd_dif",  "technical", ("close",), 26, "(EMA12-EMA26)/close",  warmup=35),
    FeatureSpec("macd_dea",  "technical", ("close",), 35, "EMA(DIF,9)/close",     warmup=35),
    FeatureSpec("macd_hist", "technical", ("close",), 35, "(DIF-DEA)*2/close",    warmup=35),
    # KDJ — already dimensionless (0–100 range)
    FeatureSpec("kdj_k", "technical", ("high","low","close"), 9, "KDJ-K(9)",  warmup=9),
    FeatureSpec("kdj_d", "technical", ("high","low","close"), 9, "KDJ-D(9)",  warmup=9),
    FeatureSpec("kdj_j", "technical", ("high","low","close"), 9, "KDJ-J(9)",  warmup=9),
    # RSI — already dimensionless (0–100)
    FeatureSpec("rsi_6",  "technical", ("close",), 6,  "RSI(6)",  warmup=6),
    FeatureSpec("rsi_12", "technical", ("close",), 12, "RSI(12)", warmup=12),
    FeatureSpec("rsi_24", "technical", ("close",), 24, "RSI(24)", warmup=24),
    # Bollinger: boll_upper → %B=(close-lower)/(upper-lower); boll_mid → close/MA20-1
    # boll_lower is dropped (redundant after %B)
    FeatureSpec("boll_upper", "technical", ("close",), 20, "%B=(close-lower)/(upper-lower)", warmup=20),
    FeatureSpec("boll_mid",   "technical", ("close",), 20, "close/MA(20)-1",                warmup=20),
    # ATR / close  (dimensionless volatility ratio)
    FeatureSpec("atr_14",    "technical", ("high","low","close"), 14, "ATR(14)/close",   warmup=14),
    # ADX — already dimensionless (0–100)
    FeatureSpec("adx_14",    "technical", ("high","low","close"), 14, "ADX(14)",         warmup=28),
    # OBV → rolling z-score of OBV.diff()  (rate of change, dimensionless)
    FeatureSpec("obv",       "technical", ("close","volume"),      20, "zscore(OBV.diff(),20)", warmup=20),
    # CCI, ROC, WillR, Stoch — already dimensionless
    FeatureSpec("cci_14",    "technical", ("high","low","close"), 14, "CCI(14)",         warmup=14),
    FeatureSpec("roc_10",    "technical", ("close",),             10, "ROC(10)",         warmup=10),
    FeatureSpec("willr_14",  "technical", ("high","low","close"), 14, "WillR(14)",       warmup=14),
    FeatureSpec("stoch_k",   "technical", ("high","low","close"), 14, "Stoch-K(14)",     warmup=14),
    FeatureSpec("stoch_d",   "technical", ("high","low","close"), 14, "Stoch-D(14)",     warmup=17),
    # T3.1 (P?-reversal): short-term reversal counter-signal to momentum.
    # Negative of the trailing 3-day return — already dimensionless (a
    # return ratio), needs no further normalisation. High past 1-3d return
    # -> low (negative) reversal_3d value, i.e. "expect mean reversion".
    FeatureSpec("reversal_3d", "technical", ("close",), 3, "-(close/close.shift(3)-1)", warmup=3),
]

# Cross-sectional features — computed across the universe on each date.
# window=0 signals "no look-back; computed per-date across symbols".
#
# NOTE: cs_rank_close and cs_zscore_close are intentionally excluded.
# Raw close price is not cross-sectionally meaningful (high-price stocks
# always rank above low-price stocks regardless of signal).  Use the
# normalised technical features (ma_5, roc_10, etc.) for cross-section instead.
CROSS_SECTIONAL_SPECS: list[FeatureSpec] = [
    FeatureSpec("cs_rank_close",     "cross_sectional", ("close",),  0, "rank(close)/N",     warmup=0),
    FeatureSpec("cs_rank_volume",    "cross_sectional", ("volume",), 0, "rank(volume)/N",    warmup=0),
    FeatureSpec("cs_rank_rsi_6",     "cross_sectional", ("rsi_6",),  0, "rank(rsi_6)/N",     warmup=0),
    FeatureSpec("cs_rank_roc_10",    "cross_sectional", ("roc_10",), 0, "rank(roc_10)/N",    warmup=0),
    FeatureSpec("cs_rank_ma_5",      "cross_sectional", ("ma_5",),   0, "rank(ma_5_ratio)/N", warmup=0),
    FeatureSpec("cs_zscore_close",   "cross_sectional", ("close",),  0, "zscore(close)",     warmup=0),
    FeatureSpec("cs_zscore_volume",  "cross_sectional", ("volume",), 0, "zscore(volume)",    warmup=0),
    FeatureSpec("cs_zscore_rsi_6",   "cross_sectional", ("rsi_6",),  0, "zscore(rsi_6)",     warmup=0),
]

# Default full spec = technical + cross-sectional
DEFAULT_SPECS: list[FeatureSpec] = TECHNICAL_SPECS + CROSS_SECTIONAL_SPECS

# ---------------------------------------------------------------------------
# P4B extended spec lists (imported lazily to avoid circular deps)
# ---------------------------------------------------------------------------
# These are imported at the module level for convenience.  Each builder
# module (features.valuation, features.industry, features.flow,
# features.margin) defines its own spec list and registers it independently.
# The FULL_SPECS convenience constant bundles everything together.

def _valuation_specs_fallback() -> list[FeatureSpec]:
    return [
        FeatureSpec("cs_log_float_mcap", "valuation", ("float_mcap_yi",), 0, "minmax(log(float_mcap_yi))", warmup=0),
        FeatureSpec("cs_pe_ttm_rank", "valuation", ("pe_ttm",), 0, "pct_rank(pe_ttm; neg->0)", warmup=0),
        FeatureSpec("cs_pb_rank", "valuation", ("pb",), 0, "pct_rank(pb)", warmup=0),
        FeatureSpec("cs_turnover_rank", "valuation", ("turnover_pct",), 0, "pct_rank(turnover_pct)", warmup=0),
        FeatureSpec("cs_log_mcap_rank", "valuation", ("total_mcap_yi",), 0, "pct_rank(log(total_mcap_yi))", warmup=0),
        FeatureSpec("pe_momentum_5d", "valuation", ("pe_ttm",), 5, "pct_rank(PE_TTM.diff(5))", warmup=5),
    ]


def _get_p4b_specs() -> list[FeatureSpec]:
    """
    Return the combined P4B spec list: valuation + industry + flow + margin.
    Imported lazily so the registry module can be loaded even when the
    P4B feature modules are not yet on the import path.
    """
    specs: list[FeatureSpec] = []
    fallback_by_attr = {"VALUATION_SPECS": _valuation_specs_fallback}
    for module_name, attr in [
        ("quant_platform.features.valuation", "VALUATION_SPECS"),
        ("quant_platform.features.industry",  "INDUSTRY_SPECS"),
        ("quant_platform.features.flow",      "FLOW_SPECS"),
        ("quant_platform.features.sector_flow", "SECTOR_FLOW_SPECS"),
        ("quant_platform.features.margin",    "MARGIN_SPECS"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_name)
            module_specs = getattr(mod, attr, [])
        except ImportError:
            module_specs = []
        if not module_specs and attr in fallback_by_attr:
            module_specs = fallback_by_attr[attr]()
        specs.extend(module_specs)
    return specs


# Full spec set including all P4B data blocks.
# Use this as: registry.register(FULL_SPECS)
FULL_SPECS: list[FeatureSpec] = DEFAULT_SPECS + _get_p4b_specs()


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
