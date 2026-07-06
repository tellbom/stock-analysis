"""
quant_platform.selection
========================
Post-inference industry-neutralized ranking and stock selection.

This module operates on ALREADY-SCORED panels.  It does NOT retrain
models, change labels, or modify features.  Existing global-ranking
fields (rank, model_score, score_percentile) are preserved untouched
alongside the new industry-neutral fields.

Components
----------
  config.py        — SelectionConfig, StrategyType
  ranker.py        — IndustryNeutralRanker (within-industry ranking)
  strategies.py    — SelectionStrategy ABC + concrete implementations
  exposure.py      — ExposureMonitor (concentration analysis)

Quick start
-----------
    from quant_platform.selection import IndustryNeutralRanker, SelectionConfig

    cfg = SelectionConfig(strategy="equal_top_k", top_k=3, max_total=50)
    ranker = IndustryNeutralRanker(cfg)
    enriched = ranker.run(scored_panel)
    # enriched now has: industry_rank, industry_neutral_score,
    #   selected, selection_reason, exposure_flag, ...
"""

from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.exposure import ExposureMonitor
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.selection.gate_fusion import (
    GateFusionConfig,
    gate_first_fusion,
    write_gate_fusion_outputs,
)
from quant_platform.selection.strategies import (
    EqualTopKStrategy,
    HybridStrategy,
    ProportionalTopKStrategy,
)

__all__ = [
    "SelectionConfig",
    "StrategyType",
    "IndustryNeutralRanker",
    "EqualTopKStrategy",
    "ProportionalTopKStrategy",
    "HybridStrategy",
    "ExposureMonitor",
    "GateFusionConfig",
    "gate_first_fusion",
    "write_gate_fusion_outputs",
]
