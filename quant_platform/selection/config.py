"""
quant_platform.selection.config
================================
Configuration for industry-neutral ranking and stock selection.

Single dataclass + enum — no dependencies on other selection modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StrategyType(str, Enum):
    """Cross-industry stock selection strategies."""

    EQUAL_TOP_K = "equal_top_k"
    PROPORTIONAL_TOP_K = "proportional_top_k"
    HYBRID = "hybrid"
    TURNOVER_AWARE = "turnover_aware"


@dataclass
class SelectionConfig:
    """
    Configuration for IndustryNeutralRanker.

    Parameters
    ----------
    strategy : StrategyType
        Which cross-industry selection strategy to apply.
    top_k : int
        Picks per industry for equal_top_k / baseline for proportional.
    max_total : int
        Hard cap on total selected stocks.
    hybrid_weight : float
        0 = all within-industry rank, 1 = all global score.
        Only used by HybridStrategy.
    min_stocks_per_industry : int
        Always pick at least this many per industry (prevents industries
        with very few stocks from being excluded entirely).
    exposure_warning_threshold : float
        Flag "industry_overweight" when any single industry exceeds this.
    exposure_diversified_threshold : float
        Flag "diversified" when all industries are below this.
    unknown_industry_label : str
        Label used for stocks with missing industry classification.
    enter_rank : int
        SR-02: a challenger not currently held is admitted only if its
        global rank (by score_col) is <= enter_rank. Only used by
        TurnoverAwareStrategy.
    keep_rank : int
        SR-02: an incumbent (already held) is kept unless its global rank
        falls outside (>) keep_rank. keep_rank >= enter_rank -- the
        enter/exit asymmetry is the no-churn band. Only used by
        TurnoverAwareStrategy.
    max_turnover : float
        SR-02: cap on one-way turnover per rebalance, i.e. the fraction of
        max_total slots that may change (adds == drops). Only used by
        TurnoverAwareStrategy.
    """

    strategy: StrategyType = StrategyType.EQUAL_TOP_K
    top_k: int = 3
    max_total: int = 50
    hybrid_weight: float = 0.5
    min_stocks_per_industry: int = 1
    exposure_warning_threshold: float = 0.30
    exposure_diversified_threshold: float = 0.15
    unknown_industry_label: str = "_UNKNOWN"
    enter_rank: int = 30
    keep_rank: int = 60
    max_turnover: float = 0.5

    def __post_init__(self) -> None:
        # Validate enum coercion
        if isinstance(self.strategy, str):
            self.strategy = StrategyType(self.strategy)

        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.max_total < 1:
            raise ValueError(f"max_total must be >= 1, got {self.max_total}")
        if not 0.0 <= self.hybrid_weight <= 1.0:
            raise ValueError(
                f"hybrid_weight must be in [0, 1], got {self.hybrid_weight}"
            )
        if self.min_stocks_per_industry < 1:
            raise ValueError(
                f"min_stocks_per_industry must be >= 1, "
                f"got {self.min_stocks_per_industry}"
            )
        if self.exposure_warning_threshold <= self.exposure_diversified_threshold:
            raise ValueError(
                f"exposure_warning_threshold ({self.exposure_warning_threshold}) "
                f"must be > exposure_diversified_threshold "
                f"({self.exposure_diversified_threshold})"
            )
        if self.enter_rank < 1 or self.keep_rank < 1:
            raise ValueError(
                f"enter_rank ({self.enter_rank}) and keep_rank ({self.keep_rank}) must be >= 1"
            )
        if self.enter_rank > self.keep_rank:
            raise ValueError(
                f"enter_rank ({self.enter_rank}) must be <= keep_rank ({self.keep_rank})"
            )
        if not 0.0 < self.max_turnover <= 1.0:
            raise ValueError(f"max_turnover must be in (0, 1], got {self.max_turnover}")
