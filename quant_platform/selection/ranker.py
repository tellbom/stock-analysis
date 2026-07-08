"""
quant_platform.selection.ranker
================================
IndustryNeutralRanker — post-inference industry-neutral ranking and selection.

Operates on an ALREADY-SCORED panel.  Does NOT retrain models, change
labels, or modify features.  Adds industry-neutral fields alongside
existing global-ranking fields.

Pipeline
--------
  1. rank()    → within-industry ranks + z-scores
  2. select()  → apply cross-industry selection strategy
  3. monitor() → exposure / concentration flags
  4. run()     → rank + select + monitor (convenience)
"""

from __future__ import annotations

import pandas as pd

from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.exposure import ExposureMonitor
from quant_platform.selection.strategies import (
    EqualTopKStrategy,
    HybridStrategy,
    ProportionalTopKStrategy,
    TurnoverAwareStrategy,
)

# Map StrategyType enum values → strategy instances
_STRATEGY_REGISTRY = {
    StrategyType.EQUAL_TOP_K: EqualTopKStrategy(),
    StrategyType.PROPORTIONAL_TOP_K: ProportionalTopKStrategy(),
    StrategyType.HYBRID: HybridStrategy(),
    StrategyType.TURNOVER_AWARE: TurnoverAwareStrategy(),
}


class IndustryNeutralRanker:
    """
    Post-inference industry-neutral ranking and stock selection.

    Parameters
    ----------
    config : SelectionConfig
    industry_col : str
        Panel column with industry code (e.g. "industry_code").
    name_col : str
        Panel column with human-readable industry name.
    score_col : str
        Panel column with raw model predictions.
    symbol_col : str
        Panel column with stock identifier.
    """

    def __init__(
        self,
        config: SelectionConfig,
        *,
        industry_col: str = "industry_code",
        name_col: str = "industry_name",
        score_col: str = "model_score",
        symbol_col: str = "symbol",
    ):
        self.config = config
        self.industry_col = industry_col
        self.name_col = name_col
        self.score_col = score_col
        self.symbol_col = symbol_col

    # ------------------------------------------------------------------
    # rank()
    # ------------------------------------------------------------------

    def rank(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute within-industry rankings.

        For each (date, industry_code) group:
          - ``industry_rank`` : integer rank descending by score
          - ``industry_neutral_score`` : z-score of score within industry
          - ``industry_size`` : number of stocks in that industry-date group
          - ``global_score`` : alias for the raw model score

        Stocks with missing / empty / ``_UNKNOWN`` industry are grouped
        into a single virtual industry.

        Returns *panel* with new columns appended.  All original columns
        are preserved.
        """
        df = panel.copy()

        # --- validate inputs ---
        if self.score_col not in df.columns:
            raise ValueError(
                f"score_col '{self.score_col}' not found in panel columns "
                f"{list(df.columns)}"
            )
        if self.symbol_col not in df.columns:
            raise ValueError(
                f"symbol_col '{self.symbol_col}' not found in panel columns "
                f"{list(df.columns)}"
            )

        # --- normalise industry labels ---
        ind_series = df[self.industry_col] if self.industry_col in df.columns else None
        if ind_series is None:
            raise ValueError(
                f"industry_col '{self.industry_col}' not found in panel. "
                f"Ensure build_industry_features() was called or set "
                f"include_industry=True in FeaturePipeline."
            )
        ind_series = ind_series.fillna(self.config.unknown_industry_label)
        ind_series = ind_series.replace("", self.config.unknown_industry_label)

        # Date column may or may not exist; handle both
        group_keys = ["date", ind_series] if "date" in df.columns else [ind_series]

        # --- within-industry rank (1 = highest score) ---
        df["industry_rank"] = (
            df.groupby(group_keys, observed=False)[self.score_col]
            .rank(method="average", ascending=False)
            .astype(int)
        )

        # --- within-industry z-score (fallback to percentile for <5 stocks) ---
        def _within_ind_zscore(grp: pd.Series) -> pd.Series:
            n = len(grp)
            if n < 5:
                # pct=True gives values in [0, 1]
                return grp.rank(pct=True)
            mu = grp.mean()
            sigma = grp.std(ddof=0)
            if sigma < 1e-12:
                return pd.Series(0.0, index=grp.index)
            return (grp - mu) / sigma

        df["industry_neutral_score"] = (
            df.groupby(group_keys, observed=False)[self.score_col]
            .transform(_within_ind_zscore)
        )

        # --- industry size ---
        df["industry_size"] = (
            df.groupby(group_keys, observed=False)[self.symbol_col]
            .transform("count")
        )

        # --- alias ---
        df["global_score"] = df[self.score_col]

        return df

    # ------------------------------------------------------------------
    # select()
    # ------------------------------------------------------------------

    def select(self, panel: pd.DataFrame, *, strategy=None) -> pd.DataFrame:
        """
        Apply the configured selection strategy.

        Requires *panel* to have been through :meth:`rank` (or at minimum
        contain the columns expected by the strategy).

        Parameters
        ----------
        strategy : SelectionStrategy | None
            Override the registry-resolved strategy for this call. Needed
            for stateful strategies (e.g. TurnoverAwareStrategy) that carry
            per-call state such as prior holdings -- the registry only
            holds one stateless default instance per StrategyType.

        Adds columns:
          ``selected`` (bool), ``selection_reason`` (str)
        """
        if strategy is None:
            strategy = _STRATEGY_REGISTRY.get(self.config.strategy)
        if strategy is None:
            raise ValueError(
                f"Unknown strategy: {self.config.strategy}. "
                f"Available: {list(_STRATEGY_REGISTRY)}"
            )

        selected_set, reasons = strategy.select(
            panel,
            self.config,
            industry_col=self.industry_col,
            score_col=self.score_col,
            symbol_col=self.symbol_col,
        )

        df = panel.copy()
        df["selected"] = df[self.symbol_col].isin(selected_set)
        df["selection_reason"] = reasons
        return df

    # ------------------------------------------------------------------
    # monitor()
    # ------------------------------------------------------------------

    def monitor(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Add exposure flags via :class:`ExposureMonitor`.

        Adds column ``exposure_flag`` (str).
        """
        df = panel.copy()
        df["exposure_flag"] = ExposureMonitor.flag(
            df,
            self.config,
            industry_col=self.industry_col,
            selected_col="selected",
        )
        return df

    # ------------------------------------------------------------------
    # run() — convenience pipeline
    # ------------------------------------------------------------------

    def run(self, panel: pd.DataFrame, *, strategy=None) -> pd.DataFrame:
        """
        Convenience: rank → select → monitor.

        *strategy* is forwarded to :meth:`select` (see its docstring).

        Returns *panel* with all new columns appended.
        """
        panel = self.rank(panel)
        panel = self.select(panel, strategy=strategy)
        panel = self.monitor(panel)
        return panel
