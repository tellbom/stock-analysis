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

# ---------------------------------------------------------------------------
# 科技板块映射（风控层）
# ---------------------------------------------------------------------------
# 依据 2026-10-14 裁决：科技板块 bet 是 D3/D5 主要亏损源（半导体 -5.7%、
# 数字芯片设计 -7.7% 等），3 只/行业的粒度防不住板块级暴露。
# 清单来自 docs/reviews/2026-09-30-controlled/all_picks.csv 的实际行业名。
# 匹配时用 industry_name 精确匹配。
TECH_SECTOR_INDUSTRIES = frozenset({
    "半导体",
    "半导体材料",
    "半导体设备",
    "数字芯片设计",
    "通信设备",
    "通信应用增值服务",
    "元件",
    "电子",
    "消费电子",
    "消费电子零部件及组装",
    "IT服务Ⅲ",
    "软件开发",
    "横向通用软件",
    "垂直应用软件",
    "其他计算机设备",
})


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

        # 风控层：科技板块合计上限（2026-10-14 裁决）。
        # 对所有策略一视同仁，不区分 arm。这是风控不是策略，
        # 不影响 ordinary 组的对照纯洁性。
        selected_set, reasons = self._apply_sector_cap(
            panel, selected_set, reasons,
        )

        df = panel.copy()
        df["selected"] = df[self.symbol_col].isin(selected_set)
        df["selection_reason"] = reasons
        return df

    def _apply_sector_cap(
        self,
        panel: pd.DataFrame,
        selected: set,
        reasons: pd.Series,
    ) -> tuple[set, pd.Series]:
        """
        科技板块合计上限过滤。

        若选中股票中科技大类（TECH_SECTOR_INDUSTRIES）数量超过
        config.sector_cap_tech，按分数从低到高剔除超出部分（保留高分）。
        sector_cap_tech=0 时不启用。
        """
        cap = self.config.sector_cap_tech
        if cap <= 0 or not selected:
            return selected, reasons

        # 用 name_col（industry_name）匹配科技板块
        name_col = self.name_col if self.name_col in panel.columns else None
        if name_col is None:
            return selected, reasons

        selected_df = panel[panel[self.symbol_col].isin(selected)]
        is_tech = selected_df[name_col].isin(TECH_SECTOR_INDUSTRIES)
        tech_df = selected_df[is_tech]

        n_tech = len(tech_df)
        if n_tech <= cap:
            return selected, reasons

        # 按分数从低到高剔除超出部分
        n_drop = n_tech - cap
        drop_df = tech_df.nsmallest(n_drop, self.score_col)
        drop_set = set(drop_df[self.symbol_col])

        selected = selected - drop_set
        reasons.loc[panel[self.symbol_col].isin(drop_set)] = "trimmed_by_sector_cap_tech"
        return selected, reasons

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
