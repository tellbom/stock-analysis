"""
quant_platform.selection.strategies
====================================
Cross-industry stock selection strategies.

Each strategy implements the SelectionStrategy ABC and returns
(set of selected symbols, reason Series) given a scored panel.

Strategies:
  EqualTopKStrategy      — top K per industry (maximises diversity)
  ProportionalTopKStrategy — proportional to industry size
  HybridStrategy         — weighted combination of ind-neutral + global score
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from quant_platform.selection.config import SelectionConfig


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SelectionStrategy(ABC):
    """Protocol for cross-industry stock selection strategies."""

    @abstractmethod
    def select(
        self,
        panel: pd.DataFrame,
        config: SelectionConfig,
        industry_col: str,
        score_col: str,
        symbol_col: str,
    ) -> tuple[set, pd.Series]:
        """
        Select stocks from *panel*.

        Returns
        -------
        (selected_symbols, reason_series)
          selected_symbols : set of symbol values (str)
          reason_series    : pd.Series aligned to panel.index
        """
        ...


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _trim_to_max_total(
    panel: pd.DataFrame,
    selected: set,
    score_col: str,
    symbol_col: str,
    max_total: int,
    reasons: pd.Series,
    trim_label: str = "trimmed_by_cap",
) -> tuple[set, pd.Series]:
    """If len(selected) > max_total, keep the highest-scoring max_total."""
    if len(selected) <= max_total:
        return selected, reasons

    selected_df = panel[panel[symbol_col].isin(selected)]
    keep_df = selected_df.nlargest(max_total, score_col)
    keep_set = set(keep_df[symbol_col])
    dropped = selected - keep_set
    reasons.loc[panel[symbol_col].isin(dropped)] = trim_label
    return keep_set, reasons


# ---------------------------------------------------------------------------
# Equal Top-K
# ---------------------------------------------------------------------------

class EqualTopKStrategy(SelectionStrategy):
    """
    Pick top K stocks (by score) from each industry.

    If an industry has fewer than K stocks, pick all of them.
    The global *max_total* cap trims the lowest-scoring picks.
    """

    def select(
        self,
        panel: pd.DataFrame,
        config: SelectionConfig,
        industry_col: str,
        score_col: str,
        symbol_col: str,
    ) -> tuple[set, pd.Series]:
        reasons = pd.Series("not_selected", index=panel.index)
        selected: set = set()

        industry_series = panel[industry_col].fillna(config.unknown_industry_label)
        industry_series = industry_series.replace("", config.unknown_industry_label)

        for _ind_label, grp in panel.groupby(industry_series, observed=False):
            top = grp.nlargest(config.top_k, score_col)
            selected.update(top[symbol_col].tolist())
            reasons.loc[top.index] = "industry_top_k"

        # Apply global cap
        selected, reasons = _trim_to_max_total(
            panel, selected, score_col, symbol_col,
            config.max_total, reasons,
        )
        return selected, reasons


# ---------------------------------------------------------------------------
# Proportional Top-K
# ---------------------------------------------------------------------------

class ProportionalTopKStrategy(SelectionStrategy):
    """
    Allocate picks proportionally to industry size.

    k_i = max(min_stocks_per_industry,
              ceil(top_k * size_i / mean_industry_size))

    The *max_total* cap trims excess picks by score.
    """

    def select(
        self,
        panel: pd.DataFrame,
        config: SelectionConfig,
        industry_col: str,
        score_col: str,
        symbol_col: str,
    ) -> tuple[set, pd.Series]:
        reasons = pd.Series("not_selected", index=panel.index)
        selected: set = set()

        industry_series = panel[industry_col].fillna(config.unknown_industry_label)
        industry_series = industry_series.replace("", config.unknown_industry_label)

        industry_sizes = industry_series.value_counts()
        mean_size = industry_sizes.mean()

        for ind_label, size in industry_sizes.items():
            k_i = max(
                config.min_stocks_per_industry,
                int(round(config.top_k * size / max(mean_size, 1))),
            )
            grp = panel[industry_series == ind_label]
            top = grp.nlargest(k_i, score_col)
            selected.update(top[symbol_col].tolist())
            reasons.loc[top.index] = "industry_top_proportional"

        selected, reasons = _trim_to_max_total(
            panel, selected, score_col, symbol_col,
            config.max_total, reasons,
        )
        return selected, reasons


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------

class HybridStrategy(SelectionStrategy):
    """
    Combine within-industry rank and global score:

        hybrid_score = (1-w) * ind_neutral_score_norm + w * global_score_pct

    Pick top *max_total* stocks, with per-industry floor at
    *min_stocks_per_industry*.

    Requires that *industry_neutral_score* is already in *panel*
    (i.e. rank() was called before select()).
    """

    def select(
        self,
        panel: pd.DataFrame,
        config: SelectionConfig,
        industry_col: str,
        score_col: str,
        symbol_col: str,
    ) -> tuple[set, pd.Series]:
        if "industry_neutral_score" not in panel.columns:
            raise ValueError(
                "HybridStrategy requires 'industry_neutral_score' column. "
                "Call IndustryNeutralRanker.rank() before select()."
            )

        industry_series = panel[industry_col].fillna(config.unknown_industry_label)
        industry_series = industry_series.replace("", config.unknown_industry_label)

        w = config.hybrid_weight
        ind_z = panel["industry_neutral_score"].clip(-3, 3) / 3  # normalise
        global_pct = panel[score_col].rank(pct=True)

        panel = panel.copy()
        panel["_hybrid_score"] = (1.0 - w) * ind_z + w * global_pct

        selected: set = set()

        # Per-industry floor
        for _ind_label, grp in panel.groupby(industry_series, observed=False):
            if len(grp) <= config.min_stocks_per_industry:
                selected.update(grp[symbol_col].tolist())
            else:
                top = grp.nlargest(config.min_stocks_per_industry, "_hybrid_score")
                selected.update(top[symbol_col].tolist())

        # Fill remaining slots
        remaining = config.max_total - len(selected)
        if remaining > 0:
            candidates = panel[~panel[symbol_col].isin(selected)]
            fill = candidates.nlargest(remaining, "_hybrid_score")
            selected.update(fill[symbol_col].tolist())

        reasons = pd.Series("not_selected", index=panel.index)
        reasons.loc[panel[symbol_col].isin(selected)] = "hybrid_selected"

        return selected, reasons
