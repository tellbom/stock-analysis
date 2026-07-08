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
  TurnoverAwareStrategy  — SR-02: rank-band hysteresis to bound one-way turnover
"""

from __future__ import annotations

import math
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


# ---------------------------------------------------------------------------
# Turnover-aware (SR-02)
# ---------------------------------------------------------------------------

class TurnoverAwareStrategy(SelectionStrategy):
    """
    Rank-band hysteresis selection: bounds one-way turnover vs a prior
    selection instead of re-selecting the top set from scratch each date.

    An incumbent (in ``prior_selected``) is kept unless its current global
    rank by ``score_col`` falls outside ``config.keep_rank``. A challenger
    (not currently held) is admitted only if its rank clears the tighter
    ``config.enter_rank``. The enter/exit asymmetry (enter_rank <=
    keep_rank) is the no-churn band. Net adds/drops are capped so one-way
    turnover never exceeds ``config.max_turnover``; if the cap would leave
    empty slots, the best-ranked incumbents that fell outside keep_rank are
    retained rather than dropped, keeping the selected set at ``max_total``.

    With an empty (or ``None``) ``prior_selected``, there is nothing to
    apply hysteresis against, so selection reduces to a plain global
    Top-``max_total`` by ``score_col`` -- i.e. the same outcome as
    EqualTopKStrategy would produce absent industry floors.

    Stateless per call: the caller supplies ``prior_selected`` explicitly
    (no hidden global state, no implicit "yesterday" lookup).
    """

    def __init__(self, prior_selected: set | None = None):
        self.prior_selected = prior_selected or set()

    def select(
        self,
        panel: pd.DataFrame,
        config: SelectionConfig,
        industry_col: str,
        score_col: str,
        symbol_col: str,
    ) -> tuple[set, pd.Series]:
        target_size = config.max_total
        reasons = pd.Series("not_selected", index=panel.index)

        rank_by_symbol: dict = dict(
            zip(
                panel[symbol_col],
                panel[score_col].rank(ascending=False, method="first").astype(int),
            )
        )

        if not self.prior_selected:
            top = panel.nlargest(target_size, score_col)
            selected = set(top[symbol_col])
            reasons.loc[top.index] = "cold_start_top_k"
            return selected, reasons

        allowed_changes = int(math.floor(config.max_turnover * target_size))

        incumbents_in_panel = sorted(
            ((rank_by_symbol[s], s) for s in self.prior_selected if s in rank_by_symbol)
        )
        kept = [s for r, s in incumbents_in_panel if r <= config.keep_rank]
        fell_out = sorted(
            (s for r, s in incumbents_in_panel if r > config.keep_rank),
            key=lambda s: rank_by_symbol[s],
        )

        if len(kept) > target_size:
            kept = sorted(kept, key=lambda s: rank_by_symbol[s])[:target_size]

        candidate_pool_sorted = sorted(
            (s for s in panel[symbol_col].unique() if s not in kept),
            key=lambda s: rank_by_symbol.get(s, float("inf")),
        )
        new_entrants = [
            s for s in candidate_pool_sorted
            if s not in self.prior_selected and rank_by_symbol.get(s, float("inf")) <= config.enter_rank
        ]

        remaining_slots = target_size - len(kept)
        additions_allowed = max(0, min(remaining_slots, allowed_changes, len(new_entrants)))
        additions = new_entrants[:additions_allowed]

        selected = set(kept) | set(additions)

        # Turnover cap left slots unfilled -- retain best-ranked incumbents
        # that fell outside keep_rank rather than dropping them, so the
        # selected set stays at target_size without extra churn.
        remaining_slots = target_size - len(selected)
        if remaining_slots > 0 and fell_out:
            retained = fell_out[:remaining_slots]
            selected |= set(retained)

        # Still short (e.g. prior smaller than target_size) -- fill from the
        # best-ranked remaining candidates, ignoring enter_rank/turnover cap
        # since there is no incumbent left to retain.
        remaining_slots = target_size - len(selected)
        if remaining_slots > 0:
            fillers = [s for s in candidate_pool_sorted if s not in selected]
            selected |= set(fillers[:remaining_slots])

        symbol_series = panel[symbol_col]
        reasons.loc[symbol_series.isin(kept)] = "kept_incumbent"
        reasons.loc[symbol_series.isin(additions)] = "new_entrant"
        reasons.loc[symbol_series.isin(selected - set(kept) - set(additions))] = "retained_over_turnover_cap"

        return selected, reasons
