"""
scripts/run_industry_neutral_backtest.py
==========================================
T4.3 — Cost-aware short-horizon backtest at the chosen horizon (3 or 5,
per T2.3's verdict), on the INDUSTRY-NEUTRAL ranked/selected portfolio
(T1.2's IndustryNeutralRanker output), using the existing
evaluation.backtest.run_backtest unmodified.

Two complementary views (task.md says "on the industry-neutral selected
portfolio"; both readings below are useful and cheap to compute together
-- neither is an optimisation target, both are reported as the arbiter):

(a) PRIMARY: run_backtest with pred_col="industry_neutral_score" instead
    of the raw global "model_score". Because industry_neutral_score is a
    within-industry z-score (IndustryNeutralRanker.rank()), the resulting
    decile long/short legs draw from many industries by construction --
    this measures whether routing the backtest signal itself through
    T1.2's industry-neutral ranking (not just the discrete "selected"
    flag) changes the cost-aware verdict vs. the old global-score
    backtest.

(b) SUPPLEMENTARY: realised, net-of-cost performance of the literal
    "selected" book from IndustryNeutralRanker.select() -- a long-only
    diagnostic of the actual tradeable basket, with non-overlapping
    rebalancing at the chosen horizon and a simple turnover estimate.
    This does NOT re-run run_backtest's decile bucketing (bucketing an
    already-curated ~30-50 name book into deciles would be a strange use
    of a machinery built for a full cross-section); it directly computes
    mean/median return, net of cost_bps round-trip, on non-overlapping
    rebalancing dates.

Reused as-is, unmodified: evaluation.backtest.run_backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.evaluation.backtest import run_backtest, BacktestResult

logger = get_logger(__name__)


@dataclass
class SelectedBookResult:
    n_rebalance_dates: int = 0
    mean_return_gross: float = float("nan")
    mean_return_net: float = float("nan")
    median_return_gross: float = float("nan")
    hit_rate: float = float("nan")
    avg_names_per_rebalance: float = float("nan")
    est_one_way_turnover: float = float("nan")  # fraction of names that change between rebalances


def _selected_book_diagnostic(
    ranked_panel: pd.DataFrame,
    return_col: str,
    horizon: int,
    cost_bps: float,
    symbol_col: str = "symbol",
    selected_col: str = "selected",
) -> SelectedBookResult:
    """Long-only realised performance of the literal `selected` basket."""
    df = ranked_panel[ranked_panel[selected_col]].copy()
    if df.empty:
        return SelectedBookResult()

    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    # Non-overlapping rebalancing dates at the chosen horizon -- same
    # subsampling principle run_backtest uses internally for horizon > 1.
    rebalance_dates = dates[::horizon] if horizon > 1 else dates

    sub = df[df["date"].isin(rebalance_dates)]
    if sub.empty or sub[return_col].dropna().empty:
        return SelectedBookResult()

    gross = sub[return_col].dropna()
    one_way_cost = cost_bps / 10_000.0
    net = gross - 2 * one_way_cost  # round-trip: enter + exit

    # Turnover proxy: fraction of names NOT repeated between consecutive
    # rebalancing dates (crude but cheap; a full turnover calc needs
    # position-level weights, which this long-only equal-weight book
    # doesn't carry beyond membership).
    names_by_date = [set(df[df["date"] == d][symbol_col]) for d in rebalance_dates]
    turnovers = []
    for prev, cur in zip(names_by_date[:-1], names_by_date[1:]):
        if not cur:
            continue
        changed = len(cur - prev)
        turnovers.append(changed / len(cur))

    return SelectedBookResult(
        n_rebalance_dates=len(rebalance_dates),
        mean_return_gross=float(gross.mean()),
        mean_return_net=float(net.mean()),
        median_return_gross=float(gross.median()),
        hit_rate=float((gross > 0).mean()),
        avg_names_per_rebalance=float(np.mean([len(s) for s in names_by_date])) if names_by_date else float("nan"),
        est_one_way_turnover=float(np.mean(turnovers)) if turnovers else float("nan"),
    )


def run_industry_neutral_backtest(
    ranked_panel: pd.DataFrame,
    return_col: str,
    horizon: int,
    cost_bps: float = 10.0,
    industry_neutral_score_col: str = "industry_neutral_score",
) -> tuple[BacktestResult, SelectedBookResult]:
    """
    T4.3 entry point. `ranked_panel` must already be the output of
    IndustryNeutralRanker.run() (T1.2) -- i.e. it has industry_rank,
    industry_neutral_score, selected, exposure_flag, etc.
    """
    required = {industry_neutral_score_col, "selected", return_col, "date", "symbol"}
    missing = required - set(ranked_panel.columns)
    if missing:
        raise ValueError(
            f"run_industry_neutral_backtest: panel missing {missing}. "
            f"Did you run it through IndustryNeutralRanker.run() (T1.2) first?"
        )

    print("\n" + "=" * 70)
    print(f"T4.3 (a) PRIMARY: run_backtest on industry_neutral_score, horizon={horizon}")
    print("=" * 70)
    bt_input = ranked_panel.dropna(subset=[industry_neutral_score_col, return_col])
    bt_result = run_backtest(
        bt_input, pred_col=industry_neutral_score_col, return_col=return_col,
        cost_bps=cost_bps, horizon=horizon,
    )
    print(f"  Sharpe (net):            {bt_result.sharpe:+.4f}")
    print(f"  Gross long-short spread: {bt_result.long_minus_short:+.4f}")
    print(f"  Net long-short spread:   {bt_result.net_long_minus_short:+.4f}")
    print(f"  Avg daily turnover:      {bt_result.avg_daily_turnover:.4f}")
    print(f"  Max drawdown:            {bt_result.max_drawdown:+.4f}")

    print("\n" + "=" * 70)
    print("T4.3 (b) SUPPLEMENTARY: realised performance of the literal 'selected' book")
    print("=" * 70)
    book_result = _selected_book_diagnostic(ranked_panel, return_col, horizon, cost_bps)
    print(f"  Rebalance dates:          {book_result.n_rebalance_dates}")
    print(f"  Mean return (gross):      {book_result.mean_return_gross:+.4f}")
    print(f"  Mean return (net of {cost_bps:.0f}bps RT): {book_result.mean_return_net:+.4f}")
    print(f"  Hit rate:                 {book_result.hit_rate:.2%}" if not np.isnan(book_result.hit_rate) else "  Hit rate: N/A")
    print(f"  Avg names / rebalance:    {book_result.avg_names_per_rebalance:.1f}")
    print(f"  Est. one-way turnover:    {book_result.est_one_way_turnover:.2%}" if not np.isnan(book_result.est_one_way_turnover) else "  Est. one-way turnover: N/A")
    print("\n  NOTE: this is the arbiter, not an optimisation target (task.md T4.3).")
    print("=" * 70 + "\n")

    return bt_result, book_result
