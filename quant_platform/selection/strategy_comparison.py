"""
selection.strategy_comparison
==============================
T1.4 — Compare cross-industry selection strategies BY MEASUREMENT.

This module adds no new selection logic. It orchestrates the existing
``IndustryNeutralRanker`` / ``SelectionConfig`` / ``ExposureMonitor`` stack
across multiple already-scored dated cross-sections and reports, per
(date, strategy):

  - n_selected
  - mean_return / excess_return_vs_benchmark of the selected set
  - run-level exposure_flag
  - max_single_industry_fraction

Then aggregates across dates so a strategy choice is made from evidence,
not a default.

Dependency (per task.md T1.4)
------------------------------
This module does NOT retrain, fetch, or generate multi-date data on its
own. It is deliberately generic: it accepts *any* mapping of
``{date: scored_panel}`` where each panel already has the columns
IndustryNeutralRanker needs (symbol, model_score, industry_code) plus a
realised return column and a benchmark return.

Until T4.1 (WalkForwardEvaluator adoption) / T4.2 (multi-date live
harness) exist, callers must supply the dated panels themselves — e.g.
from cached silver/ohlcv forward returns, one scored cross-section per
prediction date. The ``compare_strategies`` / ``aggregate_comparison``
functions are what T4.1/T4.2 will eventually feed at scale; nothing here
assumes T4.1 exists, so this can be merged and used standalone, but the
*evidence-based decision* T1.4 asks for is only as strong as the number
and quality of dates supplied (task.md asks for >= 5).

"Change one, measure, record" (T1.4 constraint)
------------------------------------------------
``diff_configs`` makes the caller's intent explicit: given two named
SelectionConfig variants, it reports exactly which fields differ. If more
than one field differs, a warning is logged (not raised — comparisons
across strategy *types* legitimately differ in several fields, e.g.
EqualTopK has no hybrid_weight), so the decision trail is visible either
way and never silently swept under a blind grid search.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.selection.config import SelectionConfig
from quant_platform.selection.exposure import ExposureMonitor
from quant_platform.selection.ranker import IndustryNeutralRanker

logger = get_logger(__name__)


def diff_configs(name_a: str, config_a: SelectionConfig,
                  name_b: str, config_b: SelectionConfig) -> dict:
    """
    Report which SelectionConfig fields differ between two named configs.

    Returns
    -------
    dict
        {field_name: (value_in_a, value_in_b)} for every field that
        differs. Logs a warning (does not raise) if more than one field
        differs, so a "blind" multi-knob comparison is visible in logs
        rather than silently accepted.
    """
    a_dict = dataclasses.asdict(config_a)
    b_dict = dataclasses.asdict(config_b)
    diffs = {
        field: (a_dict[field], b_dict[field])
        for field in a_dict
        if a_dict[field] != b_dict[field]
    }
    if len(diffs) > 1:
        logger.warning(
            "diff_configs(%s, %s): %d fields differ (%s) — confirm this is "
            "an intentional strategy-level comparison, not a blind multi-knob "
            "sweep (T1.4 constraint: change one, measure, record).",
            name_a, name_b, len(diffs), sorted(diffs),
        )
    else:
        logger.info("diff_configs(%s, %s): %s", name_a, name_b, diffs)
    return diffs


def _run_one(
    panel: pd.DataFrame,
    config: SelectionConfig,
    *,
    industry_col: str,
    score_col: str,
    symbol_col: str,
    return_col: str,
    benchmark_return_col: str | None,
) -> dict:
    """Run IndustryNeutralRanker for one (date, strategy) and summarise."""
    ranker = IndustryNeutralRanker(
        config,
        industry_col=industry_col,
        score_col=score_col,
        symbol_col=symbol_col,
    )
    ranked = ranker.run(panel)
    selected = ranked[ranked["selected"]]

    n_selected = len(selected)
    mean_return = selected[return_col].mean() if n_selected else float("nan")

    excess_return = float("nan")
    if n_selected and benchmark_return_col is not None and benchmark_return_col in panel.columns:
        excess_return = mean_return - panel[benchmark_return_col].mean()

    if n_selected:
        run_flag = selected["exposure_flag"].iloc[0]
        max_industry_fraction = (
            selected[industry_col].value_counts(normalize=True).max()
        )
    else:
        run_flag = "not_selected"
        max_industry_fraction = float("nan")

    return {
        "n_selected": n_selected,
        "mean_return": mean_return,
        "excess_return_vs_benchmark": excess_return,
        "exposure_flag": run_flag,
        "max_single_industry_fraction": max_industry_fraction,
    }


def compare_strategies(
    dated_panels: Mapping,
    configs: Mapping[str, SelectionConfig],
    *,
    industry_col: str = "industry_code",
    score_col: str = "model_score",
    symbol_col: str = "symbol",
    return_col: str = "total_return",
    benchmark_return_col: str | None = "benchmark_return",
) -> pd.DataFrame:
    """
    Run every named strategy config against every dated scored panel.

    Parameters
    ----------
    dated_panels : Mapping[date-like, pd.DataFrame]
        One already-scored cross-section per prediction date. Each panel
        must contain [symbol_col, score_col, industry_col, return_col]
        and, if excess return is wanted, benchmark_return_col.
    configs : Mapping[str, SelectionConfig]
        e.g. {"equal_top_k": SelectionConfig(strategy=StrategyType.EQUAL_TOP_K, ...),
              "hybrid":      SelectionConfig(strategy=StrategyType.HYBRID, ...)}

    Returns
    -------
    pd.DataFrame
        One row per (date, strategy) with n_selected, mean_return,
        excess_return_vs_benchmark, exposure_flag, max_single_industry_fraction.
    """
    if len(dated_panels) < 5:
        logger.warning(
            "compare_strategies: only %d dates supplied; task.md T1.4 asks "
            "for >= 5 dates for a fair comparison — treat this result as "
            "preliminary, not a final strategy decision.",
            len(dated_panels),
        )

    rows = []
    for date, panel in dated_panels.items():
        for strategy_name, config in configs.items():
            metrics = _run_one(
                panel, config,
                industry_col=industry_col, score_col=score_col,
                symbol_col=symbol_col, return_col=return_col,
                benchmark_return_col=benchmark_return_col,
            )
            rows.append({"date": date, "strategy": strategy_name, **metrics})

    return pd.DataFrame(rows).sort_values(["date", "strategy"]).reset_index(drop=True)


def aggregate_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a compare_strategies() result across dates, per strategy.

    Returns
    -------
    pd.DataFrame
        Indexed by strategy: n_dates, mean_excess_return,
        pct_dates_overweight (share of dates flagged industry_overweight),
        mean_max_single_industry_fraction.
    """
    def _agg(grp: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_dates": len(grp),
            "mean_excess_return": grp["excess_return_vs_benchmark"].mean(),
            "pct_dates_overweight": (grp["exposure_flag"] == "industry_overweight").mean(),
            "mean_max_single_industry_fraction": grp["max_single_industry_fraction"].mean(),
        })

    return (
        comparison_df.groupby("strategy", group_keys=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
