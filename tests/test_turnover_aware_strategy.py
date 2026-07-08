from __future__ import annotations

import pandas as pd

from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.strategies import TurnoverAwareStrategy


def _panel(symbols, scores):
    return pd.DataFrame({
        "symbol": symbols,
        "industry_code": ["801010"] * len(symbols),
        "model_score": scores,
    })


def _select(panel, config, prior_selected=None):
    strategy = TurnoverAwareStrategy(prior_selected=prior_selected)
    selected, reasons = strategy.select(
        panel, config,
        industry_col="industry_code",
        score_col="model_score",
        symbol_col="symbol",
    )
    return selected, reasons


def test_empty_prior_is_cold_start_top_k():
    symbols = [f"S{i:03d}" for i in range(10)]
    scores = list(range(10, 0, -1))  # S000 highest .. S009 lowest
    panel = _panel(symbols, scores)
    config = SelectionConfig(
        strategy=StrategyType.TURNOVER_AWARE,
        max_total=5, enter_rank=3, keep_rank=6, max_turnover=0.5,
    )

    selected, reasons = _select(panel, config, prior_selected=None)

    assert selected == set(symbols[:5])
    assert (reasons.loc[panel["symbol"].isin(selected)] == "cold_start_top_k").all()


def test_prior_equals_current_pool_zero_turnover():
    symbols = [f"S{i:03d}" for i in range(10)]
    scores = list(range(10, 0, -1))
    panel = _panel(symbols, scores)
    config = SelectionConfig(
        strategy=StrategyType.TURNOVER_AWARE,
        max_total=5, enter_rank=3, keep_rank=6, max_turnover=0.5,
    )
    prior = set(symbols[:5])

    selected, reasons = _select(panel, config, prior_selected=prior)

    assert selected == prior
    turnover = len(selected - prior) / config.max_total
    assert turnover == 0.0
    assert (reasons.loc[panel["symbol"].isin(selected)] == "kept_incumbent").all()


def test_turnover_bounded_across_synthetic_two_date_sequence():
    config = SelectionConfig(
        strategy=StrategyType.TURNOVER_AWARE,
        max_total=10, enter_rank=8, keep_rank=15, max_turnover=0.2,
    )

    # Date 1: cold start.
    symbols_d1 = [f"S{i:03d}" for i in range(20)]
    scores_d1 = list(range(20, 0, -1))
    panel_d1 = _panel(symbols_d1, scores_d1)
    selected_d1, _ = _select(panel_d1, config, prior_selected=None)
    assert len(selected_d1) == config.max_total

    # Date 2: scores reshuffled so many incumbents drop rank sharply, and
    # several new challengers now rank at the very top.
    symbols_d2 = symbols_d1
    new_order = symbols_d1[15:] + symbols_d1[10:15] + symbols_d1[:10]
    rank_lookup = {s: r for r, s in enumerate(new_order)}
    scores_d2 = [len(symbols_d2) - rank_lookup[s] for s in symbols_d2]
    panel_d2 = _panel(symbols_d2, scores_d2)

    selected_d2, _ = _select(panel_d2, config, prior_selected=selected_d1)

    assert len(selected_d2) == config.max_total
    one_way_turnover = len(selected_d2 - selected_d1) / config.max_total
    assert one_way_turnover <= config.max_turnover + 1e-9


def test_determinism_same_input_twice_identical_output():
    symbols = [f"S{i:03d}" for i in range(12)]
    scores = [5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2]  # ties -- must break deterministically
    panel = _panel(symbols, scores)
    config = SelectionConfig(
        strategy=StrategyType.TURNOVER_AWARE,
        max_total=6, enter_rank=4, keep_rank=8, max_turnover=0.5,
    )
    prior = {"S000", "S003", "S006"}

    selected_1, reasons_1 = _select(panel, config, prior_selected=set(prior))
    selected_2, reasons_2 = _select(panel, config, prior_selected=set(prior))

    assert selected_1 == selected_2
    assert reasons_1.equals(reasons_2)
