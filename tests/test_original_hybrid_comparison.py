import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from scripts.compare_original_hybrid import combine, scores, verify_reference
from scripts.reconcile_original_hybrid import evaluate


def test_hybrids_reuse_original_scores_and_reject_different_universe():
    pred = pd.DataFrame({"symbol": ["000001", "000002"], "date": [dt.date(2026, 9, 30)] * 2})
    base = scores(pred, np.array([0.01, 0.02]), "base")
    frozen = base.copy(deep=True)
    recent = scores(pred, np.array([0.03, -0.01]), "recent")
    first, second = combine(base, recent), combine(base, recent.iloc[::-1])
    pd.testing.assert_frame_equal(base, frozen)
    assert first.base_score.tolist() == second.base_score.tolist() == [0.01, 0.02]
    with pytest.raises(ValueError, match="universes differ"):
        combine(base, recent.iloc[:1])


def test_reference_rejects_retrained_score_drift_even_with_same_picks():
    train = pd.DataFrame({"date": [dt.date(2026, 9, 23)]})
    picks = pd.DataFrame({"symbol": ["000001"], "base_score": [0.02]})
    reference = {"horizons": {"3": {"train_rows": 1, "train_first_date": "2026-09-23",
                                    "train_last_date": "2026-09-23",
                                    "selected": [{"symbol": "000001", "model_score": 0.01}]}}}
    with pytest.raises(RuntimeError, match="scores changed"):
        verify_reference(picks, train, reference, 3)


@pytest.fixture
def experiment(tmp_path):
    (tmp_path / "training_summary.json").write_text(json.dumps({"as_of": "2026-09-30", "entry_date": "2026-10-08",
        "horizons": {"3": {"target_date": "2026-10-13"}, "5": {"target_date": "2026-10-15"}}}), encoding="utf-8")
    arms = ("ordinary", "hybrid_no_flow", "hybrid_flow", "hybrid_flow_score_gate_diagnostic")
    pd.DataFrame([{"horizon": h, "strategy": a, "strategy_name": a, "symbol": "000001",
                   "name": "test", "industry_name": "test", "recommendation_rank": 1,
                   "in_equal_count_comparison": True} for h in (3, 5) for a in arms]).to_csv(tmp_path / "all_picks.csv", index=False)
    rows = []
    for d, price in ((8, 100), (9, 100), (12, 100), (13, 90), (14, 95), (15, 110)):
        for symbol, close in (("000001", price), ("000300", 100)):
            rows.append({"symbol": symbol, "date": f"2026-10-{d:02d}", "close": close})
    path = tmp_path / "staged_prices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return tmp_path, path


def test_each_horizon_stops_at_its_target_date(experiment):
    out, path = experiment
    result = evaluate(out, path, dt.date(2026, 10, 15), 10)
    d3 = next(s for s in result["strategies"] if s["horizon"] == 3)
    d5 = next(s for s in result["strategies"] if s["horizon"] == 5)
    assert d3["end_date"] == "2026-10-13"
    assert d3["net_if_exit"] == pytest.approx(-0.102)
    assert d5["end_date"] == "2026-10-15"
    assert d5["net_if_exit"] == pytest.approx(0.098)
    assert d3["complete"] and d5["complete"]


def test_early_d5_is_not_reported_as_completed(experiment):
    out, path = experiment
    result = evaluate(out, path, dt.date(2026, 10, 13), 10)
    assert all(s["complete"] == (s["horizon"] == 3) for s in result["strategies"])


def test_missing_prices_and_broad_history_are_rejected(experiment):
    out, path = experiment
    with pytest.raises(ValueError, match="one calendar week"):
        evaluate(out, path, dt.date(2026, 10, 16), 10)
    prices = pd.read_csv(path, dtype={"symbol": str})
    prices = prices[~((prices.symbol == "000001") & (prices.date == "2026-10-12"))]
    prices.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing holding-week prices"):
        evaluate(out, path, dt.date(2026, 10, 13), 10)


def test_diagnostic_without_equal_count_rows_does_not_block_reconciliation(experiment):
    out, path = experiment
    picks = pd.read_csv(out / "all_picks.csv", dtype={"symbol": str})
    mask = picks.strategy == "hybrid_flow_score_gate"
    picks.loc[mask, "strategy"] = "hybrid_flow_score_gate_diagnostic"
    picks.loc[mask, "in_equal_count_comparison"] = False
    picks.to_csv(out / "all_picks.csv", index=False)
    result = evaluate(out, path, dt.date(2026, 10, 13), 10)
    diag = [s for s in result["strategies"] if s["strategy"].endswith("_diagnostic")]
    assert len(diag) == 2
    assert all(s["scope"] == "full_selection" for s in diag)


def test_independent_experiment_has_own_pair_and_does_not_break_frozen_scopes(experiment):
    out, path = experiment
    picks = pd.read_csv(out / "all_picks.csv", dtype={"symbol": str})
    extra = picks[picks.strategy == "ordinary"].copy()
    extra["strategy"] = "hybrid_timeweighted"
    extra["strategy_name"] = "independent experiment"
    extra["is_experimental"] = True
    extra["in_equal_count_comparison"] = False
    pd.concat([picks, extra], ignore_index=True).to_csv(out / "all_picks.csv", index=False)
    result = evaluate(out, path, dt.date(2026, 10, 13), 17.5)
    exp = [s for s in result["strategies"] if s["strategy"] == "hybrid_timeweighted"]
    assert {s["scope"] for s in exp} == {"full_selection", "experiment_pair"}
    pairs = [s for s in result["strategies"] if s["scope"] == "experiment_pair"]
    assert len(pairs) == 4 and {s["count"] for s in pairs} == {1}
    assert len([s for s in result["strategies"] if s["scope"] == "equal_count"]) == 6
    daily = pd.read_csv(out / "daily_returns.csv")
    active_d5 = daily[(daily.horizon == 5) & (daily.strategy == "ordinary") & (daily.scope == "equal_count")]
    assert active_d5.cumulative_cost.tolist() == pytest.approx([.00175] * 4)
    completed_d3 = daily[(daily.horizon == 3) & (daily.strategy == "ordinary") & (daily.scope == "equal_count")]
    assert completed_d3.cumulative_cost.tolist() == pytest.approx([.00175, .00175, .00175, .0035])
