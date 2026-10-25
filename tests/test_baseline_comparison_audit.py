import datetime as dt

import numpy as np
import pandas as pd
import pytest

from scripts.audit_d3_d5_baselines import cohort_net, control_gate_snapshot, mature_panel, paired_interval
from scripts import compare_original_hybrid as compare
from quant_platform.selection.ridge_baseline import ORIGINAL_FEATURES


def test_maturity_uses_observation_date_even_if_cached_label_is_present():
    signal = dt.date(2026, 10, 9)
    panel = pd.DataFrame({"date": [dt.date(2026, 9, 29), dt.date(2026, 9, 30)],
                          "ret_fwd_5d": [.1, 999.0],
                          "label_end_5d": [signal, dt.date(2026, 10, 14)]})
    result = mature_panel(panel, signal, 5)
    assert result.ret_fwd_5d.iloc[0] == .1
    assert pd.isna(result.ret_fwd_5d.iloc[1])
    assert panel.ret_fwd_5d.iloc[1] == 999.0


def test_complete_round_trip_cost_and_deterministic_paired_interval():
    assert cohort_net(.01, 17.5) == pytest.approx(.0065)
    assert paired_interval(np.arange(8) / 100) == paired_interval(np.arange(8) / 100)
    assert paired_interval(np.array([.1])) == (None, None)


def test_explicit_control_gate_is_read_only_and_missing_gate_fails(tmp_path):
    gate = tmp_path / "control_gate.json"
    original = b'{"status":"needs_plan_choice","blocked":true}'
    gate.write_bytes(original)
    digest = control_gate_snapshot(gate)
    assert isinstance(digest, str) and len(digest) == 64
    assert gate.read_bytes() == original
    assert control_gate_snapshot(None) is None
    with pytest.raises(FileNotFoundError):
        control_gate_snapshot(tmp_path / "missing.json")


class RecentFixtureModel:
    def predict(self, frame):
        return frame.iloc[:, :3].sum(axis=1).to_numpy() / 1000


def test_new_baseline_participates_without_changing_frozen_arm_files(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    days = list(pd.bdate_range("2025-01-01", periods=140).date)
    symbols = [str(i).zfill(6) for i in range(250)]
    index = pd.MultiIndex.from_product([symbols, days[:130]], names=["symbol", "date"])
    panel = index.to_frame(index=False)
    for feature in ORIGINAL_FEATURES:
        panel[feature] = rng.normal(size=len(panel))
    panel["close"] = 100 + rng.uniform(0, 10, size=len(panel))
    panel["industry_code"] = panel.symbol.map(lambda s: str(int(s) % 25))
    panel["industry_name"] = "fixture"
    calendar = pd.DataFrame({"date": days, "is_trading": True})
    names = pd.DataFrame({"symbol": symbols, "name": "fixture"})
    prices = {s: panel[panel.symbol == s][["symbol", "date", "close"]].copy() for s in symbols}

    def read(path, **kwargs):
        path = str(path).replace("\\", "/")
        if "trading_calendar" in path:
            return calendar.copy()
        if "membership" in path:
            return names.copy()
        return prices[path.split("/")[-1].split(".")[0]].copy()

    def flow_features(keys, *args):
        result = keys.copy()
        for i, feature in enumerate(compare.FLOW_FEATURES):
            result[feature] = .1 + i / 10
        return result

    monkeypatch.setattr(compare, "load_symbols", lambda: symbols)
    monkeypatch.setattr(compare, "build_panel", lambda *a, **kw: panel.copy())
    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(compare, "load_flow_panel", lambda *a: pd.DataFrame({"date": days[:130]}))
    monkeypatch.setattr(compare, "load_valuation_panel", lambda *a: pd.DataFrame({"date": days[:130]}))
    monkeypatch.setattr(compare, "build_flow_features", flow_features)
    monkeypatch.setattr(compare, "fit_final_model", lambda *a: RecentFixtureModel())
    monkeypatch.setattr(compare, "_feature_importance_all", lambda model, features:
                        pd.DataFrame({"feature": features, "importance": 1}))
    old, new = tmp_path / "old", tmp_path / "new"
    compare.run(days[129], old, None, 1, horizons=(3,))
    summary = compare.run(days[129], new, None, 1, horizons=(3,), include_robust_baseline=True)
    for name in ("ordinary", "hybrid_no_flow", "hybrid_flow", "hybrid_flow_score_gate_diagnostic"):
        assert (old / f"D3_{name}_picks.csv").read_bytes() == (new / f"D3_{name}_picks.csv").read_bytes()
    assert (old / "D3_original_scores.csv").read_bytes() == (new / "D3_original_scores.csv").read_bytes()
    expanded = pd.read_csv(new / "expanded_comparison_picks.csv")
    assert set(expanded.strategy) == {"ordinary", "hybrid_no_flow", "hybrid_flow", "robust_ridge_v1"}
    assert expanded.groupby("strategy").size().nunique() == 1
    assert summary["horizons"]["3"]["robust_baseline"]["enabled"]
