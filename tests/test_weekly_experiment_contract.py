"""Integration contracts for experiment controls and frozen weekly arms."""
import datetime as dt
import sys

import numpy as np
import pandas as pd
import pytest

from quant_platform.selection.ridge_baseline import fit_predict_ridge32
from scripts.compare_original_hybrid import (
    FLOW_FEATURES, experiment_recent_train, run, validate_experiment_options,
)
from scripts import run_dual_horizon_rrf as weekly


def test_time_weight_reaches_real_ridge_fit_without_changing_default():
    # Constant X isolates the fitted intercept. Labels 180 days apart must
    # have a 1:2 weight ratio, so the weighted mean is exactly 2/3.
    signal = dt.date(2026, 10, 14)
    panel = pd.DataFrame({
        "symbol": ["000001"] * 3,
        "date": [signal - dt.timedelta(days=n) for n in (195, 15, 0)],
        "factor": [0., 0., 0.], "ret_fwd_5d": [0., 1., np.nan],
    })
    base, meta, _ = fit_predict_ridge32(panel, signal, "ret_fwd_5d", ["factor"])
    weighted, weighted_meta, _ = fit_predict_ridge32(
        panel, signal, "ret_fwd_5d", ["factor"], use_time_weight=True,
    )
    assert base.base_score.iloc[0] == pytest.approx(.5)
    assert weighted.base_score.iloc[0] == pytest.approx(2/3)
    assert not meta["time_weighted"] and weighted_meta["time_weighted"]


def _train():
    dates = list(pd.bdate_range("2026-01-01", periods=130).date)
    rows = pd.DataFrame({"symbol": "000001", "date": dates})
    for col in FLOW_FEATURES:
        rows[col] = .5
    return rows


def test_requested_60_date_window_is_usable_and_keeps_latest_dates():
    train = _train()
    frozen = train.copy(deep=True)
    actual = experiment_recent_train(train, 60)
    assert actual.date.tolist() == train.date.tolist()[-60:]
    pd.testing.assert_frame_equal(train, frozen)


def test_missing_flow_date_cannot_be_silently_counted_as_usable():
    train = _train()
    train.loc[train.index[-3], FLOW_FEATURES] = np.nan
    with pytest.raises(RuntimeError, match="60 usable flow dates; got 59"):
        experiment_recent_train(train, 60)


@pytest.mark.parametrize("window", [0, -60, 59, True, 60.5])
def test_invalid_window_rejected_before_output_creation(tmp_path, window):
    out = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="integer >= 60"):
        run(dt.date(2026, 10, 14), out, None, 1, recent_window=window)
    assert not out.exists()


def test_requested_window_requires_sufficient_history():
    with pytest.raises(RuntimeError, match="140 label dates; only 130 available"):
        experiment_recent_train(_train(), 140)


def test_weekly_entry_passes_both_controls_to_canonical_runner(tmp_path, monkeypatch):
    captured = {}
    def shared(*args, **kwargs):
        captured.update(kwargs)
        return {"ok": True}
    monkeypatch.setattr(weekly, "_run_shared_weekly", shared)
    assert weekly.run(dt.date(2026, 10, 14), [3, 5], tmp_path / "run", 1,
                      use_time_weight=True, recent_window=60) == {"ok": True}
    assert captured == {"use_time_weight": True, "recent_window": 60}


def test_weekly_cli_exposes_independent_controls(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(weekly, "run", lambda *a, **kw: captured.update(kw))
    monkeypatch.setattr(sys, "argv", ["run_dual_horizon_rrf.py", "--as-of", "2026-10-14",
                                     "--out-dir", str(tmp_path / "cli"), "--recent-window", "60"])
    weekly.main()
    assert captured["recent_window"] == 60
    assert captured["use_time_weight"] is False


def test_lgbm_independent_base_rejects_unsupported_experiment(tmp_path):
    with pytest.raises(ValueError, match="require the shared ridge32"):
        weekly.run(dt.date(2026, 10, 14), [3, 5], tmp_path / "out", 1,
                   base_model="lgbm-long", use_time_weight=True)
