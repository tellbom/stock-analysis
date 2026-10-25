"""Regression checks for observable labels at holiday and sparse-price boundaries."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quant_platform.evaluation.walk_forward import WalkForwardEvaluator, _ic_at_lag
from scripts.next_week_d3_prediction import observable_training_panel
from scripts.run_walk_forward_verdict import run_primary_verdict


def _holiday_dates():
    dates = pd.bdate_range("2026-09-01", "2026-10-23")
    return dates[~((dates >= "2026-10-01") & (dates <= "2026-10-07"))]


@pytest.mark.parametrize("horizon", [3, 5])
def test_walk_forward_holiday_training_labels_are_observable(horizon):
    dates = _holiday_dates()
    panel = pd.DataFrame([
        {"symbol": str(symbol), "date": date, "x": float(symbol),
         f"ret_fwd_{horizon}d": symbol / 1000.0}
        for date in dates for symbol in range(10)
    ])
    seen = []

    class CaptureModel:
        def fit(self, x, y):
            seen.extend(x.index)
            return self

        def predict(self, x):
            return x.x.to_numpy()

    evaluator = WalkForwardEvaluator(horizon=horizon, embargo_extra=2)
    result = evaluator._run_window(
        panel, ["x"], f"ret_fwd_{horizon}d", pd.Timestamp("2026-10-09"),
        pd.Timestamp("2026-10-23"), 0, CaptureModel, dates,
    )
    assert result is not None
    max_feature = panel.loc[seen, "date"].max()
    maturity = dates[dates.get_loc(max_feature) + horizon + 1]
    assert maturity == pd.Timestamp(result.max_train_label_end)
    assert maturity <= pd.Timestamp(result.training_observable_through)
    assert maturity < pd.Timestamp("2026-10-09")
    # The previous calendar-day cutoff admitted a pre-holiday feature whose
    # outcome did not exist at the first test session.
    old_cutoff = pd.Timestamp("2026-10-09") - pd.Timedelta(days=horizon + 3)
    old_last = dates[dates < old_cutoff][-1]
    assert dates[dates.get_loc(old_last) + horizon + 1] > pd.Timestamp("2026-10-09")


@pytest.mark.parametrize("horizon", [3, 5])
def test_historical_prediction_rebuilds_exact_target_and_censors_future(horizon):
    dates = _holiday_dates()
    panel = pd.DataFrame({"symbol": "A", "date": dates, "close": np.arange(len(dates)) + 100.0})
    panel[f"ret_fwd_{horizon}d"] = 999.0  # Cached after the historical signal.
    panel[f"ret_fwd_{horizon}d_cs"] = 7
    calendar = pd.DataFrame({"date": dates, "is_trading": True})
    as_of = dt.date(2026, 10, 9)
    result = observable_training_panel(panel, calendar, as_of, horizon)
    valid = result[result[f"ret_fwd_{horizon}d"].notna()]
    assert not valid.empty
    assert valid[f"label_end_{horizon}d"].max() <= as_of
    assert result.tail(horizon + 1)[f"ret_fwd_{horizon}d"].isna().all()
    assert (result[f"ret_fwd_{horizon}d_cs"] == 7).all()
    first = valid.iloc[0]
    expected = panel.close.iloc[horizon + 1] / panel.close.iloc[1] - 1
    assert first[f"ret_fwd_{horizon}d"] == pytest.approx(expected)


def test_prediction_missing_exchange_session_invalidates_label():
    dates = pd.bdate_range("2026-09-01", periods=9)
    panel = pd.DataFrame({"symbol": "A", "date": dates.delete(2), "close": 100.0})
    calendar = pd.DataFrame({"date": dates, "is_trading": True})
    result = observable_training_panel(panel, calendar, dates[-1].date(), 3)
    assert np.isnan(result.iloc[0].ret_fwd_3d)


def test_ic_decay_lag_one_means_one_holding_day_after_entry():
    dates = pd.bdate_range("2026-09-01", periods=4)
    panel = pd.DataFrame([
        {"symbol": str(symbol), "date": date, "pred": float(symbol) if j == 0 else np.nan,
         "close": 100.0 if j < 2 else 100.0 + (symbol + 1) * 10}
        for symbol in range(3) for j, date in enumerate(dates)
    ])
    assert _ic_at_lag(panel, "pred", "close", [1], dates)[1] == pytest.approx(1.0)


def test_ic_decay_does_not_shift_across_missing_price_sessions():
    dates = pd.bdate_range("2026-09-01", periods=5)
    panel = pd.DataFrame([
        {"symbol": str(symbol), "date": date, "pred": float(symbol) if j == 0 else np.nan,
         "close": 100.0 + (symbol + 1) * j}
        for symbol in range(3) for j, date in enumerate(dates) if j != 2
    ])
    assert np.isnan(_ic_at_lag(panel, "pred", "close", [2], dates)[2])


def test_primary_verdict_rejects_wrong_target_before_training():
    with pytest.raises(ValueError, match="exact raw"):
        run_primary_verdict(pd.DataFrame(), ["x"], "ret_fwd_3d_cs", horizon=3)
    with pytest.raises(ValueError, match="exact raw"):
        run_primary_verdict(pd.DataFrame(), ["x"], "ret_fwd_3d", horizon=5)


def test_walk_forward_rejects_forward_features_before_training():
    with pytest.raises(ValueError, match="Forward labels"):
        WalkForwardEvaluator(horizon=3).run(pd.DataFrame(), ["ret_fwd_5d"], "ret_fwd_3d")


def test_fixed_model_validation_purges_unobservable_labels():
    from quant_platform.evaluation.live_multi_date_harness import run_live_multi_date_harness
    dates = _holiday_dates()
    panel = pd.DataFrame([
        {"symbol": str(s), "date": d, "x": float(s), "ret_fwd_5d": s / 1000}
        for d in dates for s in range(10)
    ])
    seen = []
    class Model:
        def fit(self, x, y):
            seen.extend(x.index)
            return self
        def predict(self, x):
            return x.x.to_numpy()
    result = run_live_multi_date_harness(panel, ["x"], "ret_fwd_5d", "2026-10-09",
                                        ["2026-10-09", "2026-10-12"], Model, horizon=5,
                                        trading_calendar=dates)
    last = panel.loc[seen, "date"].max()
    assert dates[dates.get_loc(last) + 6] < pd.Timestamp("2026-10-09")
    assert result.metadata["calendar_source"] == "exchange"
    with pytest.raises(ValueError, match="at/after"):
        run_live_multi_date_harness(panel, ["x"], "ret_fwd_5d", "2026-10-09",
                                   ["2026-09-30"], Model, horizon=5)
