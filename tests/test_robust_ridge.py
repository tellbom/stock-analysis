"""Regression tests for the frozen, mature-label Robust Ridge challenger."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from quant_platform.selection.ridge_baseline import ORIGINAL_FEATURES
from quant_platform.selection.robust_ridge import (
    ALPHA,
    ROBUST_FEATURES,
    WINDOW_TRADING_DAYS,
    build_robust_features,
    build_robust_pipeline,
    fit_predict_robust_ridge,
)


def _panel(n_dates=330, n_symbols=3):
    rng = np.random.default_rng(731)
    days = list(pd.bdate_range("2024-01-02", periods=n_dates).date)
    panel = pd.DataFrame({
        "symbol": np.tile([f"{i:06}" for i in range(n_symbols)], n_dates),
        "date": np.repeat(days, n_symbols),
    })
    panel[ROBUST_FEATURES] = rng.normal(size=(len(panel), len(ROBUST_FEATURES)))
    panel["ret_fwd_3d"] = panel["ma_5"] * 0.02 + rng.normal(0, 0.01, len(panel))
    panel["ret_fwd_5d"] = panel["rsi_6"] * 0.03 + rng.normal(0, 0.01, len(panel))
    return panel, days


@pytest.mark.parametrize("horizon", [3, 5])
def test_future_label_pollution_is_excluded_without_precensoring(horizon):
    panel, days = _panel()
    as_of = days[-10]
    label = f"ret_fwd_{horizon}d"
    expected, meta, model = fit_predict_robust_ridge(panel, as_of, label, trading_days=days)
    cutoff = days[days.index(as_of) - horizon - 1]
    polluted = panel.copy()
    polluted.loc[polluted.date > cutoff, label] = 1e100
    actual, other_meta, other_model = fit_predict_robust_ridge(
        polluted, as_of, label, trading_days=days
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert other_meta == meta
    np.testing.assert_array_equal(model.named_steps["model"].coef_, other_model.named_steps["model"].coef_)
    assert meta["train_last_date"] == cutoff.isoformat()
    assert meta["train_date_count"] == WINDOW_TRADING_DAYS
    assert meta["normalize_price_again"] is False


def test_clipping_and_scaling_are_learned_only_from_training_rows():
    panel, days = _panel()
    as_of = days[-8]
    _, meta, original = fit_predict_robust_ridge(panel, as_of, "ret_fwd_3d", trading_days=days)
    changed = panel.copy()
    cutoff = days[days.index(as_of) - 4]
    changed.loc[changed.date > cutoff, ROBUST_FEATURES] = 1e8
    scored, _, model = fit_predict_robust_ridge(changed, as_of, "ret_fwd_3d", trading_days=days)
    clip = model.named_steps["clip"]
    np.testing.assert_array_equal(clip.lower_, original.named_steps["clip"].lower_)
    np.testing.assert_array_equal(clip.upper_, original.named_steps["clip"].upper_)
    np.testing.assert_array_equal(model.named_steps["scaler"].mean_, original.named_steps["scaler"].mean_)
    train = panel[panel.date.astype(str).isin(meta["train_dates"])]
    expected_bounds = np.quantile(train[ROBUST_FEATURES], [0.01, 0.99], axis=0)
    np.testing.assert_allclose([clip.lower_, clip.upper_], expected_bounds)
    pred_features = build_robust_features(changed[changed.date == as_of])
    np.testing.assert_allclose(clip.transform(pred_features), np.tile(clip.upper_, (3, 1)))
    manual_prediction = model.named_steps["model"].predict(
        model.named_steps["scaler"].transform(np.tile(clip.upper_, (3, 1)))
    )
    np.testing.assert_allclose(scored.base_score, manual_prediction)


def test_training_window_does_not_reach_back_to_replace_missing_dates():
    panel, days = _panel()
    as_of = days[-1]
    cutoff_index = len(days) - 1 - 4
    first = days[cutoff_index - WINDOW_TRADING_DAYS + 1]
    panel.loc[panel.date == first, "ma_5"] = np.nan
    _, meta, _ = fit_predict_robust_ridge(panel, as_of, "ret_fwd_3d", trading_days=days)
    assert meta["window_first_date"] == first.isoformat()
    assert meta["train_first_date"] == days[days.index(first) + 1].isoformat()
    assert meta["train_date_count"] == 251


def test_old_rows_outside_frozen_window_cannot_change_fit():
    panel, days = _panel()
    expected, meta, _ = fit_predict_robust_ridge(panel, days[-1], "ret_fwd_5d", trading_days=days)
    old = panel.date.astype(str) < meta["window_first_date"]
    panel.loc[old, ROBUST_FEATURES + ["ret_fwd_5d"]] = 1e9
    actual, _, _ = fit_predict_robust_ridge(panel, days[-1], "ret_fwd_5d", trading_days=days)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("label", ["ret_fwd_1d", "ret_fwd_3d_cs", "excess_vs_csi300_5d"])
def test_only_exact_requested_labels_are_supported(label):
    panel, days = _panel()
    with pytest.raises(ValueError, match="exact ret_fwd_3d/ret_fwd_5d"):
        fit_predict_robust_ridge(panel, days[-1], label, trading_days=days)


def test_insufficient_mature_dates_fail_even_when_future_labels_are_finite():
    panel, days = _panel(n_dates=63)
    with pytest.raises(RuntimeError, match="requires 60 mature training dates; got 59"):
        fit_predict_robust_ridge(panel, days[-1], "ret_fwd_3d", trading_days=days)


def test_nonfinite_rows_are_excluded_and_missing_features_fail():
    panel, days = _panel()
    pred_indices = panel.index[panel.date == days[-1]]
    panel.loc[pred_indices[0], "ma_5"] = np.inf
    panel.loc[panel.index[panel.date == days[-10]][0], "ret_fwd_3d"] = -np.inf
    scored, meta, _ = fit_predict_robust_ridge(panel, days[-1], "ret_fwd_3d", trading_days=days)
    assert meta["prediction_rows"] == 2
    assert meta["prediction_excluded_rows"] == 1
    assert meta["train_rows"] == 252 * 3 - 1
    assert np.isfinite(scored.base_score).all()
    with pytest.raises(ValueError, match="missing feature columns"):
        build_robust_features(panel.drop(columns="rsi_6"))


def test_feature_contract_preserves_existing_normalisation():
    panel, _ = _panel(n_dates=2)
    actual = build_robust_features(panel)
    assert ROBUST_FEATURES == ORIGINAL_FEATURES
    pd.testing.assert_frame_equal(actual, panel[ORIGINAL_FEATURES])
    assert build_robust_pipeline().named_steps["model"].alpha == ALPHA == 10.0


def test_saved_pipeline_can_score_without_refitting(tmp_path):
    import joblib

    panel, days = _panel()
    expected, meta, model = fit_predict_robust_ridge(panel, days[-1], "ret_fwd_5d", trading_days=days)
    path = tmp_path / "robust_ridge.joblib"
    joblib.dump({"model": model, "meta": meta}, path)
    loaded = joblib.load(path)
    features = build_robust_features(panel[panel.date == days[-1]])
    np.testing.assert_array_equal(loaded["model"].predict(features), expected.base_score)
    assert loaded["meta"] == meta
    with pytest.raises(ValueError, match="feature order"):
        loaded["model"].predict(features[ROBUST_FEATURES[::-1]])


def test_calendar_must_cover_prediction_and_historical_panel_dates():
    panel, days = _panel()
    with pytest.raises(ValueError, match="absent from trading_days"):
        fit_predict_robust_ridge(panel, days[-1], "ret_fwd_3d", trading_days=days[:-1])
    with pytest.raises(ValueError, match="cover all panel dates"):
        fit_predict_robust_ridge(panel, days[-1], "ret_fwd_3d", trading_days=days[1:])
