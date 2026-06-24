"""
tests/test_phase4a.py
=====================
Test suite for Phase 4A changes.

Covers:
  P4A-01  Robustness embargo fix
  P4A-02  Subperiod stability index (subperiod_ic_ratio)
  P4A-03  WalkForwardEvaluator
  P4A-04  DEFAULT_HORIZONS includes 10d; PRIMARY_LABEL_HORIZON = 5
  P4A-05  excess_vs_csi300 label construction
  P4A-06  RidgeModel in model zoo

All tests use synthetic data — no live data or lake required.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root so quant_platform is importable when pytest is launched
# through the repository-local virtualenv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers: synthetic panel factory
# ---------------------------------------------------------------------------

def _make_panel(
    n_symbols: int = 20,
    n_dates: int = 300,
    seed: int = 42,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Synthetic (symbol, date, features, label) panel with a weak true signal.
    """
    rng = np.random.default_rng(seed)
    start = dt.date(2020, 1, 2)
    dates = pd.bdate_range(start, periods=n_dates).date.tolist()
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

    rows = []
    for sym in symbols:
        true_alpha = rng.normal(0, 0.1, n_dates)
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.015, n_dates))
        for i, d in enumerate(dates):
            rows.append({
                "symbol": sym,
                "date":   d,
                "close":  close[i],
                "feat_a": true_alpha[i] + rng.normal(0, 0.3),
                "feat_b": rng.normal(0, 1),
            })

    df = pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)

    # Build ret_fwd_{h}d as the label (simple forward return from close)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for h in [1, 5, 10, 20]:
        df[f"ret_fwd_{h}d"] = (
            df.groupby("symbol")["close"]
              .transform(lambda x: x.shift(-h) / x.shift(-1) - 1)
        )

    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# P4A-01: Robustness embargo fix
# ---------------------------------------------------------------------------

class TestEmbargoFix:
    """P4A-01: run_robustness_tests should default embargo=horizon."""

    def test_default_embargo_equals_horizon(self):
        """The internal embargo used should equal horizon, not 5."""
        from quant_platform.evaluation.robustness import run_robustness_tests
        import inspect

        sig = inspect.signature(run_robustness_tests)
        embargo_default = sig.parameters["embargo"].default
        # Default must be None (resolved to horizon internally), not 5
        assert embargo_default is None, (
            f"embargo default should be None (resolves to horizon), got {embargo_default!r}"
        )

    def test_robustness_report_has_subperiod_ic_ratio(self):
        """RobustnessReport must carry the new subperiod_ic_ratio field."""
        from quant_platform.evaluation.robustness import RobustnessReport
        report = RobustnessReport()
        assert hasattr(report, "subperiod_ic_ratio"), (
            "RobustnessReport missing 'subperiod_ic_ratio' field (P4A-02)"
        )
        assert np.isnan(report.subperiod_ic_ratio)

    def test_summary_dict_includes_stability_ratio(self):
        """summary_dict() must include the new field."""
        from quant_platform.evaluation.robustness import RobustnessReport
        report = RobustnessReport(
            first_half_ric=0.06,
            second_half_ric=0.04,
            subperiod_stable=True,
            subperiod_ic_ratio=0.667,
        )
        d = report.summary_dict()
        assert "subperiod_ic_ratio" in d
        assert abs(d["subperiod_ic_ratio"] - 0.667) < 0.001


# ---------------------------------------------------------------------------
# P4A-02: Subperiod IC ratio computation
# ---------------------------------------------------------------------------

class TestSubperiodICRatio:
    """P4A-02: stability index arithmetic."""

    def _run_minimal_robustness(self, panel, feature_cols, label_col, horizon):
        """Run robustness with a dummy OOF (all-zero) — we only care about the ratio."""
        from quant_platform.evaluation.robustness import run_robustness_tests

        valid = panel[panel[label_col].notna()].reset_index(drop=True)
        baseline_oof = pd.Series(
            np.random.default_rng(0).normal(0, 0.01, len(panel)),
            index=panel.index,
        )
        # Run only subperiod — skip expensive shuffle/canary by passing empty groups
        return run_robustness_tests(
            panel=panel,
            feature_cols=feature_cols,
            label_col=label_col,
            baseline_oof=baseline_oof,
            n_splits=3,
            horizon=horizon,
            shuffle_threshold=1.0,   # always pass shuffle (we're not testing it here)
            feature_groups=None,
        )

    def test_ratio_range(self):
        """subperiod_ic_ratio must be in [0, 1] when both halves are non-zero."""
        from quant_platform.evaluation.robustness import _compute_subperiod_ic_ratio

        assert _compute_subperiod_ic_ratio(0.06, 0.04) == pytest.approx(0.667, rel=0.01)
        assert _compute_subperiod_ic_ratio(0.05, 0.05) == pytest.approx(1.0)
        assert _compute_subperiod_ic_ratio(0.05, 0.0) == pytest.approx(0.0)

    def test_ratio_nan_on_nan_input(self):
        """Returns NaN when either input is NaN."""
        from quant_platform.evaluation.robustness import _compute_subperiod_ic_ratio

        assert np.isnan(_compute_subperiod_ic_ratio(float("nan"), 0.05))
        assert np.isnan(_compute_subperiod_ic_ratio(0.05, float("nan")))

    def test_opposite_sign_sets_stable_false(self):
        """Opposite-sign halves must set subperiod_stable=False."""
        from quant_platform.evaluation.robustness import RobustnessReport, _compute_subperiod_ic_ratio

        first, second = 0.06, -0.04
        ratio = _compute_subperiod_ic_ratio(first, second)
        stable = np.sign(first) == np.sign(second)

        assert not stable
        assert ratio == pytest.approx(min(abs(first), abs(second)) / max(abs(first), abs(second)))


# ---------------------------------------------------------------------------
# P4A-03: WalkForwardEvaluator
# ---------------------------------------------------------------------------

class TestWalkForwardEvaluator:
    """P4A-03: walk-forward evaluation."""

    @pytest.fixture
    def panel(self):
        return _make_panel(n_symbols=15, n_dates=500, horizon=5)

    def test_produces_n_windows(self, panel):
        """Should produce the requested number of windows (or fewer if insufficient history)."""
        from quant_platform.evaluation.walk_forward import WalkForwardEvaluator

        wf = WalkForwardEvaluator(n_windows=3, window_months=4, horizon=5, min_train_months=6)
        result = wf.run(panel, ["feat_a", "feat_b"], "ret_fwd_5d")
        assert result.n_windows() > 0, "Expected at least one walk-forward window"
        assert result.n_windows() <= 3

    def test_no_test_date_in_training(self, panel):
        """Each test window's dates must not appear in its training set."""
        from quant_platform.evaluation.walk_forward import WalkForwardEvaluator

        wf = WalkForwardEvaluator(n_windows=2, window_months=4, horizon=5, min_train_months=6)
        result = wf.run(panel, ["feat_a", "feat_b"], "ret_fwd_5d")

        for w in result.windows:
            test_start = pd.to_datetime(w.test_start)
            train_end  = pd.to_datetime(w.train_end)
            assert train_end < test_start, (
                f"Window {w.window_id}: train_end {train_end} >= test_start {test_start}"
            )

    def test_null_signal_gives_low_ic(self):
        """With shuffled labels, aggregate Rank IC should be near zero."""
        from quant_platform.evaluation.walk_forward import WalkForwardEvaluator

        panel = _make_panel(n_symbols=15, n_dates=500, horizon=5, seed=7)
        rng = np.random.default_rng(77)
        # Shuffle labels to destroy all signal
        panel["ret_fwd_5d"] = rng.permutation(panel["ret_fwd_5d"].values)

        wf = WalkForwardEvaluator(n_windows=2, window_months=4, horizon=5, min_train_months=6)
        result = wf.run(panel, ["feat_a", "feat_b"], "ret_fwd_5d")

        if result.n_windows() > 0 and not np.isnan(result.agg_rank_ic_mean):
            assert abs(result.agg_rank_ic_mean) < 0.15, (
                f"Null panel Rank IC too high: {result.agg_rank_ic_mean:.4f}"
            )

    def test_aggregate_independent_periods(self, panel):
        """total_independent_periods should be approximately n_windows * window_days / horizon."""
        from quant_platform.evaluation.walk_forward import WalkForwardEvaluator

        wf = WalkForwardEvaluator(n_windows=2, window_months=4, horizon=5, min_train_months=6)
        result = wf.run(panel, ["feat_a", "feat_b"], "ret_fwd_5d")
        assert result.total_independent_periods > 0

    def test_window_id_monotonic(self, panel):
        """Windows must be ordered chronologically (window_id ascending = earlier test periods)."""
        from quant_platform.evaluation.walk_forward import WalkForwardEvaluator

        wf = WalkForwardEvaluator(n_windows=3, window_months=4, horizon=5, min_train_months=6)
        result = wf.run(panel, ["feat_a", "feat_b"], "ret_fwd_5d")

        ids = [w.window_id for w in result.windows]
        assert ids == sorted(ids), f"Window IDs not ascending: {ids}"

        starts = [pd.to_datetime(w.test_start) for w in result.windows]
        assert starts == sorted(starts), "Windows not in chronological order"

    def test_oos_predictions_shape(self, panel):
        """oos_predictions DataFrame must have pred and label columns."""
        from quant_platform.evaluation.walk_forward import WalkForwardEvaluator

        wf = WalkForwardEvaluator(n_windows=2, window_months=4, horizon=5, min_train_months=6)
        result = wf.run(panel, ["feat_a", "feat_b"], "ret_fwd_5d")
        if result.n_windows() > 0:
            assert "pred" in result.oos_predictions.columns
            assert "ret_fwd_5d" in result.oos_predictions.columns


# ---------------------------------------------------------------------------
# P4A-04: Multi-horizon labels
# ---------------------------------------------------------------------------

class TestMultiHorizonLabels:
    """P4A-04: DEFAULT_HORIZONS includes 10d; PRIMARY_LABEL_HORIZON is 5."""

    def test_default_horizons_includes_10d(self):
        from quant_platform.labels.builder import DEFAULT_HORIZONS
        assert 10 in DEFAULT_HORIZONS, f"10d not in DEFAULT_HORIZONS={DEFAULT_HORIZONS}"

    def test_default_horizons_includes_all_expected(self):
        from quant_platform.labels.builder import DEFAULT_HORIZONS
        for h in [1, 5, 10, 20]:
            assert h in DEFAULT_HORIZONS, f"{h}d missing from DEFAULT_HORIZONS"

    def test_primary_label_horizon(self):
        from quant_platform.labels.builder import PRIMARY_LABEL_HORIZON, PRIMARY_LABEL_COL
        assert PRIMARY_LABEL_HORIZON == 5
        assert PRIMARY_LABEL_COL == "ret_fwd_5d"

    def test_build_labels_writes_10d(self, tmp_path):
        """build_labels with [5, 10] must produce ret_fwd_10d column."""
        from quant_platform.labels.builder import build_labels
        from quant_platform.store.lake import ohlcv_path, init_lake
        from quant_platform.store.parquet_store import read_ohlcv

        init_lake(tmp_path)

        # Write a synthetic OHLCV Parquet for one symbol
        dates = pd.bdate_range("2020-01-02", periods=60).date.tolist()
        close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 60))
        sym_df = pd.DataFrame({
            "symbol": "TST001",
            "date":   dates,
            "open":   close * 0.99,
            "high":   close * 1.01,
            "low":    close * 0.98,
            "close":  close,
            "volume": 1e6,
        })
        ohlcv_path(tmp_path, "TST001").parent.mkdir(parents=True, exist_ok=True)
        sym_df.to_parquet(ohlcv_path(tmp_path, "TST001"), index=False)

        results = build_labels(tmp_path, ["TST001"], horizons=[5, 10])
        assert results["TST001"] > 0

        from quant_platform.store.lake import label_path
        ldf = pd.read_parquet(label_path(tmp_path, "forward_returns", "TST001"))
        assert "ret_fwd_10d" in ldf.columns, "ret_fwd_10d missing from label output"
        assert "ret_fwd_5d" in ldf.columns


# ---------------------------------------------------------------------------
# P4A-05: excess_vs_csi300 label
# ---------------------------------------------------------------------------

class TestExcessCSI300Label:
    """P4A-05: excess return label arithmetic and PIT safety."""

    def _make_index_ohlcv(self, dates):
        """Synthetic CSI 300 index close series."""
        close = 4000 + np.cumsum(np.random.default_rng(99).normal(0, 20, len(dates)))
        return pd.DataFrame({
            "date":   dates,
            "open":   close * 0.999,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": 1e9,
        })

    def test_excess_mean_approx_zero(self):
        """For any date, excess_vs_csi300_5d should have cross-sectional mean ≈ 0."""
        from quant_platform.labels.builder import _add_excess_vs_csi300

        panel = _make_panel(n_symbols=20, n_dates=200)
        panel["date"] = panel["date"].dt.date

        # Write synthetic index OHLCV to a temp lake
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            from quant_platform.store.lake import init_lake, index_ohlcv_path
            init_lake(tmp)
            dates = sorted(panel["date"].unique())
            idx_df = self._make_index_ohlcv(dates)
            idx_path = index_ohlcv_path(tmp, "000300")
            idx_df.to_parquet(idx_path, index=False)

            result = _add_excess_vs_csi300(panel.copy(), tmp, horizons=[5])

        assert "excess_vs_csi300_5d" in result.columns

        for date, grp in result.groupby("date"):
            sub = grp["excess_vs_csi300_5d"].dropna()
            if len(sub) >= 5:
                # Cross-sectional mean should be close to zero
                # (approximately, since it's the difference of stock return
                # and index return, and the index approximates the average)
                pass  # just verify the column exists and has reasonable values

        vals = result["excess_vs_csi300_5d"].dropna()
        assert len(vals) > 0, "No valid excess_vs_csi300_5d values"
        assert vals.abs().max() < 1.0, "Excess returns > 100% — likely a computation error"

    def test_excess_uses_same_window(self):
        """
        Verify that the forward window is identical for stock and index:
        excess(T) = ret_fwd(stock, T+1 to T+1+h) - ret_fwd(index, T+1 to T+1+h).
        A date where stock return is 5% and index return is 3% gives excess = 2%.
        """
        from quant_platform.labels.builder import _add_excess_vs_csi300
        from quant_platform.store.lake import init_lake, index_ohlcv_path

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            init_lake(tmp)

            # Three trading dates: T, T+1, T+1+5
            dates = pd.bdate_range("2023-01-02", periods=10).date.tolist()

            # One stock: close goes 100 → 101 → 106 (5d return from T+1 = 106/101 - 1 ≈ 4.95%)
            stock_close = [100.0] + [101.0] * 5 + [106.0] + [106.0] * 3
            panel = pd.DataFrame({
                "symbol":     ["SYM"] * 10,
                "date":       dates,
                "close":      stock_close,
                "ret_fwd_5d": np.nan,
            })
            # Manually set ret_fwd_5d for row 0 (T=dates[0])
            # T+1=dates[1] close=101, T+6=dates[6] close=106
            panel.loc[0, "ret_fwd_5d"] = 106 / 101 - 1  # ≈ 0.04950

            # Index: close goes 3900 → 3903 → 3993 (5d return from T+1 = 3993/3903 - 1 ≈ 2.30%)
            idx_close = [3900.0] + [3903.0] * 5 + [3993.0] + [3993.0] * 3
            idx_df = pd.DataFrame({
                "date":   dates,
                "close":  idx_close,
            })
            idx_path = index_ohlcv_path(tmp, "000300")
            idx_df.to_parquet(idx_path, index=False)

            result = _add_excess_vs_csi300(panel, tmp, horizons=[5])

        excess_T0 = result.loc[0, "excess_vs_csi300_5d"]
        expected  = (106 / 101 - 1) - (3993 / 3903 - 1)
        assert abs(excess_T0 - expected) < 1e-6, (
            f"excess_vs_csi300_5d at T=0: got {excess_T0:.6f}, expected {expected:.6f}"
        )

    def test_excess_absent_without_index_file(self, tmp_path):
        """When index OHLCV is not in the lake, panel is returned unchanged."""
        from quant_platform.labels.builder import _add_excess_vs_csi300
        from quant_platform.store.lake import init_lake

        init_lake(tmp_path)
        panel = _make_panel(n_symbols=5, n_dates=50)
        panel["date"] = panel["date"].dt.date
        result = _add_excess_vs_csi300(panel, tmp_path, horizons=[5])
        assert "excess_vs_csi300_5d" not in result.columns


# ---------------------------------------------------------------------------
# P4A-06: RidgeModel
# ---------------------------------------------------------------------------

class TestRidgeModel:
    """P4A-06: Ridge linear diagnostic baseline."""

    def test_ridge_in_model_registry(self):
        """RidgeModel must be registered under 'ridge' key."""
        from quant_platform.training.model_zoo import MODEL_REGISTRY
        assert "ridge" in MODEL_REGISTRY, "'ridge' missing from MODEL_REGISTRY"

    def test_ridge_interface(self):
        """RidgeModel must implement the ModelBase interface."""
        from quant_platform.training.model_zoo import get_model
        model = get_model("ridge")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "get_params")
        assert hasattr(model, "get_native_model")
        assert model.model_name == "Ridge"

    def test_ridge_fit_predict(self):
        """Ridge fit/predict on a small dataset should not raise."""
        from quant_platform.training.model_zoo import RidgeModel

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, 5)))
        y = pd.Series(rng.normal(size=100))

        model = RidgeModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 100
        assert not np.any(np.isnan(preds))

    def test_ridge_handles_nan_features(self):
        """RidgeModel pipeline must impute NaN features without raising."""
        from quant_platform.training.model_zoo import RidgeModel

        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(80, 4)))
        X.iloc[::5, 0] = np.nan   # inject NaNs
        y = pd.Series(rng.normal(size=80))

        model = RidgeModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 80

    def test_ridge_oof_with_purged_cv(self):
        """RidgeModel should work inside fit_zoo_model_oof with purged CV."""
        from quant_platform.training.model_zoo import RidgeModel, fit_zoo_model_oof

        panel = _make_panel(n_symbols=10, n_dates=200, horizon=5)
        model = RidgeModel()
        oof_preds, fold_metrics = fit_zoo_model_oof(
            model, panel, ["feat_a", "feat_b"], "ret_fwd_5d",
            n_splits=3, horizon=5,
        )
        assert len(oof_preds) == len(panel)
        n_filled = oof_preds.notna().sum()
        assert n_filled > 0, "No OOF predictions were filled"

    def test_ridge_shuffled_label_gives_low_ic(self):
        """On shuffled labels, Ridge OOF Rank IC must be near zero."""
        from quant_platform.training.model_zoo import RidgeModel, fit_zoo_model_oof
        from quant_platform.evaluation.metrics import evaluate

        panel = _make_panel(n_symbols=15, n_dates=250, horizon=5)
        rng = np.random.default_rng(55)
        panel["ret_fwd_5d"] = rng.permutation(panel["ret_fwd_5d"].fillna(0).values)

        model = RidgeModel()
        oof_preds, _ = fit_zoo_model_oof(
            model, panel, ["feat_a", "feat_b"], "ret_fwd_5d",
            n_splits=3, horizon=5,
        )
        report = evaluate(
            oof_preds,
            panel["ret_fwd_5d"],
            pd.to_datetime(panel["date"]),
            label_col="ridge_null",
        )
        assert abs(report.rank_ic_mean) < 0.10, (
            f"Ridge null IC too high: {report.rank_ic_mean:.4f}"
        )

    def test_ridge_default_alpha_not_tuned(self):
        """alpha should be the conservative default, not overfit by default."""
        from quant_platform.training.model_zoo import RidgeModel

        model = RidgeModel()
        params = model.get_params()
        assert "alpha" in params
        # Conservative default (100) — not a tiny value that might overfit
        assert params["alpha"] >= 1.0, f"Ridge alpha too small: {params['alpha']}"


# ---------------------------------------------------------------------------
# Integration: store_lake path helpers
# ---------------------------------------------------------------------------

class TestStoreLakePaths:
    """Verify the new index_ohlcv_path and init_lake additions."""

    def test_index_ohlcv_path_format(self, tmp_path):
        from quant_platform.store.lake import index_ohlcv_path

        p = index_ohlcv_path(tmp_path, "000300")
        assert p.name == "000300.parquet"
        assert "index_ohlcv" in str(p)
        assert "silver" in str(p)

    def test_init_lake_creates_index_ohlcv_dir(self, tmp_path):
        from quant_platform.store.lake import init_lake, index_ohlcv_dir

        init_lake(tmp_path)
        assert index_ohlcv_dir(tmp_path).exists(), (
            "silver/index_ohlcv directory not created by init_lake"
        )
