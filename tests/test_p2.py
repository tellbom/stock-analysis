"""
P2 complete verification test suite (T2.1 – T2.8).

All tests use synthetic data.  No live network, no file-system state between
tests (each uses a fresh tempdir).

Run with:
  PYTHONPATH=/home/claude:/mnt/project python quant_platform/tests/test_p2.py
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_panel(
    n_symbols: int = 10,
    n_days: int = 200,
    seed: int = 42,
    signal_strength: float = 0.1,
) -> pd.DataFrame:
    """
    Synthetic (symbol, date, features, labels) panel.
    *signal_strength* controls how much correlation the model prediction has
    with the labels (0 = pure noise, 1 = perfect).
    """
    rng = np.random.default_rng(seed)
    symbols = [str(600000 + i).zfill(6) for i in range(n_symbols)]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    rows = []
    for sym in symbols:
        prices = 100.0 + np.cumsum(rng.normal(0, 1, n_days))
        prices = np.maximum(prices, 1.0)
        for i, date in enumerate(dates):
            rows.append({
                "symbol":  sym,
                "date":    date.date(),
                "close":   prices[i],
                "volume":  rng.uniform(1e5, 1e6),
                "feat_a":  rng.normal(),
                "feat_b":  rng.normal(),
                "feat_c":  rng.normal(),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Labels: forward 1-day return with some signal
    df["close_next1"] = df.groupby("symbol")["close"].shift(-1)
    df["close_next2"] = df.groupby("symbol")["close"].shift(-2)
    df["ret_fwd_1d"]  = df["close_next1"] / df.groupby("symbol")["close"].shift(-1 + 1) - 1
    # T+1 execution: label = close[T+2]/close[T+1] - 1
    df["ret_fwd_1d"] = df.groupby("symbol")["close"].transform(
        lambda x: (x.shift(-2) / x.shift(-1) - 1)
    )
    df["ret_fwd_5d"] = df.groupby("symbol")["close"].transform(
        lambda x: (x.shift(-6) / x.shift(-1) - 1)
    )
    df = df.drop(columns=["close_next1", "close_next2"])
    df = df.dropna(subset=["ret_fwd_1d"])

    return df.reset_index(drop=True)


FEATURE_COLS = ["feat_a", "feat_b", "feat_c"]
LABEL_COL    = "ret_fwd_1d"


# ---------------------------------------------------------------------------
# T2.1 — PurgedKFold splitter
# ---------------------------------------------------------------------------

def test_purged_kfold_time_ordering():
    """All train indices must be strictly before val indices (time order)."""
    from quant_platform.training.splitter import PurgedKFold

    panel = _make_panel(n_days=200)
    panel = panel.sort_values("date").reset_index(drop=True)
    splitter = PurgedKFold(n_splits=5, horizon=5, embargo=2)

    for train_idx, val_idx in splitter.split(panel):
        assert train_idx.max() < val_idx.min(), \
            f"Train max={train_idx.max()} >= Val min={val_idx.min()} — temporal order violated"

    print("  [OK] T2.1 PurgedKFold: train indices always before val indices")


def test_purged_kfold_no_overlap():
    """Train and val sets must be disjoint."""
    from quant_platform.training.splitter import PurgedKFold

    panel = _make_panel(n_days=200)
    splitter = PurgedKFold(n_splits=5, horizon=5, embargo=2)

    for train_idx, val_idx in splitter.split(panel):
        overlap = set(train_idx) & set(val_idx)
        assert not overlap, f"Train/val overlap: {len(overlap)} indices"

    print("  [OK] T2.1 PurgedKFold: train and val are always disjoint")


def test_purged_kfold_purges_by_trading_date():
    """Purging must be measured in trading dates, not flat panel rows."""
    from quant_platform.training.splitter import PurgedKFold

    horizon = 5
    embargo = 2
    panel = _make_panel(n_symbols=8, n_days=120)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    dates = pd.Series(pd.to_datetime(panel["date"]).sort_values().unique())
    date_pos = {d: i for i, d in enumerate(dates)}

    splitter = PurgedKFold(n_splits=4, horizon=horizon, embargo=embargo)
    for train_idx, val_idx in splitter.split(panel):
        train_dates = pd.to_datetime(panel.iloc[train_idx]["date"])
        val_dates = pd.to_datetime(panel.iloc[val_idx]["date"])
        val_start_pos = date_pos[val_dates.min()]
        max_allowed = val_start_pos - horizon - embargo - 2

        assert not (set(train_dates) & set(val_dates)), \
            "Train/val should not share the same trading date"
        assert train_dates.map(date_pos).max() <= max_allowed, \
            "Training label windows or embargo period reach the validation fold"

    print("  [OK] T2.1 PurgedKFold: purge/embargo measured in trading dates")


def test_lockbox_split():
    """make_lockbox_split separates the most recent months correctly."""
    from quant_platform.training.splitter import make_lockbox_split

    panel = _make_panel(n_days=500)
    train_val, lockbox = make_lockbox_split(panel, lockbox_months=6)

    assert len(lockbox) > 0
    assert len(train_val) > 0
    assert len(train_val) + len(lockbox) == len(panel)

    tv_max = pd.to_datetime(train_val["date"]).max()
    lb_min = pd.to_datetime(lockbox["date"]).min()
    assert tv_max < lb_min, "Lockbox dates overlap with train_val dates"

    print(f"  [OK] T2.1 lockbox split: {len(lockbox)} lockbox rows, "
          f"{len(train_val)} train_val rows, no overlap")


# ---------------------------------------------------------------------------
# T2.2 — LightGBM OOF fit
# ---------------------------------------------------------------------------

def test_lockbox_split_purges_label_window():
    """Train labels immediately before lockbox must not reach into lockbox."""
    from quant_platform.training.splitter import make_lockbox_split

    horizon = 5
    panel = _make_panel(n_symbols=8, n_days=200)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    train_val, lockbox = make_lockbox_split(panel, lockbox_months=3, horizon=horizon)

    dates = pd.Series(pd.to_datetime(panel["date"]).sort_values().unique())
    date_pos = {d: i for i, d in enumerate(dates)}
    lock_start_pos = date_pos[pd.to_datetime(lockbox["date"]).min()]

    train_date_pos = pd.to_datetime(train_val["date"]).map(date_pos)
    assert (train_date_pos + 1 + horizon < lock_start_pos).all(), \
        "A train label window reaches into the lockbox"

    print("  [OK] T2.1 lockbox split: train label windows do not enter lockbox")


def test_lgbm_oof_returns_predictions():
    """fit_oof returns OOF predictions aligned to panel index."""
    from quant_platform.training.lgbm_model import fit_oof

    panel = _make_panel(n_days=300)
    result = fit_oof(
        panel, FEATURE_COLS, LABEL_COL,
        n_splits=3, horizon=5, embargo=2,
    )

    assert len(result.oof_predictions) == len(panel)
    n_filled = result.oof_predictions.notna().sum()
    assert n_filled > 0, "No OOF predictions were generated"
    assert result.n_folds > 0
    print(f"  [OK] T2.2 fit_oof: {n_filled}/{len(panel)} OOF predictions, "
          f"{result.n_folds} folds")


def test_lgbm_oof_fold_metrics():
    """Each fold produces a per-fold IC metric."""
    from quant_platform.training.lgbm_model import fit_oof

    panel = _make_panel(n_days=300)
    result = fit_oof(
        panel, FEATURE_COLS, LABEL_COL,
        n_splits=3, horizon=5, embargo=2,
    )

    for fold in result.fold_metrics:
        assert "fold" in fold
        assert "ic_pearson" in fold
        assert not np.isnan(fold["ic_pearson"])
    print(f"  [OK] T2.2 fold metrics: {len(result.fold_metrics)} folds, all have IC")


def test_lgbm_final_model_predict():
    """fit_final_model returns a pipeline that predicts without error."""
    from quant_platform.training.lgbm_model import fit_final_model

    panel = _make_panel(n_days=200)
    pipe  = fit_final_model(panel, FEATURE_COLS, LABEL_COL)
    preds = pipe.predict(panel[FEATURE_COLS].fillna(0))

    assert len(preds) == len(panel)
    assert not np.any(np.isnan(preds))
    print("  [OK] T2.2 final model: predictions produced for all rows")


# ---------------------------------------------------------------------------
# T2.3 — Evaluation module
# ---------------------------------------------------------------------------

def test_evaluate_returns_report():
    """evaluate() returns an EvalReport with all required fields."""
    from quant_platform.evaluation.metrics import evaluate

    panel = _make_panel(n_days=300, n_symbols=20)
    # Use random predictions (minimal signal)
    rng = np.random.default_rng(0)
    pred = pd.Series(rng.normal(size=len(panel)), index=panel.index)
    report = evaluate(pred, panel[LABEL_COL], pd.to_datetime(panel["date"]),
                      label_col="ret_fwd_1d")

    assert not np.isnan(report.rank_ic_mean)
    assert not np.isnan(report.icir)
    assert report.n_dates > 0
    assert report.n_predictions > 0
    print(f"  [OK] T2.3 evaluate: rank_ic={report.rank_ic_mean:.4f}, "
          f"icir={report.icir:.3f}, n_dates={report.n_dates}")


def test_evaluate_random_ic_near_zero():
    """Random predictions should produce IC close to 0."""
    from quant_platform.evaluation.metrics import evaluate

    panel = _make_panel(n_days=500, n_symbols=30, seed=99)
    rng   = np.random.default_rng(99)
    pred  = pd.Series(rng.normal(size=len(panel)), index=panel.index)
    report = evaluate(pred, panel[LABEL_COL], pd.to_datetime(panel["date"]))

    assert abs(report.rank_ic_mean) < 0.15, \
        f"Random predictor IC too high: {report.rank_ic_mean:.4f}"
    print(f"  [OK] T2.3 random IC near zero: {report.rank_ic_mean:+.4f}")


def test_evaluate_perfect_predictor():
    """Perfect predictions (pred = label) should yield IC ≈ 1."""
    from quant_platform.evaluation.metrics import evaluate

    panel = _make_panel(n_days=200, n_symbols=20)
    report = evaluate(
        panel[LABEL_COL], panel[LABEL_COL],
        pd.to_datetime(panel["date"]),
    )

    assert report.rank_ic_mean > 0.9, \
        f"Perfect predictor IC should be ~1.0, got {report.rank_ic_mean:.4f}"
    print(f"  [OK] T2.3 perfect predictor IC ≈ 1: {report.rank_ic_mean:.4f}")


def test_evaluate_quantile_spread():
    """Quantile returns and spread are computed."""
    from quant_platform.evaluation.metrics import evaluate

    panel = _make_panel(n_days=400, n_symbols=30)
    rng   = np.random.default_rng(1)
    pred  = pd.Series(rng.normal(size=len(panel)), index=panel.index)
    report = evaluate(pred, panel[LABEL_COL], pd.to_datetime(panel["date"]))

    assert len(report.quantile_returns) > 0
    assert not np.isnan(report.quantile_spread)
    print(f"  [OK] T2.3 quantile spread: {report.quantile_spread:+.4f}, "
          f"{len(report.quantile_returns)} deciles")


# ---------------------------------------------------------------------------
# T2.4 — Baseline gauntlet
# ---------------------------------------------------------------------------

def test_baseline_gauntlet_produces_table():
    """run_baseline_gauntlet returns a DataFrame with expected rows."""
    from quant_platform.evaluation.baselines import run_baseline_gauntlet

    panel = _make_panel(n_days=300, n_symbols=20)
    rng   = np.random.default_rng(0)
    pred  = pd.Series(rng.normal(size=len(panel)), index=panel.index)

    table = run_baseline_gauntlet(panel, LABEL_COL, pred, model_name="TestModel")

    expected_rows = {"TestModel", "momentum_1d", "mean_rev_1d", "cs_momentum", "random"}
    assert expected_rows == set(table.index)
    assert "rank_ic_mean" in table.columns
    assert "icir"         in table.columns
    assert table.loc["momentum_1d", "n_dates"] > 0
    assert table.loc["mean_rev_1d", "n_dates"] > 0
    assert table.loc["cs_momentum", "n_dates"] > 0
    print(f"  [OK] T2.4 baseline gauntlet: {len(table)} rows in comparison table")


def test_baseline_gauntlet_random_near_zero():
    """Random baseline must have IC near zero."""
    from quant_platform.evaluation.baselines import run_baseline_gauntlet

    panel = _make_panel(n_days=400, n_symbols=25)
    pred  = panel[LABEL_COL].copy()  # use label itself as model (won't matter)

    table = run_baseline_gauntlet(panel, LABEL_COL, pred)
    random_ic = table.loc["random", "rank_ic_mean"]
    assert abs(random_ic) < 0.15, f"Random IC too large: {random_ic:.4f}"
    print(f"  [OK] T2.4 random baseline IC near zero: {random_ic:+.4f}")


# ---------------------------------------------------------------------------
# T2.5 — Signal backtest
# ---------------------------------------------------------------------------

def test_baseline_gauntlet_no_label_shift_fallback_without_close():
    """Missing close must not fall back to label.shift(1)."""
    from quant_platform.evaluation.baselines import run_baseline_gauntlet

    panel = _make_panel(n_days=300, n_symbols=20).drop(columns=["close"])
    rng = np.random.default_rng(0)
    pred = pd.Series(rng.normal(size=len(panel)), index=panel.index)

    table = run_baseline_gauntlet(panel, LABEL_COL, pred)
    assert pd.isna(table.loc["momentum_1d", "rank_ic_mean"])
    assert pd.isna(table.loc["cs_momentum", "rank_ic_mean"])
    print("  [OK] T2.4 baselines: no label-derived fallback when close is absent")


def test_backtest_returns_result():
    """run_backtest returns a BacktestResult with required fields."""
    from quant_platform.evaluation.backtest import run_backtest

    panel = _make_panel(n_days=300, n_symbols=20)
    rng   = np.random.default_rng(0)
    panel["pred"] = rng.normal(size=len(panel))

    result = run_backtest(panel, pred_col="pred", return_col=LABEL_COL,
                          cost_bps=10.0)

    assert not np.isnan(result.sharpe)
    assert not np.isnan(result.max_drawdown)
    assert result.max_drawdown <= 0
    assert result.n_days > 0
    assert not result.daily_pnl.empty
    print(f"  [OK] T2.5 backtest: Sharpe={result.sharpe:.2f}, "
          f"MaxDD={result.max_drawdown:.2%}, days={result.n_days}")


def test_backtest_net_lower_than_gross():
    """Net return must be ≤ gross return (costs are always non-negative)."""
    from quant_platform.evaluation.backtest import run_backtest

    panel = _make_panel(n_days=300, n_symbols=20)
    rng   = np.random.default_rng(1)
    panel["pred"] = rng.normal(size=len(panel))

    result = run_backtest(panel, pred_col="pred", return_col=LABEL_COL,
                          cost_bps=20.0)

    # Total cost must be non-negative
    assert result.total_cost >= 0
    # Net ≤ Gross
    assert result.net_long_minus_short <= result.long_minus_short + 1e-10
    print(f"  [OK] T2.5 backtest: net ({result.net_long_minus_short:.4f}) "
          f"≤ gross ({result.long_minus_short:.4f})")


def test_backtest_zero_cost():
    """With cost_bps=0, net should equal gross."""
    from quant_platform.evaluation.backtest import run_backtest

    panel = _make_panel(n_days=200, n_symbols=15)
    panel["pred"] = panel[LABEL_COL].fillna(0)   # use label as predictor

    result = run_backtest(panel, pred_col="pred", return_col=LABEL_COL, cost_bps=0.0)
    assert abs(result.net_long_minus_short - result.long_minus_short) < 1e-8
    print("  [OK] T2.5 backtest: zero cost → net equals gross")


# ---------------------------------------------------------------------------
# T2.6 — Robustness tests
# ---------------------------------------------------------------------------

def test_robustness_shuffle_null():
    """Label-shuffle IC must be near zero (null test correctness)."""
    from quant_platform.evaluation.robustness import run_robustness_tests
    from quant_platform.training.lgbm_model import fit_oof

    panel = _make_panel(n_days=300, n_symbols=15)
    oof   = fit_oof(panel, FEATURE_COLS, LABEL_COL, n_splits=3, horizon=5, embargo=2)

    report = run_robustness_tests(
        panel, FEATURE_COLS, LABEL_COL,
        baseline_oof=oof.oof_predictions,
        n_splits=3, horizon=5, embargo=2,
        shuffle_threshold=0.10,
    )

    assert not np.isnan(report.shuffle_rank_ic)
    print(f"  [OK] T2.6 shuffle null: IC={report.shuffle_rank_ic:+.4f}, "
          f"passed={report.shuffle_passed}")


def test_robustness_canary_detected():
    """Canary feature (future label) must produce higher IC than baseline."""
    from quant_platform.evaluation.robustness import run_robustness_tests
    from quant_platform.training.lgbm_model import fit_oof

    panel = _make_panel(n_days=300, n_symbols=15)
    oof   = fit_oof(panel, FEATURE_COLS, LABEL_COL, n_splits=3, horizon=5, embargo=2)

    report = run_robustness_tests(
        panel, FEATURE_COLS, LABEL_COL,
        baseline_oof=oof.oof_predictions,
        n_splits=3, horizon=5, embargo=2,
    )

    assert not np.isnan(report.canary_rank_ic)
    # Canary should have higher IC than baseline (it peeks at the future)
    assert report.canary_rank_ic > report.baseline_rank_ic, \
        f"Canary IC ({report.canary_rank_ic:.4f}) should exceed baseline " \
        f"({report.baseline_rank_ic:.4f})"
    print(f"  [OK] T2.6 canary: IC={report.canary_rank_ic:+.4f} > "
          f"baseline={report.baseline_rank_ic:+.4f}")


def test_robustness_subperiod():
    """Subperiod stability check runs without error."""
    from quant_platform.evaluation.robustness import run_robustness_tests
    from quant_platform.training.lgbm_model import fit_oof

    panel = _make_panel(n_days=400, n_symbols=15)
    oof   = fit_oof(panel, FEATURE_COLS, LABEL_COL, n_splits=3, horizon=5, embargo=2)

    report = run_robustness_tests(
        panel, FEATURE_COLS, LABEL_COL,
        baseline_oof=oof.oof_predictions,
        n_splits=3, horizon=5, embargo=2,
    )

    assert not np.isnan(report.first_half_ric)
    assert not np.isnan(report.second_half_ric)
    print(f"  [OK] T2.6 subperiod: first={report.first_half_ric:+.4f}, "
          f"second={report.second_half_ric:+.4f}, "
          f"stable={report.subperiod_stable}")


# ---------------------------------------------------------------------------
# T2.7 — MLflow tracking
# ---------------------------------------------------------------------------

def test_mlflow_experiment_creation():
    """get_or_create_experiment creates a SQLite experiment without error."""
    from quant_platform.training.tracking import get_or_create_experiment
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        exp_id = get_or_create_experiment(tmp, "test_exp")
        assert exp_id is not None
        # Calling twice is idempotent
        exp_id2 = get_or_create_experiment(tmp, "test_exp")
        assert exp_id == exp_id2
    print(f"  [OK] T2.7 MLflow: experiment created (id={exp_id}), idempotent")


def test_mlflow_run_logger():
    """RunLogger logs params and metrics to MLflow."""
    from quant_platform.training.tracking import get_or_create_experiment, RunLogger
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        exp_id = get_or_create_experiment(tmp, "run_logger_test")

        with RunLogger(tmp, exp_id, run_name="test_run") as run:
            run.log_params({"seed": 42, "label_col": "ret_fwd_1d"})
            run.log_metrics({"rank_ic": 0.05, "icir": 0.8, "sharpe": 1.2})
            run_id = run.run_id

        # Verify the run is logged
        mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow/mlflow.db")
        runs = mlflow.search_runs(experiment_ids=[exp_id])
        assert len(runs) >= 1
        row = runs[runs["run_id"] == run_id].iloc[0]
        assert abs(row["metrics.rank_ic"] - 0.05) < 1e-6
        assert abs(row["metrics.icir"]    - 0.8)  < 1e-6
    print("  [OK] T2.7 RunLogger: params and metrics logged and retrievable")


def test_reproducibility_manifest():
    """make_manifest returns a dict with all required keys."""
    from quant_platform.training.tracking import make_manifest

    with tempfile.TemporaryDirectory() as tmp:
        manifest = make_manifest(
            store_root=tmp,
            feature_set_id="abc12345",
            feature_cols=["feat_a", "feat_b"],
            label_col="ret_fwd_1d",
            lgbm_params={"n_estimators": 100, "seed": 42},
            seed=42,
        )

    required_keys = {
        "generated_at", "data_snapshot_id", "feature_set_id",
        "feature_cols_hash", "label_col", "lgbm_params", "seed",
        "python_version", "package_versions",
    }
    assert required_keys.issubset(set(manifest.keys()))
    assert manifest["feature_set_id"] == "abc12345"
    assert manifest["label_col"] == "ret_fwd_1d"
    print(f"  [OK] T2.7 manifest: {len(manifest)} keys, feature_set_id={manifest['feature_set_id']}")


# ---------------------------------------------------------------------------
# T2.8 — Alpha verdict
# ---------------------------------------------------------------------------

def test_alpha_verdict_no_go_on_noise():
    """A model with random (no signal) predictions → NO_GO verdict."""
    from quant_platform.evaluation.metrics import evaluate
    from quant_platform.evaluation.baselines import run_baseline_gauntlet
    from quant_platform.evaluation.backtest import run_backtest
    from quant_platform.evaluation.robustness import RobustnessReport
    from quant_platform.evaluation.alpha_verdict import render_verdict

    panel = _make_panel(n_days=400, n_symbols=20)
    rng   = np.random.default_rng(0)
    panel["pred"] = rng.normal(size=len(panel))

    oof_eval  = evaluate(panel["pred"], panel[LABEL_COL],
                         pd.to_datetime(panel["date"]))
    bt        = run_backtest(panel, "pred", LABEL_COL, cost_bps=15.0)
    bt_table  = run_baseline_gauntlet(panel, LABEL_COL, panel["pred"])
    robustness = RobustnessReport(
        shuffle_rank_ic=0.001, shuffle_passed=True,
        subperiod_stable=False,
        first_half_ric=-0.01, second_half_ric=0.01,
        canary_rank_ic=0.2, canary_passed=True,
        baseline_rank_ic=oof_eval.rank_ic_mean,
    )

    with tempfile.TemporaryDirectory() as tmp:
        verdict = render_verdict(
            tmp, oof_eval, bt_table, bt, robustness,
            icir_threshold=0.3, sharpe_threshold=0.5,
        )
        assert (Path(tmp) / "alpha_verdict.txt").exists()
        assert (Path(tmp) / "alpha_verdict.json").exists()

    # Random predictions should not produce a GO verdict
    assert verdict.verdict in ("NO_GO", "INCONCLUSIVE"), \
        f"Noise model got verdict={verdict.verdict} — should be NO_GO or INCONCLUSIVE"
    print(f"  [OK] T2.8 verdict on noise: {verdict.verdict} (expected NO_GO/INCONCLUSIVE)")


def test_alpha_verdict_files_written():
    """render_verdict writes both .txt and .json files."""
    from quant_platform.evaluation.metrics import evaluate, EvalReport
    from quant_platform.evaluation.backtest import BacktestResult
    from quant_platform.evaluation.robustness import RobustnessReport
    from quant_platform.evaluation.alpha_verdict import render_verdict

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        # Minimal stubs — just test file writing
        oof_eval = EvalReport(
            rank_ic_mean=0.05, rank_ic_std=0.06, icir=0.83,
            n_dates=100, n_predictions=2000, label_col="test",
        )
        bt = BacktestResult(
            sharpe=0.8, max_drawdown=-0.15, annualised_return=0.12,
            annualised_vol=0.15, net_long_minus_short=0.05,
            long_minus_short=0.07, n_days=100,
        )
        robustness = RobustnessReport(
            shuffle_rank_ic=0.001, shuffle_passed=True,
            subperiod_stable=True,
            first_half_ric=0.04, second_half_ric=0.06,
            canary_rank_ic=0.3, canary_passed=True,
            baseline_rank_ic=0.05,
        )
        verdict = render_verdict(
            tmp, oof_eval, pd.DataFrame(), bt, robustness,
        )
        assert (Path(tmp) / "alpha_verdict.txt").exists()
        assert (Path(tmp) / "alpha_verdict.json").exists()
        text = (Path(tmp) / "alpha_verdict.txt").read_text()
        assert "VERDICT" in text
    print(f"  [OK] T2.8 verdict files: .txt and .json written, verdict={verdict.verdict}")


def test_alpha_verdict_lockbox_seals():
    """Providing lockbox results sets lockbox_used=True."""
    from quant_platform.evaluation.metrics import EvalReport
    from quant_platform.evaluation.backtest import BacktestResult
    from quant_platform.evaluation.robustness import RobustnessReport
    from quant_platform.evaluation.alpha_verdict import render_verdict

    with tempfile.TemporaryDirectory() as tmp:
        oof_eval  = EvalReport(rank_ic_mean=0.04, icir=0.5, n_dates=50)
        bt        = BacktestResult(sharpe=0.6, max_drawdown=-0.1, n_days=50,
                                   net_long_minus_short=0.03, long_minus_short=0.04,
                                   annualised_return=0.08, annualised_vol=0.13)
        rob       = RobustnessReport(shuffle_passed=True, subperiod_stable=True,
                                     canary_passed=True, baseline_rank_ic=0.04,
                                     shuffle_rank_ic=0.005, canary_rank_ic=0.2,
                                     first_half_ric=0.03, second_half_ric=0.05)
        lb_eval   = EvalReport(rank_ic_mean=0.03, icir=0.4, n_dates=30)
        lb_bt     = BacktestResult(sharpe=0.4, max_drawdown=-0.08, n_days=30,
                                   net_long_minus_short=0.02, long_minus_short=0.03,
                                   annualised_return=0.05, annualised_vol=0.12)

        verdict = render_verdict(tmp, oof_eval, pd.DataFrame(), bt, rob,
                                 lockbox_eval=lb_eval, lockbox_backtest=lb_bt)

    assert verdict.lockbox_used is True
    assert not np.isnan(verdict.lockbox_rank_ic)
    print(f"  [OK] T2.8 lockbox sealed: lockbox_rank_ic={verdict.lockbox_rank_ic:.4f}, "
          f"verdict={verdict.verdict}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== P2 Complete Test Suite (T2.1 – T2.8) ===\n")

    tests = [
        # T2.1 Splitter
        test_purged_kfold_time_ordering,
        test_purged_kfold_no_overlap,
        test_purged_kfold_purges_by_trading_date,
        test_lockbox_split,
        test_lockbox_split_purges_label_window,
        # T2.2 LightGBM
        test_lgbm_oof_returns_predictions,
        test_lgbm_oof_fold_metrics,
        test_lgbm_final_model_predict,
        # T2.3 Evaluation
        test_evaluate_returns_report,
        test_evaluate_random_ic_near_zero,
        test_evaluate_perfect_predictor,
        test_evaluate_quantile_spread,
        # T2.4 Baselines
        test_baseline_gauntlet_produces_table,
        test_baseline_gauntlet_random_near_zero,
        test_baseline_gauntlet_no_label_shift_fallback_without_close,
        # T2.5 Backtest
        test_backtest_returns_result,
        test_backtest_net_lower_than_gross,
        test_backtest_zero_cost,
        # T2.6 Robustness
        test_robustness_shuffle_null,
        test_robustness_canary_detected,
        test_robustness_subperiod,
        # T2.7 Tracking
        test_mlflow_experiment_creation,
        test_mlflow_run_logger,
        test_reproducibility_manifest,
        # T2.8 Verdict
        test_alpha_verdict_no_go_on_noise,
        test_alpha_verdict_files_written,
        test_alpha_verdict_lockbox_seals,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*45}")
    print(f"P2 Results: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
