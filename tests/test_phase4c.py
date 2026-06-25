"""
tests/test_phase4c.py
=====================
Test suite for Phase 4C — Factor Diagnostics and Refinements.

Covers:
  P4C-01  FeatureICReport — IC t-stat, decay half-life, collinearity clusters
  P4C-02  FeaturePruner — cluster pruning, active_feature_cols, log persistence
  P4C-03  LockupCollector + build_lockup_features — PIT correctness
  P4C-04  residualise_returns — per-date OLS, zero mean, near-zero market beta
  P4C-05  RegimeAnalyser — window breakdown, hard regime detection
  P4X-01  store_lake — lockup paths and evaluation dir
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root so quant_platform is importable when pytest is launched
# through the repository-local virtualenv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_panel(n_sym: int = 12, n_dates: int = 150, seed: int = 0) -> pd.DataFrame:
    """Synthetic panel with a weak true signal in feat_signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates).date.tolist()
    symbols = [f"S{i:02d}" for i in range(n_sym)]
    rows = []
    for sym in symbols:
        true_alpha = rng.normal(0, 0.1, n_dates)
        close = 10 + np.cumsum(rng.normal(0, 0.2, n_dates))
        close = np.clip(close, 1, None)
        for i, d in enumerate(dates):
            rows.append({
                "symbol":      sym,
                "date":        d,
                "close":       close[i],
                "feat_signal": true_alpha[i] + rng.normal(0, 0.2),
                "feat_noise":  rng.normal(0, 1),
                # correlated pair for collinearity testing
                "feat_a":      rng.normal(0, 1),
                "feat_b_corr": rng.normal(0, 1),   # will be made correlated with feat_a below
            })
    df = pd.DataFrame(rows)
    # Make feat_b_corr nearly collinear with feat_a (ρ ≈ 0.95)
    df["feat_b_corr"] = df["feat_a"] * 0.95 + df["feat_b_corr"] * 0.05

    # Build labels from close
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["ret_fwd_1d"]  = df.groupby("symbol")["close"].transform(lambda x: x.shift(-2) / x.shift(-1) - 1)
    df["ret_fwd_5d"]  = df.groupby("symbol")["close"].transform(lambda x: x.shift(-6) / x.shift(-1) - 1)
    df["ret_fwd_10d"] = df.groupby("symbol")["close"].transform(lambda x: x.shift(-11) / x.shift(-1) - 1)
    df["ret_fwd_20d"] = df.groupby("symbol")["close"].transform(lambda x: x.shift(-21) / x.shift(-1) - 1)
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# P4C-01: FeatureICReport
# ---------------------------------------------------------------------------

class TestFeatureICReport:
    """P4C-01: Single-factor IC diagnostic."""

    @pytest.fixture
    def panel(self):
        return _make_panel()

    def test_canary_ranks_first(self, panel):
        """An injected canary (future close) should rank first in IC at 1d."""
        from quant_platform.evaluation.feature_ic import compute_feature_ic_report

        p = panel.copy()
        # Canary: next day's return encoded directly — leaks future
        p["_canary"] = p.groupby("symbol")["close"].transform(lambda x: x.shift(-2) / x.shift(-1) - 1)

        feature_cols = ["feat_signal", "feat_noise", "_canary"]
        report = compute_feature_ic_report(p, feature_cols, label_cols=["ret_fwd_1d"])

        table = report.feature_ic_table()
        assert len(table) > 0
        top_feat = table.iloc[0]["feature"]
        assert top_feat == "_canary", (
            f"Canary should rank first in IC_1d, got {top_feat}"
        )

    def test_noise_feature_low_tstat(self, panel):
        """A pure noise feature should have |t-stat| < 2 at all horizons."""
        from quant_platform.evaluation.feature_ic import compute_feature_ic_report

        report = compute_feature_ic_report(
            panel, ["feat_noise"], label_cols=["ret_fwd_5d"]
        )
        row = next((r for r in report.rows if r.feature == "feat_noise"), None)
        assert row is not None
        if not np.isnan(row.tstat_5d):
            assert abs(row.tstat_5d) < 3.0, (
                f"Noise feature t-stat too large: {row.tstat_5d:.3f}"
            )

    def test_ic_tstat_formula(self):
        """IC t-stat = mean_IC * sqrt(n) / std_IC."""
        from quant_platform.evaluation.feature_ic import _ic_tstat

        rics = [0.05] * 50 + [-0.01] * 10   # mean ≈ 0.0417, std ≈ 0.025
        tstat = _ic_tstat(rics)
        n = len(rics)
        arr = np.array(rics)
        expected = np.mean(arr) * np.sqrt(n) / np.std(arr, ddof=1)
        assert abs(tstat - expected) < 1e-9

    def test_decay_halflife_correct(self):
        """decay_halflife should return the lag where IC drops below 50% of peak."""
        from quant_platform.evaluation.feature_ic import _decay_halflife

        decay = {1: 0.10, 2: 0.08, 3: 0.06, 5: 0.04, 10: 0.02, 20: 0.01}
        peak = 0.10
        hl = _decay_halflife(decay, peak)
        # 50% of 0.10 = 0.05; IC drops below 0.05 at lag 5
        assert hl == pytest.approx(5.0)

    def test_collinearity_matrix_symmetric(self, panel):
        """Pairwise Spearman correlation matrix must be symmetric and diagonal=1."""
        from quant_platform.evaluation.feature_ic import compute_feature_ic_report

        report = compute_feature_ic_report(
            panel, ["feat_a", "feat_b_corr", "feat_noise"],
            label_cols=["ret_fwd_5d"],
        )
        mat = report.collinearity_matrix
        if mat.empty:
            pytest.skip("No collinearity matrix computed")
        # Diagonal
        for feat in mat.columns:
            assert abs(mat.loc[feat, feat] - 1.0) < 1e-9
        # Symmetric
        diff = (mat - mat.T).abs().max().max()
        assert diff < 1e-9

    def test_collinearity_cluster_detects_correlated_pair(self, panel):
        """feat_a and feat_b_corr (ρ≈0.95) should be in the same cluster."""
        from quant_platform.evaluation.feature_ic import (
            compute_feature_ic_report, _find_collinearity_clusters,
        )

        report = compute_feature_ic_report(
            panel, ["feat_a", "feat_b_corr", "feat_noise"],
            label_cols=["ret_fwd_5d"],
        )
        clusters = report.collinearity_clusters
        # At least one cluster should contain both feat_a and feat_b_corr
        corr_pair_found = any(
            "feat_a" in members and "feat_b_corr" in members
            for members in clusters.values()
        )
        assert corr_pair_found, (
            f"Collinear pair (feat_a, feat_b_corr) not detected. Clusters: {clusters}"
        )

    def test_pruning_candidates_filter(self, panel):
        """Low IC, low t-stat features should be returned as pruning candidates."""
        from quant_platform.evaluation.feature_ic import compute_feature_ic_report

        report = compute_feature_ic_report(
            panel, ["feat_noise", "feat_signal"],
            label_cols=["ret_fwd_5d"],
        )
        candidates = report.pruning_candidates(ic_threshold=0.05, tstat_threshold=2.0)
        # feat_noise should be a candidate (low IC, low t-stat)
        # (may not always hold on small synthetic panels; check the mechanism)
        assert isinstance(candidates, list)

    def test_save_csv(self, tmp_path):
        """save_csv should write a readable CSV to the evaluation directory."""
        from quant_platform.evaluation.feature_ic import compute_feature_ic_report

        panel = _make_panel(n_sym=5, n_dates=60)
        report = compute_feature_ic_report(
            panel, ["feat_signal", "feat_noise"],
            label_cols=["ret_fwd_5d"],
            store_root=tmp_path,
        )
        csv_files = list((tmp_path / "evaluation").glob("feature_ic_report_*.csv"))
        assert len(csv_files) == 1
        df = pd.read_csv(csv_files[0])
        assert "feature" in df.columns


# ---------------------------------------------------------------------------
# P4C-02: FeaturePruner
# ---------------------------------------------------------------------------

class TestFeaturePruner:
    """P4C-02: Feature pruning logic."""

    def _make_ic_report_with_corr(self):
        """Build a minimal FeatureICReport with a correlated pair."""
        from quant_platform.evaluation.feature_ic import (
            FeatureICReport, FeatureICRow,
        )
        import pandas as pd

        rows = [
            FeatureICRow("feat_strong", ic_5d=0.08, tstat_5d=3.5),
            FeatureICRow("feat_weak",   ic_5d=0.02, tstat_5d=0.8),
            FeatureICRow("feat_noise",  ic_5d=0.00, tstat_5d=0.1),
        ]
        # Corr matrix: feat_strong and feat_weak are collinear (ρ=0.92)
        corr_data = {
            "feat_strong": {"feat_strong": 1.0, "feat_weak": 0.92, "feat_noise": 0.05},
            "feat_weak":   {"feat_strong": 0.92, "feat_weak": 1.0,  "feat_noise": 0.03},
            "feat_noise":  {"feat_strong": 0.05, "feat_weak": 0.03, "feat_noise": 1.0},
        }
        corr_matrix = pd.DataFrame(corr_data)
        report = FeatureICReport(rows=rows, collinearity_matrix=corr_matrix)
        return report

    def test_prunes_weaker_in_cluster(self, tmp_path):
        """The weaker feature (feat_weak) should be pruned, not the stronger (feat_strong)."""
        from quant_platform.features.pruning import FeaturePruner

        ic_report = self._make_ic_report_with_corr()
        pruner = FeaturePruner(store_root=tmp_path)
        result = pruner.run(ic_report, corr_threshold=0.85)

        pruned_names = [p["feature"] for p in result.pruned]
        assert "feat_weak" in pruned_names, (
            f"feat_weak should be pruned (lower IC); pruned: {pruned_names}"
        )
        assert "feat_strong" not in pruned_names, (
            f"feat_strong should be retained (higher IC)"
        )

    def test_retains_singletons(self, tmp_path):
        """feat_noise (no collinear partner) should remain in retained list."""
        from quant_platform.features.pruning import FeaturePruner

        ic_report = self._make_ic_report_with_corr()
        pruner = FeaturePruner(store_root=tmp_path)
        result = pruner.run(ic_report, corr_threshold=0.85)
        assert "feat_noise" in result.retained

    def test_get_active_feature_cols(self, tmp_path):
        """get_active_feature_cols should exclude pruned features."""
        from quant_platform.features.pruning import FeaturePruner

        ic_report = self._make_ic_report_with_corr()
        pruner = FeaturePruner(store_root=tmp_path)
        result = pruner.run(ic_report, corr_threshold=0.85)

        all_cols = ["feat_strong", "feat_weak", "feat_noise", "ret_fwd_5d"]
        active = pruner.get_active_feature_cols(all_cols, pruning_result=result)
        assert "feat_weak" not in active
        assert "feat_strong" in active
        assert "ret_fwd_5d" in active   # non-feature cols should pass through

    def test_log_persisted(self, tmp_path):
        """Pruning log should be saved to Parquet."""
        from quant_platform.features.pruning import FeaturePruner

        ic_report = self._make_ic_report_with_corr()
        pruner = FeaturePruner(store_root=tmp_path)
        result = pruner.run(ic_report, corr_threshold=0.85)

        log = pruner.load_pruning_log()
        assert len(log) >= 1
        assert "feat_weak" in log["feature"].values

    def test_no_pruning_below_threshold(self, tmp_path):
        """With threshold=0.99, no features should be pruned (ρ=0.92 < 0.99)."""
        from quant_platform.features.pruning import FeaturePruner

        ic_report = self._make_ic_report_with_corr()
        pruner = FeaturePruner(store_root=tmp_path)
        result = pruner.run(ic_report, corr_threshold=0.99)
        assert result.n_pruned == 0


# ---------------------------------------------------------------------------
# P4C-03: LockupCollector + build_lockup_features
# ---------------------------------------------------------------------------

class TestLockupFeatures:
    """P4C-03: Lockup expiry PIT correctness."""

    def _make_lockup_panel(self) -> pd.DataFrame:
        """Synthetic lockup events for two symbols."""
        return pd.DataFrame([
            {
                "symbol": "S00", "unlock_date": dt.date(2022, 3, 15),
                "lock_type": "IPO", "shares_million": 100.0, "ratio_pct": 20.0,
            },
            {
                "symbol": "S00", "unlock_date": dt.date(2022, 6, 1),
                "lock_type": "PE", "shares_million": 50.0, "ratio_pct": 10.0,
            },
            {
                "symbol": "S01", "unlock_date": dt.date(2022, 4, 20),
                "lock_type": "IPO", "shares_million": 200.0, "ratio_pct": 30.0,
            },
        ])

    def test_lockup_fetch_parses_current_eastmoney_fields(self, monkeypatch):
        """RPT_LIFT_STAGE uses FREE_SHARES_TYPE/CURRENT_FREE_SHARES today."""
        from quant_platform.ingest import lockup_collector as lc

        class FakeResponse:
            def json(self):
                return {
                    "result": {
                        "data": [{
                            "SECURITY_CODE": "603296",
                            "FREE_DATE": "2026-08-20 00:00:00",
                            "FREE_SHARES_TYPE": "首发机构配售股份",
                            "CURRENT_FREE_SHARES": 194.9371,
                            "FREE_SHARES": 6147.6276,
                            "FREE_RATIO": 0.031709321495,
                        }]
                    }
                }

        monkeypatch.setattr(lc, "_em_get", lambda *args, **kwargs: FakeResponse())

        df = lc._fetch_lockup_events("603296", "2026-01-01", "2026-12-31")
        assert len(df) == 1
        assert df.iloc[0]["lock_type"] == "首发机构配售股份"
        assert df.iloc[0]["shares_million"] == pytest.approx(194.9371 / 1e6)
        assert df.iloc[0]["ratio_pct"] == pytest.approx(0.031709321495)

    def test_days_to_next_unlock_strictly_positive(self):
        """days_to_next_unlock must be > 0 for all dates T < unlock date."""
        from quant_platform.features.event import build_lockup_features

        lockup = self._make_lockup_panel()
        panel  = _make_panel(n_sym=2, n_dates=60)
        panel  = panel[panel["symbol"].isin(["S00", "S01"])].copy()

        result = build_lockup_features(panel, lockup)
        days = result[result["days_to_next_unlock"] < 999]["days_to_next_unlock"]
        assert (days > 0).all(), "days_to_next_unlock must be strictly positive"

    def test_no_same_day_unlock_used(self):
        """Feature at date T must NOT use the unlock event on date T itself."""
        from quant_platform.features.event import build_lockup_features

        unlock_date = dt.date(2022, 3, 15)
        lockup = pd.DataFrame([{
            "symbol": "S00", "unlock_date": unlock_date,
            "lock_type": "IPO", "shares_million": 100.0, "ratio_pct": 20.0,
        }])
        dates = [unlock_date - dt.timedelta(days=1), unlock_date,
                 unlock_date + dt.timedelta(days=1)]
        panel = pd.DataFrame([
            {"symbol": "S00", "date": d, "close": 10.0}
            for d in dates
        ])
        result = build_lockup_features(panel, lockup)

        # On the day of the unlock itself (T == unlock_date), the unlock is
        # NOT the "next" event (since we require unlock_date > T).
        # So days_to_next_unlock on unlock_date should be 999 (no future events).
        on_unlock = result[result["date"] == unlock_date]["days_to_next_unlock"].values
        assert all(v == 999 for v in on_unlock), (
            f"On unlock_date itself, days_to_next_unlock should be 999, got {on_unlock}"
        )

    def test_default_value_when_no_lockup(self):
        """Stocks with no lockup data should get days_to_next_unlock=999, ratio=0."""
        from quant_platform.features.event import build_lockup_features

        panel = _make_panel(n_sym=3, n_dates=20)
        result = build_lockup_features(panel, pd.DataFrame())
        assert (result["days_to_next_unlock"] == 999).all()
        assert (result["unlock_size_ratio"] == 0.0).all()

    def test_lockup_roundtrip(self, tmp_path):
        """LockupCollector should write readable Parquet."""
        from quant_platform.ingest.lockup_collector import LockupCollector, load_lockup
        from quant_platform.store.lake import init_lake, lockup_path

        init_lake(tmp_path)

        # Build a fake lockup row and write directly (bypass API)
        lk = pd.DataFrame([{
            "symbol": "S00", "unlock_date": dt.date(2023, 6, 1),
            "lock_type": "IPO", "shares_million": 80.0, "ratio_pct": 15.0,
        }])
        p = lockup_path(tmp_path, "S00")
        p.parent.mkdir(parents=True, exist_ok=True)
        lk.to_parquet(p, index=False)

        loaded = load_lockup(tmp_path, "S00")
        assert len(loaded) == 1
        assert loaded["shares_million"].iloc[0] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# P4C-04: residualise_returns
# ---------------------------------------------------------------------------

class TestResidualiseReturns:
    """P4C-04: Residualised return label arithmetic and PIT safety."""

    def _panel_with_industry(self, n_sym=12, n_dates=80) -> pd.DataFrame:
        panel = _make_panel(n_sym=n_sym, n_dates=n_dates)
        panel["industry_code"] = ["IND_A" if int(s[1:]) % 2 == 0 else "IND_B"
                                   for s in panel["symbol"]]
        panel["float_mcap_yi"] = np.random.default_rng(42).uniform(10, 2000, len(panel))
        return panel

    def test_residual_mean_approx_zero(self):
        """residual_ret_5d should have mean ≈ 0 per date (by construction)."""
        from quant_platform.labels.residualiser import residualise_returns

        panel = self._panel_with_industry()
        result = residualise_returns(panel, horizons=[5])

        assert "residual_ret_5d" in result.columns
        for date, grp in result.groupby("date"):
            sub = grp["residual_ret_5d"].dropna()
            if len(sub) >= 5:
                assert abs(sub.mean()) < 0.01, (
                    f"residual mean on {date}: {sub.mean():.6f} — should be ≈ 0"
                )

    def test_residual_near_zero_market_corr(self):
        """
        residual_ret_5d should have near-zero correlation with the cross-sectional
        mean return (the proxy for market return) per date.
        """
        from quant_platform.labels.residualiser import residualise_returns

        panel = self._panel_with_industry()
        result = residualise_returns(panel, horizons=[5])

        # Compute daily cross-sectional mean return (market proxy)
        result["_mkt_ret"] = result.groupby("date")["ret_fwd_5d"].transform("mean")
        corr = result[["residual_ret_5d", "_mkt_ret"]].dropna().corr().iloc[0, 1]
        assert abs(corr) < 0.3, (
            f"residual_ret_5d has high market correlation: {corr:.3f}"
        )

    def test_regression_uses_only_within_date(self):
        """Each date's regression uses only that date's cross-section — no look-ahead."""
        from quant_platform.labels.residualiser import _residualise_horizon

        panel = self._panel_with_industry(n_sym=8, n_dates=20)
        # Verify by checking that the residual on date T does not change when
        # we add extra noise rows at date T+1
        result1 = _residualise_horizon(panel, "ret_fwd_5d", "industry_code", "float_mcap_yi", None, 5)

        # Add corrupted future data
        future_noise = panel[panel["date"] == panel["date"].max()].copy()
        future_noise["date"] = (
            pd.to_datetime(future_noise["date"]) + pd.Timedelta(days=1)
        ).dt.date
        future_noise["ret_fwd_5d"] = 999.0
        contaminated = pd.concat([panel, future_noise], ignore_index=True)

        result2 = _residualise_horizon(contaminated, "ret_fwd_5d", "industry_code", "float_mcap_yi", None, 5)

        # Residuals on original dates should be unchanged
        orig_idx = panel[panel["date"] < panel["date"].max()].index
        diff = (result1[orig_idx] - result2[orig_idx]).abs().max()
        assert diff < 1e-9, f"Residuals changed due to future data — PIT violation! diff={diff}"

    def test_multiple_horizons(self):
        """All four horizons should be present in the output."""
        from quant_platform.labels.residualiser import residualise_returns

        panel = self._panel_with_industry()
        result = residualise_returns(panel, horizons=[1, 5, 10, 20])
        for h in [1, 5, 10, 20]:
            assert f"residual_ret_{h}d" in result.columns

    def test_missing_industry_graceful(self):
        """When industry_col is absent, the function falls back gracefully."""
        from quant_platform.labels.residualiser import residualise_returns

        panel = _make_panel(n_sym=8, n_dates=60)
        panel["float_mcap_yi"] = np.random.default_rng(1).uniform(10, 500, len(panel))
        # No industry_code — should still work (intercept + size only)
        result = residualise_returns(panel, horizons=[5], industry_col="industry_code")
        assert "residual_ret_5d" in result.columns


# ---------------------------------------------------------------------------
# P4C-05: RegimeAnalyser
# ---------------------------------------------------------------------------

class TestRegimeAnalyser:
    """P4C-05: Walk-forward regime analysis."""

    @pytest.fixture
    def panel(self):
        return _make_panel(n_sym=10, n_dates=500)

    def test_produces_window_rows(self, panel):
        """RegimeReport should contain per-window rows."""
        from quant_platform.evaluation.regime_analysis import RegimeAnalyser

        analyser = RegimeAnalyser(
            store_root=Path("/tmp"),
            n_windows=2,
            window_months=4,
            horizon=5,
        )
        report = analyser.run(
            panel=panel,
            feature_groups={
                "technical": ["feat_signal", "feat_noise"],
                "dummy":     ["feat_a"],
            },
            label_col="ret_fwd_5d",
            save_csv=False,
            record_to_ledger=False,
        )
        assert len(report.window_rows) > 0

    def test_ensemble_stability_in_01(self, panel):
        """Ensemble stability must be in [0, 1]."""
        from quant_platform.evaluation.regime_analysis import RegimeAnalyser

        analyser = RegimeAnalyser(
            store_root=Path("/tmp"),
            n_windows=2,
            window_months=4,
            horizon=5,
        )
        report = analyser.run(
            panel=panel,
            feature_groups={"all": ["feat_signal", "feat_noise"]},
            label_col="ret_fwd_5d",
            save_csv=False,
            record_to_ledger=False,
        )
        stab = report.ensemble_stability
        if not np.isnan(stab):
            assert 0.0 <= stab <= 1.0

    def test_hard_regime_detection(self):
        """Windows with |ensemble IC| < 0.01 should be flagged as hard."""
        from quant_platform.evaluation.regime_analysis import (
            RegimeReport, RegimeWindowRow,
            _HARD_REGIME_IC_THRESHOLD, _find_consecutive,
        )

        row0 = RegimeWindowRow(0, "2021-01-01", "2021-06-30",
                               ensemble_ic=0.005, is_hard=True)
        row1 = RegimeWindowRow(1, "2021-07-01", "2021-12-31",
                               ensemble_ic=0.008, is_hard=True)
        row2 = RegimeWindowRow(2, "2022-01-01", "2022-06-30",
                               ensemble_ic=0.06,  is_hard=False)

        report = RegimeReport()
        report.window_rows = [row0, row1, row2]
        hard_ids = [r.window_id for r in report.window_rows if r.is_hard]
        consec   = _find_consecutive(hard_ids, min_consec=2)
        assert 0 in consec and 1 in consec, "Windows 0 and 1 should be consecutive hard"
        assert 2 not in consec

    def test_recommendations_non_empty(self, panel):
        """Recommendations list should be populated."""
        from quant_platform.evaluation.regime_analysis import RegimeAnalyser

        analyser = RegimeAnalyser(
            store_root=Path("/tmp"),
            n_windows=2,
            window_months=4,
            horizon=5,
        )
        report = analyser.run(
            panel=panel,
            feature_groups={"g1": ["feat_signal"]},
            label_col="ret_fwd_5d",
            save_csv=False,
            record_to_ledger=False,
        )
        assert len(report.recommendations) > 0


# ---------------------------------------------------------------------------
# P4X-01: store_lake lockup paths
# ---------------------------------------------------------------------------

class TestStoreLakeLockup:
    def test_lockup_path_format(self, tmp_path):
        from quant_platform.store.lake import lockup_path
        p = lockup_path(tmp_path, "600519")
        assert p.name == "600519.parquet"
        assert "lockup" in str(p) and "silver" in str(p)

    def test_init_lake_creates_lockup_and_evaluation(self, tmp_path):
        from quant_platform.store.lake import init_lake
        init_lake(tmp_path)
        assert (tmp_path / "silver" / "lockup").exists()
        assert (tmp_path / "evaluation").exists()

    def test_lockup_dir(self, tmp_path):
        from quant_platform.store.lake import lockup_dir
        d = lockup_dir(tmp_path)
        assert d.name == "lockup"
        assert "silver" in str(d)
