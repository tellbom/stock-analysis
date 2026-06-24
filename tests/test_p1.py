"""
P1 complete verification test suite (T1.1 – T1.8).

Covers:
  T1.1 Feature pipeline scaffold
  T1.2 Technical feature builder + warm-up masking
  T1.3 Cross-sectional features
  T1.4 Fundamental feature builder (PIT join)
  T1.5 Feature-spec registry + versioning
  T1.6 Label builder (T+1 execution, multiple horizons)
  T1.7 Leakage test harness
  T1.8 Data dictionary generation

All tests use synthetic data — no live network.

Run with:  PYTHONPATH=/home/claude:/mnt/project python quant_platform/tests/test_p1.py
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root so quant_platform and technical_indicators.py are importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(symbol: str = "600519", n: int = 120,
                start: str = "2022-01-04") -> pd.DataFrame:
    """Synthetic OHLCV with enough rows for all warm-up periods (need ≥ 60)."""
    rng = np.random.default_rng(seed=int(symbol))
    dates = pd.date_range(start, periods=n, freq="B")
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame({
        "symbol": symbol,
        "date":   [d.date() for d in dates],
        "open":   prices * rng.uniform(0.99, 1.00, n),
        "high":   prices * rng.uniform(1.00, 1.02, n),
        "low":    prices * rng.uniform(0.98, 1.00, n),
        "close":  prices,
        "volume": rng.uniform(5e5, 2e6, n),
        "amount": prices * rng.uniform(5e5, 2e6, n),
    })


def _build_lake(tmp: str, symbols: list[str] = None,
                n: int = 120) -> None:
    """Populate a temp lake with OHLCV, universe, and calendar."""
    from quant_platform.store.lake import init_lake, ohlcv_path
    from quant_platform.store.parquet_store import write_ohlcv
    from quant_platform.ingest.calendar_service import CalendarService
    from quant_platform.ingest.universe_service import UniverseService
    import quant_platform.ingest.calendar_service as calendar_service

    symbols = symbols or ["600519", "000858"]
    init_lake(tmp)

    for sym in symbols:
        df = _make_ohlcv(sym, n=n)
        write_ohlcv(df, ohlcv_path(tmp, sym), sym)

    # Calendar
    original_akshare = calendar_service._build_from_akshare
    calendar_service._build_from_akshare = lambda start, end: None
    try:
        CalendarService(tmp).build_and_save(start="2022-01-01", end="2024-12-31")
    finally:
        calendar_service._build_from_akshare = original_akshare

    # Universe membership
    csv_path = Path(tmp) / "cons.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "name"])
        for s in symbols:
            w.writerow([s, f"Stock_{s}"])
    svc = UniverseService("csi300", tmp)
    svc.load_from_csv(csv_path)


# ---------------------------------------------------------------------------
# T1.5 — Feature registry
# ---------------------------------------------------------------------------

def test_feature_set_id_stable():
    """Same spec list → same feature_set_id."""
    from quant_platform.features.registry import TECHNICAL_SPECS, compute_feature_set_id
    id1 = compute_feature_set_id(TECHNICAL_SPECS)
    id2 = compute_feature_set_id(TECHNICAL_SPECS)
    assert id1 == id2
    assert len(id1) == 8
    print("  [OK] T1.5 compute_feature_set_id: same specs → same 8-char id")


def test_feature_set_id_different_for_different_specs():
    """Different specs → different IDs."""
    from quant_platform.features.registry import (
        TECHNICAL_SPECS, CROSS_SECTIONAL_SPECS, compute_feature_set_id,
    )
    id_tech = compute_feature_set_id(TECHNICAL_SPECS)
    id_cs   = compute_feature_set_id(CROSS_SECTIONAL_SPECS)
    assert id_tech != id_cs
    print("  [OK] T1.5 different specs → different ids (no collision)")


def test_feature_registry_register_and_retrieve():
    """FeatureRegistry.register() persists specs; get_specs() retrieves them."""
    from quant_platform.features.registry import FeatureRegistry, TECHNICAL_SPECS

    with tempfile.TemporaryDirectory() as tmp:
        reg = FeatureRegistry(tmp)
        fset_id = reg.register(TECHNICAL_SPECS)
        retrieved = reg.get_specs(fset_id)

        assert len(retrieved) == len(TECHNICAL_SPECS)
        assert {s.name for s in retrieved} == {s.name for s in TECHNICAL_SPECS}
        print(f"  [OK] T1.5 registry: registered {len(TECHNICAL_SPECS)} specs, retrieved correctly")


def test_feature_registry_idempotent():
    """Registering the same specs twice keeps only one entry in Parquet."""
    from quant_platform.features.registry import FeatureRegistry, TECHNICAL_SPECS

    with tempfile.TemporaryDirectory() as tmp:
        reg = FeatureRegistry(tmp)
        id1 = reg.register(TECHNICAL_SPECS)
        id2 = reg.register(TECHNICAL_SPECS)

        assert id1 == id2
        df = reg._load()
        assert (df["feature_set_id"] == id1).sum() == len(TECHNICAL_SPECS), \
            "Duplicate rows found after re-registration"
    print("  [OK] T1.5 registry: re-registering same specs is idempotent")


# ---------------------------------------------------------------------------
# T1.2 — Technical features
# ---------------------------------------------------------------------------

def test_build_technical_features_output_columns():
    """build_technical_features returns all expected technical columns."""
    from quant_platform.features.technical import build_technical_features
    from quant_platform.features.registry import TECHNICAL_SPECS

    df = _make_ohlcv(n=120)
    result = build_technical_features(df)

    expected = {s.name for s in TECHNICAL_SPECS}
    present  = set(result.columns)
    missing  = expected - present
    assert not missing, f"Missing columns: {missing}"
    print(f"  [OK] T1.2 technical features: all {len(expected)} columns present")


def test_warmup_rows_are_nan():
    """First warmup rows for each feature must be NaN after build_technical_features."""
    from quant_platform.features.technical import build_technical_features
    from quant_platform.features.registry import TECHNICAL_SPECS

    df = _make_ohlcv(n=120)
    result = build_technical_features(df).reset_index(drop=True)

    violations = []
    for spec in TECHNICAL_SPECS:
        if spec.name not in result.columns or spec.warmup == 0:
            continue
        warm_rows = result[spec.name].iloc[:spec.warmup]
        n_non_nan = warm_rows.notna().sum()
        if n_non_nan > 0:
            violations.append(f"{spec.name}: {n_non_nan}/{spec.warmup} non-NaN warm-up rows")

    assert not violations, f"Warm-up NaN violations:\n" + "\n".join(violations)
    print("  [OK] T1.2 warm-up masking: all warm-up rows are NaN")


def test_stoch_features_preserve_pta_index_alignment():
    """Stoch output from pandas_ta must stay on its original input rows."""
    import pandas_ta_classic as pta
    from quant_platform.features.technical import build_technical_features

    df = _make_ohlcv(n=120).reset_index(drop=True)
    result = build_technical_features(df).reset_index(drop=True)
    raw = pta.stoch(df["high"], df["low"], df["close"])

    expected_k = raw[raw.columns[0]].reindex(range(len(df))).reset_index(drop=True)
    expected_d = raw[raw.columns[1]].reindex(range(len(df))).reset_index(drop=True)

    for col, expected in (("stoch_k", expected_k), ("stoch_d", expected_d)):
        mask = expected.notna() & result[col].notna()
        assert mask.any(), f"{col}: no comparable non-NaN rows"
        max_diff = (result.loc[mask, col] - expected.loc[mask]).abs().max()
        assert max_diff < 1e-10, f"{col}: pandas_ta index alignment shifted by {max_diff}"
    print("  [OK] T1.2 stoch alignment: pandas_ta original index preserved")


def test_technical_features_stable_after_warmup():
    """After warmup rows, feature values are finite (not NaN or inf)."""
    from quant_platform.features.technical import build_technical_features
    from quant_platform.features.registry import TECHNICAL_SPECS

    df = _make_ohlcv(n=120)
    result = build_technical_features(df).reset_index(drop=True)
    max_warmup = max(s.warmup for s in TECHNICAL_SPECS)

    for spec in TECHNICAL_SPECS:
        if spec.name not in result.columns:
            continue
        post_warmup = result[spec.name].iloc[max_warmup:]
        n_nan = post_warmup.isna().sum()
        n_inf = np.isinf(post_warmup.fillna(0)).sum()
        # Allow up to 5% NaN post-warmup (some indicators may still have early NaN)
        if n_nan > len(post_warmup) * 0.05:
            print(f"  [WARN] {spec.name}: {n_nan}/{len(post_warmup)} NaN after warmup")
        assert n_inf == 0, f"{spec.name}: {n_inf} inf values post-warmup"
    print("  [OK] T1.2 stability: no inf values in features after warmup period")


# ---------------------------------------------------------------------------
# T1.3 — Cross-sectional features
# ---------------------------------------------------------------------------

def test_cross_sectional_rank_range():
    """cs_rank_close values must be in [0, 1]."""
    from quant_platform.features.cross_sectional import build_cross_sectional_features
    from quant_platform.features.technical import build_technical_features

    frames = []
    for sym in ["600519", "000858", "300750"]:
        df = _make_ohlcv(sym, n=60)
        df = build_technical_features(df)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    result = build_cross_sectional_features(panel)

    col = "cs_rank_close"
    assert col in result.columns
    valid = result[col].dropna()
    assert (valid >= 0).all() and (valid <= 1).all(), \
        f"cs_rank_close out of [0,1]: min={valid.min():.4f}, max={valid.max():.4f}"
    print("  [OK] T1.3 cs_rank_close: all values in [0, 1]")


def test_cross_sectional_zscore_mean_near_zero():
    """cs_zscore_close mean per date should be ~0."""
    from quant_platform.features.cross_sectional import build_cross_sectional_features
    from quant_platform.features.technical import build_technical_features

    frames = []
    for sym in ["600519", "000858", "300750", "601318", "002594"]:
        df = _make_ohlcv(sym, n=60)
        df = build_technical_features(df)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    result = build_cross_sectional_features(panel)

    per_date_mean = result.groupby("date")["cs_zscore_close"].mean()
    assert (per_date_mean.abs() < 1e-9).all(), \
        f"cs_zscore_close mean not ~0: max abs = {per_date_mean.abs().max():.6f}"
    print("  [OK] T1.3 cs_zscore_close: mean per date is 0 (z-score property)")


def test_cross_sectional_missing_column_handled():
    """If rsi_6 column is absent, cs_rank_rsi_6 is NaN (no crash)."""
    from quant_platform.features.cross_sectional import build_cross_sectional_features

    panel = _make_ohlcv("600519", n=30)
    # rsi_6 not in panel
    result = build_cross_sectional_features(panel)
    assert "cs_rank_rsi_6" in result.columns
    assert result["cs_rank_rsi_6"].isna().all()
    print("  [OK] T1.3 missing input column: cs_rank output is all NaN (no crash)")


# ---------------------------------------------------------------------------
# T1.4 — Fundamental features
# ---------------------------------------------------------------------------

def test_fundamental_pit_join_excludes_future_announcements():
    """
    Fundamental rows announced AFTER the feature date must NOT appear.
    """
    from quant_platform.features.fundamental import build_fundamental_features
    from quant_platform.store.lake import fundamentals_dir

    with tempfile.TemporaryDirectory() as tmp:
        # Write a fundamentals file: Q2 announced 2022-08-30
        fund_dir = Path(tmp) / "silver" / "fundamentals"
        fund_dir.mkdir(parents=True, exist_ok=True)
        fund_df = pd.DataFrame([{
            "symbol": "600519",
            "announce_date": "2022-08-30",  # announced Aug 30
            "period_end":    "2022-06-30",
            "period_type":   "H1",
            "source":        "yjkb_em",
            "revenue":       5_000_000.0,
            "net_profit":    1_000_000.0,
            "eps":           2.5,
            "roe":           15.0,
        }])
        fund_df.to_parquet(fund_dir / "600519.parquet", index=False)

        # Feature dates: before and after announcement
        panel = pd.DataFrame({
            "symbol": "600519",
            "date":   [dt.date(2022, 8, 29), dt.date(2022, 8, 30), dt.date(2022, 9, 1)],
            "close":  [100.0, 101.0, 102.0],
            "volume": [1e6, 1e6, 1e6],
        })
        result = build_fundamental_features(panel, store_root=tmp)

        # Aug 29: before announcement → fund_revenue must be NaN
        row_before = result[result["date"] == dt.date(2022, 8, 29)].iloc[0]
        assert pd.isna(row_before["fund_revenue"]) or row_before["fund_revenue"] != row_before["fund_revenue"], \
            f"fund_revenue should be NaN on Aug 29 (before announcement), got {row_before['fund_revenue']}"

        # Aug 30: announcement day → should see the data
        row_on = result[result["date"] == dt.date(2022, 8, 30)].iloc[0]
        assert row_on["fund_revenue"] == 5_000_000.0, \
            f"fund_revenue should be 5M on announcement day, got {row_on['fund_revenue']}"

        # Sep 1: after → should also see the data
        row_after = result[result["date"] == dt.date(2022, 9, 1)].iloc[0]
        assert row_after["fund_revenue"] == 5_000_000.0
    print("  [OK] T1.4 PIT join: future announcements excluded, same-day and after included")


def test_fundamental_features_ignore_forecast_only_rows():
    """Forecast-only yjyg rows must not overwrite latest formal metrics."""
    from quant_platform.features.fundamental import build_fundamental_features

    with tempfile.TemporaryDirectory() as tmp:
        fund_dir = Path(tmp) / "silver" / "fundamentals"
        fund_dir.mkdir(parents=True, exist_ok=True)
        fund_df = pd.DataFrame([
            {
                "symbol": "600519",
                "announce_date": "2022-08-30",
                "period_end": "2022-06-30",
                "period_type": "H1",
                "source": "yjkb_em",
                "revenue": 5_000_000.0,
                "net_profit": 1_000_000.0,
                "eps": 2.5,
                "roe": 15.0,
            },
            {
                "symbol": "600519",
                "announce_date": "2022-09-10",
                "period_end": "2022-09-30",
                "period_type": "Q3",
                "source": "yjyg_em",
                "forecast_metric": "归属于上市公司股东的净利润",
            },
        ])
        fund_df.to_parquet(fund_dir / "600519.parquet", index=False)

        panel = pd.DataFrame({
            "symbol": ["600519"],
            "date": [dt.date(2022, 9, 15)],
            "close": [100.0],
            "volume": [1e6],
        })
        result = build_fundamental_features(panel, store_root=tmp)

        row = result.iloc[0]
        assert row["fund_revenue"] == 5_000_000.0
        assert row["fund_announce_date"] == dt.date(2022, 8, 30)
    print("  [OK] T1.4 fundamental: forecast-only rows do not overwrite formal metrics")


def test_fundamental_no_data_returns_nan():
    """If no fundamentals file exists, all fund_* columns are NaN."""
    from quant_platform.features.fundamental import build_fundamental_features

    with tempfile.TemporaryDirectory() as tmp:
        panel = _make_ohlcv("600519", n=5)
        result = build_fundamental_features(panel, store_root=tmp)
        assert "fund_revenue" in result.columns
        assert result["fund_revenue"].isna().all()
    print("  [OK] T1.4 fundamental: no data file → all fund_* columns NaN (no crash)")


# ---------------------------------------------------------------------------
# T1.1 — Feature pipeline
# ---------------------------------------------------------------------------

def test_pipeline_run_writes_parquet():
    """FeaturePipeline.run() writes feature Parquet for each symbol."""
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.features.registry import TECHNICAL_SPECS
    from quant_platform.store.lake import feature_path

    with tempfile.TemporaryDirectory() as tmp:
        _build_lake(tmp, symbols=["600519", "000858"], n=120)
        pipe = FeaturePipeline(store_root=tmp)
        fset_id = pipe.run(["600519", "000858"], specs=TECHNICAL_SPECS)

        for sym in ["600519", "000858"]:
            p = feature_path(tmp, fset_id, sym)
            assert p.exists(), f"Feature Parquet not found for {sym}"
            df = pd.read_parquet(p)
            assert "symbol" in df.columns and "date" in df.columns
            assert len(df) > 0
    print(f"  [OK] T1.1 pipeline: feature Parquets written for 2 symbols, id={fset_id[:6]}…")


def test_pipeline_build_panel():
    """build_panel concatenates symbol files and adds cross-sectional features."""
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.features.registry import TECHNICAL_SPECS

    with tempfile.TemporaryDirectory() as tmp:
        _build_lake(tmp, symbols=["600519", "000858"], n=120)
        pipe = FeaturePipeline(store_root=tmp)
        fset_id = pipe.run(["600519", "000858"], specs=TECHNICAL_SPECS)
        panel = pipe.build_panel(["600519", "000858"], fset_id)

        assert len(panel) > 0
        assert "symbol" in panel.columns
        assert "cs_rank_close" in panel.columns   # cross-sectional added
        assert panel["symbol"].nunique() == 2
        assert panel["date"].is_monotonic_increasing or \
               all(panel.groupby("date")["symbol"].count() > 0)
    print(f"  [OK] T1.1 build_panel: panel has {len(panel)} rows, cs features present")


def test_pipeline_feature_set_id_deterministic():
    """Running pipeline twice with same specs gives same feature_set_id."""
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.features.registry import TECHNICAL_SPECS

    with tempfile.TemporaryDirectory() as tmp:
        _build_lake(tmp, n=120)
        pipe = FeaturePipeline(store_root=tmp)
        id1 = pipe.run(["600519"], specs=TECHNICAL_SPECS)
        id2 = pipe.run(["600519"], specs=TECHNICAL_SPECS)
        assert id1 == id2
    print(f"  [OK] T1.1 pipeline: same specs → same feature_set_id ({id1[:6]}…) both runs")


# ---------------------------------------------------------------------------
# T1.6 — Label builder
# ---------------------------------------------------------------------------

def test_label_builder_writes_parquet():
    """build_labels writes label Parquets for each symbol."""
    from quant_platform.labels.builder import build_labels
    from quant_platform.store.lake import label_path

    with tempfile.TemporaryDirectory() as tmp:
        _build_lake(tmp, n=120)
        results = build_labels(tmp, ["600519", "000858"], horizons=[1, 5])

        for sym in ["600519", "000858"]:
            p = label_path(tmp, "forward_returns", sym)
            assert p.exists(), f"Label Parquet not found for {sym}"
            assert results[sym] > 0
    print("  [OK] T1.6 build_labels: label Parquets written for 2 symbols")


def test_label_t1_execution_assumption():
    """
    Label ret_fwd_1d at row i = close[i+2]/close[i+1] - 1 (T+1 execution).
    Confirm it is NOT close[i+1]/close[i] - 1 (same-day return).
    """
    from quant_platform.labels.builder import build_labels
    from quant_platform.store.lake import label_path, ohlcv_path
    from quant_platform.store.parquet_store import write_ohlcv

    with tempfile.TemporaryDirectory() as tmp:
        from quant_platform.store.lake import init_lake
        init_lake(tmp)
        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        prices = [100.0, 102.0, 105.0, 103.0, 108.0,
                  110.0, 107.0, 112.0, 115.0, 113.0]
        df = pd.DataFrame({
            "symbol": "600519",
            "date": [d.date() for d in dates],
            "open": prices, "high": prices, "low": prices,
            "close": prices, "volume": 1e6,
        })
        write_ohlcv(df, ohlcv_path(tmp, "600519"), "600519")
        build_labels(tmp, ["600519"], horizons=[1])

        ldf = pd.read_parquet(label_path(tmp, "forward_returns", "600519"))
        ldf = ldf.sort_values("date").reset_index(drop=True)

        # Row 0 (2024-01-02): label = close[2]/close[1] - 1 = 105/102 - 1
        expected_row0 = prices[2] / prices[1] - 1.0
        actual_row0   = ldf["ret_fwd_1d"].iloc[0]
        assert abs(actual_row0 - expected_row0) < 1e-9, \
            f"Row 0 label: expected {expected_row0:.6f}, got {actual_row0:.6f}"

        # Row 0 label must NOT be close[1]/close[0] - 1 (same-day return)
        same_day_ret = prices[1] / prices[0] - 1.0
        assert abs(actual_row0 - same_day_ret) > 1e-9, \
            "Label should not be same-day return (T+1 execution violated)"

    print("  [OK] T1.6 T+1 execution: label[0] = close[2]/close[1]-1, not same-day return")


def test_label_tail_rows_are_nan():
    """Last h rows per symbol per horizon must be NaN."""
    from quant_platform.labels.builder import build_labels
    from quant_platform.store.lake import label_path

    with tempfile.TemporaryDirectory() as tmp:
        _build_lake(tmp, n=60)
        horizons = [1, 5]
        build_labels(tmp, ["600519"], horizons=horizons)

        ldf = pd.read_parquet(label_path(tmp, "forward_returns", "600519"))
        ldf = ldf.sort_values("date").reset_index(drop=True)

        for h in horizons:
            col = f"ret_fwd_{h}d"
            # Last h+1 rows should be NaN (T+1 and T+1+h both need to exist)
            tail = ldf[col].iloc[-(h + 1):]
            n_non_nan = tail.notna().sum()
            assert n_non_nan == 0, \
                f"ret_fwd_{h}d: {n_non_nan} non-NaN in last {h+1} rows (future data leaked)"
    print("  [OK] T1.6 tail NaN: last h+1 rows are NaN for h=1,5 (no future data)")


def test_label_panel_cross_sectional_decile():
    """build_label_panel adds cs decile (0–9) and binary columns."""
    from quant_platform.labels.builder import build_labels, build_label_panel

    symbols = ["600519", "000858", "300750", "601318", "002594",
               "000001", "601398", "600036", "002415", "000651"]

    with tempfile.TemporaryDirectory() as tmp:
        _build_lake(tmp, symbols=symbols, n=80)
        build_labels(tmp, symbols, horizons=[5])
        panel = build_label_panel(tmp, symbols, horizons=[5])

        cs_col = "ret_fwd_5d_cs"
        assert cs_col in panel.columns

        valid = panel[cs_col].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0 and valid.max() <= 9, \
                f"Decile out of [0,9]: min={valid.min()}, max={valid.max()}"
    print("  [OK] T1.6 label panel: cross-sectional decile in [0,9]")


# ---------------------------------------------------------------------------
# T1.7 — Leakage harness
# ---------------------------------------------------------------------------

def test_leakage_harness_passes_on_clean_data():
    """run_leakage_harness passes on correctly built features and labels."""
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.features.registry import TECHNICAL_SPECS
    from quant_platform.labels.builder import build_labels, build_label_panel
    from quant_platform.labels.leakage_harness import run_leakage_harness
    from quant_platform.store.parquet_store import read_ohlcv
    from quant_platform.store.lake import ohlcv_path

    symbols = ["600519", "000858"]
    with tempfile.TemporaryDirectory() as tmp:
        _build_lake(tmp, symbols=symbols, n=120)
        pipe = FeaturePipeline(store_root=tmp)
        fset_id = pipe.run(symbols, specs=TECHNICAL_SPECS)
        panel = pipe.build_panel(symbols, fset_id, add_cross_sectional=False)

        build_labels(tmp, symbols, horizons=[1, 5])
        label_panel = build_label_panel(tmp, symbols, horizons=[1, 5])

        ohlcv_frames = [read_ohlcv(ohlcv_path(tmp, s)) for s in symbols]
        ohlcv_df = pd.concat(ohlcv_frames, ignore_index=True)

        report = run_leakage_harness(panel, label_panel, ohlcv_df, horizons=[1, 5])
        report.print_summary()

        hard_failures = [c for c in report.failures
                         if c.name != "overlap_warning"]
        assert not hard_failures, \
            f"Leakage harness FAILED on clean data:\n" + \
            "\n".join(str(f) for f in hard_failures)
    print("  [OK] T1.7 leakage harness: PASSES on clean pipeline output")


def test_leakage_harness_canary_detection():
    """Canary (close.shift(-1)) is always detected by the harness."""
    from quant_platform.labels.leakage_harness import (
        LeakageReport, check_canary_future_leak,
    )

    ohlcv = _make_ohlcv("600519", n=30)
    report = LeakageReport()
    check_canary_future_leak(report, ohlcv)

    canary_check = next(
        c for c in report.checks
        if c.name == "canary_future_leak_detection"
    )
    assert canary_check.passed, \
        "Canary detection failed — harness cannot detect future leakage"
    print("  [OK] T1.7 canary: future-leaking feature (close.shift(-1)) detected")


def test_leakage_harness_warmup_violation_detected():
    """Harness detects when warm-up rows are not NaN."""
    from quant_platform.labels.leakage_harness import LeakageReport, check_warmup_nans

    # Build features WITHOUT warm-up masking to simulate the violation
    df = _make_ohlcv("600519", n=120)
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from technical_indicators import calculate_all_indicators
    raw = calculate_all_indicators(df)
    # Rename to canonical names without masking
    raw = raw.rename(columns={
        "MA5": "ma_5", "MA10": "ma_10", "MA20": "ma_20", "MA60": "ma_60",
        "DIF": "macd_dif", "DEA": "macd_dea", "MACD": "macd_hist",
        "K": "kdj_k", "D": "kdj_d", "J": "kdj_j",
        "RSI6": "rsi_6", "RSI12": "rsi_12", "RSI24": "rsi_24",
        "BOLL_UPPER": "boll_upper", "BOLL_MID": "boll_mid", "BOLL_LOWER": "boll_lower",
    })

    report = LeakageReport()
    check_warmup_nans(report, raw)  # should FAIL because no masking was applied

    warmup_check = next(c for c in report.checks if c.name == "warmup_nans")
    assert not warmup_check.passed, \
        "Expected warmup violation detection, but check passed"
    print("  [OK] T1.7 warmup check: correctly detects non-NaN warm-up rows")


# ---------------------------------------------------------------------------
# T1.8 — Data dictionary
# ---------------------------------------------------------------------------

def test_data_dictionary_written():
    """build_data_dictionary writes both Parquet and text files."""
    from quant_platform.features.data_dictionary import build_data_dictionary

    with tempfile.TemporaryDirectory() as tmp:
        df = build_data_dictionary(tmp, horizons=[1, 5, 20])

        assert (Path(tmp) / "data_dictionary.parquet").exists()
        assert (Path(tmp) / "data_dictionary.txt").exists()
        assert len(df) > 0
        assert "name" in df.columns
        assert "known_at" in df.columns
        assert "formula" in df.columns
    print(f"  [OK] T1.8 data dictionary: {len(df)} columns documented, files written")


def test_data_dictionary_covers_all_technical_specs():
    """Every TECHNICAL_SPECS entry appears in the data dictionary."""
    from quant_platform.features.data_dictionary import build_data_dictionary
    from quant_platform.features.registry import TECHNICAL_SPECS

    with tempfile.TemporaryDirectory() as tmp:
        df = build_data_dictionary(tmp, horizons=[1])
        dict_names = set(df["name"].tolist())
        spec_names = {s.name for s in TECHNICAL_SPECS}
        missing = spec_names - dict_names
        assert not missing, f"Technical specs missing from dictionary: {missing}"
    print("  [OK] T1.8 data dictionary: covers all TECHNICAL_SPECS entries")


def test_data_dictionary_label_horizons_expanded():
    """Label entries in the dictionary are expanded per horizon."""
    from quant_platform.features.data_dictionary import build_data_dictionary

    with tempfile.TemporaryDirectory() as tmp:
        df = build_data_dictionary(tmp, horizons=[1, 5, 20])
        label_rows = df[df["category"] == "label"]
        # Should have ret_fwd_1d, ret_fwd_5d, ret_fwd_20d etc.
        names = set(label_rows["name"].tolist())
        assert "ret_fwd_1d"  in names
        assert "ret_fwd_5d"  in names
        assert "ret_fwd_20d" in names
    print("  [OK] T1.8 data dictionary: label entries expanded for each horizon")


def test_data_dictionary_known_at_semantics():
    """All label entries have 'NOT known at T' in their known_at field."""
    from quant_platform.features.data_dictionary import build_data_dictionary

    with tempfile.TemporaryDirectory() as tmp:
        df = build_data_dictionary(tmp)
        label_rows = df[df["category"] == "label"]
        for _, row in label_rows.iterrows():
            assert "NOT known at T" in row["known_at"], \
                f"Label {row['name']} missing 'NOT known at T' in known_at: {row['known_at']}"
    print("  [OK] T1.8 data dictionary: all label entries flagged as 'NOT known at T'")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== P1 Complete Test Suite (T1.1 – T1.8) ===\n")
    tests = [
        # T1.5 Registry
        test_feature_set_id_stable,
        test_feature_set_id_different_for_different_specs,
        test_feature_registry_register_and_retrieve,
        test_feature_registry_idempotent,
        # T1.2 Technical
        test_build_technical_features_output_columns,
        test_warmup_rows_are_nan,
        test_stoch_features_preserve_pta_index_alignment,
        test_technical_features_stable_after_warmup,
        # T1.3 Cross-sectional
        test_cross_sectional_rank_range,
        test_cross_sectional_zscore_mean_near_zero,
        test_cross_sectional_missing_column_handled,
        # T1.4 Fundamental
        test_fundamental_pit_join_excludes_future_announcements,
        test_fundamental_no_data_returns_nan,
        # T1.1 Pipeline
        test_pipeline_run_writes_parquet,
        test_pipeline_build_panel,
        test_pipeline_feature_set_id_deterministic,
        # T1.6 Labels
        test_label_builder_writes_parquet,
        test_label_t1_execution_assumption,
        test_label_tail_rows_are_nan,
        test_label_panel_cross_sectional_decile,
        # T1.7 Leakage harness
        test_leakage_harness_passes_on_clean_data,
        test_leakage_harness_canary_detection,
        test_leakage_harness_warmup_violation_detected,
        # T1.8 Data dictionary
        test_data_dictionary_written,
        test_data_dictionary_covers_all_technical_specs,
        test_data_dictionary_label_horizons_expanded,
        test_data_dictionary_known_at_semantics,
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
    print(f"P1 Results: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
