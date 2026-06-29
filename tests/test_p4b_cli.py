from __future__ import annotations

import pandas as pd
import pytest


def test_model_parser_accepts_p4b_feature_flags():
    from quant_platform.cli import build_parser

    args = build_parser().parse_args([
        "model",
        "--store-root", "models/data",
        "--include-valuation",
        "--include-industry",
    ])

    assert args.include_valuation is True
    assert args.include_industry is True


def test_feature_audit_fails_when_requested_valuation_is_absent():
    from quant_platform.cli import _audit_training_features

    panel = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "volume": [1.0, 2.0],
        "ret_fwd_5d": [0.01, -0.01],
    })

    with pytest.raises(RuntimeError, match="include-valuation"):
        _audit_training_features(
            panel,
            ["volume"],
            include_valuation=True,
            include_industry=False,
        )


def test_feature_audit_reports_all_nan_exclusion(capsys):
    from quant_platform.cli import _audit_training_features, _build_feature_cols

    panel = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
        "volume": [1.0, 2.0],
        "ind_rank_main_flow": [float("nan"), float("nan")],
        "ret_fwd_5d": [0.01, -0.01],
    })
    feature_cols = _build_feature_cols(panel)
    assert "ind_rank_main_flow" not in feature_cols

    _audit_training_features(panel, feature_cols)
    out = capsys.readouterr().out
    assert "ind_rank_main_flow: all-NaN" in out


def test_cmd_model_passes_p4b_flags_to_feature_pipeline(monkeypatch, tmp_path):
    import argparse
    import datetime as dt

    import quant_platform.cli as cli
    import quant_platform.features.pipeline as pipeline_mod
    import quant_platform.labels.builder as labels_mod
    import quant_platform.store.parquet_store as parquet_mod

    captured = {}

    class FakePipeline:
        def __init__(self, store_root, project_root=None, include_valuation=False, include_industry=False):
            captured["include_valuation"] = include_valuation
            captured["include_industry"] = include_industry

        def build_panel(self, symbols, feature_set_id, add_cross_sectional=True):
            return pd.DataFrame({
                "symbol": ["000001", "000002"],
                "date": [dt.date(2023, 1, 3), dt.date(2023, 1, 3)],
                "volume": [1.0, 2.0],
            })

    monkeypatch.setattr(cli, "_resolve_symbols", lambda args, store_root: ["000001", "000002"])
    monkeypatch.setattr(cli, "_auto_detect_feature_set", lambda store_root: "deadbeef")
    monkeypatch.setattr(pipeline_mod, "FeaturePipeline", FakePipeline)
    monkeypatch.setattr(labels_mod, "build_label_panel", lambda *a, **k: pd.DataFrame({
        "symbol": ["000001", "000002"],
        "date": [dt.date(2023, 1, 3), dt.date(2023, 1, 3)],
        "ret_fwd_5d": [0.01, -0.01],
    }))
    monkeypatch.setattr(parquet_mod, "read_ohlcv", lambda path: pd.DataFrame({
        "symbol": ["000001"],
        "date": [dt.date(2023, 1, 3)],
        "close": [10.0],
    }))

    args = argparse.Namespace(
        store_root=str(tmp_path),
        universe="csi300",
        project_root=None,
        feature_set_id=None,
        label="ret_fwd_5d",
        horizon=5,
        horizons=[5],
        n_splits=2,
        use_lockbox=False,
        include_valuation=True,
        include_industry=True,
        n_estimators=10,
        learning_rate=0.05,
        num_leaves=31,
        cost_bps=10.0,
        n_windows=1,
        window_months=12,
        lockbox_months=12,
    )

    rc = cli.cmd_model(args)

    assert rc == 1
    assert captured == {"include_valuation": True, "include_industry": True}
