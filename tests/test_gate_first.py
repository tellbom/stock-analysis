from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def test_coverage_gate_separates_base_and_recent_features():
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
        select_features_by_gate,
    )

    dates = pd.bdate_range("2024-01-01", periods=90).date
    symbols = [f"{i:06d}" for i in range(1, 6)]
    rows = []
    for d in dates:
        for sym in symbols:
            rows.append({
                "symbol": sym,
                "date": d,
                "ma_5": 1.0,
                "cs_main_flow_rank_1d": 0.5,
            })
    panel = pd.DataFrame(rows)

    report = compute_feature_coverage_report(
        panel,
        ["ma_5", "cs_main_flow_rank_1d"],
        family_by_col={"ma_5": "technical", "cs_main_flow_rank_1d": "flow"},
        config=CoverageGateConfig(
            recent_symbol_threshold=4,
            recent_20d_symbol_threshold=4,
            min_recent_trading_days=80,
        ),
    )

    base_cols = select_features_by_gate(report, model_path="base")
    recent_cols = select_features_by_gate(report, model_path="recent")

    assert "ma_5" in base_cols
    assert "cs_main_flow_rank_1d" not in base_cols
    assert "cs_main_flow_rank_1d" in recent_cols
    flow_row = report[report["feature_name"] == "cs_main_flow_rank_1d"].iloc[0]
    assert "short-history-family" in flow_row["rejection_reason"]


def test_coverage_gate_marks_future_fields_prediction_only():
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
    )

    panel = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "date": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
        "future_return_hint": [0.1, -0.1],
    })
    report = compute_feature_coverage_report(
        panel,
        ["future_return_hint"],
        config=CoverageGateConfig(
            recent_symbol_threshold=1,
            recent_20d_symbol_threshold=1,
            min_recent_trading_days=1,
        ),
    )
    row = report.iloc[0]
    assert not row["is_allowed_for_base_model"]
    assert not row["is_allowed_for_recent_model"]
    assert row["is_allowed_for_prediction_only"]
    assert "future-field" in row["rejection_reason"]


def test_gate_first_fusion_tiers_and_rank_order():
    from quant_platform.selection.gate_fusion import gate_first_fusion

    scored = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E", "F"],
        "trade_date": ["2024-01-02"] * 6,
        "base_score": [1.0, 0.8, 0.9, 0.1, 0.2, 0.7],
        "base_pct": [0.90, 0.65, 0.88, 0.30, 0.40, 0.70],
        "recent_score": [0.5, 1.0, 0.1, 0.95, 0.2, 0.8],
        "recent_pct": [0.70, 0.90, 0.20, 0.95, 0.40, 0.80],
        "risk_flags": ["", "", "", "", "", "high_unlock"],
    })

    fused = gate_first_fusion(scored)
    tier_by_symbol = dict(zip(fused["symbol"], fused["gate_tier"]))

    assert tier_by_symbol["A"] == "A_MAIN"
    assert tier_by_symbol["B"] == "B_SHORT_BOOST"
    assert tier_by_symbol["C"] == "C_DOWNGRADE_OBSERVE"
    assert tier_by_symbol["D"] == "D_OBSERVE"
    assert tier_by_symbol["E"] == "E_REJECT"
    assert tier_by_symbol["F"] == "RISK_VETO"
    assert fused.iloc[0]["symbol"] == "A"
    assert fused["final_rank"].tolist() == list(range(1, len(fused) + 1))


def test_fuse_parser_accepts_ranked_inputs():
    from quant_platform.cli import build_parser

    args = build_parser().parse_args([
        "fuse",
        "--base-ranked", "base.csv",
        "--recent-ranked", "recent.csv",
        "--output-dir", "reports",
    ])
    assert args.command == "fuse"
    assert args.prefix == "D3_fused"
