"""P0 风险与分歧可观测性测试（Claude 2026-10-15 最终裁决）。

- combine() 透传上游风险标记；缺失时标 UNKNOWN，不默认为安全。
- disagreement_flag 分歧标记触发逻辑。
- annotate_picks 补齐 8 个审计字段，不改变选股行为。

纯 pandas/numpy 逻辑；combine 相关用例复用 scripts.compare_original_hybrid。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from quant_platform.selection.risk_audit import (
    AUDIT_COLUMNS,
    RISK_RULE_VERSION,
    RISK_UNKNOWN,
    annotate_picks,
    coalesce_flag_values,
    disagreement_flag,
    disagreement_threshold,
    reason_codes,
    risk_data_status,
)
from scripts.compare_original_hybrid import combine, scores


def _frames(**kwargs):
    pred = pd.DataFrame({
        "symbol": ["000001", "000002", "000003"],
        "date": [dt.date(2026, 10, 14)] * 3,
    })
    base = scores(pred, np.array([0.05, 0.03, -0.02]), "base")
    recent = scores(pred, np.array([0.04, -0.01, 0.02]), "recent")
    for col, vals in kwargs.items():
        target, _, name = col.partition("__")
        (base if target == "base" else recent)[name] = vals
    return base, recent


def test_combine_passes_through_upstream_risk_flags():
    base, recent = _frames(base__risk_flags=["DRAWDOWN@2026-10-01", "", ""])
    out = combine(base, recent)
    assert out["risk_flags"].tolist() == ["DRAWDOWN@2026-10-01", RISK_UNKNOWN, RISK_UNKNOWN]
    assert out["event_flags"].tolist() == [RISK_UNKNOWN] * 3


def test_combine_falls_back_to_recent_flags_when_base_empty():
    base, recent = _frames(recent__event_flags=["", "EARNINGS@2026-10-10", ""])
    out = combine(base, recent)
    assert out["risk_flags"].tolist() == [RISK_UNKNOWN] * 3
    assert out["event_flags"].tolist() == [RISK_UNKNOWN, "EARNINGS@2026-10-10", RISK_UNKNOWN]


def test_combine_marks_unknown_instead_of_defaulting_safe():
    """P0 核心：空不再是""（曾被误读为无风险），而是显式 UNKNOWN。"""
    base, recent = _frames()
    out = combine(base, recent)
    assert (out["risk_flags"] == RISK_UNKNOWN).all()
    assert (out["event_flags"] == RISK_UNKNOWN).all()
    # 选股列不受影响
    assert out["base_score"].tolist() == [0.05, 0.03, -0.02]


def test_coalesce_prefers_primary_then_secondary_then_unknown():
    p = pd.Series(["A", "", None, ""])
    s = pd.Series(["X", "B", "C", None])
    assert coalesce_flag_values(p, s).tolist() == ["A", "B", "C", RISK_UNKNOWN]


def test_disagreement_threshold_is_pool_median():
    fused = pd.DataFrame({"base_rank": [1, 2, 3, 4], "recent_rank": [4, 3, 2, 1]})
    assert disagreement_threshold(fused) == 2.0  # |diff| = 3,1,1,3


def test_disagreement_flag_rank_gap_and_sign_clash():
    frame = pd.DataFrame({
        "base_rank": [1, 2, 3, 4],
        "recent_rank": [10, 3, 4, 5],
        "base_score": [0.05, 0.04, -0.01, 0.02],
        "recent_score": [0.06, 0.03, 0.02, 0.01],
    })
    flags = disagreement_flag(frame, 2.0)
    assert flags.tolist() == [True, False, True, False]
    # row0: 排名差 9 > 2；row2: 一负一正符号相反；row1/row3: 无分歧


def test_disagreement_flag_false_without_recent_model():
    """ordinary 臂无 recent_rank：分歧标记全 False（不是 NA，不是报错）。"""
    frame = pd.DataFrame({"base_rank": [1, 2], "base_score": [0.05, 0.03]})
    assert disagreement_flag(frame, None).tolist() == [False, False]
    assert disagreement_flag(frame, 1.0).tolist() == [False, False]


def test_reason_codes_merges_flags_and_unknown_when_empty():
    frame = pd.DataFrame({
        "risk_flags": ["A", "", RISK_UNKNOWN, ""],
        "event_flags": ["", "B", "", RISK_UNKNOWN],
    })
    assert reason_codes(frame).tolist() == ["A", "B", RISK_UNKNOWN, RISK_UNKNOWN]


def test_risk_data_status_levels():
    assert risk_data_status(pd.Series([RISK_UNKNOWN, RISK_UNKNOWN])) == "UNKNOWN"
    assert risk_data_status(pd.Series(["A", RISK_UNKNOWN])) == "PARTIAL"
    assert risk_data_status(pd.Series(["A", "B"])) == "OK"


def test_annotate_picks_adds_all_audit_columns_without_changing_selection():
    picks = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "recommendation_rank": [1, 2],
        "base_rank": [1, 50],
        "recent_rank": [60, 51],
        "base_score": [0.05, 0.01],
        "recent_score": [0.02, 0.015],
        "risk_flags": ["", ""],
        "event_flags": ["", ""],
    })
    frozen = picks.copy(deep=True)
    ann = annotate_picks(picks, as_of=dt.date(2026, 10, 14), rank_gap_threshold=10.0)
    pd.testing.assert_frame_equal(picks, frozen)  # 不改变输入
    for col in AUDIT_COLUMNS:
        assert col in ann.columns, f"missing audit column {col}"
    assert ann["disagreement_flag"].tolist() == [True, False]  # |1-60|=59>10
    assert (ann["risk_as_of"] == "2026-10-14").all()
    assert (ann["risk_data_status"] == "UNKNOWN").all()
    assert (ann["risk_reason_codes"] == RISK_UNKNOWN).all()
    assert ann["eligible"].all()
    assert ann["pre_filter_rank"].tolist() == [1, 2]
    assert ann["post_filter_rank"].tolist() == [1, 2]
    assert (ann["rule_version"] == RISK_RULE_VERSION).all()
