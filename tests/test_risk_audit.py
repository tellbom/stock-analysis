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
    RISK_INVALID_TIME,
    RISK_RULE_VERSION,
    RISK_UNKNOWN,
    annotate_picks,
    attach_audit_columns,
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


def test_coalesce_union_semantics():
    """R1：独立风险事件做并集合并，UNKNOWN 视同缺失回退对方。"""
    p = pd.Series(["A", "", None, "", "A", "A"])
    s = pd.Series(["X", "B", "C", None, "A", RISK_UNKNOWN])
    assert coalesce_flag_values(p, s).tolist() == [
        "A;X",          # 双有效不同代码：并集合并，primary 在前
        "B",            # primary 空：secondary 补位
        "C",            # primary None：secondary 补位
        RISK_UNKNOWN,   # 双空：UNKNOWN
        "A",            # 重复代码：去重
        "A",            # secondary UNKNOWN 视同缺失
    ]


def test_coalesce_unknown_falls_back_to_secondary_veto():
    """R1 最小复现：base=UNKNOWN + recent=有效否决码，否决码不得被吞。"""
    p = pd.Series([RISK_UNKNOWN])
    s = pd.Series(["extreme_outflow@2026-10-14"])
    assert coalesce_flag_values(p, s).tolist() == ["extreme_outflow@2026-10-14"]


def test_coalesce_two_valid_different_codes_keeps_both_with_time():
    """R1：双有效不同代码并集，保留各自时间和来源。"""
    p = pd.Series(["coverage@2026-10-14"])
    s = pd.Series(["extreme_outflow@2026-10-14"])
    assert coalesce_flag_values(p, s).tolist() == [
        "coverage@2026-10-14;extreme_outflow@2026-10-14"
    ]


def test_coalesce_multi_code_strings_union_dedup():
    p = pd.Series(["A@2026-10-01;B@2026-10-02"])
    s = pd.Series(["B@2026-10-02;C@2026-10-03"])
    assert coalesce_flag_values(p, s).tolist() == ["A@2026-10-01;B@2026-10-02;C@2026-10-03"]


def test_coalesce_aligns_misaligned_index():
    """R1：错位索引按 primary 索引对齐，secondary 缺失位置视为空。"""
    p = pd.Series(["A", ""], index=[10, 20])
    s = pd.Series(["X"], index=[10])
    assert coalesce_flag_values(p, s).tolist() == ["A;X", RISK_UNKNOWN]


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
    as_of = dt.date(2026, 10, 14)
    assert risk_data_status(pd.Series([RISK_UNKNOWN, RISK_UNKNOWN]), as_of) == "UNKNOWN"
    assert risk_data_status(pd.Series(["A", RISK_UNKNOWN]), as_of) == "PARTIAL"
    assert risk_data_status(pd.Series(["A", "B"]), as_of) == "OK"


def test_risk_data_status_rejects_future_timestamp():
    """R3 最小复现：extreme_outflow@2026-10-16 在信号日 10-14 不得标 OK。"""
    as_of = dt.date(2026, 10, 14)
    codes = pd.Series(["extreme_outflow@2026-10-16"])
    assert risk_data_status(codes, as_of) == RISK_INVALID_TIME


def test_risk_data_status_rejects_unparsable_timestamp():
    as_of = dt.date(2026, 10, 14)
    codes = pd.Series(["extreme_outflow@tomorrow"])
    assert risk_data_status(codes, as_of) == RISK_INVALID_TIME
    codes = pd.Series(["extreme_outflow@2026-13-45"])
    assert risk_data_status(codes, as_of) == RISK_INVALID_TIME


def test_risk_data_status_accepts_past_and_untimestamped_codes():
    """R3 最小版：过去时间戳 OK；无 @时间戳的代码不裁决（P1 再定）。"""
    as_of = dt.date(2026, 10, 14)
    assert risk_data_status(pd.Series(["extreme_outflow@2026-10-14"]), as_of) == "OK"
    assert risk_data_status(pd.Series(["extreme_outflow@2026-10-13"]), as_of) == "OK"
    assert risk_data_status(pd.Series(["DRAWDOWN"]), as_of) == "OK"


def test_risk_data_status_future_code_among_unknowns_is_invalid_not_partial():
    as_of = dt.date(2026, 10, 14)
    codes = pd.Series(["extreme_outflow@2026-10-16", RISK_UNKNOWN])
    assert risk_data_status(codes, as_of) == RISK_INVALID_TIME


def test_risk_data_status_accepts_datetime_as_of():
    as_of = dt.datetime(2026, 10, 14, 15, 0, 0)
    assert risk_data_status(pd.Series(["A@2026-10-14"]), as_of) == "OK"
    assert risk_data_status(pd.Series(["A@2026-10-15"]), as_of) == RISK_INVALID_TIME


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


def _audit_source(symbols=("000001", "000002")):
    """模拟比较名单 D{h}_hybrid_flow_picks.csv（含 8 个审计列，CSV 往返形态）。"""
    return pd.DataFrame({
        "symbol": list(symbols),
        "fusion_score_col": [0.9, 0.8][: len(symbols)],
        "disagreement_flag": ["True", "False"][: len(symbols)],
        "risk_as_of": ["2026-10-14"] * len(symbols),
        "risk_data_status": ["UNKNOWN"] * len(symbols),
        "risk_reason_codes": [RISK_UNKNOWN] * len(symbols),
        "eligible": ["True"] * len(symbols),
        "pre_filter_rank": [1, 2][: len(symbols)],
        "post_filter_rank": [1, 2][: len(symbols)],
        "rule_version": [RISK_RULE_VERSION] * len(symbols),
    })


def test_attach_audit_columns_merges_all_eight_fields():
    """R2：每周 picks 按 symbol 拿回 8 个审计列，逐字段与比较名单一致。"""
    picks = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "fusion_score_col": [0.9, 0.8],
        "recommendation_rank": [1, 2],
    })
    source = _audit_source()
    out = attach_audit_columns(picks, source)
    for col in AUDIT_COLUMNS:
        assert col in out.columns, f"missing {col}"
        assert out[col].tolist() == source[col].tolist() or (
            out[col].astype(str).tolist() == source[col].astype(str).tolist()
        ), f"column {col} mismatch"
    # 选股内容不变
    assert out["symbol"].tolist() == ["000001", "000002"]
    assert out["fusion_score_col"].tolist() == [0.9, 0.8]
    # 布尔列从 CSV 字符串还原为 bool
    assert out["disagreement_flag"].tolist() == [True, False]
    assert out["eligible"].tolist() == [True, True]


def test_attach_audit_columns_rejects_missing_columns():
    """R2：上游审计接线断了（缺列）必须抛错，不能静默写出无审计列名单。"""
    picks = pd.DataFrame({"symbol": ["000001"], "fusion_score_col": [0.9]})
    source = _audit_source().drop(columns=["risk_reason_codes"])
    try:
        attach_audit_columns(picks, source)
    except ValueError as e:
        assert "risk_reason_codes" in str(e)
    else:
        raise AssertionError("expected ValueError for missing audit column")


def test_attach_audit_columns_rejects_symbol_mismatch():
    """R2：symbol 对不上必须抛错。"""
    picks = pd.DataFrame({"symbol": ["000001", "999999"], "fusion_score_col": [0.9, 0.1]})
    try:
        attach_audit_columns(picks, _audit_source())
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for symbol mismatch")
