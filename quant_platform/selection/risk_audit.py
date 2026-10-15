"""quant_platform.selection.risk_audit
=====================================
P0 风险与分歧可观测性（Claude 2026-10-15 最终裁决）。

只做观测、不做过滤：
- ``risk_flags`` / ``event_flags`` 透传上游；缺失时标记 UNKNOWN，不默认为安全。
- ``disagreement_flag`` 标记 base/recent 排名分歧，供审计与后续 P1 规则使用。
- ``annotate_picks`` 给选股输出补齐审计字段（eligible / pre-post filter rank /
  rule_version），为 P1 风险过滤器预留合同字段。

本模块只依赖 pandas/numpy，无 sklearn/lightgbm 依赖，可离线测试。
"""

from __future__ import annotations

import pandas as pd

#: 上游无风险数据时的显式标记。空字符串历史曾被误读为"无风险"，
#: P0 裁决：未知必须显式标记，不默认为安全。
RISK_UNKNOWN = "UNKNOWN"

#: 风险规则合同版本。P0 只有观测、无过滤规则；P1 加规则时升级版本。
RISK_RULE_VERSION = "risk-observability-v1"

#: 风险标记列（透传口径）。
RISK_FLAG_COLUMNS = ("risk_flags", "event_flags")

#: annotate_picks 新增的审计字段（顺序即 CSV 列顺序）。
AUDIT_COLUMNS = (
    "disagreement_flag",
    "risk_as_of",
    "risk_data_status",
    "risk_reason_codes",
    "eligible",
    "pre_filter_rank",
    "post_filter_rank",
    "rule_version",
)


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def coalesce_flag_values(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    """合并两路风险标记：primary 优先，空位由 secondary 补，仍空则 UNKNOWN。"""
    p = _clean(primary)
    s = _clean(secondary)
    merged = p.where(p != "", s)
    return merged.where(merged != "", RISK_UNKNOWN)


def unknown_flag_series(index) -> pd.Series:
    """全 UNKNOWN 的标记列（上游没有任何风险数据时）。"""
    return pd.Series(RISK_UNKNOWN, index=index)


def disagreement_threshold(fused: pd.DataFrame) -> float:
    """全池 base/recent 排名差绝对值的中位数（50% 分位）。

    阈值必须在选股前的全池上计算，不能在已截断的 picks 上算。
    """
    diff = (fused["base_rank"].astype(float) - fused["recent_rank"].astype(float)).abs()
    return float(diff.median())


def disagreement_flag(frame: pd.DataFrame, threshold: float | None) -> pd.Series:
    """分歧标记：排名差超全池 50% 分位，或两模型预测符号相反。

    threshold=None（无 recent 模型，如 ordinary 臂）时全为 False。
    """
    if threshold is None or "recent_rank" not in frame.columns:
        return pd.Series(False, index=frame.index)
    rank_gap = (frame["base_rank"].astype(float) - frame["recent_rank"].astype(float)).abs() > threshold
    if "base_score" in frame.columns and "recent_score" in frame.columns:
        sign_clash = (frame["base_score"].astype(float) < 0) != (frame["recent_score"].astype(float) < 0)
        return (rank_gap | sign_clash).astype(bool)
    return rank_gap.astype(bool)


def reason_codes(frame: pd.DataFrame) -> pd.Series:
    """每行的有效风险原因码：risk_flags + event_flags 去空合并，无则 UNKNOWN。"""
    codes = pd.Series("", index=frame.index)
    for col in RISK_FLAG_COLUMNS:
        if col not in frame.columns:
            continue
        vals = _clean(frame[col])
        vals = vals.where((vals != "") & (vals != RISK_UNKNOWN), "")
        needs_sep = (codes != "") & (vals != "")
        codes = codes.where(~needs_sep, codes + ";" + vals)
        codes = codes.where((codes != "") | (vals == ""), vals)
    return codes.where(codes != "", RISK_UNKNOWN)


def risk_data_status(codes: pd.Series) -> str:
    """整池风险数据状态：全 UNKNOWN / 部分 / 完整。"""
    unknown = codes == RISK_UNKNOWN
    if bool(unknown.all()):
        return RISK_UNKNOWN
    if bool(unknown.any()):
        return "PARTIAL"
    return "OK"


def annotate_picks(
    picks: pd.DataFrame,
    *,
    as_of,
    rank_gap_threshold: float | None,
    rule_version: str = RISK_RULE_VERSION,
) -> pd.DataFrame:
    """给选股输出补齐 P0 审计字段。

    P0 无过滤：eligible 全 True，pre/post_filter_rank 均为 recommendation_rank。
    不改变任何选股行为，只追加列。
    """
    out = picks.copy()
    out["disagreement_flag"] = disagreement_flag(out, rank_gap_threshold)
    out["risk_as_of"] = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
    codes = reason_codes(out)
    out["risk_reason_codes"] = codes
    out["risk_data_status"] = risk_data_status(codes)
    out["eligible"] = True
    if "recommendation_rank" in out.columns:
        out["pre_filter_rank"] = out["recommendation_rank"]
        out["post_filter_rank"] = out["recommendation_rank"]
    else:
        out["pre_filter_rank"] = pd.NA
        out["post_filter_rank"] = pd.NA
    out["rule_version"] = rule_version
    return out
