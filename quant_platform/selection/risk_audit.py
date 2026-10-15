"""quant_platform.selection.risk_audit
=====================================
P0 风险与分歧可观测性（Claude 2026-10-15 最终裁决）。

只做观测、不做过滤：
- ``risk_flags`` / ``event_flags`` 透传上游；缺失时标记 UNKNOWN，不默认为安全。
- ``disagreement_flag`` 标记 base/recent 排名分歧，供审计与后续 P1 规则使用。
- ``annotate_picks`` 给选股输出补齐审计字段（eligible / pre-post filter rank /
  rule_version），为 P1 风险过滤器预留合同字段。

本模块只依赖 pandas/numpy，无 sklearn/lightgbm 依赖，可离线测试。

2026-10-15 R1/R2/R3 修复（Codex 本机验证发现，Claude 裁决必须修）：
- R1：coalesce_flag_values 改为并集语义——UNKNOWN 视同缺失回退对方；
  双有效不同代码做并集合并去重，不再静默丢弃 secondary 的有效否决码。
- R2：attach_audit_columns 供每周出口按 symbol 把比较名单的 8 个审计列
  merge 回顶层 picks/JSON，保证双出口逐字段一致。
- R3：risk_data_status 最小版时间校验——代码时间戳不可解析或晚于 as_of
  时标 INVALID，不伪造"当时可得"证明。

作用域（写死）：risk_data_status 是池级聚合不是逐股状态；eligible=True
不得解读为通过风险检查；disagreement_flag=False 不得解读为股票安全。
"""

from __future__ import annotations

import datetime as dt
import re

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

#: risk_data_status 的无效时间状态：任一代码时间戳不可解析或晚于 as_of。
#: P0 最小版（2026-10-15 R3）：不能伪造"当时可得"证明。
#: 完整 known_at/observed_at 合同时序等 P1 有真实风险生产者时再定。
RISK_INVALID_TIME = "INVALID"

#: 代码尾部 @时间戳（如 extreme_outflow@2026-10-14）。
_TS_TAIL_RE = re.compile(r"@([^@;]*?)\s*$")

#: CSV 往返后的布尔字符串还原。
_BOOL_LIKE = {"True": True, "False": False}


def _clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _split_codes(value: str) -> list[str]:
    """把可能含 ';' 的标记串拆成去重后的有效代码（丢弃空值和 UNKNOWN）。"""
    seen: list[str] = []
    for part in str(value).split(";"):
        part = part.strip()
        if not part or part == RISK_UNKNOWN or part in seen:
            continue
        seen.append(part)
    return seen


def coalesce_flag_values(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    """合并两路风险标记（独立风险事件并集语义）。

    - 空值和 UNKNOWN 均视为缺失，由对方补位；
    - 两路都有有效代码时做并集去重（独立风险事件不是互斥选项），
      保留各自时间和来源，primary 的代码排在前面；
    - 结果仍空则 UNKNOWN。

    R1 修复（2026-10-15）：旧实现只把 "" 当缺失，UNKNOWN 作为非空值
    被优先保留，会静默吞掉 secondary 的有效否决码。
    """
    p = _clean(primary)
    s = _clean(secondary.reindex(p.index))
    out = []
    for pv, sv in zip(p.tolist(), s.tolist()):
        codes = _split_codes(pv)
        codes += [c for c in _split_codes(sv) if c not in codes]
        out.append(";".join(codes) if codes else RISK_UNKNOWN)
    return pd.Series(out, index=p.index)


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


def _as_date(value) -> dt.date:
    """as_of 归一化为 date（接受 date/datetime/ISO 字符串）。"""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _future_or_bad_codes(code: str, as_of: dt.date) -> list[str]:
    """返回 code 串中时间戳无效的子代码：不可解析，或晚于 as_of。

    无 @时间戳的子代码在 P0 最小版中不裁决（P1 有真实风险生产者时
    再定完整合同时序）；"当时可得"的未来事件（如已公告事项）也不应
    一律剔除——这里只拦"时间戳晚于信号日却标 OK"的伪造证明行为。
    """
    bad = []
    for part in str(code).split(";"):
        part = part.strip()
        if not part or part == RISK_UNKNOWN:
            continue
        m = _TS_TAIL_RE.search(part)
        if not m:
            continue  # 无时间戳：最小版不裁决
        try:
            ts = dt.date.fromisoformat(m.group(1).strip())
        except ValueError:
            bad.append(part)
            continue
        if ts > as_of:
            bad.append(part)
    return bad


def risk_data_status(codes: pd.Series, as_of) -> str:
    """整池风险数据状态（池级聚合，不是逐股状态）。

    - 全 UNKNOWN -> UNKNOWN（无风险数据）
    - 任一代码时间戳不可解析或晚于 as_of -> INVALID
      （不能伪造"当时可得"证明；R3 最小版，2026-10-15）
    - 部分 UNKNOWN -> PARTIAL
    - 否则 OK
    """
    as_of = _as_date(as_of)
    unknown = codes == RISK_UNKNOWN
    if bool(unknown.all()):
        return RISK_UNKNOWN
    for code in codes[~unknown].tolist():
        if _future_or_bad_codes(code, as_of):
            return RISK_INVALID_TIME
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

    作用域声明（写死）：
    - risk_data_status 是整池聚合，不是逐股状态；
    - eligible=True 只表示"本版无风险过滤"，不得解读为"已通过风险检查"；
    - disagreement_flag=False 只表示未达到本版分歧条件，不得解读为"股票安全"。
    """
    out = picks.copy()
    out["disagreement_flag"] = disagreement_flag(out, rank_gap_threshold)
    out["risk_as_of"] = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
    codes = reason_codes(out)
    out["risk_reason_codes"] = codes
    out["risk_data_status"] = risk_data_status(codes, as_of)
    out["eligible"] = True
    if "recommendation_rank" in out.columns:
        out["pre_filter_rank"] = out["recommendation_rank"]
        out["post_filter_rank"] = out["recommendation_rank"]
    else:
        out["pre_filter_rank"] = pd.NA
        out["post_filter_rank"] = pd.NA
    out["rule_version"] = rule_version
    return out


def attach_audit_columns(picks: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """R2：按 symbol 把 source（含审计列）的 AUDIT_COLUMNS merge 回 picks。

    要求：
    - source 必须含全部 8 个审计列；缺失即抛错——说明上游比较入口的
      审计接线断了，静默写出无审计列的名单比报错更危险；
    - picks 与 source 的 symbol 必须一一对应（调用方在 merge 前已做
      symbol 顺序 + 融合分数一致性校验）。

    返回带审计列的 picks 副本；disagreement_flag/eligible 从 CSV 的
    "True"/"False" 字符串还原为 bool，保持 JSON 输出类型正确。
    不改变行顺序与选股内容，只追加列。
    """
    missing = [c for c in AUDIT_COLUMNS if c not in source.columns]
    if missing:
        raise ValueError(f"Audit source missing columns: {missing}")
    merged = picks.merge(
        source[["symbol", *AUDIT_COLUMNS]], on="symbol", how="left",
        validate="one_to_one",
    )
    if merged[list(AUDIT_COLUMNS)].isna().any().any():
        raise ValueError("Audit columns failed to attach for some symbols")
    for col in ("disagreement_flag", "eligible"):
        vals = merged[col]
        if vals.dtype == object:
            mapped = vals.map(_BOOL_LIKE)
            if mapped.notna().all():
                merged[col] = mapped.astype(bool)
    return merged
