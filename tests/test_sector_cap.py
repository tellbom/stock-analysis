"""Tests for the tech-sector cap (风控层) in IndustryNeutralRanker.

- cap disabled (None/0): historical-reproduction aperture, no filtering.
- cap enabled: tech trimmed to cap, lowest scores dropped first.
- weekly_selection_config(): ordinary and hybrid share one aperture.
"""
from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.selection.config import SelectionConfig, weekly_selection_config
from quant_platform.selection.ranker import IndustryNeutralRanker, TECH_SECTOR_INDUSTRIES

TECHS = ["半导体", "数字芯片设计", "通信设备", "消费电子", "软件开发"]


def _panel():
    rows = []
    k = 0
    for ti, t in enumerate(TECHS):
        for i in range(5):
            rows.append({
                "symbol": f"{k:06d}", "industry_code": f"t{ti}",
                "industry_name": t, "score": 100 - k,
            })
            k += 1
    for i in range(5):
        rows.append({
            "symbol": f"8{i:05d}", "industry_code": "b0",
            "industry_name": "银行", "score": 10 - i,
        })
    return pd.DataFrame(rows)


def _run(df, **cfg_kwargs):
    ranker = IndustryNeutralRanker(
        weekly_selection_config(**cfg_kwargs),
        industry_col="industry_code", name_col="industry_name", score_col="score",
    )
    out = ranker.run(df)
    return out, ranker.last_sector_cap_stats


def test_cap_trims_tech_to_cap_dropping_lowest_scores():
    out, stats = _run(_panel(), sector_cap_tech=10)
    sel = out[out.selected]
    assert stats["enabled"] is True
    assert stats["cap"] == 10
    assert stats["pre_count"] == 18
    assert stats["post_count"] == 13
    assert sel["industry_name"].isin(TECH_SECTOR_INDUSTRIES).sum() == 10
    # 被剔除的是分数最低的 5 只科技股
    assert sorted(stats["dropped_symbols"]) == ["000016", "000017", "000020", "000021", "000022"]
    dropped_rows = out[out.symbol.isin(stats["dropped_symbols"])]
    assert (dropped_rows["selection_reason"] == "trimmed_by_sector_cap_tech").all()
    assert not sel["symbol"].isin(stats["dropped_symbols"]).any()


def test_cap_disabled_by_none_or_zero():
    df = _panel()
    for off in (None, 0):
        out, stats = _run(df, sector_cap_tech=off)
        assert stats["enabled"] is False
        assert stats["pre_count"] == stats["post_count"] == 18
        assert out.selected.sum() == 18


def test_no_trim_when_under_cap():
    df = _panel().head(8)  # 8 tech + 0 bank -> under cap after industry top_k
    out, stats = _run(df, sector_cap_tech=10)
    assert stats["enabled"] is True
    assert stats["pre_count"] == stats["post_count"]
    assert stats["dropped_symbols"] == []


def test_weekly_config_is_single_aperture():
    cfg = weekly_selection_config()
    assert cfg.top_k == 3 and cfg.max_total == 50
    assert cfg.strategy.value == "equal_top_k"
    assert cfg.sector_cap_tech == 10
    assert isinstance(cfg.version, str) and cfg.version
    with pytest.raises(ValueError):
        SelectionConfig(sector_cap_tech=-1)
