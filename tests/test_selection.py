"""Tests for quant_platform.selection — industry-neutral ranking & selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.selection.config import SelectionConfig, StrategyType
from quant_platform.selection.exposure import ExposureMonitor
from quant_platform.selection.ranker import IndustryNeutralRanker
from quant_platform.selection.strategies import (
    EqualTopKStrategy,
    HybridStrategy,
    ProportionalTopKStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_panel():
    """Panel with 3 industries x 10 stocks = 30 rows, one date."""
    np.random.seed(42)
    symbols = []
    industries = []
    scores = []
    for ind in ["半导体", "银行", "食品饮料"]:
        for i in range(10):
            symbols.append(f"{ind[:2]}{i:04d}")
            industries.append(ind)
            scores.append(np.random.randn())

    return pd.DataFrame({
        "symbol": symbols,
        "date": pd.Timestamp("2026-06-18"),
        "industry_code": industries,
        "industry_name": [f"{i}_name" for i in industries],
        "model_score": scores,
    })


@pytest.fixture
def panel_with_unknown(basic_panel):
    """Panel with some UNKNOWN industry stocks."""
    df = basic_panel.copy()
    df.loc[0:2, "industry_code"] = None
    df.loc[3, "industry_code"] = ""
    return df


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestSelectionConfig:
    def test_defaults(self):
        cfg = SelectionConfig()
        assert cfg.strategy == StrategyType.EQUAL_TOP_K
        assert cfg.top_k == 3
        assert cfg.max_total == 50
        assert cfg.hybrid_weight == 0.5

    def test_rejects_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            SelectionConfig(top_k=0)

    def test_rejects_invalid_hybrid_weight(self):
        with pytest.raises(ValueError, match="hybrid_weight"):
            SelectionConfig(hybrid_weight=1.5)

    def test_rejects_invalid_thresholds(self):
        with pytest.raises(ValueError, match="exposure_warning"):
            SelectionConfig(
                exposure_warning_threshold=0.1,
                exposure_diversified_threshold=0.2,
            )

    def test_coerces_str_to_enum(self):
        cfg = SelectionConfig(strategy="hybrid")  # type: ignore[arg-type]
        assert cfg.strategy == StrategyType.HYBRID


# ---------------------------------------------------------------------------
# rank() tests
# ---------------------------------------------------------------------------

class TestIndustryNeutralRankerRank:
    def test_adds_expected_columns(self, basic_panel):
        ranker = IndustryNeutralRanker(SelectionConfig())
        result = ranker.rank(basic_panel)
        for col in ["industry_rank", "industry_neutral_score", "global_score", "industry_size"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_preserves_original_columns(self, basic_panel):
        ranker = IndustryNeutralRanker(SelectionConfig())
        result = ranker.rank(basic_panel)
        for col in basic_panel.columns:
            assert col in result.columns

    def test_industry_rank_within_groups(self, basic_panel):
        """Best score in each industry should get industry_rank=1."""
        ranker = IndustryNeutralRanker(SelectionConfig())
        result = ranker.rank(basic_panel)

        for ind in basic_panel["industry_code"].unique():
            grp = result[result["industry_code"] == ind]
            best = grp.loc[grp["model_score"].idxmax()]
            assert best["industry_rank"] == 1, f"{ind}: expected rank=1, got {best['industry_rank']}"

    def test_unknown_industry_grouped_together(self, panel_with_unknown):
        ranker = IndustryNeutralRanker(SelectionConfig())
        result = ranker.rank(panel_with_unknown)

        unknown_rows = result[
            result["industry_code"].isna() | (result["industry_code"] == "")
        ]
        # Should all share ranks 1..4 within the unknown group
        ranks = unknown_rows["industry_rank"].values
        assert set(ranks) == {1, 2, 3, 4}

    def test_zscore_fallback_for_small_groups(self):
        """<5 stocks in industry → percentile fallback (values in [0,1])."""
        df = pd.DataFrame({
            "symbol": ["A", "B"],
            "date": pd.Timestamp("2026-06-18"),
            "industry_code": ["ind1", "ind1"],
            "industry_name": ["i1", "i1"],
            "model_score": [10.0, 20.0],
        })
        ranker = IndustryNeutralRanker(SelectionConfig())
        result = ranker.rank(df)
        zscores = result["industry_neutral_score"].values
        # percentile: higher score → higher percentile (0.5, 1.0 for 2 values)
        assert zscores[0] < zscores[1]

    def test_raises_on_missing_score_col(self):
        df = pd.DataFrame({"symbol": ["A"], "industry_code": ["X"]})
        ranker = IndustryNeutralRanker(SelectionConfig())
        with pytest.raises(ValueError, match="model_score"):
            ranker.rank(df)

    def test_raises_on_missing_industry_col(self):
        df = pd.DataFrame({"symbol": ["A"], "model_score": [0.5]})
        ranker = IndustryNeutralRanker(SelectionConfig())
        with pytest.raises(ValueError, match="industry_code"):
            ranker.rank(df)

    def test_handles_empty_panel(self):
        df = pd.DataFrame(columns=["symbol", "date", "industry_code", "model_score"])
        ranker = IndustryNeutralRanker(SelectionConfig())
        result = ranker.rank(df)
        assert len(result) == 0
        assert "industry_rank" in result.columns


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestEqualTopK:
    def test_selects_top_k_per_industry(self, basic_panel):
        cfg = SelectionConfig(strategy="equal_top_k", top_k=3, max_total=50)
        selected, _reasons = EqualTopKStrategy().select(
            basic_panel, cfg, "industry_code", "model_score", "symbol",
        )
        # 3 industries x 3 each = 9
        assert len(selected) == 9

    def test_respects_max_total_cap(self, basic_panel):
        cfg = SelectionConfig(strategy="equal_top_k", top_k=5, max_total=6)
        selected, _reasons = EqualTopKStrategy().select(
            basic_panel, cfg, "industry_code", "model_score", "symbol",
        )
        assert len(selected) == 6


class TestProportionalTopK:
    def test_selects_something(self, basic_panel):
        cfg = SelectionConfig(strategy="proportional_top_k", top_k=3, max_total=50)
        selected, _reasons = ProportionalTopKStrategy().select(
            basic_panel, cfg, "industry_code", "model_score", "symbol",
        )
        # Each industry has 10 stocks, proportional k=3 → 3 each → 9 total
        assert 6 <= len(selected) <= 12


class TestHybrid:
    def test_requires_industry_neutral_score(self, basic_panel):
        cfg = SelectionConfig(strategy="hybrid", hybrid_weight=0.5)
        with pytest.raises(ValueError, match="industry_neutral_score"):
            HybridStrategy().select(
                basic_panel, cfg, "industry_code", "model_score", "symbol",
            )

    def test_selects_up_to_max_total(self, basic_panel):
        # First rank, then hybrid select
        ranker = IndustryNeutralRanker(SelectionConfig())
        ranked = ranker.rank(basic_panel)

        cfg = SelectionConfig(strategy="hybrid", top_k=3, max_total=15)
        selected, _reasons = HybridStrategy().select(
            ranked, cfg, "industry_code", "model_score", "symbol",
        )
        assert len(selected) <= 15
        assert len(selected) >= 3  # at least min_stocks_per_industry


# ---------------------------------------------------------------------------
# ExposureMonitor tests
# ---------------------------------------------------------------------------

class TestExposureMonitor:
    def test_flag_overweight(self, basic_panel):
        cfg = SelectionConfig(exposure_warning_threshold=0.30)

        # Mark all semiconductor stocks as selected → 100% concentration
        panel = basic_panel.copy()
        panel["selected"] = panel["industry_code"] == "半导体"

        flags = ExposureMonitor.flag(panel, cfg, "industry_code", "selected")
        selected_flags = flags[panel["selected"]]
        assert (selected_flags == "industry_overweight").all()

    def test_flag_balanced(self, basic_panel):
        # Use a higher warning threshold so 33% per industry reads as balanced
        cfg = SelectionConfig(exposure_warning_threshold=0.40)
        panel = basic_panel.copy()
        # Select 3 per industry → 33% each (under 40% warning threshold)
        ranked = panel.copy()
        ranked["selected"] = False
        for ind in panel["industry_code"].unique():
            grp = panel[panel["industry_code"] == ind].head(3)
            ranked.loc[grp.index, "selected"] = True

        flags = ExposureMonitor.flag(ranked, cfg, "industry_code", "selected")
        selected_flags = flags[ranked["selected"]]
        assert (selected_flags == "balanced").all()

    def test_flag_diversified(self):
        cfg = SelectionConfig(
            exposure_warning_threshold=0.30,
            exposure_diversified_threshold=0.15,
        )
        # 10 industries x 2 stocks each = 10% per industry
        rows = []
        for i in range(10):
            rows.append({"symbol": f"A{i}", "industry_code": f"ind{i}", "selected": True})
        panel = pd.DataFrame(rows)

        flags = ExposureMonitor.flag(panel, cfg, "industry_code", "selected")
        assert (flags == "diversified").all()

    def test_concentration_report(self, basic_panel):
        cfg = SelectionConfig()
        panel = basic_panel.copy()
        panel["selected"] = True  # select all

        report = ExposureMonitor.concentration_report(
            panel, "industry_code", "industry_name", "selected", "symbol",
        )
        assert len(report) == 3  # 3 industries
        for info in report.values():
            assert "count" in info
            assert "fraction" in info
            assert "symbols" in info


# ---------------------------------------------------------------------------
# End-to-end run() tests
# ---------------------------------------------------------------------------

class TestIndustryNeutralRankerRun:
    def test_full_pipeline_output(self, basic_panel):
        cfg = SelectionConfig(strategy="equal_top_k", top_k=3, max_total=9)
        ranker = IndustryNeutralRanker(cfg)
        result = ranker.run(basic_panel)

        expected_new_cols = [
            "industry_rank", "industry_neutral_score", "global_score",
            "industry_size", "selected", "selection_reason", "exposure_flag",
        ]
        for col in expected_new_cols:
            assert col in result.columns, f"Missing: {col}"

        assert result["selected"].sum() == 9
        assert result["exposure_flag"].notna().all()

    def test_no_industry_exceeds_top_k_in_equal_mode(self, basic_panel):
        """EqualTopK: no industry should have more than top_k selected."""
        cfg = SelectionConfig(strategy="equal_top_k", top_k=2, max_total=30)
        ranker = IndustryNeutralRanker(cfg)
        result = ranker.run(basic_panel)

        sel = result[result["selected"]]
        for ind, grp in sel.groupby("industry_code"):
            assert len(grp) <= cfg.top_k, f"{ind}: {len(grp)} > {cfg.top_k}"

    def test_all_returns_in_valid_range(self, basic_panel):
        cfg = SelectionConfig()
        ranker = IndustryNeutralRanker(cfg)
        result = ranker.run(basic_panel)

        # industry_rank should be positive integers
        assert (result["industry_rank"] >= 1).all()
        assert result["industry_rank"].dtype in (np.dtype("int64"), np.dtype("int32"))

        # global_score == model_score
        assert (result["global_score"] == result["model_score"]).all()

    def test_works_without_date_column(self):
        """Ranker should work on panels without a date column."""
        df = pd.DataFrame({
            "symbol": [f"S{i}" for i in range(20)],
            "industry_code": ["A"] * 10 + ["B"] * 10,
            "industry_name": ["IndA"] * 10 + ["IndB"] * 10,
            "model_score": np.random.randn(20),
        })
        cfg = SelectionConfig(top_k=3, max_total=20)
        ranker = IndustryNeutralRanker(cfg)
        result = ranker.run(df)
        assert "industry_rank" in result.columns
        assert result["selected"].sum() <= cfg.max_total


# ---------------------------------------------------------------------------
# Smoke test with real industry map (if available)
# ---------------------------------------------------------------------------

def test_smoke_with_industry_map():
    """Verify ranker does not crash with real industry data (if present)."""
    from pathlib import Path

    store_root = Path("E:/stock-analysis/models/data")
    imap_path = store_root / "silver" / "industry_map.parquet"

    if not imap_path.exists():
        pytest.skip("No industry_map.parquet — skipping smoke test")

    imap = pd.read_parquet(imap_path)
    # Get active industry assignments
    active = imap[imap["out_date"].isna()].head(100)
    if len(active) < 5:
        pytest.skip("Too few active industry records")

    panel = pd.DataFrame({
        "symbol": active["symbol"].tolist(),
        "date": pd.Timestamp("2026-06-18"),
        "model_score": np.random.randn(len(active)),
        "industry_code": active["industry_code"].fillna("_UNKNOWN").tolist(),
        "industry_name": active["industry_name"].fillna("未知").tolist(),
    })

    cfg = SelectionConfig(strategy="equal_top_k", top_k=3, max_total=30)
    ranker = IndustryNeutralRanker(cfg)
    result = ranker.run(panel)

    assert "selected" in result.columns
    assert result["selected"].sum() <= cfg.max_total
    assert "exposure_flag" in result.columns
    # All flags should be valid
    valid_flags = {"not_selected", "industry_overweight", "balanced", "diversified"}
    assert set(result["exposure_flag"].unique()) <= valid_flags
