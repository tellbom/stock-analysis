"""
tests/test_phase4b.py
=====================
Test suite for Phase 4B — Data Enrichment.

Covers:
  P4B-01  ValuationCollector + Tencent API parsing
  P4B-02  build_valuation_features — cross-sectional rank properties
  P4B-03  IndustryCollector — SCD logic + PIT query
  P4B-04  build_industry_features — within-industry rank properties
  P4B-05  build_excess_vs_industry_labels — arithmetic + PIT safety
  P4B-06  FundFlowCollector — parse/store
  P4B-07  build_flow_features — normalisation + cross-sectional rank
  P4B-08  MarginCollector + build_margin_features — 1-day lag correctness
  Store   new lake path helpers + init_lake directories
  Schemas new schema validators

All tests use synthetic data — no live Eastmoney / Tencent connections.
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
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_panel(n_symbols: int = 10, n_dates: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates).date.tolist()
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    rows = []
    for sym in symbols:
        close = 10 + np.cumsum(rng.normal(0, 0.2, n_dates))
        close = np.clip(close, 1, None)
        for i, d in enumerate(dates):
            rows.append({
                "symbol": sym, "date": d,
                "open": close[i] * 0.99, "high": close[i] * 1.01,
                "low": close[i] * 0.98,  "close": close[i],
                "volume": rng.integers(1_000, 100_000),
            })
    return pd.DataFrame(rows)


def _make_valuation_panel(n_symbols: int = 10, n_dates: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates).date.tolist()
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    rows = []
    for sym in symbols:
        for d in dates:
            rows.append({
                "symbol": sym, "date": d,
                "pe_ttm":        float(rng.uniform(10, 80)),
                "pb":            float(rng.uniform(0.5, 8)),
                "total_mcap_yi": float(rng.uniform(10, 5000)),
                "float_mcap_yi": float(rng.uniform(5, 4000)),
                "turnover_pct":  float(rng.uniform(0.1, 10)),
            })
    return pd.DataFrame(rows)


def _make_industry_map(n_symbols: int = 10, n_industries: int = 3) -> pd.DataFrame:
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    industries = [f"IND{j:02d}" for j in range(n_industries)]
    rows = []
    for i, sym in enumerate(symbols):
        rows.append({
            "symbol":         sym,
            "industry_code":  industries[i % n_industries],
            "industry_name":  f"Industry {i % n_industries}",
            "concept_tags":   f"tag_a|tag_b",
            "effective_date": dt.date(2020, 1, 1),
            "out_date":       None,
        })
    return pd.DataFrame(rows)


def _make_flow_panel(n_symbols: int = 10, n_dates: int = 120, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates).date.tolist()
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    rows = []
    for sym in symbols:
        for d in dates:
            rows.append({
                "symbol":    sym, "date": d,
                "main_net":  float(rng.normal(0, 5e7)),
                "small_net": float(rng.normal(0, 2e7)),
                "mid_net":   float(rng.normal(0, 3e7)),
                "large_net": float(rng.normal(0, 4e7)),
                "super_net": float(rng.normal(0, 6e7)),
            })
    return pd.DataFrame(rows)


def _make_margin_panel(n_symbols: int = 6, n_dates: int = 60, seed: int = 3) -> pd.DataFrame:
    """Only 6/10 symbols are margin-eligible."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates).date.tolist()
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    rows = []
    base = 1e8
    for sym in symbols:
        rzye = base + np.cumsum(rng.normal(0, 1e6, n_dates))
        for i, d in enumerate(dates):
            rows.append({
                "symbol": sym, "date": d,
                "rzye":   float(rzye[i]),
                "rzmre":  float(abs(rng.normal(5e5, 1e5))),
                "rzche":  float(abs(rng.normal(4e5, 1e5))),
                "rqye":   float(abs(rng.normal(2e6, 5e5))),
                "rqmcl":  0.0, "rqchl": 0.0,
                "rzrqye": float(rzye[i] + abs(rng.normal(2e6, 5e5))),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# P4B Store path helpers
# ---------------------------------------------------------------------------

class TestStoreLakePaths:
    """New lake paths must resolve correctly and init_lake must create dirs."""

    def test_valuation_path(self, tmp_path):
        from quant_platform.store.lake import valuation_path
        p = valuation_path(tmp_path, "600519")
        assert p.name == "600519.parquet"
        assert "valuation" in str(p) and "silver" in str(p)

    def test_industry_map_path(self, tmp_path):
        from quant_platform.store.lake import industry_map_path
        p = industry_map_path(tmp_path)
        assert p.name == "industry_map.parquet"
        assert "silver" in str(p)

    def test_fund_flow_path(self, tmp_path):
        from quant_platform.store.lake import fund_flow_path
        p = fund_flow_path(tmp_path, "000858")
        assert p.name == "000858.parquet"
        assert "fund_flow" in str(p)

    def test_margin_path(self, tmp_path):
        from quant_platform.store.lake import margin_path
        p = margin_path(tmp_path, "600036")
        assert p.name == "600036.parquet"
        assert "margin" in str(p)

    def test_init_lake_creates_all_p4b_dirs(self, tmp_path):
        from quant_platform.store.lake import init_lake
        init_lake(tmp_path)
        for sub in ("silver/valuation", "silver/fund_flow", "silver/margin"):
            assert (tmp_path / sub).exists(), f"Missing: {sub}"


# ---------------------------------------------------------------------------
# P4B Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_enforce_valuation_ok(self):
        from quant_platform.store.schemas import enforce_valuation
        df = pd.DataFrame([{
            "symbol": "600519", "date": "2023-01-10",
            "pe_ttm": 35.5, "pb": 9.2,
            "total_mcap_yi": 2800.0, "float_mcap_yi": 2500.0,
            "turnover_pct": 1.2,
        }])
        clean = enforce_valuation(df, "600519")
        assert clean["symbol"].iloc[0] == "600519"
        assert isinstance(clean["date"].iloc[0], dt.date)

    def test_enforce_valuation_missing_cols_raises(self):
        from quant_platform.store.schemas import enforce_valuation
        df = pd.DataFrame([{"symbol": "600519", "date": "2023-01-10", "pe_ttm": 10}])
        with pytest.raises(ValueError, match="missing"):
            enforce_valuation(df, "600519")

    def test_enforce_fund_flow_fills_missing_optional(self):
        from quant_platform.store.schemas import enforce_fund_flow
        df = pd.DataFrame([{"symbol": "000858", "date": "2023-01-10",
                             "main_net": 1e6, "small_net": -5e5}])
        clean = enforce_fund_flow(df, "000858")
        assert "super_net" in clean.columns
        assert clean["super_net"].iloc[0] == 0.0

    def test_enforce_margin_deduplicate(self):
        from quant_platform.store.schemas import enforce_margin
        df = pd.DataFrame([
            {"symbol": "SYM", "date": "2023-01-10", "rzye": 1e8, "rzmre": 2e6},
            {"symbol": "SYM", "date": "2023-01-10", "rzye": 1.1e8, "rzmre": 2.1e6},
        ])
        clean = enforce_margin(df, "SYM")
        assert len(clean) == 1  # deduplicated


# ---------------------------------------------------------------------------
# P4B-01: ValuationCollector — Tencent API parsing
# ---------------------------------------------------------------------------

class TestTencentParsing:
    """Test the Tencent field parsing in isolation without network calls."""

    def _fake_tencent_response(self) -> str:
        """
        Minimal fake GBK Tencent response for SH600519.
        Field layout (0-indexed): 1=name, 38=turnover, 39=pe_ttm,
        43=振幅%(NOT PB), 44=total_mcap, 45=float_mcap, 46=PB.
        """
        # Build ~88 fields; most are 0
        fields = [""] * 88
        fields[1]  = "贵州茅台"
        fields[3]  = "1700.00"
        fields[38] = "1.50"      # turnover_pct
        fields[39] = "32.5"      # PE_TTM
        fields[43] = "3.20"      # 振幅% (NOT PB — must NOT be read as PB)
        fields[44] = "21000.0"   # total_mcap_yi
        fields[45] = "21000.0"   # float_mcap_yi
        fields[46] = "11.3"      # PB — correct field

        vals_str = "~".join(fields)
        line = f'v_sh600519="{vals_str}"'
        return line + ";"

    def test_pb_comes_from_field_46_not_43(self):
        """PB must be parsed from field 46, NOT field 43."""
        from quant_platform.ingest.valuation_collector import _fetch_tencent_batch

        fake_response = self._fake_tencent_response().encode("gbk")
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = fake_response
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp

            result = _fetch_tencent_batch(["600519"])

        assert "600519" in result
        q = result["600519"]
        assert abs(q["pb"] - 11.3) < 0.01, f"PB expected 11.3 (field 46), got {q['pb']}"
        assert abs(q["pe_ttm"] - 32.5) < 0.01
        assert abs(q["total_mcap_yi"] - 21000.0) < 1.0
        assert abs(q["turnover_pct"] - 1.50) < 0.01
        # Verify 振幅% (field 43 = 3.20) was NOT used as PB
        assert abs(q["pb"] - 3.20) > 0.01, "BUG: field 43 (振幅%) was misread as PB"

    def test_write_and_read_roundtrip(self, tmp_path):
        """ValuationCollector._write_one should produce a readable Parquet."""
        from quant_platform.ingest.valuation_collector import ValuationCollector
        vc = ValuationCollector(store_root=tmp_path)
        quote = {
            "pe_ttm": 32.5, "pb": 11.3,
            "total_mcap_yi": 21000.0, "float_mcap_yi": 21000.0,
            "turnover_pct": 1.5,
        }
        vc._write_one("600519", dt.date(2023, 3, 10), quote, overwrite=False)

        from quant_platform.ingest.valuation_collector import load_valuation
        df = load_valuation(tmp_path, "600519")
        assert len(df) == 1
        assert abs(df["pb"].iloc[0] - 11.3) < 0.01

    def test_incremental_no_duplicate(self, tmp_path):
        """Running _write_one twice for the same date should not duplicate."""
        from quant_platform.ingest.valuation_collector import ValuationCollector, load_valuation
        vc = ValuationCollector(store_root=tmp_path)
        q = {"pe_ttm": 30.0, "pb": 10.0, "total_mcap_yi": 500.0,
             "float_mcap_yi": 400.0, "turnover_pct": 1.0}
        vc._write_one("000858", dt.date(2023, 4, 1), q, overwrite=False)
        vc._write_one("000858", dt.date(2023, 4, 1), q, overwrite=False)  # second write
        df = load_valuation(tmp_path, "000858")
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"

    def test_stock_value_em_mapping_uses_column_names(self):
        from quant_platform.ingest.valuation_collector import normalise_stock_value_em

        raw = pd.DataFrame({
            "PB": [1.2, 1.3],
            "总市值": [100_000_000_000.0, 101_000_000_000.0],
            "数据日期": ["2023-01-03", "2023-01-04"],
            "流通市值": [80_000_000_000.0, 81_000_000_000.0],
            "PE(TTM)": [10.0, 10.5],
        })
        turnover = pd.DataFrame({
            "date": [dt.date(2023, 1, 3), dt.date(2023, 1, 4)],
            "turnover_pct": [0.5, 0.6],
        })

        df = normalise_stock_value_em(
            raw, "600519", "2023-01-03", "2023-01-04", turnover
        )

        assert list(df.columns) == [
            "symbol", "date", "pe_ttm", "pb",
            "total_mcap_yi", "float_mcap_yi", "turnover_pct",
        ]
        assert df["total_mcap_yi"].iloc[0] == pytest.approx(1000.0)
        assert df["float_mcap_yi"].iloc[0] == pytest.approx(800.0)
        assert df["turnover_pct"].notna().all()


# ---------------------------------------------------------------------------
# P4B-02: build_valuation_features
# ---------------------------------------------------------------------------

class TestValuationFeatures:
    """Cross-sectional properties of valuation features."""

    @pytest.fixture
    def panels(self):
        panel = _make_ohlcv_panel()
        val = _make_valuation_panel()
        return panel, val

    def test_cs_ranks_in_01(self, panels):
        from quant_platform.features.valuation import build_valuation_features
        panel, val = panels
        result = build_valuation_features(panel, val)
        for col in ("cs_pe_ttm_rank", "cs_pb_rank", "cs_turnover_rank", "cs_log_float_mcap"):
            valid = result[col].dropna()
            assert (valid >= 0).all() and (valid <= 1).all(), f"{col} out of [0,1]"

    def test_pe_negative_ranks_last(self, panels):
        """Stocks with negative PE should rank at 0 (bottom)."""
        from quant_platform.features.valuation import build_valuation_features
        panel, val = panels
        # Force one stock to have negative PE on all dates
        val.loc[val["symbol"] == "SYM000", "pe_ttm"] = -5.0
        result = build_valuation_features(panel, val)
        sym000 = result[result["symbol"] == "SYM000"]["cs_pe_ttm_rank"].dropna()
        assert (sym000 == 0.0).all() or sym000.empty, (
            "Negative PE should give rank 0"
        )

    def test_size_rank_monotone_with_mcap(self, panels):
        """
        Rank of log(float_mcap) should be positively correlated with the
        raw float_mcap within a date.
        """
        from quant_platform.features.valuation import build_valuation_features
        panel, val = panels
        result = build_valuation_features(panel.copy(), val)
        merged = result.merge(val[["symbol", "date", "float_mcap_yi"]], on=["symbol", "date"])
        corr = merged["cs_log_float_mcap"].corr(np.log(merged["float_mcap_yi"].clip(1e-6)))
        assert corr > 0.9, f"Size rank should be strongly correlated with mcap, got {corr:.3f}"

    def test_empty_valuation_returns_nan_columns(self, panels):
        from quant_platform.features.valuation import build_valuation_features, VALUATION_SPECS
        panel, _ = panels
        result = build_valuation_features(panel, pd.DataFrame())
        for spec in VALUATION_SPECS:
            assert spec.name in result.columns
            assert result[spec.name].isna().all()


# ---------------------------------------------------------------------------
# P4B-03: IndustryCollector SCD logic + PIT query
# ---------------------------------------------------------------------------

class TestIndustrySCD:
    def test_slist_industry_filters_region_and_concepts(self, monkeypatch):
        from quant_platform.ingest import industry_collector as ic

        class FakeResponse:
            def json(self):
                return {
                    "data": {
                        "diff": [
                            {"f12": "BK0475", "f14": "食品饮料"},
                            {"f12": "BK0477", "f14": "白酒Ⅱ"},
                            {"f12": "BK9999", "f14": "贵州板块"},
                            {"f12": "BK0001", "f14": "酿酒概念"},
                            {"f12": "BK0002", "f14": "HS300_"},
                        ]
                    }
                }

        monkeypatch.setattr(ic, "_em_get", lambda *args, **kwargs: FakeResponse())

        info = ic._fetch_em_slist_industry("600519")
        assert info["industry_name"] == "食品饮料"
        assert info["industry_code"] == "BK0475"
        assert "贵州板块" in info["concept_tags"]

    def test_run_fails_when_industry_coverage_is_too_low(self, tmp_path, monkeypatch):
        from quant_platform.ingest import industry_collector as ic

        def fake_slist(symbol):
            if symbol == "000001":
                return {
                    "industry_name": "银行Ⅱ",
                    "industry_code": "BK0470",
                    "concept_tags": "银行Ⅱ|广东板块",
                }
            return {}

        monkeypatch.setattr(ic, "_fetch_em_slist_industry", fake_slist)
        monkeypatch.setattr(ic, "_fetch_em_stock_info", lambda symbol: {})

        collector = ic.IndustryCollector(tmp_path, fetch_concepts=False)
        with pytest.raises(RuntimeError, match="Industry coverage too low"):
            collector.run(["000001", "000002", "000063"], as_of=dt.date(2026, 6, 25))

    def test_enforce_industry_map_keeps_active_same_day_correction(self):
        from quant_platform.store.schemas import enforce_industry_map

        imap = pd.DataFrame([
            {
                "symbol": "000858",
                "industry_code": "四川板块",
                "industry_name": "白酒Ⅱ",
                "concept_tags": "",
                "effective_date": dt.date(2026, 6, 25),
                "out_date": dt.date(2026, 6, 25),
            },
            {
                "symbol": "000858",
                "industry_code": "BK0438",
                "industry_name": "食品饮料",
                "concept_tags": "食品饮料|白酒Ⅱ|四川板块",
                "effective_date": dt.date(2026, 6, 25),
                "out_date": None,
            },
        ])

        fixed = enforce_industry_map(imap)
        assert len(fixed) == 1
        assert fixed.iloc[0]["industry_code"] == "BK0438"
        assert pd.isna(fixed.iloc[0]["out_date"])

    def test_run_repairs_same_day_active_bad_industry(self, tmp_path, monkeypatch):
        from quant_platform.ingest import industry_collector as ic
        from quant_platform.store.lake import industry_map_path

        existing = pd.DataFrame([{
            "symbol": "000858",
            "industry_code": "四川板块",
            "industry_name": "白酒Ⅱ",
            "concept_tags": "",
            "effective_date": dt.date(2026, 6, 25),
            "out_date": None,
        }])
        p = industry_map_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existing.to_parquet(p, index=False)

        monkeypatch.setattr(
            ic,
            "_fetch_em_slist_industry",
            lambda symbol: {
                "industry_name": "食品饮料",
                "industry_code": "BK0438",
                "concept_tags": "食品饮料|白酒Ⅱ|四川板块",
            },
        )
        monkeypatch.setattr(ic, "_fetch_em_stock_info", lambda symbol: {})

        collector = ic.IndustryCollector(tmp_path, fetch_concepts=False)
        fixed = collector.run(["000858"], as_of=dt.date(2026, 6, 25))

        assert len(fixed) == 1
        assert fixed.iloc[0]["industry_code"] == "BK0438"
        assert fixed.iloc[0]["industry_name"] == "食品饮料"
        assert pd.isna(fixed.iloc[0]["out_date"])

    def test_get_industry_as_of_current(self):
        from quant_platform.ingest.industry_collector import get_industry_as_of
        imap = _make_industry_map(n_symbols=5, n_industries=2)
        info = get_industry_as_of(imap, "SYM000", dt.date(2023, 6, 1))
        assert info["industry_code"] in ("IND00", "IND01")

    def test_get_industry_as_of_before_effective_date(self):
        """Before the SCD record's effective_date, should return empty."""
        from quant_platform.ingest.industry_collector import get_industry_as_of
        imap = pd.DataFrame([{
            "symbol": "SYM000", "industry_code": "IND01",
            "industry_name": "Test", "concept_tags": "",
            "effective_date": dt.date(2022, 6, 1),
            "out_date": None,
        }])
        info = get_industry_as_of(imap, "SYM000", dt.date(2021, 1, 1))
        assert info["industry_code"] == ""

    def test_scd_two_records_respects_out_date(self):
        """After a change, the new record is returned; before change, the old one."""
        from quant_platform.ingest.industry_collector import get_industry_as_of
        imap = pd.DataFrame([
            {
                "symbol": "SYM000", "industry_code": "IND_OLD",
                "industry_name": "Old Industry", "concept_tags": "",
                "effective_date": dt.date(2020, 1, 1),
                "out_date": dt.date(2022, 1, 1),
            },
            {
                "symbol": "SYM000", "industry_code": "IND_NEW",
                "industry_name": "New Industry", "concept_tags": "",
                "effective_date": dt.date(2022, 1, 1),
                "out_date": None,
            },
        ])
        before_change = get_industry_as_of(imap, "SYM000", dt.date(2021, 6, 1))
        after_change  = get_industry_as_of(imap, "SYM000", dt.date(2023, 1, 1))
        assert before_change["industry_code"] == "IND_OLD"
        assert after_change["industry_code"]  == "IND_NEW"

    def test_cninfo_events_to_scd_generates_out_dates_and_source(self):
        from quant_platform.ingest.industry_collector import (
            industry_events_to_scd,
            get_industry_as_of,
            normalise_cninfo_industry_events,
        )

        raw = pd.DataFrame({
            "行业名称": ["Old", "New"],
            "行业编码": ["IND_OLD", "IND_NEW"],
            "行业级别": ["主要行业", "主要行业"],
            "行业分类标准": ["申万", "申万"],
            "变更日期": ["2020-01-01", "2022-01-01"],
        })
        events = normalise_cninfo_industry_events(raw, "600519")
        scd = industry_events_to_scd(events, "600519")

        assert scd.iloc[0]["out_date"] == dt.date(2022, 1, 1)
        assert pd.isna(scd.iloc[1]["out_date"])
        assert set(scd["source"]) == {"cninfo_stock_industry_change"}
        assert get_industry_as_of(scd, "600519", dt.date(2021, 6, 1))["industry_code"] == "IND_OLD"
        assert get_industry_as_of(scd, "600519", dt.date(2023, 6, 1))["industry_code"] == "IND_NEW"

    def test_load_save_roundtrip(self, tmp_path):
        from quant_platform.store.lake import init_lake, industry_map_path
        from quant_platform.store.schemas import enforce_industry_map
        init_lake(tmp_path)
        imap = _make_industry_map()
        imap = enforce_industry_map(imap)
        p = industry_map_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        imap.to_parquet(p, index=False)

        from quant_platform.ingest.industry_collector import load_industry_map
        loaded = load_industry_map(tmp_path)
        assert len(loaded) == len(imap)


# ---------------------------------------------------------------------------
# P4B-04: build_industry_features
# ---------------------------------------------------------------------------

class TestIndustryFeatures:
    @pytest.fixture
    def data(self):
        panel = _make_ohlcv_panel(n_symbols=9)
        imap  = _make_industry_map(n_symbols=9, n_industries=3)
        return panel, imap

    def test_ind_rank_in_01(self, data):
        from quant_platform.features.valuation import build_valuation_features, load_valuation_panel
        from quant_platform.features.industry import build_industry_features
        panel, imap = data
        val = _make_valuation_panel(n_symbols=9)
        panel = build_valuation_features(panel, val)
        result = build_industry_features(panel, imap)
        for col in ("ind_rank_rsi_6", "ind_rank_turnover"):
            valid = result[col].dropna()
            if len(valid) > 0:
                assert (valid >= 0).all() and (valid <= 1).all(), f"{col} out of [0,1]"

    def test_industry_code_added(self, data):
        from quant_platform.features.industry import build_industry_features
        panel, imap = data
        result = build_industry_features(panel, imap)
        assert "industry_code" in result.columns
        assert result["industry_code"].notna().any()

    def test_empty_imap_gives_nan(self, data):
        from quant_platform.features.industry import build_industry_features, INDUSTRY_SPECS
        panel, _ = data
        result = build_industry_features(panel, pd.DataFrame())
        for spec in INDUSTRY_SPECS:
            assert spec.name in result.columns


# ---------------------------------------------------------------------------
# P4B-05: excess_vs_industry label
# ---------------------------------------------------------------------------

class TestExcessVsIndustryLabel:
    def test_industry_excess_mean_approx_zero_per_industry(self):
        """
        Within each (date, industry) group, excess_vs_industry should sum to ~0
        because it equals stock_ret − industry_mean.
        """
        from quant_platform.features.industry import build_excess_vs_industry_labels

        n, n_d = 12, 80
        panel = _make_ohlcv_panel(n_symbols=n, n_dates=n_d)
        imap  = _make_industry_map(n_symbols=n, n_industries=3)

        # Build forward return labels manually
        panel["date"] = pd.to_datetime(panel["date"]).dt.date
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        panel["ret_fwd_5d"] = panel.groupby("symbol")["close"].transform(
            lambda x: x.shift(-6) / x.shift(-1) - 1
        )

        result = build_excess_vs_industry_labels(panel, imap, horizons=[5])
        assert "excess_vs_industry_5d" in result.columns

        grp = result.groupby(["date", "industry_code"])["excess_vs_industry_5d"]
        sums = grp.sum()
        # Each group sum should be near 0 (sum of deviations from mean)
        assert (sums.abs() < 1e-9).all(), f"Non-zero group sums: {sums[sums.abs() > 1e-9]}"

    def test_unknown_industry_gives_nan(self):
        """Stocks with _UNKNOWN industry should have NaN excess label."""
        from quant_platform.features.industry import build_excess_vs_industry_labels

        n, n_d = 6, 40
        panel = _make_ohlcv_panel(n_symbols=n, n_dates=n_d)
        # Partial industry map — leaves SYM005 unknown
        imap = _make_industry_map(n_symbols=5, n_industries=2)

        panel["date"] = pd.to_datetime(panel["date"]).dt.date
        panel["ret_fwd_5d"] = 0.02   # constant label
        result = build_excess_vs_industry_labels(panel, imap, horizons=[5])

        sym5 = result[result["symbol"] == "SYM005"]["excess_vs_industry_5d"]
        assert sym5.isna().all(), "SYM005 with _UNKNOWN industry should have NaN excess"


# ---------------------------------------------------------------------------
# P4B-06 / P4B-07: FundFlowCollector + build_flow_features
# ---------------------------------------------------------------------------

class TestFundFlow:
    def test_fetch_push2his_parse(self):
        """_fetch_push2his correctly parses a fake API response."""
        from quant_platform.ingest.flow_collector import _fetch_push2his

        fake_klines = [
            "2023-01-10,5000000,-1000000,2000000,3000000,1500000",
            "2023-01-11,-2000000,800000,1000000,-3000000,0",
            "BADINPUT",     # should be skipped
        ]
        fake_response = {"data": {"klines": fake_klines}}

        with patch("quant_platform.ingest.flow_collector._em_get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = fake_response
            mock_get.return_value = mock_resp
            df = _fetch_push2his("600519")

        assert len(df) == 2
        assert df.iloc[0]["main_net"] == pytest.approx(5_000_000)
        assert df.iloc[1]["main_net"] == pytest.approx(-2_000_000)
        assert df.iloc[1]["super_net"] == pytest.approx(0.0)

    def test_flow_features_rank_in_01(self):
        from quant_platform.features.flow import build_flow_features
        panel = _make_ohlcv_panel(n_symbols=10)
        flow  = _make_flow_panel(n_symbols=10)
        val   = _make_valuation_panel(n_symbols=10)
        result = build_flow_features(panel, flow, val)
        for col in ("cs_main_flow_rank_1d", "cs_main_flow_rank_5d",
                    "cs_small_flow_rank_1d", "cs_super_flow_rank_1d"):
            valid = result[col].dropna()
            if len(valid) > 0:
                assert (valid >= 0).all() and (valid <= 1).all(), f"{col} out of [0,1]"

    def test_flow_reversal_range(self):
        """cs_flow_reversal_5d = 1d_rank − 5d_rank, so range is (−1, 1)."""
        from quant_platform.features.flow import build_flow_features
        panel = _make_ohlcv_panel(n_symbols=10)
        flow  = _make_flow_panel(n_symbols=10)
        val   = _make_valuation_panel(n_symbols=10)
        result = build_flow_features(panel, flow, val)
        rev = result["cs_flow_reversal_5d"].dropna()
        if len(rev) > 0:
            assert rev.abs().max() <= 1.001, f"Reversal out of (-1,1): {rev.abs().max()}"

    def test_empty_flow_returns_nan(self):
        from quant_platform.features.flow import build_flow_features, FLOW_SPECS
        panel = _make_ohlcv_panel(n_symbols=5)
        result = build_flow_features(panel, pd.DataFrame())
        for spec in FLOW_SPECS:
            assert spec.name in result.columns
            assert result[spec.name].isna().all()

    def test_flow_normalised_by_mcap(self):
        """
        Normalisation means a stock with 10× larger float_mcap but same absolute
        flow should rank LOWER (lower intensity).
        """
        from quant_platform.features.flow import build_flow_features

        # Two symbols: SYM000 tiny (10 亿), SYM001 large (1000 亿)
        # Same main_net for both. SYM000 should rank higher.
        dates = [dt.date(2023, 1, i + 3) for i in range(5)]
        panel = pd.DataFrame([
            {"symbol": sym, "date": d, "close": 10.0, "open": 10.0,
             "high": 10.1, "low": 9.9, "volume": 1000}
            for sym in ["SYM000", "SYM001"] for d in dates
        ])
        flow = pd.DataFrame([
            {"symbol": sym, "date": d, "main_net": 1e7,
             "small_net": 0.0, "mid_net": 0.0, "large_net": 0.0, "super_net": 0.0}
            for sym in ["SYM000", "SYM001"] for d in dates
        ])
        val = pd.DataFrame([
            {"symbol": "SYM000", "date": d, "pe_ttm": 20, "pb": 2,
             "total_mcap_yi": 10.0, "float_mcap_yi": 10.0, "turnover_pct": 1.5}
            for d in dates
        ] + [
            {"symbol": "SYM001", "date": d, "pe_ttm": 20, "pb": 2,
             "total_mcap_yi": 1000.0, "float_mcap_yi": 1000.0, "turnover_pct": 1.5}
            for d in dates
        ])
        result = build_flow_features(panel, flow, val)
        sym0 = result[result["symbol"] == "SYM000"]["cs_main_flow_rank_1d"].dropna()
        sym1 = result[result["symbol"] == "SYM001"]["cs_main_flow_rank_1d"].dropna()
        if len(sym0) > 0 and len(sym1) > 0:
            assert sym0.mean() > sym1.mean(), (
                "Tiny-cap stock with same absolute flow should have higher intensity rank"
            )


# ---------------------------------------------------------------------------
# P4B-08: build_margin_features — 1-day lag
# ---------------------------------------------------------------------------

class TestMarginFeatures:
    def test_margin_1d_lag(self):
        """
        Margin features at date T should use margin data from T-1 (lag=1).
        If margin data only starts on 2023-01-04, features should be NaN
        on 2023-01-03.
        """
        from quant_platform.features.margin import build_margin_features

        dates = [dt.date(2023, 1, i + 3) for i in range(10)]
        symbols = ["SYM000", "SYM001"]
        panel = pd.DataFrame([
            {"symbol": sym, "date": d, "close": 10.0}
            for sym in symbols for d in dates
        ])
        # Margin starts from dates[1] (second date)
        margin = pd.DataFrame([
            {"symbol": sym, "date": dates[i], "rzye": 1e8 + i * 1e6,
             "rzmre": 1e6, "rzche": 5e5, "rqye": 2e6,
             "rqmcl": 0.0, "rqchl": 0.0, "rzrqye": 1.02e8}
            for sym in symbols for i in range(1, len(dates))  # skip first date
        ])
        result = build_margin_features(panel, margin)

        # First date (2023-01-03) has no T-1 margin data → NaN
        first_day = result[result["date"] == dates[0]]["cs_rzrq_ratio_rank"]
        assert first_day.isna().all(), (
            "Margin features on first date should be NaN (no T-1 data)"
        )

    def test_margin_ranks_in_01(self):
        from quant_platform.features.margin import build_margin_features
        panel = _make_ohlcv_panel(n_symbols=6)
        margin = _make_margin_panel(n_symbols=6)
        val = _make_valuation_panel(n_symbols=6)
        result = build_margin_features(panel, margin, val)
        for col in ("cs_margin_balance_change_5d", "cs_rzrq_ratio_rank"):
            valid = result[col].dropna()
            if len(valid) > 0:
                assert (valid >= 0).all() and (valid <= 1).all(), f"{col} out of [0,1]"

    def test_empty_margin_returns_nan(self):
        from quant_platform.features.margin import build_margin_features, MARGIN_SPECS
        panel = _make_ohlcv_panel(n_symbols=5)
        result = build_margin_features(panel, pd.DataFrame())
        for spec in MARGIN_SPECS:
            assert spec.name in result.columns


# ---------------------------------------------------------------------------
# Feature registry: P4B spec lists registered
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_full_specs_includes_p4b(self):
        from quant_platform.features.registry import FULL_SPECS
        families = {s.family for s in FULL_SPECS}
        assert "valuation" in families, "FULL_SPECS missing valuation family"
        assert "flow" in families, "FULL_SPECS missing flow family"
        assert "industry" in families, "FULL_SPECS missing industry family"
        assert "margin" in families, "FULL_SPECS missing margin family"

    def test_full_specs_unique_names(self):
        from quant_platform.features.registry import FULL_SPECS
        names = [s.name for s in FULL_SPECS]
        assert len(names) == len(set(names)), "Duplicate feature names in FULL_SPECS"

    def test_valuation_specs_declared(self):
        from quant_platform.features.valuation import VALUATION_SPECS
        assert len(VALUATION_SPECS) >= 5
        names = [s.name for s in VALUATION_SPECS]
        assert "cs_pe_ttm_rank" in names
        assert "cs_log_float_mcap" in names

    def test_flow_specs_declared(self):
        from quant_platform.features.flow import FLOW_SPECS
        names = [s.name for s in FLOW_SPECS]
        assert "cs_main_flow_rank_1d" in names
        assert "cs_flow_reversal_5d" in names

    def test_industry_specs_declared(self):
        from quant_platform.features.industry import INDUSTRY_SPECS
        names = [s.name for s in INDUSTRY_SPECS]
        assert "ind_rank_rsi_6" in names

    def test_margin_specs_declared(self):
        from quant_platform.features.margin import MARGIN_SPECS
        names = [s.name for s in MARGIN_SPECS]
        assert "cs_rzrq_ratio_rank" in names
