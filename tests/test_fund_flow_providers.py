from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def test_native_provider_canonical_schema():
    from quant_platform.ingest.fund_flow_providers import (
        CANONICAL_FUND_FLOW_COLUMNS,
        NativeEastmoneyFundFlowProvider,
    )

    fake = pd.DataFrame({
        "date": ["2026-07-03"],
        "main_net": [1.0],
        "small_net": [-1.0],
        "mid_net": [2.0],
        "large_net": [3.0],
        "super_net": [4.0],
    })
    with patch(
        "quant_platform.ingest.fund_flow_providers._fetch_push2his_result",
        return_value=(fake, None, 0),
    ):
        result = NativeEastmoneyFundFlowProvider().fetch_symbol("600000.SH")

    assert result.ok
    assert list(result.frame.columns) == CANONICAL_FUND_FLOW_COLUMNS
    assert result.frame.iloc[0]["symbol"] == "600000"
    assert "main_net_rate" in result.missing_fields


def test_adata_provider_missing_dependency():
    from quant_platform.ingest.fund_flow_providers import ADataFundFlowProvider

    with patch.dict(sys.modules, {"adata": None}):
        with pytest.raises(ImportError, match="adata is not installed"):
            ADataFundFlowProvider().fetch_symbol("000001")


def test_qstock_provider_smoke_with_mock_module(monkeypatch):
    from quant_platform.ingest.fund_flow_providers import QStockFundFlowProvider

    mod = types.SimpleNamespace()

    def fund_flow(code):
        return pd.DataFrame({
            "日期": ["2026-07-03"],
            "主力净流入": [10.0],
            "小单净流入": [-3.0],
        })

    mod.fund_flow = fund_flow
    monkeypatch.setitem(sys.modules, "qstock", mod)
    result = QStockFundFlowProvider().fetch_symbol("000001.SZ")
    assert result.ok
    assert result.frame.iloc[0]["main_net"] == 10.0
    assert result.frame.iloc[0]["small_net"] == -3.0


def test_collector_routes_to_second_provider_and_reports_first_failure(tmp_path, monkeypatch):
    from quant_platform.ingest.flow_collector import FundFlowCollector

    class BadProvider:
        name = "adata"

        def fetch_symbol(self, symbol, days=120):
            raise RuntimeError("boom")

    class GoodProvider:
        name = "qstock"

        def fetch_symbol(self, symbol, days=120):
            from quant_platform.ingest.fund_flow_providers import ProviderResult

            return ProviderResult(pd.DataFrame({
                "symbol": ["000001"],
                "trade_date": [pd.Timestamp("2026-07-03").date()],
                "main_net": [1.0],
                "main_net_rate": [pd.NA],
                "super_net": [pd.NA],
                "super_net_rate": [pd.NA],
                "large_net": [pd.NA],
                "large_net_rate": [pd.NA],
                "medium_net": [pd.NA],
                "medium_net_rate": [pd.NA],
                "small_net": [-1.0],
                "small_net_rate": [pd.NA],
                "source": ["qstock"],
                "raw_update_time": [None],
                "fetched_at": ["2026-07-06T00:00:00+00:00"],
            }), missing_fields=["main_net_rate"])

    monkeypatch.setattr(
        "quant_platform.ingest.fund_flow_providers.default_fund_flow_providers",
        lambda: [BadProvider(), GoodProvider()],
    )
    collector = FundFlowCollector(tmp_path)
    result = collector.run(["000001"])
    assert result["000001"] == 1
    stored = pd.read_parquet(tmp_path / "silver/fund_flow/000001.parquet")
    assert stored.iloc[0]["source"] == "qstock"
    failed = pd.read_csv(tmp_path / "reports/fund_flow_failed_symbols.csv")
    assert failed.iloc[0]["provider"] == "adata"
    assert failed.iloc[0]["latest_success_provider"] == "qstock"


def test_sector_flow_feature_builder_keeps_proxy_names():
    from quant_platform.features.sector_flow import build_sector_flow_features

    panel = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "date": [pd.Timestamp("2026-07-03").date()] * 2,
        "industry_name": ["银行", "银行"],
    })
    sf = pd.DataFrame({
        "name": ["银行"],
        "trade_date": [pd.Timestamp("2026-07-03").date()],
        "sector_main_net": [100.0],
        "sector_main_net_rate": [1.2],
    })
    out = build_sector_flow_features(panel, sf)
    assert "sector_main_flow_rank" in out.columns
    assert "stock_industry_flow_strength" in out.columns
    assert "main_net" not in out.columns
