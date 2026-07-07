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


def test_emdatah5_provider_canonical_schema():
    from quant_platform.ingest.fund_flow_providers import (
        CANONICAL_FUND_FLOW_COLUMNS,
        EMDATAH5_SOURCE,
        EastmoneyH5FundFlowProvider,
    )

    class FakeResponse:
        status = 200
        data = (
            b'{"data":{"klines":["2026-07-03,1,-1,2,3,4,0.1,-0.1,0.2,0.3,0.4,10.5,1.2"]}}'
        )

    with patch(
        "quant_platform.ingest.fund_flow_providers._EMDATAH5_HTTP.request",
        return_value=FakeResponse(),
    ):
        result = EastmoneyH5FundFlowProvider().fetch_symbol("600000.SH")

    assert result.ok
    assert list(result.frame.columns) == CANONICAL_FUND_FLOW_COLUMNS
    row = result.frame.iloc[0]
    assert row["symbol"] == "600000"
    assert row["source"] == EMDATAH5_SOURCE
    assert row["medium_net"] == 2.0
    assert row["mid_net"] == 2.0


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


def test_qstock_provider_does_not_route_ths_money_into_canonical_fields(monkeypatch):
    """
    Review finding #2: ths_money must NEVER be one of fetch_symbol()'s
    canonical-mapping candidates, even if it's the only recognised
    function qstock exposes. fetch_symbol() must fail loudly rather than
    silently canonicalise THS-methodology data.
    """
    from quant_platform.ingest.fund_flow_providers import QStockFundFlowProvider

    mod = types.SimpleNamespace()

    def ths_money(code):
        return pd.DataFrame({
            "日期": ["2026-07-03"],
            "主力净流入": [999.0],  # would look like a plausible canonical value
        })

    mod.ths_money = ths_money
    monkeypatch.setitem(sys.modules, "qstock", mod)

    assert "ths_money" not in QStockFundFlowProvider.CANONICAL_CANDIDATES
    with pytest.raises(RuntimeError, match="no recognised fund-flow function"):
        QStockFundFlowProvider().fetch_symbol("000001.SZ")


def test_qstock_ths_proxy_uses_proxy_fields_only(monkeypatch):
    """
    Review finding #2: fetch_ths_proxy() must return ths_*-prefixed proxy
    fields only, never CANONICAL_FUND_FLOW_COLUMNS names.
    """
    from quant_platform.ingest.fund_flow_providers import (
        CANONICAL_FUND_FLOW_COLUMNS,
        THS_PROXY_COLUMNS,
        QStockFundFlowProvider,
    )

    mod = types.SimpleNamespace()

    def ths_money(code):
        return pd.DataFrame({
            "日期": ["2026-07-03"],
            "流入资金": [12.0],
            "流出资金": [5.0],
            "净流入": [7.0],
        })

    mod.ths_money = ths_money
    monkeypatch.setitem(sys.modules, "qstock", mod)

    out = QStockFundFlowProvider().fetch_ths_proxy("000001.SZ")
    assert list(out.columns) == THS_PROXY_COLUMNS
    assert "main_net" not in out.columns
    assert not (set(out.columns) & set(CANONICAL_FUND_FLOW_COLUMNS) - {"symbol", "trade_date", "source", "raw_update_time", "fetched_at"})
    assert out.iloc[0]["ths_in_amount"] == 12.0
    assert out.iloc[0]["ths_out_amount"] == 5.0
    assert out.iloc[0]["ths_net_amount"] == 7.0
    assert out.iloc[0]["source"] == "qstock_ths_money_proxy"


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


def test_default_provider_prefers_emdatah5_only():
    from quant_platform.ingest.fund_flow_providers import (
        EastmoneyH5FundFlowProvider,
        default_fund_flow_providers,
    )

    providers = default_fund_flow_providers()
    assert [type(p) for p in providers] == [EastmoneyH5FundFlowProvider]


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
