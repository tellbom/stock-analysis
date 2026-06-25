"""
features.pipeline
=================
Feature pipeline scaffold (T1.1).

Reads gold OHLCV from the lake, applies technical → cross-sectional →
fundamental → valuation → industry → flow → margin builders per symbol,
and writes per-symbol feature Parquet to
``features/<feature_set_id>/<symbol>.parquet``.

The pipeline is:
  - Idempotent: re-running with the same feature_set_id and date range
    overwrites existing files.
  - Incremental-friendly: callers can pass only symbols that need updating.
  - Leakage-safe: only data available at each date T is used for features
    at T (warm-up masks, PIT fundamental join, no future OHLCV).

P4B additions
-------------
New optional parameters on FeaturePipeline.__init__:
  include_valuation  : bool  join PE/PB/size/turnover (requires ValuationCollector)
  include_industry   : bool  join within-industry ranks (requires IndustryCollector)
  include_flow       : bool  join capital flow ranks (requires FundFlowCollector)
  include_margin     : bool  join margin features (requires MarginCollector)

These default to False so the pipeline is backward-compatible with P0–P3
deployments that have not run the P4B collectors.

Panel-level builders (valuation, industry, flow, margin) are run in
``build_panel()`` after all per-symbol Parquets are loaded — they operate
on the full universe panel, not per-symbol.

Usage
-----
    from quant_platform.features.pipeline import FeaturePipeline
    from quant_platform.features.registry import DEFAULT_SPECS

    pipeline = FeaturePipeline(
        store_root="/data/lake",
        include_valuation=True,
        include_industry=True,
        include_flow=True,
    )
    feature_set_id = pipeline.run(
        symbols=["600519", "000858"],
        specs=DEFAULT_SPECS,
    )
    panel = pipeline.build_panel(symbols, feature_set_id)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.features.registry import (
    FeatureSpec, FeatureRegistry, compute_feature_set_id, DEFAULT_SPECS,
)
from quant_platform.features.technical import build_technical_features
from quant_platform.features.cross_sectional import build_cross_sectional_features
from quant_platform.features.fundamental import build_fundamental_features
from quant_platform.store.lake import (
    ohlcv_path, feature_path, feature_set_dir, init_lake,
)
from quant_platform.store.parquet_store import read_ohlcv

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Orchestrates the full feature engineering pipeline.

    Parameters
    ----------
    store_root : Path | str
        Root of the Parquet data lake.
    project_root : Path | str | None
        Directory containing ``technical_indicators.py``.
        Defaults to the current working directory.
    include_fundamentals : bool
        Whether to run the PIT fundamental builder.  Default False.
    include_valuation : bool
        Whether to join PE/PB/size/turnover valuation features (P4B-02).
        Requires ValuationCollector to have run.  Default False.
    include_industry : bool
        Whether to join within-industry rank features (P4B-04).
        Requires IndustryCollector to have run.  Default False.
    include_flow : bool
        Whether to join capital flow rank features (P4B-07).
        Requires FundFlowCollector to have run.  Default False.
    include_margin : bool
        Whether to join margin trading features (P4B-08).
        Requires MarginCollector to have run.  Default False.
    """

    def __init__(
        self,
        store_root: Path | str,
        project_root: Path | str | None = None,
        include_fundamentals: bool = False,
        include_valuation: bool = False,
        include_industry: bool = False,
        include_flow: bool = False,
        include_margin: bool = False,
    ) -> None:
        self.store_root          = Path(store_root)
        self.project_root        = Path(project_root or Path.cwd())
        self.include_fundamentals = include_fundamentals
        self.include_valuation   = include_valuation
        self.include_industry    = include_industry
        self.include_flow        = include_flow
        self.include_margin      = include_margin
        self.registry            = FeatureRegistry(store_root)
        init_lake(self.store_root)

    def run(
        self,
        symbols: list[str],
        specs: list[FeatureSpec] | None = None,
        start_date: dt.date | str | None = None,
        end_date:   dt.date | str | None = None,
    ) -> str:
        """
        Run the feature pipeline for all symbols.

        Parameters
        ----------
        symbols : list[str]
            6-digit A-share codes to process.
        specs : list[FeatureSpec] | None
            Feature definitions.  Defaults to DEFAULT_SPECS.
        start_date, end_date : date | str | None
            Optional date filter applied to OHLCV before feature computation.

        Returns
        -------
        str
            The ``feature_set_id`` (8-char hex) for this spec list.
        """
        specs = specs or DEFAULT_SPECS
        feature_set_id = self.registry.register(specs)

        logger.info(
            "FeaturePipeline.run: %d symbols, feature_set_id=%s, "
            "fundamentals=%s",
            len(symbols), feature_set_id, self.include_fundamentals,
        )

        succeeded, failed = 0, 0
        for symbol in symbols:
            try:
                self._process_symbol(symbol, specs, feature_set_id,
                                     start_date, end_date)
                succeeded += 1
            except Exception as exc:
                logger.error("Feature pipeline failed for %s: %s", symbol, exc)
                failed += 1

        logger.info(
            "Feature pipeline done: %d succeeded, %d failed, id=%s",
            succeeded, failed, feature_set_id,
        )
        return feature_set_id

    def _process_symbol(
        self,
        symbol: str,
        specs: list[FeatureSpec],
        feature_set_id: str,
        start_date,
        end_date,
    ) -> None:
        # 1. Load OHLCV
        path = ohlcv_path(self.store_root, symbol)
        df = read_ohlcv(path)
        if df.empty:
            logger.warning("%s: no OHLCV data — skipping", symbol)
            return

        # Date filter
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if start_date:
            sd = pd.to_datetime(start_date).date() if isinstance(start_date, str) else start_date
            df = df[df["date"] >= sd]
        if end_date:
            ed = pd.to_datetime(end_date).date() if isinstance(end_date, str) else end_date
            df = df[df["date"] <= ed]

        if df.empty:
            logger.warning("%s: no OHLCV rows in date range — skipping", symbol)
            return

        df = df.sort_values("date").reset_index(drop=True)

        # 2. Technical features
        df = build_technical_features(df, self.project_root)

        # 3. Fundamental features (optional)
        if self.include_fundamentals:
            df = build_fundamental_features(df, self.store_root)

        # 4. Keep only spec columns + metadata
        meta_cols = ["symbol", "date"]
        tech_cols = [s.name for s in specs if s.family in ("technical",)]
        fund_cols = [
            "fund_revenue","fund_net_profit","fund_eps","fund_roe",
            "fund_period_end","fund_period_type","fund_lag_days","fund_announce_date",
        ] if self.include_fundamentals else []
        base_cols = ["close", "volume"]
        keep = meta_cols + [c for c in base_cols + tech_cols + fund_cols if c in df.columns]
        df = df[keep].copy()

        # 5. Write feature Parquet
        out_path = feature_path(self.store_root, feature_set_id, symbol)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.debug(
            "%s: wrote %d feature rows → %s", symbol, len(df), out_path
        )

    def build_panel(
        self,
        symbols: list[str],
        feature_set_id: str,
        add_cross_sectional: bool = True,
    ) -> pd.DataFrame:
        """
        Load per-symbol feature Parquets and concatenate into a universe panel.
        Optionally adds cross-sectional and P4B panel-level features.

        P4B builders (valuation, industry, flow, margin) run here on the
        full panel — they require cross-symbol data and cannot be applied
        per-symbol.

        Returns
        -------
        pd.DataFrame
            Panel with columns: symbol, date, <all feature cols>.
            Sorted by (date, symbol).
        """
        frames = []
        for symbol in symbols:
            p = feature_path(self.store_root, feature_set_id, symbol)
            if p.exists():
                frames.append(pd.read_parquet(p))
            else:
                logger.debug("No feature file for %s (id=%s)", symbol, feature_set_id)

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"]).dt.date
        panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)

        if add_cross_sectional:
            panel = build_cross_sectional_features(panel)

        # --- P4B panel-level builders ---

        if self.include_valuation:
            try:
                from quant_platform.features.valuation import (
                    build_valuation_features, load_valuation_panel,
                )
                val_panel = load_valuation_panel(self.store_root, symbols)
                panel = build_valuation_features(panel, val_panel)
                logger.info("Valuation features added to panel")
            except ImportError:
                logger.warning("features.valuation not available — skipping")

        if self.include_industry:
            try:
                from quant_platform.features.industry import build_industry_features
                from quant_platform.ingest.industry_collector import load_industry_map
                industry_map = load_industry_map(self.store_root)
                panel = build_industry_features(panel, industry_map)
                logger.info("Industry features added to panel")
            except ImportError:
                logger.warning("features.industry not available — skipping")

        if self.include_flow:
            try:
                from quant_platform.features.flow import (
                    build_flow_features, load_flow_panel,
                )
                flow_panel = load_flow_panel(self.store_root, symbols)
                val_panel_for_flow = None
                if self.include_valuation:
                    from quant_platform.features.valuation import load_valuation_panel
                    val_panel_for_flow = load_valuation_panel(self.store_root, symbols)
                panel = build_flow_features(panel, flow_panel, val_panel_for_flow)
                logger.info("Flow features added to panel")
            except ImportError:
                logger.warning("features.flow not available — skipping")

        if self.include_margin:
            try:
                from quant_platform.features.margin import (
                    build_margin_features, load_margin_panel,
                )
                margin_panel = load_margin_panel(self.store_root, symbols)
                val_panel_for_margin = None
                if self.include_valuation:
                    from quant_platform.features.valuation import load_valuation_panel
                    val_panel_for_margin = load_valuation_panel(self.store_root, symbols)
                panel = build_margin_features(panel, margin_panel, val_panel_for_margin)
                logger.info("Margin features added to panel")
            except ImportError:
                logger.warning("features.margin not available — skipping")

        return panel
