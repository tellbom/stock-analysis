"""
quant_platform.selection.exposure
==================================
Industry concentration analysis for selected stock sets.

ExposureMonitor provides static methods to:
  - flag stocks based on industry concentration
  - generate a human-readable concentration report
"""

from __future__ import annotations

import pandas as pd

from quant_platform.selection.config import SelectionConfig


class ExposureMonitor:
    """
    Analyse industry concentration in the selected stock set.

    Flags describe the selected set as a whole; all selected stocks
    within a given run receive the same flag.

    Flag values
    -----------
    industry_overweight
        Any single industry exceeds exposure_warning_threshold.
    balanced
        No industry exceeds exposure_warning_threshold.
    diversified
        All industries are below exposure_diversified_threshold.
    not_selected
        Stock was not selected by the strategy.
    """

    @staticmethod
    def flag(
        panel: pd.DataFrame,
        config: SelectionConfig,
        industry_col: str = "industry_code",
        selected_col: str = "selected",
    ) -> pd.Series:
        """
        Return per-stock exposure flag as a Series aligned to *panel*.

        Parameters
        ----------
        panel : DataFrame
            Must contain *industry_col* and a boolean *selected_col*.
        config : SelectionConfig
        industry_col : str
        selected_col : str

        Returns
        -------
        pd.Series[str]
            One of {not_selected, industry_overweight, balanced, diversified}
            per row.
        """
        flags = pd.Series("not_selected", index=panel.index)

        sel = panel[panel[selected_col] == True]
        if sel.empty:
            return flags

        industry_series = sel[industry_col].fillna(config.unknown_industry_label)
        industry_series = industry_series.replace("", config.unknown_industry_label)

        ind_counts = industry_series.value_counts()
        total = len(sel)
        max_frac = ind_counts.max() / total if total > 0 else 0.0

        if max_frac > config.exposure_warning_threshold:
            flag_value = "industry_overweight"
        elif all(
            (c / total) < config.exposure_diversified_threshold
            for c in ind_counts
        ):
            flag_value = "diversified"
        else:
            flag_value = "balanced"

        flags.loc[sel.index] = flag_value
        return flags

    @staticmethod
    def concentration_report(
        panel: pd.DataFrame,
        industry_col: str = "industry_code",
        name_col: str = "industry_name",
        selected_col: str = "selected",
        symbol_col: str = "symbol",
        top_n_symbols: int = 5,
    ) -> dict:
        """
        Build a dict summarising industry concentration.

        Returns
        -------
        dict
            {industry_name: {"count": N, "fraction": f, "symbols": [...]}, ...}
            Sorted by count descending.
        """
        sel = panel[panel[selected_col] == True]
        if sel.empty:
            return {}

        report: dict = {}
        total = len(sel)

        for ind_code, grp in sel.groupby(industry_col):
            ind_name = str(grp[name_col].iloc[0]) if name_col in grp.columns else str(ind_code)
            report[ind_name] = {
                "count": len(grp),
                "fraction": len(grp) / total,
                "symbols": grp[symbol_col].tolist()[:top_n_symbols],
            }

        return dict(
            sorted(report.items(), key=lambda x: x[1]["count"], reverse=True)
        )
