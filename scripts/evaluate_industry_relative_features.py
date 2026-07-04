"""
scripts/evaluate_industry_relative_features.py
==================================================
T5.2 (optional, evidence-gated) — Should ind_rank_rsi_6, ind_rank_turnover,
sector_momentum_10d (features.industry.INDUSTRY_SPECS) be ADDED to the
active feature set? Answered by measurement (T4.1 walk-forward), not by
assumption, per task.md's own verify step: "any added industry
feature/label improves walk-forward aggregate IC before adoption."

Constraints honored explicitly
--------------------------------
- ind_rank_main_flow depends on flow features that may be absent
  (features.industry docstring: populated "only after flow features
  exist"). This module checks whether it is entirely NaN in the supplied
  panel and DROPS it from the candidate set if so, rather than assuming
  it is usable -- it never silently trains on an all-NaN column.
- excess_vs_csi300 is NOT used here. NOTE: task.md/plan.md state
  excess_vs_csi300 is [NOT FOUND IN CODEBASE], but the current
  labels.builder actually has an `_add_excess_vs_csi300` /
  `build_label_panel(add_excess_csi300=True)` path (P4A-05). This is the
  same kind of stale-doc drift already found for `include_industry`
  wiring (T1.1) and `DEFAULT_HORIZONS` (T2.1/T2.2) -- the doc predates a
  later merge. This module still does NOT use excess_vs_csi300, per
  task.md's explicit instruction; that's a separate, un-gated decision
  for a human to make deliberately, not something to fold in here as a
  side effect.

Reused as-is: features.industry.{INDUSTRY_SPECS, build_industry_features},
scripts.run_walk_forward_verdict.run_primary_verdict (T4.1).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.features.industry import INDUSTRY_SPECS, build_industry_features
from scripts.run_walk_forward_verdict import run_primary_verdict

logger = get_logger(__name__)


def _usable_industry_cols(panel: pd.DataFrame) -> list[str]:
    """
    Return INDUSTRY_SPECS column names that are actually present AND not
    entirely NaN in *panel* -- explicitly guards against ind_rank_main_flow
    being all-NaN when flow features haven't been computed yet.
    """
    usable = []
    for spec in INDUSTRY_SPECS:
        if spec.name not in panel.columns:
            logger.info("industry feature '%s' not in panel -- excluded from candidate set", spec.name)
            continue
        if panel[spec.name].notna().sum() == 0:
            logger.warning(
                "industry feature '%s' is entirely NaN in this panel (likely missing "
                "an upstream dependency, e.g. flow features for ind_rank_main_flow) -- "
                "excluded from candidate set, not assumed usable.", spec.name,
            )
            continue
        usable.append(spec.name)
    return usable


def evaluate_industry_relative_features(
    panel_with_industry_map: pd.DataFrame,
    industry_map: pd.DataFrame,
    base_feature_cols: list[str],
    label_col: str,
    horizon: int,
    model_factory: Callable,
    n_windows: int = 5,
    window_months: int = 12,
) -> dict:
    """
    Build the industry-relative features onto the panel, then compare
    walk-forward aggregate IC/ICIR of (base features) vs
    (base + usable industry features), model held fixed.

    Returns
    -------
    dict with keys: usable_industry_cols, base_rank_ic, base_icir,
    with_industry_rank_ic, with_industry_icir, adopt (bool), reason (str)
    """
    panel = build_industry_features(panel_with_industry_map.copy(), industry_map)
    usable_cols = _usable_industry_cols(panel)

    print(f"\nUsable industry feature columns this run: {usable_cols}")
    if not usable_cols:
        return {
            "usable_industry_cols": [], "base_rank_ic": float("nan"), "base_icir": float("nan"),
            "with_industry_rank_ic": float("nan"), "with_industry_icir": float("nan"),
            "adopt": False, "reason": "no usable industry feature columns in this panel (all missing/NaN)",
        }

    print("\n--- Walk-forward: BASE features only ---")
    base_result = run_primary_verdict(
        panel, base_feature_cols, label_col, horizon=horizon,
        n_windows=n_windows, window_months=window_months, model_factory=model_factory,
    )

    print("\n--- Walk-forward: BASE + industry-relative features ---")
    with_industry_result = run_primary_verdict(
        panel, base_feature_cols + usable_cols, label_col, horizon=horizon,
        n_windows=n_windows, window_months=window_months, model_factory=model_factory,
    )

    base_ic, base_icir = base_result.agg_rank_ic_mean, base_result.agg_icir
    wi_ic, wi_icir = with_industry_result.agg_rank_ic_mean, with_industry_result.agg_icir

    if np.isnan(base_ic) or np.isnan(wi_ic):
        adopt, reason = False, "one or both runs produced NaN aggregate IC -- insufficient evidence, do not adopt"
    elif wi_ic > base_ic and wi_icir > base_icir:
        adopt, reason = True, (
            f"industry features improve BOTH Rank IC ({wi_ic:+.4f} > {base_ic:+.4f}) "
            f"and ICIR ({wi_icir:+.4f} > {base_icir:+.4f})"
        )
    else:
        adopt, reason = False, (
            f"industry features did not clearly improve both metrics "
            f"(RankIC {wi_ic:+.4f} vs {base_ic:+.4f}, ICIR {wi_icir:+.4f} vs {base_icir:+.4f}) "
            f"-- no assumption, do not adopt"
        )

    result = {
        "usable_industry_cols": usable_cols,
        "base_rank_ic": base_ic, "base_icir": base_icir,
        "with_industry_rank_ic": wi_ic, "with_industry_icir": wi_icir,
        "adopt": adopt, "reason": reason,
    }

    print("\n" + "=" * 70)
    print("T5.2 — industry-relative feature adoption decision")
    print("=" * 70)
    print(f"  Base:            RankIC={base_ic:+.4f}  ICIR={base_icir:+.4f}")
    print(f"  Base+Industry:   RankIC={wi_ic:+.4f}  ICIR={wi_icir:+.4f}")
    print(f"  ADOPT: {adopt}")
    print(f"  REASON: {reason}")
    print("=" * 70 + "\n")

    return result
