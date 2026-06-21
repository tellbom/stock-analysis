"""
evaluation.selection
====================
Champion/challenger selection protocol (T3.4).

Pre-registered promotion criteria
----------------------------------
Promotion thresholds are written down BEFORE any comparison is run and stored
in a JSON file (``<store_root>/promotion_criteria.json``).  A challenger
replaces the champion only when ALL registered criteria are satisfied
simultaneously.  Criteria cannot be changed retroactively.

Significance test
-----------------
IC series comparison uses a **paired Wilcoxon signed-rank test** on the
per-day Rank IC series of champion vs. challenger.  Wilcoxon is preferred
over a t-test because daily IC series are not normally distributed.

The test is one-sided: challenger must be *significantly better*, not
merely different.

Default promotion thresholds
-----------------------------
  icir_delta_min  : challenger ICIR must exceed champion by ≥ 0.1
  p_value_max     : Wilcoxon p-value ≤ 0.05
  sharpe_positive : challenger backtest Sharpe > 0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

_CRITERIA_FILE = "promotion_criteria.json"

DEFAULT_CRITERIA = {
    "icir_delta_min":  0.1,
    "p_value_max":     0.05,
    "sharpe_positive": True,
}


# ---------------------------------------------------------------------------
# Criteria management
# ---------------------------------------------------------------------------

def register_criteria(
    store_root: Path | str,
    criteria:   dict | None = None,
    overwrite:  bool = False,
) -> dict:
    """
    Persist promotion criteria.  Once written, criteria are immutable unless
    ``overwrite=True`` (which requires explicit intent).

    Returns the active criteria dict.
    """
    path = Path(store_root) / _CRITERIA_FILE
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text())
        logger.info("Promotion criteria already registered: %s", existing)
        return existing

    active = criteria or DEFAULT_CRITERIA
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(active, indent=2))
    logger.info("Promotion criteria registered: %s", active)
    return active


def load_criteria(store_root: Path | str) -> dict:
    """Load registered criteria; raises FileNotFoundError if not set."""
    path = Path(store_root) / _CRITERIA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Promotion criteria not found at {path}. "
            "Call register_criteria() before any model comparison."
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------

@dataclass
class PromotionDecision:
    promoted:           bool
    reason:             str
    champion_icir:      float = float("nan")
    challenger_icir:    float = float("nan")
    icir_delta:         float = float("nan")
    wilcoxon_pvalue:    float = float("nan")
    challenger_sharpe:  float = float("nan")
    criteria_used:      dict  = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PROMOTED ✓" if self.promoted else "REJECTED ✗"
        return (
            f"[{status}] {self.reason} | "
            f"ΔICIR={self.icir_delta:+.4f}  p={self.wilcoxon_pvalue:.4f}  "
            f"challenger_sharpe={self.challenger_sharpe:+.2f}"
        )


def evaluate_promotion(
    store_root:           Path | str,
    champion_daily_ic:    pd.Series,
    challenger_daily_ic:  pd.Series,
    champion_icir:        float,
    challenger_icir:      float,
    challenger_sharpe:    float,
) -> PromotionDecision:
    """
    Decide whether the challenger should replace the champion.

    Parameters
    ----------
    champion_daily_ic / challenger_daily_ic
        Per-day Rank IC series (same dates, same fold configuration).
        Must be aligned by date index for the paired Wilcoxon test.
    champion_icir / challenger_icir
        Pre-computed ICIR values.
    challenger_sharpe
        Backtest Sharpe of the challenger.
    """
    criteria = load_criteria(store_root)
    icir_delta_min  = criteria.get("icir_delta_min",  0.1)
    p_value_max     = criteria.get("p_value_max",     0.05)
    sharpe_positive = criteria.get("sharpe_positive", True)

    icir_delta = challenger_icir - champion_icir

    # Align IC series by date
    common_idx = champion_daily_ic.index.intersection(challenger_daily_ic.index)
    champ_ic = champion_daily_ic.reindex(common_idx).dropna()
    chal_ic  = challenger_daily_ic.reindex(common_idx).dropna()
    common   = champ_ic.index.intersection(chal_ic.index)
    champ_ic, chal_ic = champ_ic.loc[common], chal_ic.loc[common]

    # Wilcoxon signed-rank test (one-sided: challenger > champion)
    pvalue = 1.0
    if len(common) >= 10:
        try:
            diff = chal_ic.values - champ_ic.values
            if np.any(diff != 0):
                stat, pvalue = wilcoxon(diff, alternative="greater")
            else:
                pvalue = 1.0
        except Exception as exc:
            logger.warning("Wilcoxon test failed: %s", exc)
            pvalue = 1.0

    # Apply all criteria
    fails = []
    if icir_delta < icir_delta_min:
        fails.append(f"ΔICIR={icir_delta:+.4f} < {icir_delta_min}")
    if pvalue > p_value_max:
        fails.append(f"p={pvalue:.4f} > {p_value_max}")
    if sharpe_positive and challenger_sharpe <= 0:
        fails.append(f"challenger_sharpe={challenger_sharpe:.2f} ≤ 0")

    promoted = len(fails) == 0
    reason   = "All criteria met" if promoted else "; ".join(fails)

    decision = PromotionDecision(
        promoted=promoted,
        reason=reason,
        champion_icir=champion_icir,
        challenger_icir=challenger_icir,
        icir_delta=icir_delta,
        wilcoxon_pvalue=pvalue,
        challenger_sharpe=challenger_sharpe,
        criteria_used=criteria,
    )
    logger.info("Promotion decision: %s", decision)
    return decision
