"""
quant_platform.selection.confidence
====================================
SR-03: best-effort confidence signal for the strategy layer's daily banner.

PIT-safe: the confidence signal is derived only from features already
available at decision time -- cross-sectional dispersion of the fusion
score, and (when present) agreement between the base and recent model's
percentile scores. It NEVER reads any forward-label column.

Report-banner only (per the SR task doc's open-question default):
confidence does not shrink or reorder the actionable shortlist -- it is
metadata attached alongside it.

Regime is best-effort/optional only: ``RegimeAnalyser``
(``evaluation/regime_analysis.py``) needs forward labels via walk-forward
IC and cannot run PIT at live-recommendation time. This module never calls
it live -- callers may pass a ``regime_hint`` sourced from an existing
offline/historical regime report.
"""

from __future__ import annotations

import pandas as pd

CONFIDENCE_LABELS = ("low", "normal", "high")

_LOW_THRESHOLD = 0.35
_HIGH_THRESHOLD = 0.65

# Cap for normalizing the cross-sectional std of a min-max-scaled score
# into roughly [0, 1]. 0.5 is the std of the most dispersed possible
# [0, 1]-bounded sample (a two-point {0, 1} split).
_DISPERSION_STD_CAP = 0.5

_REGIME_NUDGE = 0.1


def _dispersion_signal(scores: pd.Series) -> float:
    scores = scores.astype(float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return 0.0
    normalized = (scores - lo) / (hi - lo)
    return float(min(normalized.std(ddof=0) / _DISPERSION_STD_CAP, 1.0))


def _agreement_signal(panel: pd.DataFrame) -> float | None:
    if "base_pct" not in panel.columns or "recent_pct" not in panel.columns:
        return None
    disagreement = (
        panel["base_pct"].astype(float) - panel["recent_pct"].astype(float)
    ).abs().mean()
    return float(max(0.0, 1.0 - disagreement))


def _label(confidence: float) -> str:
    if confidence < _LOW_THRESHOLD:
        return "low"
    if confidence > _HIGH_THRESHOLD:
        return "high"
    return "normal"


def compute_confidence(
    panel_at_t: pd.DataFrame,
    *,
    score_col: str = "fusion_score_col",
    regime_hint: str | None = None,
) -> tuple[float, str]:
    """
    Compute a best-effort confidence score + label for a single date's
    actionable pool.

    Blends two PIT-safe signals:
      * cross-sectional dispersion of ``score_col`` -- a well-separated
        pool (clear winners vs losers) is more trustworthy than a flat
        one.
      * base/recent model agreement (``base_pct`` vs ``recent_pct``, when
        both are present) -- higher agreement is more trustworthy.

    ``regime_hint`` is optional and sourced from an already-generated,
    offline/historical regime report -- never computed live here. When
    provided, "weak" nudges confidence down and "strong" nudges it up by a
    small fixed amount; any other value (including ``None``) is ignored.

    Returns
    -------
    (confidence, label) where confidence is in [0, 1] and label is one of
    ``CONFIDENCE_LABELS``.
    """
    if panel_at_t.empty:
        raise ValueError("compute_confidence requires a non-empty panel")
    if score_col not in panel_at_t.columns:
        raise ValueError(f"compute_confidence requires column '{score_col}'")

    dispersion = _dispersion_signal(panel_at_t[score_col])
    agreement = _agreement_signal(panel_at_t)
    confidence = dispersion if agreement is None else 0.5 * dispersion + 0.5 * agreement

    if regime_hint == "weak":
        confidence = max(0.0, confidence - _REGIME_NUDGE)
    elif regime_hint == "strong":
        confidence = min(1.0, confidence + _REGIME_NUDGE)

    return confidence, _label(confidence)
