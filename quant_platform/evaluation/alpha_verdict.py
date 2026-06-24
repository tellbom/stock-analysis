"""
evaluation.alpha_verdict
========================
Alpha verdict document generator (T2.8).

Synthesises evidence from T2.3 (evaluation), T2.4 (baselines), T2.5 (backtest),
and T2.6 (robustness) into a single go/no-go verdict with a documented
evidence chain.

The lockbox result is included at the end.  Touching the lockbox seals P2:
no further tuning is allowed once the lockbox verdict is rendered.

Output
------
  <store_root>/alpha_verdict.txt     — human-readable verdict
  <store_root>/alpha_verdict.json    — machine-readable for MLflow logging
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.evaluation.metrics import EvalReport
from quant_platform.evaluation.backtest import BacktestResult
from quant_platform.evaluation.robustness import RobustnessReport
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AlphaVerdict:
    """
    Structured alpha verdict.

    ``verdict`` is "GO", "NO_GO", or "INCONCLUSIVE".
    ``lockbox_used`` tracks whether the test set has been spent.

    P4A additions
    -------------
    ``walk_forward_used``     : True when a WalkForwardResult is provided.
    ``wf_agg_rank_ic``        : Aggregate OOS Rank IC from walk-forward.
    ``wf_agg_icir``           : Aggregate OOS ICIR from walk-forward.
    ``wf_agg_sharpe``         : Aggregate OOS Sharpe from walk-forward.
    ``wf_ic_sign_stability``  : Fraction of windows with positive Rank IC.
    ``wf_total_indep_periods``: Total independent forward periods.
    ``subperiod_ic_ratio``    : Stability index from RobustnessReport.
    ``subperiod_interpretation``: Human-readable stability note.
    """
    verdict:         str   = "INCONCLUSIVE"   # "GO" | "NO_GO" | "INCONCLUSIVE"
    confidence:      str   = "LOW"            # "LOW" | "MEDIUM" | "HIGH"
    evidence:        list[str] = field(default_factory=list)
    caveats:         list[str] = field(default_factory=list)

    # Legacy lockbox
    lockbox_used:    bool  = False
    lockbox_rank_ic: float = float("nan")
    lockbox_sharpe:  float = float("nan")

    # P4A: walk-forward evaluation
    walk_forward_used:       bool  = False
    wf_agg_rank_ic:          float = float("nan")
    wf_agg_icir:             float = float("nan")
    wf_agg_sharpe:           float = float("nan")
    wf_ic_sign_stability:    float = float("nan")
    wf_total_indep_periods:  int   = 0

    # P4A: subperiod stability index
    subperiod_ic_ratio:      float = float("nan")
    subperiod_interpretation: str  = ""

    generated_at:    str   = ""

    def to_dict(self) -> dict:
        return {
            "verdict":                  self.verdict,
            "confidence":               self.confidence,
            "lockbox_used":             self.lockbox_used,
            "lockbox_rank_ic":          round(self.lockbox_rank_ic, 6),
            "lockbox_sharpe":           round(self.lockbox_sharpe, 4),
            "walk_forward_used":        self.walk_forward_used,
            "wf_agg_rank_ic":           round(self.wf_agg_rank_ic, 6),
            "wf_agg_icir":              round(self.wf_agg_icir, 4),
            "wf_agg_sharpe":            round(self.wf_agg_sharpe, 4),
            "wf_ic_sign_stability":     round(self.wf_ic_sign_stability, 4),
            "wf_total_indep_periods":   self.wf_total_indep_periods,
            "subperiod_ic_ratio":       round(self.subperiod_ic_ratio, 4),
            "subperiod_interpretation": self.subperiod_interpretation,
            "evidence":                 self.evidence,
            "caveats":                  self.caveats,
            "generated_at":             self.generated_at,
        }


def render_verdict(
    store_root: Path | str,
    oof_eval:           EvalReport,
    baseline_table:     pd.DataFrame,
    backtest:           BacktestResult,
    robustness:         RobustnessReport,
    lockbox_eval:       EvalReport | None = None,
    lockbox_backtest:   BacktestResult | None = None,
    walk_forward_result=None,   # WalkForwardResult | None (optional import)
    icir_threshold:     float = 0.3,
    sharpe_threshold:   float = 0.5,
    wf_icir_threshold:  float = 0.3,
) -> AlphaVerdict:
    """
    Synthesise a go/no-go alpha verdict.

    The verdict criteria (conservative defaults):
    - GO requires ALL of:
        - OOF Rank ICIR > ``icir_threshold`` (default 0.3)
        - Backtest net Sharpe > ``sharpe_threshold`` (default 0.5)
        - Beats all trivial baselines on Rank IC
        - Label-shuffle null test passes (shuffle IC ≈ 0)
        - Subperiod stable (both halves positive)
    - NO_GO if any critical condition fails.
    - INCONCLUSIVE if data is insufficient to judge.

    P4A additions
    -------------
    When ``walk_forward_result`` is provided it becomes the primary OOS
    evidence, replacing the legacy lockbox.  The walkforward aggregate
    ICIR must exceed ``wf_icir_threshold`` for a GO verdict.

    When the lockbox is provided (legacy path), it is the final evidence
    and seals P2.

    P4A-02: the subperiod stability index (``robustness.subperiod_ic_ratio``)
    is now surfaced in the evidence with a human-readable interpretation.
    """
    verdict = AlphaVerdict(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
    )
    store_root = Path(store_root)

    ev   = verdict.evidence
    cav  = verdict.caveats
    pass_count = 0
    fail_count = 0

    # --- 1. OOF IC quality ---
    ric  = oof_eval.rank_ic_mean
    icir = oof_eval.icir
    ev.append(f"OOF Rank IC = {ric:+.4f}, ICIR = {icir:+.3f}")
    if icir > icir_threshold:
        ev.append(f"  ✓ ICIR {icir:.3f} > threshold {icir_threshold}")
        pass_count += 1
    elif icir > 0:
        cav.append(f"  ⚠ ICIR {icir:.3f} is positive but below threshold {icir_threshold}")
        fail_count += 1
    else:
        ev.append(f"  ✗ ICIR {icir:.3f} ≤ 0 — no signal")
        fail_count += 1

    # --- 2. Beats baselines ---
    if not baseline_table.empty:
        model_name = baseline_table.index[0]
        model_ric  = baseline_table.loc[model_name, "rank_ic_mean"]
        other_rics = baseline_table.drop(model_name)["rank_ic_mean"]
        n_beaten   = (model_ric > other_rics).sum()
        n_total    = len(other_rics)
        ev.append(f"Beats {n_beaten}/{n_total} trivial baselines on Rank IC")
        if n_beaten == n_total:
            pass_count += 1
        else:
            cav.append(f"Does not beat all baselines: {n_beaten}/{n_total}")
            fail_count += 1

    # --- 3. Cost-aware backtest ---
    sharpe  = backtest.sharpe
    net_spread = backtest.net_long_minus_short
    ev.append(f"Backtest: Sharpe = {sharpe:+.2f}, net L-S spread = {net_spread:+.4f}")
    if sharpe > sharpe_threshold:
        ev.append(f"  ✓ Net Sharpe {sharpe:.2f} > threshold {sharpe_threshold}")
        pass_count += 1
    elif sharpe > 0:
        cav.append(f"  ⚠ Sharpe {sharpe:.2f} positive but below threshold {sharpe_threshold}")
        fail_count += 1
    else:
        ev.append(f"  ✗ Sharpe {sharpe:.2f} ≤ 0 — costs destroy the signal")
        fail_count += 1

    # --- 4. Null tests ---
    if robustness.shuffle_passed:
        ev.append(f"  ✓ Label-shuffle IC ≈ 0 ({robustness.shuffle_rank_ic:+.4f}) — not a pipeline artefact")
        pass_count += 1
    else:
        ev.append(f"  ✗ Label-shuffle IC = {robustness.shuffle_rank_ic:+.4f} — too large, possible bug")
        fail_count += 1

    # --- 5. Subperiod stability (P4A-02: now includes stability index) ---
    if robustness.subperiod_stable:
        ev.append(
            f"  ✓ Subperiod stable: "
            f"first={robustness.first_half_ric:+.4f}, "
            f"second={robustness.second_half_ric:+.4f}"
        )
        pass_count += 1
    else:
        cav.append(
            f"  ⚠ Subperiod instability: "
            f"first={robustness.first_half_ric:+.4f}, "
            f"second={robustness.second_half_ric:+.4f}"
        )
        fail_count += 1

    # P4A-02: surface subperiod_ic_ratio with interpretation
    ratio = robustness.subperiod_ic_ratio
    if not np.isnan(ratio):
        verdict.subperiod_ic_ratio = ratio
        interp = _stability_interpretation(ratio, robustness.subperiod_stable)
        verdict.subperiod_interpretation = interp
        ev.append(f"  Stability index: {ratio:.3f} — {interp}")
        if not robustness.subperiod_stable:
            cav.append(
                "  Opposite-sign subperiods suggest genuine regime sensitivity. "
                "Consider whether the lockbox/walk-forward period differs from train/val regime."
            )

    # --- 6. Walk-forward OOS evaluation (P4A-03 — primary OOS instrument) ---
    if walk_forward_result is not None:
        verdict.walk_forward_used       = True
        verdict.wf_agg_rank_ic          = walk_forward_result.agg_rank_ic_mean
        verdict.wf_agg_icir             = walk_forward_result.agg_icir
        verdict.wf_agg_sharpe           = walk_forward_result.agg_sharpe
        verdict.wf_ic_sign_stability    = walk_forward_result.ic_sign_stability
        verdict.wf_total_indep_periods  = walk_forward_result.total_independent_periods

        ev.append(
            f"WALK-FORWARD OOS ({walk_forward_result.n_windows()} windows, "
            f"{walk_forward_result.total_independent_periods} indep. periods): "
            f"Rank IC = {walk_forward_result.agg_rank_ic_mean:+.4f}, "
            f"ICIR = {walk_forward_result.agg_icir:+.4f}, "
            f"Sharpe = {walk_forward_result.agg_sharpe:+.4f}"
        )
        ev.append(
            f"  IC sign stability = {walk_forward_result.ic_sign_stability:.2f} "
            f"({walk_forward_result.ic_sign_stability * walk_forward_result.n_windows():.0f}/"
            f"{walk_forward_result.n_windows()} windows positive)"
        )

        wf_icir = walk_forward_result.agg_icir
        if wf_icir > wf_icir_threshold:
            ev.append(f"  ✓ Walk-forward ICIR {wf_icir:.3f} > threshold {wf_icir_threshold}")
            pass_count += 1
        elif wf_icir > 0:
            cav.append(
                f"  ⚠ Walk-forward ICIR {wf_icir:.3f} positive but below threshold {wf_icir_threshold}"
            )
            fail_count += 1
        else:
            ev.append(f"  ✗ Walk-forward ICIR {wf_icir:.3f} ≤ 0 — signal does not generalise OOS")
            fail_count += 1

        if walk_forward_result.agg_sharpe > 0:
            ev.append(f"  ✓ Walk-forward Sharpe positive ({walk_forward_result.agg_sharpe:+.3f})")
            pass_count += 1
        else:
            cav.append(
                f"  ⚠ Walk-forward Sharpe negative ({walk_forward_result.agg_sharpe:+.3f}) — "
                "check cost assumptions"
            )

    # --- 7. Lockbox (legacy — seals P2 when used) ---
    if lockbox_eval is not None:
        verdict.lockbox_used    = True
        verdict.lockbox_rank_ic = lockbox_eval.rank_ic_mean
        verdict.lockbox_sharpe  = lockbox_backtest.sharpe if lockbox_backtest else float("nan")
        ev.append(
            f"LOCKBOX: Rank IC = {lockbox_eval.rank_ic_mean:+.4f}, "
            f"ICIR = {lockbox_eval.icir:+.3f}, "
            f"Sharpe = {verdict.lockbox_sharpe:+.2f}"
        )
        cav.append(
            "Lockbox has been used — P2 is sealed. "
            "No further tuning is permitted on this data split."
        )
        if lockbox_eval.rank_ic_mean > 0 and (lockbox_backtest is None or lockbox_backtest.sharpe > 0):
            ev.append("  ✓ Lockbox verdict: POSITIVE")
            pass_count += 1
        else:
            ev.append("  ✗ Lockbox verdict: NEGATIVE — signal did not generalise")
            fail_count += 1

    # --- Determine verdict ---
    if pass_count == 0 and fail_count == 0:
        verdict.verdict    = "INCONCLUSIVE"
        verdict.confidence = "LOW"
    elif fail_count == 0:
        verdict.verdict    = "GO"
        verdict.confidence = "HIGH" if pass_count >= 5 else "MEDIUM"
    elif pass_count > fail_count:
        verdict.verdict    = "INCONCLUSIVE"
        verdict.confidence = "MEDIUM"
    else:
        verdict.verdict    = "NO_GO"
        verdict.confidence = "HIGH" if fail_count >= 3 else "MEDIUM"

    # Standard caveat
    cav.append(
        "A-share market characteristics (circuit breakers, liquidity, T+1 settlement) "
        "may affect live performance beyond what this backtest captures."
    )
    if not verdict.lockbox_used and not verdict.walk_forward_used:
        cav.append(
            "Neither lockbox nor walk-forward OOS evaluation has been used. "
            "This verdict is based on OOF evaluation only — it cannot measure "
            "whether the signal generalises to genuinely new data."
        )
    if not verdict.walk_forward_used and verdict.lockbox_used:
        cav.append(
            "Using legacy static lockbox.  Consider switching to walk-forward "
            "evaluation (WalkForwardEvaluator) for more statistical power."
        )

    # Write files
    _write_verdict(verdict, store_root)
    logger.info(
        "Alpha verdict: %s (confidence=%s, pass=%d, fail=%d)",
        verdict.verdict, verdict.confidence, pass_count, fail_count,
    )
    return verdict


def _stability_interpretation(ratio: float, same_sign: bool) -> str:
    """Human-readable interpretation of subperiod IC ratio."""
    if not same_sign:
        return "REGIME-SENSITIVE (opposite signs — genuine decay likely)"
    if ratio >= 0.7:
        return "REGIME-STABLE (consistent magnitude)"
    if ratio >= 0.4:
        return "MODERATE regime sensitivity (magnitude varies)"
    return "HIGH regime sensitivity (magnitude differs strongly)"


def _write_verdict(verdict: AlphaVerdict, store_root: Path) -> None:
    """Write human-readable text and machine-readable JSON."""
    lines = [
        "=" * 65,
        f"ALPHA VERDICT — {verdict.generated_at}",
        f"VERDICT: {verdict.verdict}  (confidence: {verdict.confidence})",
        f"Lockbox used: {'YES — P2 sealed' if verdict.lockbox_used else 'NO'}",
        f"Walk-forward used: {'YES' if verdict.walk_forward_used else 'NO'}",
        "=" * 65,
        "",
        "EVIDENCE:",
    ]
    for e in verdict.evidence:
        lines.append(f"  {e}")
    lines += ["", "CAVEATS:"]
    for c in verdict.caveats:
        lines.append(f"  {c}")
    lines += ["", "=" * 65]

    txt_path  = store_root / "alpha_verdict.txt"
    json_path = store_root / "alpha_verdict.json"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(verdict.to_dict(), indent=2), encoding="utf-8")
    logger.info("Verdict written → %s, %s", txt_path, json_path)
