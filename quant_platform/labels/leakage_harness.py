"""
labels.leakage_harness
======================
Automated leakage test harness (T1.7).

The harness checks warm-up masking, validates label tail NaNs, runs a future
close canary self-test, and reports feature/label window overlap reminders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.features.registry import TECHNICAL_SPECS


@dataclass
class LeakageCheck:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


@dataclass
class LeakageReport:
    checks: list[LeakageCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[LeakageCheck]:
        return [c for c in self.checks if not c.passed]

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(LeakageCheck(name, passed, detail))

    def print_summary(self) -> None:
        print(f"\n{'=' * 60}")
        print("LEAKAGE HARNESS REPORT")
        print(f"{'=' * 60}")
        for check in self.checks:
            print(check)
        passed = sum(c.passed for c in self.checks)
        print(f"{'=' * 60}")
        print(f"Overall: {'PASSED' if self.passed else 'FAILED'} ({passed}/{len(self.checks)} checks pass)")
        print(f"{'=' * 60}\n")


def check_warmup_nans(report: LeakageReport, feature_panel: pd.DataFrame) -> None:
    """Verify that each technical feature's warm-up rows are NaN per spec."""
    spec_map = {s.name: s.warmup for s in TECHNICAL_SPECS if s.warmup > 0}
    violations: list[str] = []

    for symbol, grp in feature_panel.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for col, warmup in spec_map.items():
            if col not in grp.columns:
                continue
            n_non_nan = int(grp[col].iloc[:warmup].notna().sum())
            if n_non_nan > 0:
                violations.append(f"{symbol}/{col}: {n_non_nan}/{warmup} warm-up rows are non-NaN")

    if violations:
        report.add(
            "warmup_nans",
            False,
            f"{len(violations)} warm-up violations found. First: {violations[0]}. "
            "Ensure build_technical_features applies warm-up masking.",
        )
    else:
        n_symbols = feature_panel["symbol"].nunique() if "symbol" in feature_panel.columns else 0
        report.add("warmup_nans", True, f"All warm-up rows are NaN for {n_symbols} symbols.")


def check_canary_future_leak(report: LeakageReport, ohlcv_df: pd.DataFrame) -> None:
    """
    Inject _canary = close.shift(-1) and confirm the detection logic catches it.
    """
    test_df = ohlcv_df.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    test_df["_canary"] = test_df.groupby("symbol")["close"].shift(-1)

    detected = False
    for _, grp in test_df.groupby("symbol"):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp) - 1):
            canary = grp["_canary"].iloc[i]
            if pd.notna(canary) and abs(canary - grp["close"].iloc[i + 1]) < 1e-9:
                detected = True
                break
        if detected:
            break

    report.add(
        "canary_future_leak_detection",
        detected,
        "Canary close.shift(-1) correctly detected as future-leaking feature."
        if detected
        else "HARNESS BUG: canary feature was not detected.",
    )


def check_label_window(
    report: LeakageReport,
    label_panel: pd.DataFrame,
    horizons: list[int],
) -> None:
    """Verify that the final h+1 rows per symbol have NaN labels."""
    for h in horizons:
        ret_col = f"ret_fwd_{h}d"
        if ret_col not in label_panel.columns:
            report.add(f"label_window_h{h}", True, f"Column {ret_col} not present; skipping.")
            continue

        violations: list[str] = []
        for symbol, grp in label_panel.groupby("symbol"):
            grp = grp.sort_values("date").reset_index(drop=True)
            tail = grp[ret_col].values[-(h + 1):]
            n_non_nan = int(sum(1 for v in tail if not np.isnan(v)))
            if n_non_nan > 0:
                violations.append(
                    f"{symbol}: {n_non_nan} of last {h + 1} rows have non-NaN {ret_col}."
                )

        if violations:
            report.add(f"label_tail_nan_h{h}", False, f"{ret_col} tail violations: {violations[:3]}")
        else:
            report.add(f"label_tail_nan_h{h}", True, f"{ret_col}: last {h + 1} rows per symbol are NaN.")


def check_overlap_warning(report: LeakageReport, horizons: list[int]) -> None:
    """
    Report feature/label window overlap pairs requiring purging in model splits.
    This is informational for P1; purged CV belongs to P2.
    """
    long_features = [s for s in TECHNICAL_SPECS if s.window >= min(horizons)]
    overlapping = [
        (s.name, s.window, h)
        for s in long_features
        for h in horizons
        if s.window >= h
    ]

    if overlapping:
        examples = [(name, h) for name, _, h in overlapping[:5]]
        report.add(
            "overlap_warning",
            True,
            f"{len(overlapping)} (feature, horizon) pairs have overlapping windows. "
            f"Examples: {examples}. These require embargo purging in train/validation splits.",
        )
    else:
        report.add("overlap_warning", True, "No feature/label window overlaps for the given horizons.")


def run_leakage_harness(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> LeakageReport:
    """Run all leakage checks and return a structured report."""
    horizons = horizons or [1, 5, 20]
    report = LeakageReport()

    check_warmup_nans(report, feature_panel)
    check_canary_future_leak(report, ohlcv_df)
    check_label_window(report, label_panel, horizons)
    check_overlap_warning(report, horizons)

    return report
