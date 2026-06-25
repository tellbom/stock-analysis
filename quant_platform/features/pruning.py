"""
features.pruning
================
Technical feature collinearity pruning engine (P4C-02).

Uses the collinearity analysis from ``evaluation.feature_ic`` to identify
groups of technical features with pairwise Spearman rank correlation above
a threshold.  For each group, retains the feature with the highest single-
factor Rank IC at the primary label horizon (5d) and marks the rest as
``active=False`` in the feature registry.

Pruning is a structured, documented decision — not an ad-hoc deletion.
The research ledger records the pruning event, and pruned features:
  - Are retained in all existing Parquet files (no data destruction).
  - Are still *computed* by the technical builder (no code deletion).
  - Are excluded from the training feature list by ``get_active_feature_cols``.

Expected outcome on the 27 current technical features: ~10–12 independent
signals remain after removing the most collinear duplicates.

Usage
-----
    from quant_platform.features.pruning import FeaturePruner

    pruner = FeaturePruner(store_root=Path("/data/lake"))
    result = pruner.run(
        ic_report=feature_ic_report,
        primary_label="ret_fwd_5d",
        corr_threshold=0.85,
    )
    result.print_summary()
    active_cols = pruner.get_active_feature_cols(all_feature_cols)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

_PRUNING_LOG_FILE = "feature_pruning_log.parquet"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PruningResult:
    """Results of one pruning run."""
    generated_at:     str = ""
    n_features_before: int = 0
    n_features_after:  int = 0
    n_pruned:          int = 0
    corr_threshold:    float = 0.85
    primary_label:     str = "ret_fwd_5d"

    # List of pruned features with reasons
    pruned: list[dict] = field(default_factory=list)
    # List of retained features
    retained: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print(f"FEATURE PRUNING RESULT  [{self.generated_at}]")
        print(f"  Before: {self.n_features_before}  →  After: {self.n_features_after}")
        print(f"  Pruned: {self.n_pruned}  (|ρ| > {self.corr_threshold:.0%})")
        print(f"  Label:  {self.primary_label}")
        print()
        if self.pruned:
            print(f"  {'Feature':25s}  {'Corr_group':25s}  {'IC_5d':>8s}  {'Reason'}")
            for p in self.pruned:
                print(
                    f"  {p['feature']:25s}  {p['corr_partner']:25s}  "
                    f"{p.get('ic_5d', float('nan')):>+8.4f}  "
                    f"{p.get('reason', '')}"
                )
        print("=" * 70 + "\n")

    def to_ledger_note(self) -> str:
        return (
            f"Pruning run {self.generated_at}: "
            f"{self.n_features_before} → {self.n_features_after} features. "
            f"Threshold: |ρ| > {self.corr_threshold:.0%}. "
            f"Pruned: {[p['feature'] for p in self.pruned]}"
        )


# ---------------------------------------------------------------------------
# Pruning engine
# ---------------------------------------------------------------------------

class FeaturePruner:
    """
    Prune collinear features from the feature registry.

    Parameters
    ----------
    store_root : Path | str
        Root of the data lake.  Used to persist the pruning log.
    """

    def __init__(self, store_root: Path | str) -> None:
        self.store_root = Path(store_root)
        self._log_path  = self.store_root / "evaluation" / _PRUNING_LOG_FILE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ic_report,                         # FeatureICReport from feature_ic.py
        primary_label: str = "ret_fwd_5d",
        corr_threshold: float = 0.85,
        ic_label_col: str = "ic_5d",
    ) -> PruningResult:
        """
        Identify and mark collinear features for pruning.

        For each collinearity cluster:
          1. Retrieve IC values from the ic_report.
          2. Retain the feature with the highest absolute IC.
          3. Mark the rest as pruned (recorded in the pruning log).

        Note: this method does NOT modify the FeatureRegistry directly —
        it returns a PruningResult that can be inspected.  Call
        ``apply_pruning(result, registry)`` to persist the active flags.

        Parameters
        ----------
        ic_report : FeatureICReport
            Output from ``compute_feature_ic_report()``.
        primary_label : str
            Label horizon used to compare ICs within a cluster.
        corr_threshold : float
            Pairs above this threshold are collinear.
        ic_label_col : str
            Column name in feature IC table to use for IC comparison.

        Returns
        -------
        PruningResult
        """
        from quant_platform.evaluation.feature_ic import (
            FeatureICReport, _find_collinearity_clusters, COLLINEARITY_THRESHOLD,
        )

        result = PruningResult(
            generated_at=dt.datetime.now().isoformat(timespec="seconds"),
            corr_threshold=corr_threshold,
            primary_label=primary_label,
        )

        # Build IC lookup from the report
        ic_table = ic_report.feature_ic_table()
        if ic_table.empty:
            logger.warning("FeaturePruner: empty IC table — nothing to prune")
            return result

        ic_lookup: dict[str, float] = {}
        if ic_label_col in ic_table.columns:
            for _, row in ic_table.iterrows():
                ic_lookup[row["feature"]] = float(row[ic_label_col])

        # Use the collinearity matrix from the ic_report
        corr_matrix = ic_report.collinearity_matrix
        if corr_matrix.empty:
            logger.info("FeaturePruner: no collinearity matrix — nothing to prune")
            result.n_features_before = len(ic_table)
            result.n_features_after  = len(ic_table)
            result.retained = list(ic_table["feature"])
            return result

        # Re-derive clusters at the requested threshold (may differ from
        # the threshold used in ic_report)
        clusters = _find_collinearity_clusters(corr_matrix, threshold=corr_threshold)

        result.n_features_before = len(corr_matrix.columns)
        pruned_set: set[str] = set()

        for cid, members in clusters.items():
            # Sort members by absolute IC (descending); retain the best
            ics = {m: abs(ic_lookup.get(m, 0.0)) for m in members}
            sorted_members = sorted(ics, key=ics.get, reverse=True)
            retain   = sorted_members[0]
            to_prune = sorted_members[1:]

            for feat in to_prune:
                # Find the most correlated retained partner
                if retain in corr_matrix.columns and feat in corr_matrix.columns:
                    partner_corr = float(corr_matrix.loc[feat, retain])
                else:
                    partner_corr = float("nan")

                pruned_set.add(feat)
                result.pruned.append({
                    "feature":      feat,
                    "corr_partner": retain,
                    "corr_value":   round(partner_corr, 4),
                    "ic_5d":        round(ic_lookup.get(feat, float("nan")), 6),
                    "ic_retained":  round(ic_lookup.get(retain, float("nan")), 6),
                    "reason":       f"collinear with {retain} (|ρ|={abs(partner_corr):.3f} > {corr_threshold:.0%})",
                    "cluster_id":   cid,
                })

        result.n_pruned = len(pruned_set)
        result.n_features_after = result.n_features_before - result.n_pruned
        result.retained = [
            f for f in corr_matrix.columns if f not in pruned_set
        ]

        self._save_pruning_log(result)
        logger.info(
            "FeaturePruner: pruned %d/%d features  (threshold=%.0f%%)",
            result.n_pruned, result.n_features_before, corr_threshold * 100,
        )
        return result

    def get_active_feature_cols(
        self,
        all_feature_cols: list[str],
        pruning_result: PruningResult | None = None,
    ) -> list[str]:
        """
        Return the subset of ``all_feature_cols`` that should be passed to
        the model, excluding pruned features.

        If ``pruning_result`` is None, loads the most recent pruning log
        from disk.  If no log exists, returns all features unchanged.

        Parameters
        ----------
        all_feature_cols : list[str]
            Full list of feature columns in the panel.
        pruning_result : PruningResult | None
            If provided, use this result directly.

        Returns
        -------
        list[str]
            Active (non-pruned) feature columns.
        """
        pruned_feats = self._load_pruned_features(pruning_result)
        active = [f for f in all_feature_cols if f not in pruned_feats]
        n_removed = len(all_feature_cols) - len(active)
        if n_removed > 0:
            logger.info(
                "get_active_feature_cols: %d/%d features active (%d pruned)",
                len(active), len(all_feature_cols), n_removed,
            )
        return active

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_pruning_log(self, result: PruningResult) -> None:
        """Append the pruning result to the pruning log Parquet."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "generated_at":   result.generated_at,
                "feature":        p["feature"],
                "corr_partner":   p["corr_partner"],
                "corr_value":     p["corr_value"],
                "ic_5d":          p.get("ic_5d", float("nan")),
                "ic_retained":    p.get("ic_retained", float("nan")),
                "reason":         p["reason"],
                "cluster_id":     p.get("cluster_id", -1),
                "active":         False,
            }
            for p in result.pruned
        ]
        if not rows:
            return
        new_df = pd.DataFrame(rows)
        if self._log_path.exists():
            existing = pd.read_parquet(self._log_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(self._log_path, index=False)
        logger.info("Pruning log saved → %s (%d entries)", self._log_path, len(combined))

    def _load_pruned_features(
        self,
        pruning_result: PruningResult | None,
    ) -> set[str]:
        """Return the set of pruned feature names (most recent run)."""
        if pruning_result is not None:
            return {p["feature"] for p in pruning_result.pruned}
        if not self._log_path.exists():
            return set()
        try:
            df = pd.read_parquet(self._log_path)
            if df.empty:
                return set()
            # Use only the most recent pruning run
            last_run = df["generated_at"].max()
            return set(df[df["generated_at"] == last_run]["feature"].tolist())
        except Exception as exc:
            logger.warning("Could not load pruning log: %s", exc)
            return set()

    def load_pruning_log(self) -> pd.DataFrame:
        """Load the full pruning history log."""
        if not self._log_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self._log_path)
