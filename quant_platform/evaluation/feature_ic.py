"""
evaluation.feature_ic
=====================
Single-factor IC diagnostic framework (P4C-01).

For every feature in the feature panel, computes:
  (a) Mean daily Rank IC vs each label horizon (1d, 5d, 10d, 20d)
  (b) IC t-statistic:  mean_IC * sqrt(n_dates) / std_IC
  (c) IC decay curve:  IC at lags 1, 2, 3, 5, 10, 20 trading days
  (d) Pairwise Spearman correlation matrix for collinearity clustering

Outputs
-------
  FeatureICReport       — structured results dataclass
  feature_ic_table      — DataFrame sorted by |IC_5d| descending
  collinearity_matrix   — pairwise Spearman correlation DataFrame
  CSV exported to       — <store_root>/evaluation/feature_ic_<date>.csv

This framework is a permanent diagnostic: run it after every data
integration and before any pruning decision.  The decay half-life is
the primary indicator of a feature's short-horizon relevance:

  decay_halflife ≈ lag at which IC falls to 50% of its peak value.

For a short-horizon strategy the ideal feature has a half-life of 1–5d.
A feature with a half-life > 20d and t-stat < 1.5 at 5d is a candidate
for pruning.

Usage
-----
    from quant_platform.evaluation.feature_ic import compute_feature_ic_report

    report = compute_feature_ic_report(
        panel=panel,
        feature_cols=feature_cols,
        label_cols=["ret_fwd_1d", "ret_fwd_5d", "ret_fwd_10d", "ret_fwd_20d"],
        store_root=Path("/data/lake"),
    )
    report.print_summary(top_n=15)
    report.save_csv(Path("/data/lake"))
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)

# Default label columns to evaluate IC against
DEFAULT_LABEL_COLS = ["ret_fwd_1d", "ret_fwd_5d", "ret_fwd_10d", "ret_fwd_20d"]

# Default IC decay lags (trading days)
DEFAULT_DECAY_LAGS = [1, 2, 3, 5, 10, 20]

# Collinearity threshold — pairs above this are candidates for pruning
COLLINEARITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# IC helper
# ---------------------------------------------------------------------------

def _daily_rank_ic(
    panel: pd.DataFrame,
    feature_col: str,
    label_col: str,
    min_stocks: int = 5,
) -> list[float]:
    """
    Compute per-date cross-sectional Spearman Rank IC between one feature
    and one label.  Returns a list of daily IC values (NaN dates skipped).
    """
    daily_rics: list[float] = []
    sub = panel[["date", feature_col, label_col]].dropna()
    for _, grp in sub.groupby("date"):
        if len(grp) < min_stocks:
            continue
        ric, _ = spearmanr(grp[feature_col], grp[label_col])
        if not np.isnan(ric):
            daily_rics.append(float(ric))
    return daily_rics


def _ic_tstat(daily_rics: list[float]) -> float:
    """IC t-statistic: mean_IC * sqrt(n) / std_IC."""
    if len(daily_rics) < 2:
        return float("nan")
    arr = np.array(daily_rics)
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return float("nan")
    return float(np.mean(arr) * np.sqrt(len(arr)) / std)


def _decay_halflife(decay_dict: dict[int, float], peak_ic: float) -> float:
    """
    Estimate the lag at which IC falls to 50% of the peak value.
    Returns the smallest lag where IC / peak_ic < 0.5, or 999 if never.
    Half-life is undefined (NaN) when peak_ic ≈ 0.
    """
    if abs(peak_ic) < 1e-9:
        return float("nan")
    half = abs(peak_ic) * 0.5
    for lag in sorted(decay_dict):
        ic = decay_dict[lag]
        if np.isnan(ic):
            continue
        if abs(ic) < half:
            return float(lag)
    return 999.0


# ---------------------------------------------------------------------------
# Decay curve via close price
# ---------------------------------------------------------------------------

def _compute_ic_decay_from_close(
    panel: pd.DataFrame,
    feature_col: str,
    lags: list[int],
    min_stocks: int = 5,
) -> dict[int, float]:
    """
    Compute IC decay curve using the 'close' column to build forward returns
    on the fly (T+1 … T+1+lag convention).  Falls back to NaN when 'close'
    is absent.
    """
    if "close" not in panel.columns:
        return {lag: float("nan") for lag in lags}

    df = panel[["symbol", "date", "close", feature_col]].dropna(subset=[feature_col]).copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    result: dict[int, float] = {}
    for lag in lags:
        # Forward return: close(T+1+lag) / close(T+1) - 1
        df[f"_fwd_{lag}"] = df.groupby("symbol")["close"].transform(
            lambda x: x.shift(-lag) / x.shift(-1) - 1
        )
        sub = df[[feature_col, f"_fwd_{lag}", "date"]].dropna()
        daily_rics: list[float] = []
        for _, grp in sub.groupby("date"):
            if len(grp) < min_stocks:
                continue
            ric, _ = spearmanr(grp[feature_col], grp[f"_fwd_{lag}"])
            if not np.isnan(ric):
                daily_rics.append(float(ric))
        result[lag] = float(np.mean(daily_rics)) if daily_rics else float("nan")
        df = df.drop(columns=[f"_fwd_{lag}"])

    return result


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

@dataclass
class FeatureICRow:
    """Per-feature IC metrics."""
    feature:       str
    ic_1d:         float = float("nan")
    ic_5d:         float = float("nan")
    ic_10d:        float = float("nan")
    ic_20d:        float = float("nan")
    tstat_1d:      float = float("nan")
    tstat_5d:      float = float("nan")
    tstat_20d:     float = float("nan")
    decay_halflife: float = float("nan")   # lag where IC < 50% peak
    n_dates:       int   = 0
    decay_curve:   dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "feature":        self.feature,
            "ic_1d":          round(self.ic_1d, 6),
            "ic_5d":          round(self.ic_5d, 6),
            "ic_10d":         round(self.ic_10d, 6),
            "ic_20d":         round(self.ic_20d, 6),
            "tstat_1d":       round(self.tstat_1d, 3),
            "tstat_5d":       round(self.tstat_5d, 3),
            "tstat_20d":      round(self.tstat_20d, 3),
            "decay_halflife": round(self.decay_halflife, 1),
            "n_dates":        self.n_dates,
        }


@dataclass
class FeatureICReport:
    """Full per-feature IC diagnostic report."""
    rows:                list[FeatureICRow] = field(default_factory=list)
    collinearity_matrix: pd.DataFrame      = field(default_factory=pd.DataFrame)
    generated_at:        str               = ""
    n_features:          int               = 0
    n_dates:             int               = 0

    # Cluster groups: {cluster_id: [feature_names]}
    collinearity_clusters: dict[int, list[str]] = field(default_factory=dict)

    def feature_ic_table(self) -> pd.DataFrame:
        """Return a DataFrame sorted by available absolute IC, preferring 5d."""
        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame([r.to_dict() for r in self.rows])
        ic_cols = [c for c in ("ic_5d", "ic_1d", "ic_10d", "ic_20d") if c in df.columns]
        sort_col = None
        for col in ic_cols:
            abs_col = f"abs_{col}"
            df[abs_col] = df[col].abs()
            if sort_col is None and df[abs_col].notna().any():
                sort_col = abs_col
        if sort_col is None:
            return df.drop(columns=[c for c in df.columns if c.startswith("abs_ic_")], errors="ignore")
        abs_cols = [c for c in df.columns if c.startswith("abs_ic_")]
        return df.sort_values(sort_col, ascending=False).drop(columns=abs_cols)

    def top_features(self, n: int = 15, horizon: str = "ic_5d") -> list[str]:
        """Return the top-n features by absolute IC at the specified horizon."""
        df = self.feature_ic_table()
        if df.empty:
            return []
        return df.nlargest(n, f"abs_{horizon}" if f"abs_{horizon}" in df.columns
                           else horizon).head(n)["feature"].tolist()

    def pruning_candidates(
        self,
        ic_threshold: float = 0.01,
        tstat_threshold: float = 1.5,
        horizon: str = "5d",
    ) -> list[str]:
        """
        Return features with |IC_{horizon}| < ic_threshold AND
        |t-stat_{horizon}| < tstat_threshold — weak candidates for pruning.
        """
        df = self.feature_ic_table()
        ic_col    = f"ic_{horizon}"
        tstat_col = f"tstat_{horizon}"
        if ic_col not in df.columns or tstat_col not in df.columns:
            return []
        mask = (df[ic_col].abs() < ic_threshold) & (df[tstat_col].abs() < tstat_threshold)
        return df[mask]["feature"].tolist()

    def print_summary(self, top_n: int = 20) -> None:
        print("\n" + "=" * 80)
        print(f"FEATURE IC DIAGNOSTIC  [{self.generated_at}]")
        print(f"  {self.n_features} features  ×  {self.n_dates} dates")
        print("=" * 80)
        df = self.feature_ic_table()
        if df.empty:
            print("  (no data)")
        else:
            print(df.head(top_n).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        if self.collinearity_clusters:
            print(f"\n  Collinearity clusters (>{COLLINEARITY_THRESHOLD:.0%} Spearman):")
            for cid, members in self.collinearity_clusters.items():
                print(f"    Cluster {cid}: {members}")
        print("=" * 80 + "\n")

    def save_csv(self, store_root: Path | str) -> Path:
        """Export the feature IC table to CSV for archiving."""
        store_root = Path(store_root)
        eval_dir   = store_root / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        date_str = dt.date.today().isoformat()
        path = eval_dir / f"feature_ic_report_{date_str}.csv"
        df = self.feature_ic_table()
        if not df.empty:
            df.to_csv(path, index=False)
            logger.info("Feature IC report saved → %s", path)
        return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_feature_ic_report(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_cols: list[str] | None = None,
    decay_lags: list[int] | None = None,
    store_root: Path | str | None = None,
    min_stocks: int = 5,
) -> FeatureICReport:
    """
    Compute the full per-feature IC diagnostic report.

    Parameters
    ----------
    panel : pd.DataFrame
        Universe panel with columns [symbol, date, close (optional),
        <feature_cols>, <label_cols>].  One row per (symbol, date).
    feature_cols : list[str]
        Feature columns to evaluate.
    label_cols : list[str] | None
        Label columns.  Defaults to DEFAULT_LABEL_COLS (filters to those
        that actually exist in the panel).
    decay_lags : list[int] | None
        Lags for IC decay curve.  Default DEFAULT_DECAY_LAGS.
    store_root : Path | str | None
        If provided, save the CSV report automatically.
    min_stocks : int
        Minimum stocks per date for a cross-section to be included.

    Returns
    -------
    FeatureICReport
    """
    label_cols = [c for c in (label_cols or DEFAULT_LABEL_COLS) if c in panel.columns]
    decay_lags = decay_lags or DEFAULT_DECAY_LAGS

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date

    report = FeatureICReport(
        generated_at=dt.datetime.now().isoformat(timespec="seconds"),
        n_features=len(feature_cols),
        n_dates=panel["date"].nunique(),
    )

    # Horizon → column name mapping
    horizon_map: dict[str, str] = {}
    for lc in label_cols:
        for h in ["1d", "5d", "10d", "20d"]:
            if lc.endswith(h):
                horizon_map[h] = lc

    logger.info(
        "feature_ic: computing IC for %d features × %d dates × %d labels",
        len(feature_cols), report.n_dates, len(label_cols),
    )

    # --- Per-feature IC and decay ---
    rows: list[FeatureICRow] = []
    for i, feat in enumerate(feature_cols):
        if feat not in panel.columns:
            logger.debug("Skipping feature not in panel: %s", feat)
            continue

        row = FeatureICRow(feature=feat)

        # IC at each horizon
        daily_rics_by_horizon: dict[str, list[float]] = {}
        for h_key, lc in horizon_map.items():
            rics = _daily_rank_ic(panel, feat, lc, min_stocks=min_stocks)
            daily_rics_by_horizon[h_key] = rics
            if rics:
                mean_ic = float(np.mean(rics))
                setattr(row, f"ic_{h_key}", mean_ic)
                setattr(row, f"tstat_{h_key}", _ic_tstat(rics))
                row.n_dates = len(rics)

        # IC decay curve from close prices
        row.decay_curve = _compute_ic_decay_from_close(panel, feat, decay_lags, min_stocks)

        # Decay half-life (using 5d IC as reference)
        peak_ic = row.ic_5d if not np.isnan(row.ic_5d) else row.ic_1d
        row.decay_halflife = _decay_halflife(row.decay_curve, peak_ic)

        rows.append(row)

        if (i + 1) % 10 == 0:
            logger.info("  ... feature IC %d/%d", i + 1, len(feature_cols))

    report.rows = rows

    # --- Collinearity matrix ---
    avail_feats = [r.feature for r in rows if r.feature in panel.columns]
    if len(avail_feats) >= 2:
        feat_data = panel[avail_feats].dropna()
        if not feat_data.empty:
            report.collinearity_matrix = feat_data.corr(method="spearman")
            report.collinearity_clusters = _find_collinearity_clusters(
                report.collinearity_matrix,
                threshold=COLLINEARITY_THRESHOLD,
            )

    # --- Save CSV ---
    if store_root is not None:
        report.save_csv(Path(store_root))

    logger.info(
        "feature_ic: done — %d features, %d collinearity clusters",
        len(rows), len(report.collinearity_clusters),
    )
    return report


def _find_collinearity_clusters(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.85,
) -> dict[int, list[str]]:
    """
    Group features into clusters where any pair has |Spearman corr| > threshold.

    Uses single-linkage clustering:  if A↔B > threshold and B↔C > threshold,
    A, B, C are in the same cluster.

    Returns {cluster_id: [feature_names]}.  Singletons (no collinear partner)
    are omitted.
    """
    features = list(corr_matrix.columns)
    n = len(features)
    union_find: dict[int, int] = {i: i for i in range(n)}

    def _find(x: int) -> int:
        while union_find[x] != x:
            union_find[x] = union_find[union_find[x]]
            x = union_find[x]
        return x

    def _union(a: int, b: int) -> None:
        union_find[_find(a)] = _find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                _union(i, j)

    # Group by cluster root
    from collections import defaultdict
    groups: dict[int, list[str]] = defaultdict(list)
    for i, feat in enumerate(features):
        root = _find(i)
        groups[root].append(feat)

    # Only return clusters with >1 member
    result: dict[int, list[str]] = {}
    for cid, (root, members) in enumerate(
        (k, v) for k, v in groups.items() if len(v) > 1
    ):
        result[cid] = members

    return result
