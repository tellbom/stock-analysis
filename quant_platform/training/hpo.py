"""
training.hpo
============
Hyperparameter optimisation with Optuna (T3.1).

Design rules
------------
- Objective = OOF Rank ICIR computed entirely inside the purged CV harness.
  The lockbox is NEVER touched during HPO.
- Studies are persisted to SQLite (same db as MLflow, separate file) so
  they can be resumed after interruption.
- Pruning: MedianPruner with a warm-up period — cheap early stopping on bad trials.
- Each trial is logged to MLflow as a child run of the parent study run.
- The search space is defined in one place (``DEFAULT_SEARCH_SPACE``) and
  can be overridden by the caller — adding a parameter is a one-line change.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from quant_platform.training.lgbm_model import fit_oof
from quant_platform.evaluation.metrics import evaluate
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Default LightGBM search space
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE: dict = {
    "n_estimators":      ("int",   100, 600),
    "learning_rate":     ("float", 0.01, 0.15, True),   # log-scale
    "num_leaves":        ("int",   16, 128),
    "min_child_samples": ("int",   10, 60),
    "subsample":         ("float", 0.5, 1.0),
    "colsample_bytree":  ("float", 0.5, 1.0),
    "reg_alpha":         ("float", 1e-4, 10.0, True),
    "reg_lambda":        ("float", 1e-4, 10.0, True),
}


def _suggest_params(trial: optuna.Trial, space: dict) -> dict:
    """Translate search space dict into Optuna suggestions."""
    params: dict = {"verbose": -1, "n_jobs": -1, "random_state": 42}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            log = len(spec) > 3 and spec[3]
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


# ---------------------------------------------------------------------------
# HPO study
# ---------------------------------------------------------------------------

class HPOStudy:
    """
    Manages one Optuna HPO study for a given panel + feature/label spec.

    Parameters
    ----------
    store_root : Path | str
        Root of the lake — study SQLite is stored at
        ``<store_root>/mlflow/optuna.db``.
    study_name : str
        Unique name for this study (e.g. ``"lgbm_ret_fwd_20d"``).
    n_splits : int
        PurgedKFold folds for each objective evaluation.
    horizon : int
        Label horizon (used for purging).
    embargo : int
        Embargo gap rows.
    search_space : dict | None
        Override ``DEFAULT_SEARCH_SPACE``.
    """

    def __init__(
        self,
        store_root: Path | str,
        study_name: str,
        n_splits:   int = 5,
        horizon:    int = 20,
        embargo:    int = 5,
        search_space: dict | None = None,
    ) -> None:
        self.store_root   = Path(store_root)
        self.study_name   = study_name
        self.n_splits     = n_splits
        self.horizon      = horizon
        self.embargo      = embargo
        self.search_space = search_space or DEFAULT_SEARCH_SPACE

        db_path = self.store_root / "mlflow" / "optuna.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage = f"sqlite:///{db_path}"

    def _make_objective(
        self,
        panel:        pd.DataFrame,
        feature_cols: list[str],
        label_col:    str,
    ):
        """Return an Optuna objective function (closure over panel + cols)."""
        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial, self.search_space)
            try:
                oof = fit_oof(
                    panel, feature_cols, label_col,
                    n_splits=self.n_splits,
                    horizon=self.horizon,
                    embargo=self.embargo,
                    lgbm_params=params,
                )
                if not oof.fold_metrics:
                    return float("nan")

                # Evaluate OOF predictions
                valid = panel[panel[label_col].notna()]
                rep = evaluate(
                    oof.oof_predictions.reindex(valid.index),
                    valid[label_col],
                    pd.to_datetime(valid["date"]),
                    label_col=label_col,
                )
                icir = rep.icir
                # Optuna minimises; we maximise ICIR → return negative
                return -icir if not np.isnan(icir) else float("nan")
            except Exception as exc:
                logger.warning("Trial %d failed: %s", trial.number, exc)
                return float("nan")

        return objective

    def run(
        self,
        panel:        pd.DataFrame,
        feature_cols: list[str],
        label_col:    str,
        n_trials:     int = 30,
        n_jobs:       int = 1,
    ) -> optuna.Study:
        """
        Run HPO for ``n_trials`` trials.

        Studies are persistent — calling run() again resumes from where
        the study left off (same study_name + same SQLite storage).

        Returns the completed Optuna study (best params accessible at
        ``study.best_params``).
        """
        sampler = TPESampler(seed=42)
        pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        study = optuna.create_study(
            direction="minimize",
            storage=self._storage,
            study_name=self.study_name,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        objective = self._make_objective(panel, feature_cols, label_col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                show_progress_bar=False,
            )

        best = study.best_params
        best_icir = -study.best_value
        logger.info(
            "HPO study '%s': %d trials, best ICIR=%.4f, params=%s",
            self.study_name, len(study.trials), best_icir, best,
        )
        return study

    def best_params(self) -> dict:
        """Load the best params from a completed study."""
        study = optuna.load_study(
            study_name=self.study_name, storage=self._storage
        )
        return study.best_params

    def trial_dataframe(self) -> pd.DataFrame:
        """Return all trials as a DataFrame for the leaderboard / ledger."""
        study = optuna.load_study(
            study_name=self.study_name, storage=self._storage
        )
        return study.trials_dataframe()
