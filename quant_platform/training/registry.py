"""
training.registry
=================
Model registry with lineage and model cards (T3.7).

Uses the MLflow Model Registry (v3.x alias API) for versioning.
Every registered model carries:
  - Full lineage JSON: data_snapshot_id → feature_set_id → label_col →
    model_name → run_id → metrics
  - A model card (structured markdown) describing the model, its intended
    use, limitations, and evaluation summary.

Champion promotion uses the alias API:
  - ``champion`` alias points to the currently promoted model version.
  - Previous champion is re-aliased to ``retired_v{N}``.
  - Promotion state is queryable without loading the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


def _client(store_root: Path | str) -> MlflowClient:
    """Return an MLflow client pointing at the project SQLite."""
    db = Path(store_root) / "mlflow" / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    return MlflowClient()


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def _build_model_card(
    model_name:     str,
    run_id:         str,
    feature_set_id: str,
    label_col:      str,
    eval_metrics:   dict,
    lineage:        dict,
    caveats:        list[str] | None = None,
) -> str:
    """Return a model card as a markdown string."""
    caveats_text = "\n".join(f"- {c}" for c in (caveats or [
        "Signal is based on historical A-share data; past performance does not guarantee future results.",
        "Transaction costs and market impact are approximated; live costs may differ.",
        "Universe is CSI 300 (current constituents); survivorship bias may be present.",
    ]))
    metrics_text = "\n".join(f"- **{k}**: {v}" for k, v in eval_metrics.items())

    return f"""# Model Card: {model_name}

## Overview
| Field | Value |
|---|---|
| Model name | `{model_name}` |
| Run ID | `{run_id}` |
| Feature set | `{feature_set_id}` |
| Label | `{label_col}` |

## Evaluation Metrics
{metrics_text}

## Lineage
```json
{json.dumps(lineage, indent=2)}
```

## Intended Use
Cross-sectional equity ranking for A-share universe (CSI 300).
Produces a daily score per stock; higher score = higher predicted forward return.

## Limitations and Caveats
{caveats_text}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_model(
    store_root:     Path | str,
    model_name:     str,
    run_id:         str,
    feature_set_id: str,
    label_col:      str,
    eval_metrics:   dict,
    lineage:        dict,
    registered_name: str = "quant_platform_model",
) -> str:
    """
    Register a model version in the MLflow Model Registry.

    Returns the model version string (e.g. ``"1"``).
    """
    client = _client(store_root)

    # Log model card as artifact on the run
    card_text = _build_model_card(
        model_name, run_id, feature_set_id, label_col, eval_metrics, lineage
    )
    card_path = Path(store_root) / "mlflow" / "tmp" / f"model_card_{run_id[:8]}.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card_text)
    mlflow.set_tracking_uri(f"sqlite:///{Path(store_root)/'mlflow'/'mlflow.db'}")
    client.log_artifact(run_id, str(card_path))

    # Log lineage JSON
    lineage_path = card_path.parent / f"lineage_{run_id[:8]}.json"
    lineage_path.write_text(json.dumps(lineage, indent=2, default=str))
    client.log_artifact(run_id, str(lineage_path))

    # Register the run as a model version
    model_uri  = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, registered_name)
        version = mv.version
        logger.info(
            "Registered '%s' as version %s (run=%s)",
            registered_name, version, run_id[:8],
        )
    except mlflow.exceptions.MlflowException:
        # Model artifact may not have been logged — register metadata only
        try:
            client.create_registered_model(registered_name)
        except Exception:
            pass
        mv = client.create_model_version(
            name=registered_name, source=model_uri, run_id=run_id
        )
        version = mv.version
        logger.info("Created model version %s for %s", version, registered_name)

    # Store lineage as model version tag
    client.set_model_version_tag(
        registered_name, version, "feature_set_id", feature_set_id
    )
    client.set_model_version_tag(
        registered_name, version, "label_col", label_col
    )
    client.set_model_version_tag(
        registered_name, version, "icir",
        str(round(eval_metrics.get("icir", float("nan")), 4)),
    )

    return version


def promote_champion(
    store_root:      Path | str,
    version:         str,
    registered_name: str = "quant_platform_model",
) -> None:
    """
    Promote *version* to the ``champion`` alias.
    Retires the previous champion to ``retired_v{old_version}``.
    """
    client = _client(store_root)

    # Retire previous champion if one exists
    try:
        prev = client.get_model_version_by_alias(registered_name, "champion")
        client.set_registered_model_alias(
            registered_name, f"retired_v{prev.version}", prev.version
        )
        logger.info(
            "Retired previous champion: version %s → alias 'retired_v%s'",
            prev.version, prev.version,
        )
    except mlflow.exceptions.MlflowException:
        pass  # no previous champion

    client.set_registered_model_alias(registered_name, "champion", version)
    logger.info("Promoted version %s to champion alias", version)


def load_champion(
    store_root:      Path | str,
    registered_name: str = "quant_platform_model",
) -> tuple[str, dict]:
    """
    Return (version_str, tags_dict) for the current champion.
    Raises MlflowException if no champion is registered.
    """
    client = _client(store_root)
    mv = client.get_model_version_by_alias(registered_name, "champion")
    # MLflow 3.x: tags is already a dict
    tags = mv.tags if isinstance(mv.tags, dict) else {t.key: t.value for t in (mv.tags or [])}
    return mv.version, tags


def get_model_card(
    store_root:      Path | str,
    run_id:          str,
) -> str | None:
    """Return the model card text for a given run_id, or None if not found."""
    card_path = Path(store_root) / "mlflow" / "tmp" / f"model_card_{run_id[:8]}.md"
    if card_path.exists():
        return card_path.read_text()
    return None
