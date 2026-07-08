from __future__ import annotations

import inspect

import pandas as pd
import pytest

from quant_platform.selection import confidence as confidence_module
from quant_platform.selection.confidence import compute_confidence


def test_flat_score_yields_low_confidence():
    panel = pd.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(20)],
        "fusion_score_col": [0.5] * 20,
    })

    conf, label = compute_confidence(panel)

    assert conf == 0.0
    assert label == "low"


def test_separated_score_yields_high_confidence():
    panel = pd.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(20)],
        "fusion_score_col": [0.0] * 10 + [1.0] * 10,
        "base_pct": [i / 19 for i in range(20)],
        "recent_pct": [i / 19 for i in range(20)],
    })

    conf, label = compute_confidence(panel)

    assert conf > 0.65
    assert label == "high"


def test_regime_hint_nudges_confidence():
    # Graduated (non-extreme) dispersion so confidence sits away from the
    # [0, 1] bounds -- otherwise a nudge has nothing room to move.
    panel = pd.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(20)],
        "fusion_score_col": [i / 19 for i in range(20)],
    })

    conf_none, _ = compute_confidence(panel)
    conf_weak, _ = compute_confidence(panel, regime_hint="weak")
    conf_strong, _ = compute_confidence(panel, regime_hint="strong")

    assert conf_weak < conf_none < conf_strong


def test_missing_score_col_raises():
    panel = pd.DataFrame({"symbol": ["S000"]})
    with pytest.raises(ValueError):
        compute_confidence(panel)


def test_empty_panel_raises():
    panel = pd.DataFrame({"fusion_score_col": []})
    with pytest.raises(ValueError):
        compute_confidence(panel)


def test_pit_canary_no_forward_label_usage():
    # The module source must never reference forward-label columns.
    source = inspect.getsource(confidence_module)
    assert "ret_fwd" not in source
    assert "_fwd_" not in source

    # And it must run fine on a panel that has no label columns at all.
    panel = pd.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(5)],
        "fusion_score_col": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    conf, label = compute_confidence(panel)
    assert 0.0 <= conf <= 1.0
    assert label in ("low", "normal", "high")
