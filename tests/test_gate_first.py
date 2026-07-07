from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def test_coverage_gate_separates_base_and_recent_features():
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
        select_features_by_gate,
    )

    dates = pd.bdate_range("2024-01-01", periods=90).date
    symbols = [f"{i:06d}" for i in range(1, 6)]
    rows = []
    for d in dates:
        for sym in symbols:
            rows.append({
                "symbol": sym,
                "date": d,
                "ma_5": 1.0,
                "cs_main_flow_rank_1d": 0.5,
            })
    panel = pd.DataFrame(rows)

    report = compute_feature_coverage_report(
        panel,
        ["ma_5", "cs_main_flow_rank_1d"],
        family_by_col={"ma_5": "technical", "cs_main_flow_rank_1d": "flow"},
        config=CoverageGateConfig(
            recent_symbol_threshold=4,
            recent_20d_symbol_threshold=4,
            min_recent_trading_days=80,
        ),
    )

    base_cols = select_features_by_gate(report, model_path="base")
    recent_cols = select_features_by_gate(report, model_path="recent")

    assert "ma_5" in base_cols
    assert "cs_main_flow_rank_1d" not in base_cols
    assert "cs_main_flow_rank_1d" in recent_cols
    flow_row = report[report["feature_name"] == "cs_main_flow_rank_1d"].iloc[0]
    assert "short-history-family" in flow_row["rejection_reason"]


def test_coverage_gate_marks_future_fields_prediction_only():
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
    )

    panel = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "date": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
        "future_return_hint": [0.1, -0.1],
    })
    report = compute_feature_coverage_report(
        panel,
        ["future_return_hint"],
        config=CoverageGateConfig(
            recent_symbol_threshold=1,
            recent_20d_symbol_threshold=1,
            min_recent_trading_days=1,
        ),
    )
    row = report.iloc[0]
    assert not row["is_allowed_for_base_model"]
    assert not row["is_allowed_for_recent_model"]
    assert row["is_allowed_for_prediction_only"]
    assert "future-field" in row["rejection_reason"]


def test_gate_first_fusion_tiers_and_rank_order():
    from quant_platform.selection.gate_fusion import gate_first_fusion

    scored = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E", "F"],
        "trade_date": ["2024-01-02"] * 6,
        "base_score": [1.0, 0.8, 0.9, 0.1, 0.2, 0.7],
        "base_pct": [0.90, 0.65, 0.88, 0.30, 0.40, 0.70],
        "recent_score": [0.5, 1.0, 0.1, 0.95, 0.2, 0.8],
        "recent_pct": [0.70, 0.90, 0.20, 0.95, 0.40, 0.80],
        "risk_flags": ["", "", "", "", "", "high_unlock@2024-01-02"],
    })

    fused = gate_first_fusion(scored)
    tier_by_symbol = dict(zip(fused["symbol"], fused["gate_tier"]))

    assert tier_by_symbol["A"] == "A_MAIN"
    assert tier_by_symbol["B"] == "B_SHORT_BOOST"
    assert tier_by_symbol["C"] == "C_DOWNGRADE_OBSERVE"
    assert tier_by_symbol["D"] == "D_OBSERVE"
    assert tier_by_symbol["E"] == "E_REJECT"
    assert tier_by_symbol["F"] == "RISK_VETO"
    assert fused.iloc[0]["symbol"] == "A"
    assert fused["final_rank"].tolist() == list(range(1, len(fused) + 1))


def _typed_panel(cols, n_symbols=5, n_days=90):
    """A fully-populated panel with the given feature columns."""
    dates = pd.bdate_range("2024-01-01", periods=n_days).date
    symbols = [f"{i:06d}" for i in range(1, n_symbols + 1)]
    rows = []
    for d in dates:
        for s in symbols:
            row = {"symbol": s, "date": d}
            for c in cols:
                row[c] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_typed_metadata_gates_base_admission():
    """GF-04: base admission requires registration + pit_safe + a known_at rule.
    Name-token detection remains a secondary guard."""
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
    )
    from quant_platform.features.registry import FeatureMetadata

    def meta(name, *, pit_safe=True, known_at="post_close_T"):
        return FeatureMetadata(
            name=name, family="technical", source="feature_panel",
            pit_safe=pit_safe, known_at=known_at, history_start=None,
        )

    cols = ["ma_5", "leaky_full", "no_rule", "future_return_hint", "unreg"]
    panel = _typed_panel(cols)
    feature_metadata = {
        "ma_5": meta("ma_5"),
        "leaky_full": meta("leaky_full", pit_safe=False),   # innocuous name, unsafe
        "no_rule": meta("no_rule", known_at=""),            # no known_at rule
        "future_return_hint": meta("future_return_hint"),   # metadata "safe" but name leaks
        # "unreg" deliberately absent -> unregistered
    }
    report = compute_feature_coverage_report(
        panel, cols,
        feature_metadata=feature_metadata,
        config=CoverageGateConfig(),
    )
    by = {r["feature_name"]: r for _, r in report.iterrows()}

    assert by["ma_5"]["is_allowed_for_base_model"]
    assert not by["leaky_full"]["is_allowed_for_base_model"]
    assert "not-pit-safe" in by["leaky_full"]["rejection_reason"]
    assert not by["no_rule"]["is_allowed_for_base_model"]
    assert "known-at-missing" in by["no_rule"]["rejection_reason"]
    assert not by["unreg"]["is_allowed_for_base_model"]
    assert by["unreg"]["is_allowed_for_prediction_only"]
    assert "unregistered" in by["unreg"]["rejection_reason"]
    # name-token guard still fires even though the metadata claims pit_safe
    assert not by["future_return_hint"]["is_allowed_for_base_model"]
    assert "future-field" in by["future_return_hint"]["rejection_reason"]


def test_typed_metadata_gates_recent_alpha():
    """GF-04b: recent-alpha admission has the same typed-PIT bar as base --
    registered + pit_safe + known_at, plus name-token guard."""
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
    )
    from quant_platform.features.registry import FeatureMetadata

    def meta(name, *, pit_safe=True, known_at="post_close_T"):
        return FeatureMetadata(
            name=name, family="technical", source="feature_panel",
            pit_safe=pit_safe, known_at=known_at, history_start=None,
        )

    cols = ["ma_5", "leaky_full", "no_rule", "future_return_hint", "unreg"]
    panel = _typed_panel(cols)
    feature_metadata = {
        "ma_5": meta("ma_5"),
        "leaky_full": meta("leaky_full", pit_safe=False),
        "no_rule": meta("no_rule", known_at=""),
        "future_return_hint": meta("future_return_hint"),
    }
    report = compute_feature_coverage_report(
        panel, cols, feature_metadata=feature_metadata, config=CoverageGateConfig()
    )
    by = {r["feature_name"]: r for _, r in report.iterrows()}

    assert by["ma_5"]["is_allowed_for_recent_model"]
    assert not by["leaky_full"]["is_allowed_for_recent_model"]   # pit_safe=False
    assert "not-pit-safe" in by["leaky_full"]["rejection_reason"]
    assert not by["no_rule"]["is_allowed_for_recent_model"]      # known_at missing
    assert "known-at-missing" in by["no_rule"]["rejection_reason"]
    assert not by["unreg"]["is_allowed_for_recent_model"]        # unregistered
    assert not by["future_return_hint"]["is_allowed_for_recent_model"]  # name token


def test_name_tokens_block_base_and_recent_even_if_pit_safe():
    """GF-04b: future/fwd/label/target name tokens block base AND recent even
    when the typed metadata claims pit_safe=True."""
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
    )
    from quant_platform.features.registry import FeatureMetadata

    cols = ["fwd_ret_5d", "target_next", "label_up", "ma_5"]
    panel = _typed_panel(cols)
    feature_metadata = {
        c: FeatureMetadata(c, "technical", "feature_panel", True, "post_close_T", None)
        for c in cols
    }
    report = compute_feature_coverage_report(
        panel, cols, feature_metadata=feature_metadata, config=CoverageGateConfig()
    )
    by = {r["feature_name"]: r for _, r in report.iterrows()}
    for leaky in ("fwd_ret_5d", "target_next", "label_up"):
        assert not by[leaky]["is_allowed_for_base_model"], leaky
        assert not by[leaky]["is_allowed_for_recent_model"], leaky
    assert by["ma_5"]["is_allowed_for_base_model"]
    assert by["ma_5"]["is_allowed_for_recent_model"]


def test_event_family_feature_not_recent_alpha():
    """GF-04b req #3: event-family features never drive recent alpha -- they are
    routed to the risk/veto channel only."""
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
    )
    from quant_platform.features.registry import FeatureMetadata

    panel = _typed_panel(["dt_net_buy"])
    feature_metadata = {
        "dt_net_buy": FeatureMetadata(
            "dt_net_buy", "dragon_tiger", "datacenter-web", True, "announce_date", None
        ),
    }
    report = compute_feature_coverage_report(
        panel, ["dt_net_buy"], feature_metadata=feature_metadata, config=CoverageGateConfig()
    )
    row = report.iloc[0]
    assert not row["is_allowed_for_base_model"]
    assert not row["is_allowed_for_recent_model"]
    assert row["is_allowed_for_prediction_only"]
    assert "event-family-risk-channel-only" in row["rejection_reason"]


def test_event_flag_requires_structured_timing_for_risk_channel():
    """GF-04b req #3: an event flag drives veto/downgrade only when it is a
    registered code carrying an explicit @known_at; an untimed flag does not."""
    from quant_platform.selection.gate_fusion import gate_first_fusion

    scored = pd.DataFrame({
        "symbol": ["TIMED", "UNTIMED"],
        "trade_date": ["2024-01-02"] * 2,
        "base_score": [0.9, 0.9],
        "base_pct": [0.90, 0.90],
        "recent_score": [0.9, 0.9],
        "recent_pct": [0.90, 0.90],
        "risk_flags": ["high_unlock@2024-01-10", "high_unlock"],
    })
    fused = gate_first_fusion(scored)
    tier = dict(zip(fused["symbol"], fused["gate_tier"]))
    assert tier["TIMED"] == "RISK_VETO"          # registered + timed -> veto
    assert tier["UNTIMED"] != "RISK_VETO"        # known_at missing -> not admitted


def test_event_flag_drives_veto_not_recent_pct():
    """GF-04b req #3: a structured event flag routes a name to the veto channel
    without ever altering recent_pct."""
    from quant_platform.selection.gate_fusion import gate_first_fusion

    scored = pd.DataFrame({
        "symbol": ["X"],
        "trade_date": ["2024-01-02"],
        "base_score": [0.9],
        "base_pct": [0.90],
        "recent_score": [0.9],
        "recent_pct": [0.90],
        "event_flags": ["risk_warning@2024-01-02"],
    })
    fused = gate_first_fusion(scored)
    assert fused.iloc[0]["gate_tier"] == "RISK_VETO"
    assert fused.iloc[0]["recent_pct"] == 0.90   # flag did not touch recent_pct


def test_all_feature_families_have_provenance():
    """GF-04 hardening: every spec family is declared in _FAMILY_PROVENANCE, so
    no legitimately-specced feature is silently fail-closed; the check flags an
    unregistered family loudly."""
    from quant_platform.features.registry import (
        build_feature_metadata,
        feature_metadata_lookup,
        _families_without_provenance,
    )

    # all current families are provenanced -> build does not raise, and no
    # entry hit the fail-closed fallback (empty known_at)
    meta = build_feature_metadata()
    assert meta  # non-empty
    assert all(m.known_at for m in meta.values())

    # the pure check catches an unregistered family
    assert _families_without_provenance(["technical", "made_up_family"]) == ["made_up_family"]
    assert _families_without_provenance([m.family for m in meta.values()]) == []

    # feature_metadata_lookup delegates to build and stays consistent
    assert set(feature_metadata_lookup()) == set(meta)


def test_registry_metadata_lookup_covers_known_families():
    """GF-04: the registry exposes typed metadata for real feature columns."""
    from quant_platform.features.registry import feature_metadata_lookup

    meta = feature_metadata_lookup()
    assert "ma_5" in meta
    ma = meta["ma_5"]
    assert ma.family == "technical"
    assert ma.pit_safe is True
    assert ma.known_at  # non-empty rule
    assert meta["volume"].family == "raw_aux"


def test_universe_ratio_threshold_scales_with_universe():
    """GF-06: the coverage threshold is a fraction of the universe, so the same
    absolute coverage passes in a small universe and fails in a larger one."""
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        _effective_threshold,
        compute_feature_coverage_report,
    )

    assert _effective_threshold(0.83, 250, 300) == 0.83 * 300
    assert _effective_threshold(0.83, 250, 1000) == 830.0
    assert _effective_threshold(None, 250, 300) == 250.0     # no ratio -> absolute
    assert _effective_threshold(0.83, 250, 0) == 250.0       # unknown universe -> absolute

    def build(n_symbols, covered):
        dates = pd.bdate_range("2024-01-01", periods=90).date
        symbols = [f"{i:06d}" for i in range(1, n_symbols + 1)]
        rows = []
        for d in dates:
            for i, s in enumerate(symbols):
                val = 1.0 if i < covered else np.nan
                rows.append({"symbol": s, "date": d, "ma_5": val})
        return pd.DataFrame(rows)

    cfg = CoverageGateConfig()  # ratio 0.83
    narrow = compute_feature_coverage_report(
        build(10, 9), ["ma_5"], family_by_col={"ma_5": "technical"}, config=cfg
    )
    wide = compute_feature_coverage_report(
        build(12, 9), ["ma_5"], family_by_col={"ma_5": "technical"}, config=cfg
    )
    assert narrow.iloc[0]["is_allowed_for_base_model"]
    assert not wide.iloc[0]["is_allowed_for_base_model"]
    assert "base-recent-coverage-low" in wide.iloc[0]["rejection_reason"]


def test_untagged_feature_is_fail_closed():
    """GF-04: a feature with no family tag cannot reach the base model."""
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
    )

    dates = pd.bdate_range("2024-01-01", periods=90).date
    symbols = [f"{i:06d}" for i in range(1, 6)]
    rows = [
        {"symbol": s, "date": d, "mystery_feature": 1.0}
        for d in dates
        for s in symbols
    ]
    panel = pd.DataFrame(rows)

    report = compute_feature_coverage_report(
        panel,
        ["mystery_feature"],
        family_by_col={},  # untagged
        config=CoverageGateConfig(
            recent_symbol_threshold=4,
            recent_20d_symbol_threshold=4,
            min_recent_trading_days=80,
        ),
    )
    row = report.iloc[0]
    assert row["feature_family"] == "unknown"
    assert not row["is_allowed_for_base_model"]
    assert not row["is_allowed_for_recent_model"]
    assert row["is_allowed_for_prediction_only"]
    assert "untagged-fail-closed" in row["rejection_reason"]


def test_single_day_coverage_glitch_does_not_flip_gate():
    """GF-06: one empty as-of day must not reject an otherwise well-covered
    stable feature -- the gate uses the 20-day average, not the single-day
    snapshot."""
    from quant_platform.evaluation.coverage_gate import (
        CoverageGateConfig,
        compute_feature_coverage_report,
    )

    dates = list(pd.bdate_range("2024-01-01", periods=90).date)
    symbols = [f"{i:06d}" for i in range(1, 6)]
    rows = []
    for d in dates:
        for s in symbols:
            val = np.nan if d == dates[-1] else 1.0  # glitch on the as-of day
            rows.append({"symbol": s, "date": d, "ma_5": val})
    panel = pd.DataFrame(rows)

    report = compute_feature_coverage_report(
        panel,
        ["ma_5"],
        family_by_col={"ma_5": "technical"},
        config=CoverageGateConfig(
            recent_symbol_threshold=4,
            recent_20d_symbol_threshold=4,
            min_recent_trading_days=80,
        ),
    )
    row = report.iloc[0]
    assert row["recent_symbol_coverage"] == 0   # the glitch day is genuinely empty
    assert row["is_allowed_for_base_model"]     # ...but the gate survives it


def test_fuse_parser_accepts_ranked_inputs():
    from quant_platform.cli import build_parser

    args = build_parser().parse_args([
        "fuse",
        "--base-ranked", "base.csv",
        "--recent-ranked", "recent.csv",
        "--output-dir", "reports",
    ])
    assert args.command == "fuse"
    assert args.prefix == "D3_fused"


def test_tier_map_has_no_dead_last_hole_and_veto_is_last():
    """GF-01/GF-02: strong/mid-base names in UNCLASSIFIED are not dumped below
    reject/veto, and RISK_VETO is strictly last."""
    from quant_platform.selection.gate_fusion import gate_first_fusion

    scored = pd.DataFrame({
        "symbol": ["H1", "H2", "REJ", "VETO"],
        "trade_date": ["2024-01-02"] * 4,
        "base_score": [0.85, 0.70, 0.30, 0.90],
        "base_pct": [0.85, 0.70, 0.30, 0.90],
        "recent_score": [0.45, 0.50, 0.30, 0.90],
        "recent_pct": [0.45, 0.50, 0.30, 0.90],
        "risk_flags": ["", "", "", "high_unlock@2024-01-02"],
    })
    fused = gate_first_fusion(scored)
    tier = dict(zip(fused["symbol"], fused["gate_tier"]))
    rank = dict(zip(fused["symbol"], fused["final_rank"]))

    assert tier["H1"] == "UNCLASSIFIED"
    assert tier["H2"] == "UNCLASSIFIED"
    assert tier["REJ"] == "E_REJECT"
    assert tier["VETO"] == "RISK_VETO"
    # holes rank above reject and veto (not dead last)
    assert rank["H1"] < rank["REJ"] < rank["VETO"]
    assert rank["H2"] < rank["REJ"]
    # veto strictly last
    assert rank["VETO"] == fused["final_rank"].max()


def test_within_tier_sort_is_recent_led_for_d_observe():
    """GF-07: D_OBSERVE (recent strong, base weak) orders by recent_pct, not
    base_pct."""
    from quant_platform.selection.gate_fusion import gate_first_fusion

    scored = pd.DataFrame({
        "symbol": ["Dlow", "Dhigh"],
        "trade_date": ["2024-01-02"] * 2,
        "base_score": [0.45, 0.10],
        "base_pct": [0.45, 0.10],       # Dlow has the higher base
        "recent_score": [0.91, 0.99],
        "recent_pct": [0.91, 0.99],     # Dhigh has the higher recent
    })
    fused = gate_first_fusion(scored)
    assert set(fused["gate_tier"]) == {"D_OBSERVE"}
    # recent-led: Dhigh ranks first despite the lower base_pct
    assert fused.iloc[0]["symbol"] == "Dhigh"


def test_flag_codes_exact_match_not_substring():
    """GF-05: 'coverage_ok' must not trip the 'coverage' downgrade token; a
    parameterized 'unlock:10d' code still downgrades."""
    from quant_platform.selection.gate_fusion import gate_first_fusion

    scored = pd.DataFrame({
        "symbol": ["OK", "SOFT", "COV"],
        "trade_date": ["2024-01-02"] * 3,
        "base_score": [0.9, 0.9, 0.9],
        "base_pct": [0.90, 0.90, 0.90],
        "recent_score": [0.9, 0.9, 0.9],
        "recent_pct": [0.90, 0.90, 0.90],
        "event_flags": ["coverage_ok@2024-01-02", "unlock:10d@2024-01-02", "coverage@2024-01-02"],
    })
    fused = gate_first_fusion(scored)
    tier = dict(zip(fused["symbol"], fused["gate_tier"]))

    assert tier["OK"] == "A_MAIN"            # coverage_ok does not downgrade
    assert tier["SOFT"] == "RISK_DOWNGRADE"  # unlock:10d -> 'unlock' code
    assert tier["COV"] == "RISK_DOWNGRADE"   # bare 'coverage' code


def test_output_pools_separated(tmp_path):
    """GF-03: main ranked CSV holds only actionable A/B tiers; observe/veto/
    reject go to a separate file."""
    from quant_platform.selection.gate_fusion import (
        gate_first_fusion,
        write_gate_fusion_outputs,
    )

    scored = pd.DataFrame({
        "symbol": ["A", "B", "C", "D", "E", "F"],
        "trade_date": ["2024-01-02"] * 6,
        "base_score": [1.0, 0.8, 0.9, 0.1, 0.2, 0.7],
        "base_pct": [0.90, 0.65, 0.88, 0.30, 0.40, 0.70],
        "recent_score": [0.5, 1.0, 0.1, 0.95, 0.2, 0.8],
        "recent_pct": [0.70, 0.90, 0.20, 0.95, 0.40, 0.80],
        "risk_flags": ["", "", "", "", "", "high_unlock@2024-01-02"],
    })
    fused = gate_first_fusion(scored)
    csv_path, _ = write_gate_fusion_outputs(fused, tmp_path, prefix="T")

    ranked = pd.read_csv(csv_path)
    observe = pd.read_csv(tmp_path / "T_observe.csv")

    assert set(ranked["gate_tier"]) <= {"A_MAIN", "B_SHORT_BOOST"}
    assert "RISK_VETO" not in set(ranked["gate_tier"])
    assert "RISK_VETO" in set(observe["gate_tier"])
    # partition: every fused row is in exactly one pool
    assert len(ranked) + len(observe) == len(fused)
