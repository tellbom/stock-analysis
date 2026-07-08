# GF-08 Phase A Smoke Acceptance

- command: `.venv/bin/python scripts/run_d3_gate_once.py`
- requested_as_of_date: `2026-07-06`
- actual_as_of_date: `2026-07-03`
- recent_status: `trained`

## Checks

| Check | Status | Evidence |
|---|---|---|
| base/recent/gate outputs written | PASS | `D3_base_ranked_2026-07-03.csv`, `D3_recent_ranked_2026-07-03.csv`, `D3_gate_fused_ranked_2026-07-03.csv` |
| ranked output contains only A/B tiers | PASS | `D3_gate_fused_2026-07-03_ranked.csv` tiers: `A_MAIN`, `B_SHORT_BOOST` |
| observe output contains non-actionable tiers | PASS | `C_DOWNGRADE_OBSERVE`, `D_OBSERVE`, `E_REJECT`, `UNCLASSIFIED` |
| `RISK_VETO` max final rank in observe pool | NOT OBSERVED | no `RISK_VETO` rows on 2026-07-03 |
| timed veto/downgrade fires | NOT OBSERVED | `_risk_flags` produced no actionable risk rows; `silver/lockup` has `0/300` files |
| event-family features excluded from recent model | PASS | no recent feature names matching event/unlock/announcement/dragon/block/risk tokens |
| typed rejection reasons visible in coverage reports | NOT OBSERVED | current candidate set did not emit the expected rejection reason tokens |

## Finding

The smoke run proved the base/recent/gate routing and output split, but did not
exercise the veto/downgrade channel because the current local data lake has no
lockup files. Phase B proceeds with this limitation recorded; veto value should
be interpreted as unavailable or zero-trigger in this sample, not as evidence
that the veto channel is effective.
