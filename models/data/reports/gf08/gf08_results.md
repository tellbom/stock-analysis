# GF-08 Bake-off Results

- decision: **reject**
- gate_icir: 2.591251
- max_alternative_icir: 2.725919
- veto_realized_excess: nan
- best_smooth_arm: fixed_weight_0.3
- gate_top20_net: 0.024167
- best_smooth_top20_net: 0.027998
- wilcoxon_p_gate_vs_best_smooth: 0.202450

## Summary Metrics

| Metric | Arm | Mean | Std | N | ICIR |
|---|---|---:|---:|---:|---:|
| precision_at_20_excess | base_only | 0.570000 | 0.138981 | 20 | nan |
| precision_at_20_excess | fixed_weight_0.3 | 0.607500 | 0.152415 | 20 | nan |
| precision_at_20_excess | fixed_weight_0.5 | 0.602500 | 0.155153 | 20 | nan |
| precision_at_20_excess | fixed_weight_0.7 | 0.615000 | 0.153982 | 20 | nan |
| precision_at_20_excess | gate_first | 0.587500 | 0.144982 | 20 | nan |
| precision_at_20_excess | gate_first_noveto | 0.587500 | 0.144982 | 20 | nan |
| precision_at_20_excess | recent_heavy | 0.607500 | 0.152415 | 20 | nan |
| precision_at_20_excess | recent_only | 0.632500 | 0.153276 | 20 | nan |
| precision_at_20_excess | rrf | 0.602500 | 0.140932 | 20 | nan |
| rank_ic_ret_fwd_3d | base_only | 0.078803 | 0.138680 | 20 | 2.541216 |
| rank_ic_ret_fwd_3d | fixed_weight_0.3 | 0.098176 | 0.166880 | 20 | 2.630953 |
| rank_ic_ret_fwd_3d | fixed_weight_0.5 | 0.096062 | 0.160506 | 20 | 2.676551 |
| rank_ic_ret_fwd_3d | fixed_weight_0.7 | 0.091273 | 0.149742 | 20 | 2.725919 |
| rank_ic_ret_fwd_3d | gate_first | 0.098517 | 0.170026 | 20 | 2.591251 |
| rank_ic_ret_fwd_3d | gate_first_noveto | 0.098517 | 0.170026 | 20 | 2.591251 |
| rank_ic_ret_fwd_3d | recent_heavy | 0.098176 | 0.166880 | 20 | 2.630953 |
| rank_ic_ret_fwd_3d | recent_only | 0.095549 | 0.167628 | 20 | 2.549140 |
| rank_ic_ret_fwd_3d | rrf | 0.106596 | 0.178104 | 20 | 2.676595 |
| top20_ret_fwd_3d_net | base_only | 0.023009 | 0.046786 | 20 | nan |
| top20_ret_fwd_3d_net | fixed_weight_0.3 | 0.027998 | 0.045813 | 20 | nan |
| top20_ret_fwd_3d_net | fixed_weight_0.5 | 0.026878 | 0.047137 | 20 | nan |
| top20_ret_fwd_3d_net | fixed_weight_0.7 | 0.027987 | 0.047229 | 20 | nan |
| top20_ret_fwd_3d_net | gate_first | 0.024167 | 0.048517 | 20 | nan |
| top20_ret_fwd_3d_net | gate_first_noveto | 0.024167 | 0.048517 | 20 | nan |
| top20_ret_fwd_3d_net | recent_heavy | 0.027998 | 0.045813 | 20 | nan |
| top20_ret_fwd_3d_net | recent_only | 0.030117 | 0.041531 | 20 | nan |
| top20_ret_fwd_3d_net | rrf | 0.027400 | 0.045124 | 20 | nan |
| top50_ret_fwd_3d_net | base_only | 0.013615 | 0.037961 | 20 | nan |
| top50_ret_fwd_3d_net | fixed_weight_0.3 | 0.015600 | 0.037527 | 20 | nan |
| top50_ret_fwd_3d_net | fixed_weight_0.5 | 0.015968 | 0.038905 | 20 | nan |
| top50_ret_fwd_3d_net | fixed_weight_0.7 | 0.015668 | 0.039302 | 20 | nan |
| top50_ret_fwd_3d_net | gate_first | 0.015575 | 0.038560 | 20 | nan |
| top50_ret_fwd_3d_net | gate_first_noveto | 0.015575 | 0.038560 | 20 | nan |
| top50_ret_fwd_3d_net | recent_heavy | 0.015600 | 0.037527 | 20 | nan |
| top50_ret_fwd_3d_net | recent_only | 0.015690 | 0.038379 | 20 | nan |
| top50_ret_fwd_3d_net | rrf | 0.016023 | 0.038705 | 20 | nan |
| turnover_top20 | base_only | 0.660526 | nan | 1 | nan |
| turnover_top20 | fixed_weight_0.3 | 0.650000 | nan | 1 | nan |
| turnover_top20 | fixed_weight_0.5 | 0.660526 | nan | 1 | nan |
| turnover_top20 | fixed_weight_0.7 | 0.684211 | nan | 1 | nan |
| turnover_top20 | gate_first | 0.686842 | nan | 1 | nan |
| turnover_top20 | gate_first_noveto | 0.686842 | nan | 1 | nan |
| turnover_top20 | recent_heavy | 0.650000 | nan | 1 | nan |
| turnover_top20 | recent_only | 0.581579 | nan | 1 | nan |
| turnover_top20 | rrf | 0.652632 | nan | 1 | nan |

## Date Runs

| Date | Rows | Label Rows | Base Features | Recent Features |
|---|---:|---:|---:|---:|
| 2026-04-02 | 300 | 300 | 47 | 47 |
| 2026-04-08 | 300 | 300 | 47 | 47 |
| 2026-04-13 | 300 | 300 | 47 | 47 |
| 2026-04-16 | 300 | 300 | 47 | 47 |
| 2026-04-21 | 299 | 299 | 47 | 52 |
| 2026-04-24 | 299 | 299 | 47 | 52 |
| 2026-04-29 | 299 | 299 | 47 | 52 |
| 2026-05-07 | 300 | 300 | 47 | 52 |
| 2026-05-12 | 300 | 300 | 47 | 52 |
| 2026-05-15 | 300 | 300 | 47 | 52 |
| 2026-05-20 | 300 | 300 | 47 | 52 |
| 2026-05-25 | 300 | 300 | 47 | 52 |
| 2026-05-28 | 300 | 300 | 47 | 52 |
| 2026-06-02 | 300 | 300 | 47 | 52 |
| 2026-06-05 | 300 | 300 | 47 | 52 |
| 2026-06-10 | 300 | 300 | 47 | 52 |
| 2026-06-15 | 300 | 299 | 47 | 52 |
| 2026-06-18 | 300 | 299 | 47 | 52 |
| 2026-06-24 | 285 | 284 | 46 | 42 |
| 2026-06-29 | 298 | 298 | 35 | 40 |