# Gate-First 融合方案与测评数据评审稿

本文档用于交给 Claude 评审当前 Gate 方案是否合理，重点不是复述全部实现，而是把“方案意图、数据证据、测评口径、关键代码节选、待审问题”放在同一处。

相关主文档：`docs/gate.md`

## 1. 当前结论摘要

当前 D3 方案采用 Gate-first，而不是固定比例加权，也不是默认 RRF。

核心分工：

- `base_model_d3`：长历史稳定底座，使用覆盖稳定、PIT 风险较低的因子。
- `recent_enhanced_model_d3`：近窗口增强模型，允许短历史但最近覆盖足够的资金流、事件类因子进入。
- `gate_first_fusion`：用 recent 对 base 做确认、增强、降级或观察；事件/风险标记可否决。

当前已有两条评审样本：

| 路径 | 报告 | 重点 |
|---|---|---|
| 资金流增强 Gate | `models/data/reports/D3_gate_emdatah5_fund_flow_run_report_2026-07-03.md` | 验证 `fund_flow` 只进 recent，不进 base |
| Event3 增强 Gate | `models/data/reports/D3_gate_event3_run_report_2026-07-03.md` | 验证公告、龙虎榜、大宗交易只进 recent，并参与风险否决 |

## 2. 数据与模型路径

```text
silver 数据层
  -> feature panel
  -> coverage gate
  -> base feature set / recent feature set
  -> base_model_d3 / recent_enhanced_model_d3
  -> Gate-first fusion
  -> D3_gate_fused_ranked_YYYY-MM-DD.csv
```

关键隔离原则：

1. 资金流、事件等短历史因子不得默认进入长历史 base model。
2. 只有 recent coverage gate 通过后，短历史因子才进入 recent model。
3. 带未来收益、事后表现、PIT 风险标记的字段不得进入训练。
4. risk/event flags 不作为普通 alpha 混入排序，而是在 Gate 阶段降级或否决。

## 3. Coverage Gate 口径

实现文件：`quant_platform/evaluation/coverage_gate.py`

当前 gate 是“数据形态门禁”，不是 alpha 判断器。它判断某个特征可以进入 base、recent，还是只能 prediction-only 观察。

关键阈值：

```python
@dataclass(frozen=True)
class CoverageGateConfig:
    recent_window_days: int = 120
    recent_symbol_threshold: int = 250
    recent_20d_symbol_threshold: int = 250
    base_missing_threshold: float = 0.30
    latest_lag_days: int = 2
    min_recent_trading_days: int = 80
```

base model 允许条件节选：

```python
base_allowed = (
    stable_family
    and overall_missing <= cfg.base_missing_threshold
    and recent_symbol_coverage >= cfg.recent_symbol_threshold
    and not has_future_field
    and not has_pit_risk
)
```

recent model 允许条件节选：

```python
recent_allowed = (
    (stable_family or short_family)
    and recent_symbol_coverage >= cfg.recent_symbol_threshold
    and recent_20d_avg >= cfg.recent_20d_symbol_threshold
    and available_days >= cfg.min_recent_trading_days
    and latest_ok
    and not has_future_field
    and not has_pit_risk
)
```

短历史 family 包括：

```python
SHORT_HISTORY_FAMILIES = {
    "flow",
    "sector_flow",
    "concept_flow",
    "proxy_flow",
    "event",
    "announcement",
    "announcement_events",
    "dragon_tiger",
    "block_trade",
}
```

需要 Claude 重点评审：

- `recent_window_days=120` 是否适合 D3；是否应改成交易日窗口而不是自然日窗口。
- `recent_symbol_threshold=250/300` 是否足够，是否需要按实际 universe 动态比例。
- `overall_missing_rate` 对短历史资金流会很高，但 recent coverage 仍可通过，这个设计是否合理。
- PIT 风险当前主要靠字段名 token 和上游 builder 保证，是否需要更强的元数据约束。

## 4. Gate-First 融合规则

实现文件：`quant_platform/selection/gate_fusion.py`

配置节选：

```python
@dataclass(frozen=True)
class GateFusionConfig:
    tier_a_base_pct: float = 0.80
    tier_a_recent_pct: float = 0.60
    tier_b_base_pct: float = 0.60
    tier_b_recent_pct: float = 0.85
    tier_c_base_pct: float = 0.80
    tier_c_recent_pct: float = 0.35
    tier_d_recent_pct: float = 0.90
    tier_d_base_pct: float = 0.50
    tier_e_base_pct: float = 0.60
    tier_e_recent_pct: float = 0.60
```

核心逻辑节选：

```python
df["short_boost"] = df["recent_pct"] - df["base_pct"]

if veto:
    tiers.append("RISK_VETO")
elif downgrade:
    tiers.append("RISK_DOWNGRADE")
elif base_pct >= cfg.tier_a_base_pct and recent_pct >= cfg.tier_a_recent_pct:
    tiers.append("A_MAIN")
elif base_pct >= cfg.tier_b_base_pct and recent_pct >= cfg.tier_b_recent_pct:
    tiers.append("B_SHORT_BOOST")
elif base_pct >= cfg.tier_c_base_pct and recent_pct < cfg.tier_c_recent_pct:
    tiers.append("C_DOWNGRADE_OBSERVE")
elif recent_pct >= cfg.tier_d_recent_pct and base_pct < cfg.tier_d_base_pct:
    tiers.append("D_OBSERVE")
elif base_pct < cfg.tier_e_base_pct and recent_pct < cfg.tier_e_recent_pct:
    tiers.append("E_REJECT")
else:
    tiers.append("UNCLASSIFIED")
```

最终排序节选：

```python
df = df.sort_values(
    ["_tier_priority", "base_pct", "recent_pct", "short_boost"],
    ascending=[True, False, False, False],
)
df["final_rank"] = range(1, len(df) + 1)
```

当前 tier 优先级：

```python
_TIER_PRIORITY = {
    "A_MAIN": 0,
    "B_SHORT_BOOST": 1,
    "C_DOWNGRADE_OBSERVE": 2,
    "D_OBSERVE": 3,
    "RISK_DOWNGRADE": 4,
    "E_REJECT": 5,
    "RISK_VETO": 6,
    "UNCLASSIFIED": 7,
}
```

需要 Claude 重点评审：

- `RISK_VETO` 当前排序优先级在 `E_REJECT` 之后、`UNCLASSIFIED` 之前，虽然不是 Top 池，但语义上是否应放到最后。
- `C_DOWNGRADE_OBSERVE` 和 `D_OBSERVE` 仍排在 `E_REJECT` 前，这符合观察池逻辑，但是否应从最终推荐 CSV 中另行分层输出。
- `UNCLASSIFIED` 放在最后是否过严；某些中间状态是否应有更明确 tier。
- `A_MAIN` 内部排序优先 `base_pct`，`B_SHORT_BOOST` 内部目前也仍按统一排序字段排序，是否应对 B 单独优先 `recent_pct`。

## 5. 当前测评数据样本

### 5.1 资金流增强 Gate 样本

报告：`models/data/reports/D3_gate_emdatah5_fund_flow_run_report_2026-07-03.md`

关键结果：

```text
requested_as_of_date: 2026-07-06
actual_as_of_date: 2026-07-03
actual date reason: requested_as_of_date 2026-07-06 has OHLCV coverage 0/300;
using latest complete date 2026-07-03 with coverage 298/300
label: ret_fwd_3d
```

资金流覆盖：

```text
silver/fund_flow files: 300
symbol_file_coverage: 300/300
as_of_files: 299
latest_min: 2026-06-26
latest_max: 2026-07-03
```

feature gate：

```text
base candidates: 52; base admitted: 35
recent candidates: 57; recent admitted: 40
fund flow entered recent model: yes
fund flow forbidden from base model: yes
fund_flow coverage gate:
  recent_symbol_coverage: 299
  recent_20d_avg_symbol_coverage: 292.1
  available_trading_days: 130
  is_allowed_for_recent_model: True
```

模型训练：

```text
base_model_d3:
  train_start: 2023-01-03
  train_end: 2026-06-29
  prediction_date: 2026-07-03
  x_train_rows: 250172
  feature_count: 35

recent_enhanced_model_d3:
  train_start: 2025-12-25
  train_end: 2026-06-29
  prediction_date: 2026-07-03
  x_train_rows: 35874
  feature_count: 40
```

Gate 结果：

```text
tier counts:
  E_REJECT: 140
  UNCLASSIFIED: 85
  A_MAIN: 49
  C_DOWNGRADE_OBSERVE: 9
  B_SHORT_BOOST: 8
  D_OBSERVE: 7
risk veto count: 0
top20 intersections:
  base_recent_top20: 3
  base_gate_top20: 16
  recent_gate_top20: 5
  all_three_top20: 3
```

解读：资金流 recent 模型能影响最终排序，但 Top20 仍主要由 base 强且 recent 确认的股票构成，符合“base 定方向，recent 做确认/增强”的非对等融合思路。

### 5.2 Event3 增强 Gate 样本

报告：`models/data/reports/D3_gate_event3_run_report_2026-07-03.md`

事件数据覆盖：

```text
announcement_events: files 300/300, as_of_files 6, latest_max 2026-07-03
dragon_tiger: files 300/300, as_of_files 8, latest_max 2026-07-03
block_trade: files 300/300, as_of_files 15, latest_max 2026-07-03
```

feature gate：

```text
base candidates: 50; base admitted: 35
recent candidates: 69; recent admitted: 54
announcement_events entered recent model: yes
dragon_tiger entered recent model: yes
block_trade entered recent model: yes
event3 features entered base model: none
```

进入 recent 的 event3 特征包括：

```text
announcement_count_3d
announcement_count_5d
announcement_count_10d
has_announcement_3d
has_major_event_10d
has_risk_announcement_5d
has_financial_report_30d
has_reduction_notice_30d
has_dragon_tiger_5d
dragon_tiger_count_10d
dragon_tiger_net_buy_5d
dragon_tiger_net_buy_rank_5d
institution_net_buy_5d
institution_net_buy_rank_5d
block_trade_count_20d
block_trade_amount_20d
block_trade_amount_rank_20d
block_trade_discount_mean_20d
has_large_discount_block_trade_20d
```

模型训练：

```text
base_model_d3:
  train_start: 2023-01-03
  train_end: 2026-06-29
  x_train_rows: 250172
  feature_count: 35

recent_enhanced_model_d3:
  train_start: 2025-12-25
  train_end: 2026-06-29
  x_train_rows: 35874
  feature_count: 54
```

Gate 结果：

```text
tier counts:
  E_REJECT: 127
  UNCLASSIFIED: 91
  A_MAIN: 46
  RISK_VETO: 10
  C_DOWNGRADE_OBSERVE: 9
  B_SHORT_BOOST: 8
  D_OBSERVE: 7
risk veto count: 10
top20 intersections:
  base_recent_top20: 1
  base_gate_top20: 13
  recent_gate_top20: 1
  all_three_top20: 0
```

解读：事件类增强模型对 Top 池更克制，主要价值在风险否决和降级。需要重点评审的是事件稀疏性是否会导致 recent 模型学习不稳定，以及 `risk_flags` 是否足够 PIT-safe。

## 6. 测评与验收建议

建议 Claude 按以下层次评审，而不是只看 Top20：

1. PIT 检查

   - 资金流 T 日数据是否确认为收盘后可用；若用于预测 T+1/T+3，训练样本的 as-of 是否严格一致。
   - 公告、龙虎榜、大宗交易是否有明确 `announce_date` 或交易后发布时间约束。
   - `risk_flags` 是否仅由 as-of 当时可知数据构造。

2. Feature Gate 检查

   - 查看 `D3_base_X_train_columns_2026-07-03.csv`，确认 flow/event3 不在 base。
   - 查看 `D3_recent_*_X_train_columns_2026-07-03.csv`，确认 flow/event3 只在 recent。
   - 查看 `D3_recent_coverage_gate_2026-07-03.csv`，确认 recent coverage 满足阈值。

3. Fusion 规则检查

   - 验证 `base_pct`、`recent_pct` 的 percentile 方向一致，越大越好。
   - 验证 `short_boost = recent_pct - base_pct` 没有被当成收益或概率解释。
   - 验证 `RISK_VETO` 不会进入主推荐池。

4. OOS 测评检查

   - Gate 方案当前样本是单日推荐融合，不足以证明 alpha。
   - 应补充多日期 walk-forward 或 live multi-date harness，对比 base-only、recent-only、Gate-first、RRF experiment。
   - 指标至少包括 D3 Rank IC、TopK hit/excess、turnover、风险否决后的收益差异。

## 7. 建议补充的对照实验

| 实验 | 目的 |
|---|---|
| base-only vs Gate-first | 判断 Gate 是否比长历史底座更好 |
| recent-only vs Gate-first | 判断 recent 是否过拟合短期噪声 |
| Gate-first vs fixed weighted ensemble | 验证不采用固定比例加权是否合理 |
| Gate-first vs RRF experiment | 验证 RRF 是否会过度放大 recent |
| with risk veto vs without risk veto | 衡量事件风险否决是否有真实价值 |
| flow-only recent vs event3-only recent vs flow+event3 | 拆分资金流和事件增强来源 |

建议输出格式：

```text
date
method
rank_ic_d3
top20_excess_d3
top50_excess_d3
hit_rate_top20
turnover_top20
risk_veto_count
risk_veto_realized_excess
base_recent_top20_intersection
```

## 8. 关键文件清单

代码：

- `quant_platform/selection/gate_fusion.py`
- `quant_platform/evaluation/coverage_gate.py`
- `scripts/run_d3_gate_once.py`
- `scripts/run_d3_gate_event3_once.py`
- `quant_platform/cli.py` 中 `fuse` 子命令
- `tests/test_gate_first.py`

报告与数据：

- `models/data/reports/D3_gate_emdatah5_fund_flow_run_report_2026-07-03.md`
- `models/data/reports/D3_gate_event3_run_report_2026-07-03.md`
- `models/data/reports/D3_base_coverage_gate_2026-07-03.csv`
- `models/data/reports/D3_recent_coverage_gate_2026-07-03.csv`
- `models/data/reports/D3_base_X_train_columns_2026-07-03.csv`
- `models/data/reports/D3_recent_emdatah5_X_train_columns_2026-07-03.csv`
- `models/data/reports/D3_recent_event3_X_train_columns_2026-07-03.csv`
- `models/data/reports/D3_gate_fused_ranked_2026-07-03.csv`

## 9. 请 Claude 重点回答的问题

1. Gate-first 作为默认融合方案是否比固定比例加权/RRF 更合理？
2. 当前 coverage gate 是否足够阻止短历史资金流或事件因子误入 base model？
3. `recent_window_days=120`、`min_recent_trading_days=80`、`250/300` 覆盖阈值是否合理？
4. 事件类特征稀疏时，进入 recent model 是否会带来过拟合或不稳定？
5. `RISK_VETO`、`RISK_DOWNGRADE` 的排序优先级是否应调整为更严格的末尾输出？
6. `A_MAIN`、`B_SHORT_BOOST`、`C_DOWNGRADE_OBSERVE`、`D_OBSERVE` 的阈值是否符合 D3 短线推荐语义？
7. 当前单日 Gate 样本是否只能作为 smoke test；正式结论是否必须依赖多日期 OOS 对照实验？
8. 是否需要把 feature family、source purity、PIT-safe 元数据写入更强 schema，而不是靠字段名 token 和上游约定？

## 10. 我的初步判断

方案方向是合理的：短历史资金流和事件数据不强行混入长历史 base，而是通过 recent model 与 Gate 影响最终排序，这比固定加权更可解释，也更容易表达风险否决。

主要风险不在 Gate 代码本身，而在三处：

1. PIT 证明链还需要更硬的元数据和测试。
2. 单日样本不能证明 Gate 有稳定增益。
3. `RISK_VETO` 和观察池的最终排序/输出语义需要更清晰，避免被误读成可直接买入排序。

