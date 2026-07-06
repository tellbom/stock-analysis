# P0 Data Backfill + Gate-First Fusion Plan

## 0. Background

当前项目是 A 股短线量化预测系统，目标偏向 D+1 / D+3 / D+5，当前重点推进 D3。已有审计结论显示：

1. 技术因子、横截面因子、估值/规模、行业相对、融资融券等已有基础能力。
2. 资金面因子已有代码或规格，但本地 `silver/fund_flow` 覆盖严重不足，导致主力净流入、超大单/大单/中单/小单净流入等因子无法稳定进入训练。
3. 基本面因子已有代码或接口预留，但营收、利润、ROE、EPS 等字段覆盖不足，且必须严格使用 `announce_date/pubDate <= as_of_date` 做 PIT join。
4. 解禁、公告、业绩预告、财报事件等事件因子已有方向，但当前 `silver/lockup`、公告事件和业绩事件数据不足，且需要严格区分 `announce_date` 与 `event_date`。
5. 近 100 个交易日资金流虽然历史较短，但对短线 D+1 / D+3 有较高价值；不能简单混入三年长历史主模型中，否则可能让模型学习到“字段是否缺失”这种时间段信号。
6. 第一版融合策略不采用固定比例加权，也不以 RRF 作为主方案，而采用 Gate-first 路线：长历史模型做稳定底座，近窗口模型做短线确认、增强、降级或风控否决。

本计划目标是让 Codex 在当前项目中完成 P0 改造：先补数据、建立覆盖率监控、预留近窗口增强模型与 Gate 融合路径。不要直接重构主模型，不要把新因子强行塞入长历史主模型。

---

## 1. Overall Goals

本次 P0 开发目标分为五层：

```text
数据源接入
  ↓
silver parquet 统一落库
  ↓
coverage gate 覆盖率与 PIT 检查
  ↓
base / recent enhanced 双特征集路径
  ↓
Gate-first 融合输出
```

### 1.1 必须完成

1. 新增或修正资金流 collector。
2. 新增或修正基本面 collector。
3. 新增或修正解禁事件 collector。
4. 新增或修正业绩预告 / 财报事件 collector。
5. 所有外部数据必须统一落到项目自己的 `silver` parquet 数据层。
6. 新增覆盖率报告和训练前门禁。
7. 明确区分长历史 base feature set 与近窗口 recent enhanced feature set。
8. 设计并实现第一版 Gate-first 融合逻辑。
9. 输出清晰的运行命令、报告文件和验收结果。

### 1.2 本阶段不做

1. 不重构 LightGBM 核心训练逻辑。
2. 不修改标签定义。
3. 不把资金流等短历史因子直接强行接入三年长历史主模型。
4. 不做公告 PDF 解析、OCR、RAG、LLM 情绪分析。
5. 不把 cookie、token、账号密码写入代码。
6. 不采用绕过鉴权或反爬的方案作为默认数据源。
7. 不直接复制第三方项目大量代码；只允许通过依赖、适配器、字段映射或接口封装实现。

---

## 2. Code Discovery Requirements

执行前请先完整扫描当前项目结构，定位以下模块或等价实现：

1. 数据目录：

   * `models/data/silver/`
   * `models/data/features/`
   * `models/data/labels/`
   * `models/data/universe/`
   * 如项目实际路径不同，以实际路径为准。

2. 当前 collector / ingest 模块：

   * 检查是否已有 `fund_flow`、`fundamentals`、`lockup`、`announcement`、`events` 相关 collector。
   * 如果已有，优先在原模块最小改造。
   * 如果没有，再新增对应 collector。

3. 当前 feature pipeline：

   * 检查是否已有 `build_flow_features`。
   * 检查是否已有 `build_fundamental_features`。
   * 检查是否已有 `build_lockup_features`。
   * 检查是否已有 event feature builder。
   * 检查 registry/specs 中是否包含这些因子。
   * 检查 `FULL_SPECS` 是否包含对应因子族。
   * 检查 model CLI 是否暴露 `include-flow`、`include-fundamentals`、`include-lockup`、`include-events` 等参数。

4. 当前训练脚本：

   * 定位 D3 训练脚本。
   * 定位长历史训练逻辑。
   * 定位预测输出脚本。
   * 定位最终 ranked.csv / report 生成逻辑。
   * 定位最终 `X_train.columns` 输出位置，如无则补充。

5. 当前缺失率过滤：

   * 搜索类似逻辑：

```python
if panel[c].isna().mean() < 0.90:
    valid_cols.append(c)
```

* 不要删除原逻辑，但需要扩展为更完整的 coverage gate。

---

## 3. P0 Data Source Strategy

### 3.1 Fund Flow

优先数据源：

1. AKShare `stock_individual_fund_flow`
2. 可选增强：`a-stock-data` 或 `adata` 作为 P1 交叉验证源，不作为第一版唯一主依赖。

资金流第一版只要求覆盖最近约 100 个交易日。该数据可以用于短线近窗口模型，但不直接进入三年长历史主模型。

目标输出目录：

```text
models/data/silver/fund_flow/{symbol}.parquet
```

统一 schema：

```text
symbol
trade_date
close
pct_change
main_net
main_net_rate
super_net
super_net_rate
large_net
large_net_rate
medium_net
medium_net_rate
small_net
small_net_rate
source
raw_update_time
fetched_at
```

字段要求：

1. `trade_date` 必须标准化为交易日期。
2. 金额字段必须统一单位，明确是“元 / 万元 / 亿元”。
3. 比率字段必须统一为小数或百分比，并在 schema 文档中说明。
4. 原始字段如果无法映射，保留在 raw 层或附加 metadata，不直接进入训练。
5. 每只股票一个 parquet，便于增量更新。
6. 允许最近 100 日历史，不要求多年历史。
7. 必须限流、重试、失败记录。

资金流覆盖率门槛：

```text
最近有效交易日覆盖股票数 >= 250/300
最近 20 个交易日平均覆盖股票数 >= 250/300
最新可用日期不晚于最近交易日 T-1 或 T-2
字段缺失率不可异常升高
```

若资金流只覆盖最近 100 天，不应以三年全历史 panel 缺失率作为唯一入模判断。

---

### 3.2 Fundamentals

优先数据源：

1. BaoStock `query_profit_data`
2. BaoStock `query_growth_data`
3. 可选增强：AKShare 财务指标、a-stock-data 季报快照。

目标输出目录：

```text
models/data/silver/fundamentals/{symbol}.parquet
```

统一 schema：

```text
symbol
period_end
announce_date
report_type
roe
eps
net_profit
revenue
gross_margin
net_margin
revenue_yoy
profit_yoy
source
raw_update_time
fetched_at
```

关键要求：

1. BaoStock 的 `pubDate` 映射为 `announce_date`。
2. BaoStock 的 `statDate` 映射为 `period_end`。
3. 所有基本面 join 必须使用：

```text
announce_date <= as_of_date
```

4. 禁止使用仅按 `period_end` 对齐的逻辑。
5. 如果某些字段没有公告日期，默认不得进入训练。
6. 基本面第一版更多作为质量过滤、风格暴露或辅助因子，不作为 D3 主 alpha。

基本面覆盖率门槛：

```text
覆盖 CSI300 >= 250/300
核心字段 announce_date 非空率接近 100%
核心字段 period_end 非空率接近 100%
至少覆盖最近多个报告期
```

---

### 3.3 Lockup / Restricted Shares

优先数据源：

1. AKShare `stock_restricted_release_detail_em`
2. 可选增强：a-stock-data 解禁日历。

目标输出目录：

```text
models/data/silver/lockup/{symbol}.parquet
```

统一 schema：

```text
symbol
unlock_date
announce_date
lockup_type
unlock_shares
actual_unlock_shares
unlock_market_value
unlock_float_mcap_ratio
source
raw_update_time
fetched_at
```

关键要求：

1. 必须区分：

   * `announce_date`：市场知道这件事的日期。
   * `unlock_date`：解禁实际发生日期。
2. 构造特征时只能使用：

```text
announce_date <= as_of_date
unlock_date > as_of_date
```

3. 如果数据源没有明确公告日期，必须标注 PIT 风险，默认只能用于预测侧监控或风控，不进入训练。
4. 明确黑名单字段：

   * 解禁后 20 日涨跌幅
   * 公告后涨跌幅
   * 事件后收益
   * future_return
   * 任何未来表现字段

第一版解禁因子：

```text
days_to_next_unlock
unlock_size_ratio
has_unlock_in_3d
has_unlock_in_5d
has_unlock_in_10d
```

---

### 3.4 Earnings Forecast / Financial Report Events

优先数据源：

1. AKShare 业绩预告相关接口。
2. AKShare 财报披露预约相关接口。
3. 可选增强：巨潮公告、a-stock-data cninfo 公告。

目标输出目录：

```text
models/data/silver/announcement_events/{symbol}.parquet
```

统一 schema：

```text
symbol
event_type
event_date
announce_date
period_end
title
forecast_type
forecast_change_pct
forecast_lower_bound
forecast_upper_bound
report_disclosure_date
source
raw_update_time
fetched_at
```

关键要求：

1. 必须保留 `announce_date`。
2. 必须保留 `event_date`。
3. 对于财报披露预约：

   * `event_date` 可以是预约披露日期。
   * `announce_date` 是该预约信息被市场知道的日期，如没有则标注 PIT 风险。
4. 第一版只做结构化事件，不做 NLP。

第一版事件因子：

```text
has_announcement_3d
announcement_count_5d
has_earnings_forecast_30d
earnings_forecast_type
earnings_forecast_change_pct
days_since_earnings_forecast
days_to_next_report
```

---

## 4. Silver Layer Rules

所有 collector 输出必须满足：

```text
source
fetched_at
raw_update_time
```

如无 `raw_update_time`，可为空，但必须保留字段。

所有日期字段必须标准化：

```text
YYYY-MM-DD
```

所有 symbol 必须与项目现有 symbol 标准一致，例如：

```text
600000.SH
000001.SZ
```

如果外部接口返回：

```text
600000
sh600000
600000.SH
```

必须统一映射为项目内部标准。

禁止外部接口结果直接进入训练矩阵。

正确路径：

```text
external source
  ↓
collector
  ↓
silver parquet
  ↓
feature builder
  ↓
feature panel
  ↓
coverage gate
  ↓
training / prediction
```

---

## 5. Coverage Gate Design

当前项目如果只使用整体缺失率：

```text
overall_missing_rate < 0.90
```

不足以处理近 100 日资金流数据。

请新增或扩展 coverage gate，至少输出以下指标：

```text
feature_name
feature_family
source
latest_available_date
overall_missing_rate
recent_symbol_coverage
recent_20d_avg_symbol_coverage
available_trading_days
date_coverage_rate
is_allowed_for_base_model
is_allowed_for_recent_model
is_allowed_for_prediction_only
rejection_reason
```

### 5.1 Base Model Gate

长历史 base model 只允许使用覆盖稳定的因子。

建议规则：

```text
overall_missing_rate <= 0.30
available_trading_days 覆盖长历史训练窗口大部分日期
recent_symbol_coverage >= 250/300
无 PIT 风险
无未来字段
```

资金流近 100 日数据默认不进入 base model。

---

### 5.2 Recent Enhanced Model Gate

近窗口 enhanced model 允许使用短历史因子，但必须在近窗口内覆盖足够。

建议规则：

```text
训练窗口 = 最近 100 / 120 / 180 个交易日
recent_symbol_coverage >= 250/300
recent_20d_avg_symbol_coverage >= 250/300
available_trading_days >= 80
latest_available_date 不晚于 T-1 或 T-2
无 PIT 风险
无未来字段
```

资金流可以进入 recent enhanced model。

---

### 5.3 Prediction Only Gate

如果某个因子：

1. 有预测侧数据。
2. 历史训练不足。
3. PIT 信息不完整。
4. 覆盖率不足以训练。

则标记为：

```text
is_allowed_for_prediction_only = true
```

用于报告、风控提示或观察，不进入训练。

---

## 6. Feature Set Architecture

请预留两个 feature set 路径：

### 6.1 Base Feature Set

用途：长历史稳定模型。

训练窗口：

```text
近 2～3 年或项目现有长历史窗口
```

允许因子：

```text
技术因子
横截面技术/量能
估值/规模
行业相对
融资融券
其他长历史覆盖稳定因子
```

默认不包含：

```text
近 100 日资金流
短历史公告事件
PIT 风险未解决的解禁事件
覆盖不足基本面字段
```

输出命名建议：

```text
base_feature_set_id
```

---

### 6.2 Recent Enhanced Feature Set

用途：近窗口短线增强模型。

训练窗口：

```text
最近 100 / 120 / 180 个交易日
```

允许因子：

```text
base 稳定技术因子
资金流因子
近期事件因子
解禁因子
业绩预告因子
短线行业/概念资金因子
```

输出命名建议：

```text
recent_enhanced_feature_set_id
```

要求：

1. 明确记录训练窗口起止日期。
2. 明确记录新因子覆盖率。
3. 明确记录哪些因子进入 `X_train.columns`。
4. 明确记录哪些因子被排除及原因。

---

## 7. Model Strategy

本阶段不要把双模型融合写成固定比例模型。

目标结构：

```text
base_model_d3
  ↓
base_score / base_rank / base_pct

recent_model_d3
  ↓
recent_score / recent_rank / recent_pct

gate_fusion
  ↓
final_ranked.csv
```

### 7.1 Base Model

标签：

```text
ret_fwd_3d
```

输入：

```text
base_feature_set
```

输出：

```text
D3_base_ranked.csv
```

必须输出字段：

```text
symbol
trade_date
base_score
base_rank
base_pct
```

---

### 7.2 Recent Enhanced Model

标签：

```text
ret_fwd_3d
```

输入：

```text
recent_enhanced_feature_set
```

训练窗口：

```text
最近 100 / 120 / 180 个交易日
```

输出：

```text
D3_recent_ranked.csv
```

必须输出字段：

```text
symbol
trade_date
recent_score
recent_rank
recent_pct
```

---

## 8. Gate-First Fusion Design

第一版不采用固定比例加权，不采用 RRF 作为默认主方案。

Gate 的核心思想：

```text
长历史模型定方向
近窗口模型做确认 / 增强 / 降级
事件因子做风险否决
```

### 8.1 Required Inputs

融合模块输入：

```text
symbol
trade_date
base_score
base_rank
base_pct
recent_score
recent_rank
recent_pct
risk_flags
event_flags
```

计算字段：

```text
short_boost = recent_pct - base_pct
```

---

### 8.2 Gate Tier Rules

第一版建议规则如下，可放入配置文件，避免写死：

#### Tier A: 主推荐

条件：

```text
base_pct >= 0.80
recent_pct >= 0.60
无重大 event risk
```

含义：

```text
长模型强，短线模型也确认。
```

处理：

```text
进入主推荐池。
```

排序：

```text
优先 base_pct，其次 recent_pct，其次 short_boost。
```

---

#### Tier B: 短线增强候选

条件：

```text
base_pct >= 0.60
recent_pct >= 0.85
无重大 event risk
```

含义：

```text
长模型不差，短线资金/事件明显增强。
```

处理：

```text
进入增强候选池。
```

排序：

```text
优先 recent_pct，其次 base_pct。
```

---

#### Tier C: 长模型强但短线弱，降级观察

条件：

```text
base_pct >= 0.80
recent_pct < 0.35
```

含义：

```text
长历史模型看好，但近窗口资金/事件不确认。
```

处理：

```text
从主推荐降级为观察。
```

---

#### Tier D: 短模型强但长模型弱，只观察

条件：

```text
recent_pct >= 0.90
base_pct < 0.50
```

含义：

```text
短线模型很强，但长历史底座不支持，可能是短期噪声。
```

处理：

```text
只进入观察池，不进入主推荐。
```

---

#### Tier E: 双弱剔除

条件：

```text
base_pct < 0.60
recent_pct < 0.60
```

处理：

```text
剔除。
```

---

#### Tier Risk: 风控否决

触发条件示例：

```text
未来 3/5 日存在高比例解禁
近期有风险提示公告
业绩预告重大下修
资金流极端流出
数据源 PIT 风险未解决
```

处理：

```text
直接剔除或降级为观察。
```

输出：

```text
risk_flags
gate_tier = RISK_VETO 或 RISK_DOWNGRADE
gate_reason
```

---

### 8.3 Fusion Output

输出文件：

```text
D3_fused_ranked.csv
D3_fusion_report.md
```

输出字段：

```text
symbol
trade_date
base_score
base_rank
base_pct
recent_score
recent_rank
recent_pct
short_boost
gate_tier
gate_reason
risk_flags
event_flags
final_rank
```

报告必须解释：

1. 主推荐池数量。
2. 短线增强候选数量。
3. 长强短弱降级数量。
4. 短强长弱观察数量。
5. 风控否决数量。
6. Top10 / Top20 / Top50 列表。
7. 每只 Top 股票的 gate_reason。

---

## 9. RRF Position

RRF 本阶段只作为可选对照实验，不作为默认主方案。

原因：

1. recent model 训练窗口短，还没有足够 walk-forward 证明稳定性。
2. RRF 会让 recent model 对最终排序产生较强影响，容易放大短期噪声。
3. Gate 更适合当前“base 稳定、recent 短线确认”的非对等模型结构。
4. Gate 更容易表达事件风险否决逻辑。

如实现 RRF，请放在实验模块中，默认关闭。

建议输出：

```text
D3_rrf_experiment_ranked.csv
```

但不要替代：

```text
D3_fused_ranked.csv
```

---

## 10. CLI / Script Design

请结合现有 CLI 风格新增或扩展命令。

建议命令形态，仅供参考，以项目现有规范为准：

### 10.1 Backfill Fund Flow

```bash
python -m quant_platform.cli backfill-fund-flow --universe csi300 --days 120 --source akshare
```

### 10.2 Backfill Fundamentals

```bash
python -m quant_platform.cli backfill-fundamentals --universe csi300 --source baostock
```

### 10.3 Backfill Lockup

```bash
python -m quant_platform.cli backfill-lockup --universe csi300 --source akshare
```

### 10.4 Backfill Events

```bash
python -m quant_platform.cli backfill-events --universe csi300 --source akshare
```

### 10.5 Build Coverage Report

```bash
python -m quant_platform.cli coverage-report --universe csi300 --as-of 2026-07-03
```

### 10.6 Build Recent Enhanced Feature Set

```bash
python -m quant_platform.cli build-features --universe csi300 --mode recent-enhanced --window 120 --include-flow --include-events --include-lockup
```

### 10.7 Train Base D3

```bash
python -m quant_platform.cli train-d3 --mode base --feature-set <base_feature_set_id>
```

### 10.8 Train Recent Enhanced D3

```bash
python -m quant_platform.cli train-d3 --mode recent-enhanced --feature-set <recent_feature_set_id> --window 120
```

### 10.9 Run Gate Fusion

```bash
python -m quant_platform.cli fuse-d3 --method gate --base-report <path> --recent-report <path>
```

---

## 11. Reports To Generate

必须生成以下报告：

### 11.1 p0_data_backfill_plan.md

包含：

1. 数据源选择。
2. collector 文件清单。
3. 字段映射。
4. silver schema。
5. PIT 规则。
6. 黑名单字段。
7. 回填命令。
8. 验收标准。

---

### 11.2 p0_data_coverage_report.md

包含：

1. 资金流覆盖情况。
2. 基本面覆盖情况。
3. 解禁覆盖情况。
4. 业绩事件覆盖情况。
5. 最新可用日期。
6. 最近 20 日横截面覆盖。
7. 缺失率。
8. 是否允许进入 base model。
9. 是否允许进入 recent enhanced model。
10. 是否仅允许 prediction-only。

---

### 11.3 feature_gate_report.md

包含：

1. 每个候选特征。
2. 所属因子族。
3. 缺失率。
4. 横截面覆盖。
5. 可用交易日数量。
6. 是否进入 base model。
7. 是否进入 recent enhanced model。
8. 排除原因。

---

### 11.4 d3_gate_fusion_report.md

包含：

1. base model 信息。
2. recent model 信息。
3. Gate 规则配置。
4. 各 gate tier 数量。
5. 风控否决数量。
6. TopK 排名。
7. 每个 TopK 的 gate_reason。
8. 是否建议继续使用 Gate 作为默认融合方案。

---

## 12. Tests And Validation

### 12.1 Unit Tests

新增或补充测试：

1. symbol 标准化测试。
2. 日期字段标准化测试。
3. fund_flow 字段映射测试。
4. fundamentals `pubDate/statDate` 映射测试。
5. lockup `announce_date/unlock_date` 映射测试。
6. event `announce_date/event_date` 映射测试。
7. blacklist 字段过滤测试。
8. Gate tier 规则测试。

---

### 12.2 PIT Tests

必须覆盖：

1. 基本面：`announce_date > as_of_date` 的记录不能 join 到当前样本。
2. 解禁：`announce_date > as_of_date` 的未来补录事件不能使用。
3. 解禁：`unlock_date <= as_of_date` 的已发生解禁不应作为未来风险事件。
4. 业绩事件：公告日期晚于样本日期的事件不能进入样本。

---

### 12.3 Coverage Tests

必须覆盖：

1. 最近交易日覆盖不足 250/300 时，资金流不能进入训练。
2. 最近 20 日平均覆盖不足 250/300 时，资金流不能进入训练。
3. 只有预测侧数据但训练历史不足时，标记 prediction-only。
4. 近 100 日资金流不能被误判为长历史 base model 可用因子。

---

### 12.4 Fusion Tests

必须覆盖：

1. `base_pct >= 0.80` 且 `recent_pct >= 0.60` → Tier A。
2. `base_pct >= 0.60` 且 `recent_pct >= 0.85` → Tier B。
3. `base_pct >= 0.80` 且 `recent_pct < 0.35` → Tier C。
4. `recent_pct >= 0.90` 且 `base_pct < 0.50` → Tier D。
5. 双弱 → Tier E。
6. 风险事件触发 → Risk veto or downgrade。

---

## 13. Acceptance Criteria

P0 验收标准：

### 13.1 数据层

1. `silver/fund_flow` 覆盖 CSI300 最近有效交易日至少 250/300。
2. `silver/fund_flow` 最近 20 个交易日平均覆盖不少于 250/300。
3. `silver/fundamentals` 覆盖 CSI300 至少 250/300。
4. `silver/fundamentals` 每条有效记录必须有 `announce_date` 和 `period_end`。
5. `silver/lockup` 至少能产生 parquet 文件，并保留 `announce_date` 和 `unlock_date`。
6. `silver/announcement_events` 至少能产生 parquet 文件，并保留 `announce_date` 和 `event_date`。
7. 所有新数据都有 `source` 和 `fetched_at`。

---

### 13.2 安全与 PIT

1. 所有未来结果字段进入 blacklist。
2. 基本面 join 使用 `announce_date <= as_of_date`。
3. 解禁 join 使用 `announce_date <= as_of_date` 且 `unlock_date > as_of_date`。
4. 业绩事件 join 使用 `announce_date <= as_of_date`。
5. 无 cookie/token 硬编码。
6. 无绕过鉴权或反爬逻辑作为默认实现。

---

### 13.3 模型与融合

1. base model 与 recent enhanced model 路径清晰分离。
2. 近 100 日资金流默认不得进入三年长历史 base model。
3. recent enhanced model 可以在近窗口内使用资金流和事件因子。
4. Gate fusion 能输出 `gate_tier`、`gate_reason`、`risk_flags`。
5. 默认融合方法为 Gate，不是固定比例加权，不是 RRF。
6. RRF 如实现，仅作为实验输出，默认关闭。

---

## 14. Claude Review Handoff

代码完成后，请准备给 Claude 的评审材料，包括：

1. `p0_data_backfill_plan.md`
2. `p0_data_coverage_report.md`
3. `feature_gate_report.md`
4. `d3_gate_fusion_report.md`
5. 新增 collector 文件清单。
6. 修改的 feature pipeline 文件清单。
7. 修改的 CLI / scripts 文件清单。
8. 新增测试文件清单。
9. 关键运行命令。
10. 关键日志截图或文本。
11. 需要 Claude 重点评审的问题：

```text
1. 是否存在未来函数或 PIT 泄漏？
2. 资金流近 100 日数据是否被错误纳入长历史 base model？
3. coverage gate 是否足够防止低覆盖因子入模？
4. Gate fusion 是否清晰可解释？
5. event risk 是否能正确降级或否决？
6. collector 是否存在反爬、token、cookie、硬编码风险？
7. silver schema 是否稳定，是否能支持后续历史补齐？
8. 训练最终 X_train.columns 是否能证明新因子真实进入 recent enhanced model？
```

---

## 15. Implementation Order

建议执行顺序：

```text
Step 1: 阅读项目 README / CLAUDE.md / 现有 pipeline 文档
Step 2: 定位现有 collector、feature、training、report 模块
Step 3: 输出 p0_data_backfill_plan.md
Step 4: 实现 fund_flow collector
Step 5: 实现 fundamentals collector
Step 6: 实现 lockup collector
Step 7: 实现 announcement_events collector
Step 8: 统一 silver schema 与 symbol/date 标准化
Step 9: 实现 coverage report
Step 10: 实现 feature gate
Step 11: 预留 base / recent enhanced feature set
Step 12: 实现 recent enhanced model 训练路径
Step 13: 实现 Gate-first fusion
Step 14: 新增测试
Step 15: 生成全部报告
Step 16: 准备 Claude review handoff
```

---

## 16. Final Principle

本次 P0 的核心不是“立刻提升模型分数”，而是建立一条可靠的数据与融合链路：

```text
先保证数据真实、覆盖、可追溯
再保证 PIT 不泄漏
再保证新因子进入正确模型路径
最后用 Gate 让短线资金/事件模型影响最终排序
```

最终目标是：

```text
长历史模型负责稳定底座
近窗口模型负责短线确认
事件因子负责风险否决
Gate 输出最终可解释排序
```
