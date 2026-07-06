# Alternative A-Share Data Source Research

> **Date**: 2026-07-06
> **Context**: 除 Tushare 以外的 A 股数据补齐方案调研，目标补齐资金面、基本面、事件因子三类数据缺口。
> **Status**: 调研报告，待评审。未修改业务代码。

---

## 1. Executive Summary

### 核心结论

当前项目已有直接对接东方财富 push2his / datacenter 的 collector（`flow_collector.py`、`lockup_collector.py`、`fundamentals_collector.py`），不需要引入重量级第三方库替换现有链路。补齐策略是：**为现有 collector 增加 fallback 数据源 + 扩展新数据类型（公告、财报事件）的 collector**。

### 最值得接入的数据源（按优先级）

| 优先级 | 数据源 | 用途 | 理由 |
|--------|--------|------|------|
| **P0** | **AKShare** | 资金流 / 基本面 / 解禁 / 公告 | 覆盖面最广，接口最丰富，已在 fundamentals_collector 中部分使用 |
| **P0** | **a-stock-data** | 资金流 120 日 / 季度快照 / 解禁日历 / 巨潮公告 | 零 akshare 依赖，直连 HTTP，2026 年极活跃，v3.3.0 |
| **P0** | **BaoStock** | 基本面（季频盈利能力/成长能力）带 `pubDate` | PIT-safe 天然支持，免费无需注册，数据质量好 |
| **P1** | **adata** | 资金流 / 基本面 / 解禁交叉验证 | 多源融合，可作为交叉验证源 |
| **P2** | **mootdx** | 离线财务文件（gpcw*.zip） | 通达信 TCP 直连，无 IP 封禁风险，适合批量下载历史 |
| **P3** | **efinance** | 历史资金流向 | 作者已宣布退役，不推荐长期依赖 |
| **P3** | **easyquotation** | 实时行情验证 | 仅限实时快照，无历史深度 |
| **P3** | **qstock** | 问财选股 + 资金流排名 | 部分功能需付费会员，不推荐核心链路依赖 |
| **P3** | **pywencai / iwencai-cli** | 自然语言查询 | 需 Cookie / Node.js，反爬风险高，仅限人工验证 |

### 不建议接入的源

- **efinance**: 作者已宣布退出（2025-06），服务端计划未启动，反爬风险极高。
- **easyquotation**: 仅实时快照，无历史数据，无法做因子回填。
- **pywencai**: 依赖 Cookie + Node.js 解密，反爬严格，不适合自动化。

---

## 2. Source Comparison Table

| source | github | data_provider | covers_fund_flow | covers_fundamental | covers_events | stability | automation_fit | legal_risk | recommendation |
|--------|--------|---------------|------------------|--------------------|----------------|-----------|----------------|------------|----------------|
| **AKShare** | [akfamily/akshare](https://github.com/akfamily/akshare) | 东方财富/新浪/同花顺/巨潮 | ✅ 主力/超大单/大单/中单/小单 | ✅ 营收/净利润/ROE/EPS/三表 | ✅ 解禁/公告/业绩预告/快报 | ⭐⭐⭐⭐ 活跃 | ⭐⭐⭐⭐ 适合 | MIT，数据源反爬风险 | **P0 主力** |
| **a-stock-data** | [simonlin1212/a-stock-data](https://github.com/simonlin1212/a-stock-data) | mootdx/腾讯/东财/巨潮/新浪/同花顺 | ✅ 120日个股资金流 | ✅ 季报37字段/新浪三表/F10 | ✅ 解禁日历(未来90天)/巨潮全量公告 | ⭐⭐⭐⭐⭐ 极活跃 | ⭐⭐⭐⭐⭐ 极适合 | Apache-2.0，多源防封 | **P0 主力** |
| **BaoStock** | [BaoStock](http://baostock.com) | 自建数据库 | ❌ 无资金流数据 | ✅ ROE/净利/EPS(TTM)/毛利率/营收（带 pubDate） | ❌ 无 | ⭐⭐⭐⭐ 稳定 | ⭐⭐⭐⭐ 适合 | 免费无注册，无合规风险 | **P0 基本面** |
| **adata** | [1nchaos/adata](https://github.com/1nchaos/adata) | 东财/同花顺/百度/腾讯/新浪 | ✅ 分钟级/日级/概念板块资金流 | ⚠️ 仅核心指标 | ⚠️ 解禁(近1月) | ⭐⭐⭐ 活跃 | ⭐⭐⭐ 可接受 | Apache-2.0，多源代理 | **P1 交叉验证** |
| **qstock** | [tkfy920/qstock](https://github.com/tkfy920/qstock) | 东财/同花顺/新浪 | ✅ 日内/历史/N日/同花顺/北向资金流 | ✅ 指标/报表/EPS预测 | ⚠️ 龙虎榜/异动/新闻 | ⭐⭐⭐ 一般 | ⭐⭐ 受限（部分付费） | MIT，部分功能需会员 | **P3 参考** |
| **mootdx** | [mootdx/mootdx](https://github.com/mootdx/mootdx) | 通达信(TDX) TCP | ❌ 无资金流 | ✅ 财务zip文件(利润/资产/现金流表) | ❌ 无 | ⭐⭐ 低活跃 | ⭐⭐⭐ 可接受 | MIT，TCP直连无封IP | **P2 备份** |
| **pywencai** | [zsrl/pywencai](https://github.com/zsrl/pywencai) | 同花顺问财 | ⚠️ 可通过自然语言查询 | ⚠️ 可通过自然语言查询 | ⚠️ 可通过自然语言查询 | ⭐⭐ 反爬风险 | ⭐ 不适合自动化 | MIT，作者不赞成商用，需Cookie | **P3 人工验证** |
| **iwencai-cli** | [shaw-baobao/iwencai-cli](https://github.com/shaw-baobao/iwencai-cli) | 同花顺问财 (Playwright) | ⚠️ 可通过自然语言查询 | ⚠️ 可通过自然语言查询 | ⚠️ 可通过自然语言查询 | ⭐⭐⭐ 较稳定 | ⭐⭐ 需Chrome | MIT，需真实浏览器 | **P3 人工验证** |
| **efinance** | [Micro-sheep/efinance](https://github.com/Micro-sheep/efinance) | 东方财富 | ✅ 历史资金流向(get_history_bill) | ❌ 无 | ❌ 无 | ⭐ 已退役 | ⭐ 不可用 | MIT，作者已退出 | **不推荐** |
| **easyquotation** | [shidenggui/easyquotation](https://github.com/shidenggui/easyquotation) | 新浪/腾讯/集思录 | ❌ 仅实时快照 | ❌ 无 | ❌ 无 | ⭐⭐⭐ 稳定 | ⭐⭐ 仅实时 | MIT，纯公开接口 | **不推荐** |

---

## 3. Fund Flow Data Plan

### 3.1 推荐数据源

| 优先级 | 数据源 | 接口 | 覆盖内容 |
|--------|--------|------|----------|
| **Primary** | AKShare `stock_individual_fund_flow()` | 东方财富 datacenter | 主力净流入-净额/净占比、超大单/大单/中单/小单净流入-净额/净占比，近100交易日 |
| **Primary** | a-stock-data 资金面层 | 东方财富 push2 | 120日个股资金流，主力/大单/中单/小单分类 |
| **Fallback** | adata `stock.market.get_capital_flow()` | 东方财富 | 日级历史资金流 |
| **Cross-check** | AKShare `stock_fund_flow_individual(symbol="即时")` | 同花顺 | 当日实时资金流排行，含净额 |

### 3.2 字段映射到 `silver/fund_flow`

现有 schema（`flow_collector.py`）：

```
symbol, date, main_net, small_net, mid_net, large_net, super_net
```

AKShare `stock_individual_fund_flow` 字段映射：

| AKShare field | silver field | 备注 |
|---------------|-------------|------|
| 日期 | date | YYYY-MM-DD |
| 主力净流入-净额 | main_net | 主力 = 超大单 + 大单 |
| 超大单净流入-净额 | super_net | |
| 大单净流入-净额 | large_net | |
| 中单净流入-净额 | mid_net | |
| 小单净流入-净额 | small_net | |
| 主力净流入-净占比 | main_net_pct | **新增字段** |
| 收盘价 | close | **新增字段**（数据完整性校验用） |
| 涨跌幅 | pct_change | **新增字段** |

a-stock-data 资金流字段类似，均来自东方财富 push2，口径一致。

### 3.3 数据补齐策略

```
现状: flow_collector.py → push2his (120日窗口)
问题: 部分 CSI300 成分股 push2his 返回空/不完整
方案:
  1. Primary: 保持现有 push2his 链路
  2. Fallback: 对 push2his 空/缺失的 symbol，用 AKShare stock_individual_fund_flow 补齐
  3. Merge: 去重后按 date 排序写入 silver/fund_flow/<symbol>.parquet
  4. 行业/概念资金流: 新增 ak.stock_sector_fund_flow_rank() 作为 market-wide 特征
```

### 3.4 预期覆盖率

- CSI300 个股资金流: ≥ 280/300 (push2his + AKShare fallback)
- 最新日期: ≤ T-1（当日资金流 T 日收盘后可用）

---

## 4. Fundamental Data Plan

### 4.1 推荐数据源

| 优先级 | 数据源 | 接口 | 覆盖内容 | PIT 支持 |
|--------|--------|------|----------|----------|
| **Primary** | BaoStock `query_profit_data()` | 自建库 | ROE(平均)、净利润、EPS(TTM)、毛利率、营收、总股本 | ✅ `pubDate` + `statDate` |
| **Primary** | BaoStock `query_growth_data()` | 自建库 | 净利润同比、净资产同比、总资产同比增长率 | ✅ `pubDate` + `statDate` |
| **Primary** | AKShare `stock_financial_analysis_indicator()` | 新浪财经 | EPS/ROE/毛利率/净利率/资产负债率等 30+ 指标 | ⚠️ 仅报告期，需补充公告日期 |
| **Primary** | AKShare `stock_yjyg_em()` / `stock_yjkb_em()` | 东方财富 | 业绩预告 / 业绩快报（含公告日期） | ✅ 有 `公告日期` |
| **扩展** | a-stock-data 季报快照 | 东方财富 | 37字段季度快照包含 EPS/ROE/净利润/主营收入 | ⚠️ 需确认是否有公告日期 |
| **扩展** | a-stock-data 新浪三表 | 新浪财经 | 资产负债表/利润表/现金流量表 | ⚠️ 需确认 PIT |
| **备份** | mootdx `Affair.parse()` | 通达信 | gpcw*.zip 财务文件（利润表/资产负债表/现金流） | ❌ 仅报告期，无公告日期 |

### 4.2 避免未来函数（PIT 设计）

当前 `fundamentals_collector.py` 已有的 PIT 设计原则继续沿用并强化：

```
必填字段:
  announce_date    ← 财报披露日期（JOIN KEY，不能为 null）
  period_end       ← 报告期截止日（e.g. 2025-03-31）

数据接入规则:
  1. 所有基本面数据必须以 announce_date 为准入键
  2. 没有 announce_date 的源只作为 secondary check
  3. BaoStock 的 pubDate 可直接映射为 announce_date
  4. AKShare stock_yjyg_em / stock_yjkb_em 的 公告日期 → announce_date
  5. 无法确定 announce_date 的数据禁止进入训练特征
```

### 4.3 数据补齐策略

```
现状: fundamentals_collector.py → AKShare (多 endpoint)
问题: 覆盖率不足，缺失率高
方案:
  1. Primary fundamentals (ROE/EPS/净利润/毛利率):
     BaoStock query_profit_data + query_growth_data
     优势: pubDate 天然支持 PIT，免费，稳定
  2. 扩展指标 (资产负债率/周转率/偿债能力等):
     AKShare stock_financial_analysis_indicator
     注意: 需额外查公告日期，不能仅用 period_end
  3. 条件补齐 (业绩预告/快报):
     AKShare stock_yjyg_em / stock_yjkb_em (已有 announce_date)
  4. Merge: 多源按 announce_date 合并，标注 source 字段
  5. PIT 过滤: 写 Parquet 前检查 announce_date 不为 null
```

### 4.4 预期覆盖率

- CSI300 至少 250/300 有 ROE/EPS/净利润数据
- announce_date 覆盖率 100%（无 announce_date 的行不入库）
- 最新财报周期: ≤ T-1（考虑公告延迟通常 1-4 周）

---

## 5. Event Data Plan

### 5.1 推荐数据源

| 数据类型 | 优先级 | 数据源 | 接口 | PIT 支持 |
|----------|--------|--------|------|----------|
| **限售解禁** | P0 | a-stock-data 信号层 | 解禁日历 (历史 + 未来90天预警) | ✅ 解禁日期天然 PIT-safe |
| **限售解禁** | P0 | AKShare `stock_restricted_release_detail_em()` | 东方财富 | ✅ 解禁日期天然 PIT-safe |
| **限售解禁** | P1 | adata `sentiment.stock_lifting_last_month()` | 同花顺 | ⚠️ 仅近1月 |
| **全量公告** | P0 | a-stock-data 公告层 | 巨潮 cninfo 沪深北全量公告 | ✅ 公告日期可映射 |
| **全量公告** | P1 | AKShare `stock_notice_report()` | 巨潮 cninfo | ⚠️ 需验证字段 |
| **业绩预告** | P0 | AKShare `stock_yjyg_em()` | 东方财富 | ✅ 有公告日期 |
| **业绩快报** | P0 | AKShare `stock_yjkb_em()` | 东方财富 | ✅ 有公告日期 |
| **财报披露计划** | P1 | AKShare / 巨潮 cninfo | — | ⚠️ 需确认披露日程表 |
| **风险警示** | P2 | a-stock-data 舆情层 | 扫雷功能 | ⚠️ 来源为通达信 |

### 5.2 事件因子构造

#### 5.2.1 解禁事件因子 → `silver/lockup`

现有 `lockup_collector.py` 增强方案：

```python
# 字段（现有 + 新增）
symbol, unlock_date, lock_type, shares_million, ratio_pct, announce_date

# 新增事件因子（在 features/event.py 或 labels/ 中构造）:
days_to_next_unlock   # 距下一解禁日天数（负数=已过，正数=未到）
unlock_size_ratio     # 解禁量 / 流通股本（已有时 ratio_pct）
unlock_weighted_score # 解禁规模加权评分
days_since_last_unlock # 距上一解禁日天数
```

#### 5.2.2 公告事件因子 → `silver/announcement_events`（新建）

```python
# schema
symbol, announce_date, title, category, pdf_url, source

# 事件因子
has_announcement_3d      # 近3日是否有公告
has_announcement_5d      # 近5日是否有公告
has_announcement_20d     # 近20日是否有公告
announcement_count_5d    # 近5日公告数
announcement_count_20d   # 近20日公告数
has_risk_warning_5d      # 近5日是否有风险警示公告
```

#### 5.2.3 财报事件因子 → `silver/financial_events`（新建）

```python
# schema
symbol, announce_date, period_end, event_type, source
# event_type: "业绩预告", "业绩快报", "年报", "一季报", "中报", "三季报"

# 事件因子
days_to_next_fin_report   # 距下一财报披露日天数
has_earnings_surprise     # 业绩预告是否超预期（预告 vs 一致预期）
days_since_last_fin_report # 距上次财报披露天数
```

### 5.3 数据补齐策略

```
1. 解禁数据:
   - Primary: a-stock-data 解禁日历 (history + 90d forward)
   - Fallback: AKShare stock_restricted_release_detail_em
   - 写入 silver/lockup/<symbol>.parquet

2. 公告数据 (新建 collector):
   - Primary: a-stock-data 巨潮 cninfo 全量公告
   - 增量: 每日拉取所有 CSI300 成分股的最新公告
   - 写入 silver/announcement_events/<symbol>.parquet
   - 注意: 仅写 announce_date ≥ T 日收盘后的公告，避免未来函数

3. 财报事件 (新建 collector):
   - 业绩预告: AKShare stock_yjyg_em (已有)
   - 业绩快报: AKShare stock_yjkb_em (已有)
   - 财报计划: 从 cninfo 或东方财富财报披露日程获取
   - 写入 silver/financial_events/<symbol>.parquet

4. PIT 过滤规则:
   - 所有事件因子构造时，as_of 日期 < announce_date 的行必须过滤
   - 构造事件因子使用 "截至 T 日已知信息" 原则
   - 训练前 verify: 对每个 (symbol, date) 检查事件因子是否使用了未来数据
```

---

## 6. Risk Assessment

### 6.1 反爬与限流风险

| 风险等级 | 数据源 | 风险描述 | 缓解措施 |
|----------|--------|----------|----------|
| 🔴 高 | efinance (东财) | IP 限流封禁常态化，作者已退出 | **不接入** |
| 🔴 高 | pywencai (同花顺问财) | 需 Cookie + JS 解密，反爬严格 | **仅人工验证** |
| 🟡 中 | AKShare (东财模块) | 东财 API 有风控 (>5次/秒触发临时封禁) | 沿用现有 `_em_get()` 限流，≥1s per call |
| 🟡 中 | a-stock-data (东财模块) | 同上，已内置 em_get() 防封 | 串行请求，控制并发 |
| 🟡 中 | adata (多源) | 依赖多个公开接口，任一源变化可能影响数据 | 已有动态代理切换 |
| 🟢 低 | BaoStock | 自建数据库，无反爬 | 正常使用即可 |
| 🟢 低 | mootdx | TCP 直连通达信服务器，无 IP 频率限制 | 正常使用即可 |
| 🟢 低 | easyquotation | 新浪/腾讯公开接口 | 控制请求频率 |

### 6.2 License 与商业使用风险

| 数据源 | License | 商业使用 | 备注 |
|--------|---------|----------|------|
| AKShare | MIT | ✅ 允许 | 底层数据源（东财/同花顺）有各自 ToS，需自行评估 |
| a-stock-data | Apache-2.0 | ✅ 允许 | 同上，数据来自公开接口 |
| BaoStock | 未明确 | ⚠️ 未声明 | 免费无注册，商业使用需咨询平台方 |
| adata | Apache-2.0 | ✅ 允许 | 同上 |
| qstock | MIT | ⚠️ 部分需付费会员 | 免费功能 MIT，高级功能付费 |
| mootdx | MIT | ✅ 允许 | 通达信数据协议需自查 |
| pywencai | MIT | ❌ 作者不赞成商用 | 明确不赞成商用 |
| efinance | MIT | ❌ 已退役 | 不可用 |
| easyquotation | MIT | ✅ 允许 | 新浪/腾讯公开接口 |

**⚠️ 重要提示**: MIT/Apache-2.0 License 仅覆盖代码本身。底层数据（东方财富、同花顺、巨潮、通达信等）的知识产权和数据使用条款独立于开源协议。本项目所有数据源标注来源、不重新分发原始数据、仅供模型训练内部使用。

### 6.3 数据口径变化风险

- **东方财富**: 大单阈值可能调整（当前为 20万-100万），需定期校验字段定义。
- **同花顺**: 资金流分类口径与东方财富不同，不做直接比较，仅作为独立因子使用。
- **BaoStock**: `netProfit` 在季报中可能是累计值而非单季值，需明确标注。
- **巨潮 cninfo**: orgId 映射可能变化，a-stock-data v3.2.2 已解决此问题，需保持同步更新。
- **新浪财经**: 接口字段偶尔变动，需监控 AKShare 版本更新。

### 6.4 合规建议

1. **不要把 cookie / token 写进代码** — 从环境变量或配置文件读取，`.gitignore` 排除。
2. **不要使用需要绕过鉴权或规避反爬的方案作为默认实现** — 优先选择公开无鉴权的接口。
3. **所有数据源都必须标注来源、抓取时间、更新时间、字段口径** — silver Parquet 的 metadata 中记录。
4. **所有基本面和事件数据必须保留 announce_date / disclosure_date** — 避免未来函数是系统正确性的底线。
5. **不重新分发原始数据** — 仅 internal use within the research/training pipeline。

---

## 7. Recommended Implementation Roadmap

### P0: 最先接入（1-2 周）

**目标**: 补齐三类因子的核心数据，达到 CSI300 250/300 覆盖率。

| 任务 | 数据源 | 产出 |
|------|--------|------|
| 1.1 资金流 fallback 补齐 | AKShare `stock_individual_fund_flow` | 对 push2his 返回空的 symbol 用 AKShare 补齐，合并写入 `silver/fund_flow` |
| 1.2 基本面 BaoStock 集成 | BaoStock `query_profit_data` + `query_growth_data` | 新建 `BaoStockFundamentalsSource`，以 pubDate 为 announce_date 写入 `silver/fundamentals` |
| 1.3 解禁数据增强 | a-stock-data 解禁日历 | 扩展现有 `lockup_collector.py`，增加 90 天前向预警字段 |
| 1.4 公告事件 collector | a-stock-data 巨潮 cninfo | 新建 `announcement_collector.py`，写入 `silver/announcement_events` |
| 1.5 覆盖率和缺失率报告 | — | 训练前自动输出每个数据源的覆盖率/缺失率 |

### P1: 第二阶段增强（2-4 周）

**目标**: 提高数据完整性和因子丰富度。

| 任务 | 数据源 | 产出 |
|------|--------|------|
| 2.1 财报事件因子 | AKShare `stock_yjyg_em` / `stock_yjkb_em` | 新建 `financial_event_collector.py`，构造 `days_to_next_fin_report` 等因子 |
| 2.2 解禁因子构造 | 已有 + enhanced collector | `days_to_next_unlock`, `unlock_size_ratio`, `days_since_last_unlock` |
| 2.3 公告因子构造 | 已有 + announcement_collector | `has_announcement_3d/5d/20d`, `announcement_count_5d`, `has_risk_warning_5d` |
| 2.4 多源交叉验证 | adata | 对资金流和基本面做 adata vs AKShare 口径对比，标注差异 |

### P2: 交叉验证源（4-6 周）

**目标**: 建立数据质量监控和备份链路。

| 任务 | 数据源 | 产出 |
|------|--------|------|
| 3.1 通达信离线财务备份 | mootdx `Affair` | 批量下载 gpcw*.zip，解析为 Parquet 备份 |
| 3.2 行业/概念资金流 | AKShare / a-stock-data | 新增 market-wide 资金流特征 |
| 3.3 数据质量监控 | — | 每日自动对比多源数据，检测口径漂移 |

### P3: 参考/人工验证（不接入自动链路）

| 任务 | 数据源 | 用途 |
|------|--------|------|
| 问财自然语言查询 | pywencai / iwencai-cli | 人工验证特定选股条件结果 |
| 实时行情抽查 | easyquotation | 对比 silver OHLCV 与新浪/腾讯实时快照的一致性 |

---

## 8. Acceptance Criteria

### 8.1 覆盖率标准

```
[ ] fund_flow: CSI300 至少 250/300 symbol 有近 60 交易日数据
[ ] fund_flow: 最新日期 ≤ T-1（不晚于最近交易日）
[ ] fundamentals: CSI300 至少 250/300 symbol 有 ROE/EPS/净利润数据
[ ] fundamentals: 所有入库行的 announce_date 不为 null
[ ] lockup: CSI300 全量 symbol 有解禁历史及未来 90 天预警
[ ] announcement_events: CSI300 至少 200/300 symbol 有近 1 年公告记录
[ ] financial_events: CSI300 至少 250/300 symbol 有最近 4 季度财报事件
```

### 8.2 数据质量标准

```
[ ] 所有 silver Parquet 包含 metadata: source, fetch_time, schema_version
[ ] 所有事件因子均基于 announce_date/event_date 做 PIT 过滤
[ ] 训练前自动输出 per-source 覆盖率/缺失率报告
[ ] 每日 collector 运行日志包含: 成功数/失败数/缺失数/耗时
[ ] 字段口径文档: 每个字段标注来源、定义、单位、更新频率
```

### 8.3 禁止项

```
[ ] 禁止 cookie/token 硬编码在代码中
[ ] 禁止使用需要绕过鉴权的方案作为默认实现
[ ] 禁止将第三方库返回的 DataFrame 直接喂给训练（必须经过 silver Parquet）
[ ] 禁止在未标注 announce_date 的情况下使用基本面数据构造因子
[ ] 禁止在未做 PIT 过滤的情况下使用事件因子
[ ] 禁止并行运行多个东财数据源 collector（共用 _em_get 限流）
```

---

## Appendix A: Detailed Source Profiles

### A.1 AKShare

| 属性 | 值 |
|------|-----|
| GitHub | https://github.com/akfamily/akshare |
| Stars | 21k |
| License | MIT |
| 最新版本 | v1.18.64 (2026-05-27) |
| 活跃度 | ⭐⭐⭐⭐ 极活跃（838 commits, 185 releases） |
| 数据来源 | 东方财富、新浪财经、同花顺、巨潮资讯网、腾讯财经等 |
| 鉴权 | 无需账号/API Key/Cookie/Token（免费） |
| 反爬风险 | 底层数据源（东方财富）有频率限制，均通过模块内置重试 |
| 自动化 | ✅ 适合，需控制频率 |
| 商业使用 | MIT License 覆盖代码；底层数据 ToS 需自行评估 |

**关键接口**:
- `stock_individual_fund_flow(stock, market)` — 个股近100日资金流
- `stock_fund_flow_individual(symbol)` — 同花顺资金流实时排行
- `stock_individual_fund_flow_rank(indicator)` — 东财资金流排名
- `stock_financial_analysis_indicator(stock)` — 新浪财务指标（30+ 字段含 ROE/EPS）
- `stock_financial_abstract_ths(symbol, indicator)` — 同花顺财务摘要
- `stock_yjyg_em()` / `stock_yjkb_em()` — 业绩预告/快报（含公告日期）
- `stock_restricted_release_detail_em(start, end)` — 限售解禁明细
- `stock_notice_report()` — 巨潮公告

### A.2 a-stock-data

| 属性 | 值 |
|------|-----|
| GitHub | https://github.com/simonlin1212/a-stock-data |
| Stars | 6.6k |
| License | Apache-2.0 |
| 最新版本 | v3.3.0 (2026-06-28) |
| 活跃度 | ⭐⭐⭐⭐⭐ 极活跃（2026年持续高频更新） |
| 数据来源 | 13 个直连源: mootdx/腾讯/东财/巨潮/新浪/同花顺/百度/财联社 |
| 鉴权 | 零鉴权（除 iwencai 需 API Key 外完全免费） |
| 反爬风险 | 优先用 mootdx/腾讯（不封IP），东财仅用于独有数据 |
| 自动化 | ✅ 极适合，已内置 em_get() 防封 + 数据源优先级 |
| 商业使用 | Apache-2.0 License；底层数据 ToS 需自行评估 |

**关键接口**:
- 资金流层: 120日个股资金流（主力/大单/中单/小单分类）
- 基本面层: 季报快照 37字段（EPS/ROE/净利润/主营收入）、新浪三表、F10
- 信号层: 限售解禁日历（历史+未来90天预警）
- 公告层: 巨潮 cninfo 沪深北全量公告（动态 orgId）

### A.3 BaoStock

| 属性 | 值 |
|------|-----|
| 官网 | http://baostock.com |
| License | 未明确声明 |
| 数据更新 | 每日 17:30 更新日 K 线 |
| 活跃度 | ⭐⭐⭐⭐ 稳定运行，数据持续更新至 2025-2026 |
| 数据来源 | 自建数据库（整合公开数据源） |
| 鉴权 | 无需注册，匿名登录 `bs.login()` |
| 反爬风险 | 无 |
| 自动化 | ✅ 极适合，专为 API 调用设计 |
| 商业使用 | ⚠️ 未明确声明，需咨询平台方 |

**关键接口**:
- `query_profit_data(code, year, quarter)` — 季频盈利能力（roeAvg, netProfit, epsTTM, gpMargin, npMargin, MBRevenue）
- `query_growth_data(code, year, quarter)` — 季频成长能力（YOYNI, YOYEquity, YOYAsset）
- `query_operation_data(code, year, quarter)` — 季频营运能力
- `query_balance_data(code, year, quarter)` — 季频偿债能力
- `query_cash_flow_data(code, year, quarter)` — 季频现金流量
- `query_dupont_data(code, year, quarter)` — 季频杜邦分析
- `query_history_k_data_plus()` — 历史 K 线

**PIT 特性**: 所有财务 API 返回 `pubDate`（披露日期）和 `statDate`（统计截止日），天然支持 PIT join。这是 BaoStock 相对于其他免费源的最大优势。

### A.4 adata

| 属性 | 值 |
|------|-----|
| GitHub | https://github.com/1nchaos/adata |
| Stars | 4.9k |
| License | Apache-2.0 |
| 最新版本 | v2.9.0 (2025-04-02) |
| 活跃度 | ⭐⭐⭐ 活跃 |
| 数据来源 | 东方财富/同花顺/百度/腾讯/新浪（多源融合+动态代理） |
| 鉴权 | 无需 API Key，免费 |
| 反爬风险 | 多源融合 + 动态代理降低风险 |
| 自动化 | ✅ 适合 |
| 商业使用 | Apache-2.0 License |

**关键接口**:
- `stock.market.get_capital_flow_min(code)` — 分钟级资金流
- `stock.market.get_capital_flow(code)` — 日级历史资金流
- `stock.market.all_capital_flow_east()` — 全概念板块资金流
- `stock.finance.get_core_index(code)` — 核心财务指标
- `sentiment.stock_lifting_last_month()` — 近月解禁列表

### A.5 mootdx

| 属性 | 值 |
|------|-----|
| GitHub | https://github.com/mootdx/mootdx |
| Stars | ~1.5k |
| License | MIT |
| 最新版本 | v0.11.7（近2年未更新） |
| 活跃度 | ⭐⭐ 低活跃（无新版本，无 PR 活动） |
| 数据来源 | 通达信 TCP 服务器（端口 7709） |
| 鉴权 | 无需 API Key，TCP 直连 |
| 反爬风险 | 无（TCP 协议，非 HTTP 接口） |
| 自动化 | ✅ 适合批量下载 |
| 商业使用 | MIT License；通达信数据协议需自查 |

**关键接口**:
- `Quotes.bars(symbol, frequency, offset)` — K 线数据
- `Affair.files()` / `Affair.fetch()` / `Affair.parse()` — 财务文件下载解析
- `Financial.to_data(path)` — 财务 zip 解析为 DataFrame
- `Reader.daily(symbol)` — 离线读取本地数据

**限制**: 每页 800 条上限，最多 25 页（~10 年日线）；不提供资金流数据；不提供公告数据。

### A.6 qstock

| 属性 | 值 |
|------|-----|
| GitHub | https://github.com/tkfy920/qstock |
| Stars | 1.9k |
| License | MIT |
| 最新版本 | v1.3.8 (2025-03) |
| 活跃度 | ⭐⭐⭐ 一般 |
| 数据来源 | 东方财富/同花顺/新浪财经 |
| 鉴权 | 免费功能无需鉴权；高级策略需付费会员（知识星球） |
| 反爬风险 | 同花顺问财需 Node.js + jsdom |
| 自动化 | ⚠️ 部分功能受限 |
| 商业使用 | MIT License；部分功能需付费 |

**关键接口**:
- `qs.intraday_money(code)` / `qs.hist_money(code)` / `qs.stock_money(code, ndays)`
- `qs.ths_money('个股'/ '行业'/ '概念', n)` — 同花顺资金流
- `qs.north_money('个股'/ '行业', n)` — 北向资金
- `qs.stock_basics(code_list)` — 关键指标
- `qs.financial_statement(flag, date)` — 财务报表

### A.7 pywencai / iwencai-cli

| 属性 | pywencai | iwencai-cli |
|------|----------|-------------|
| GitHub | zsrl/pywencai | shaw-baobao/iwencai-cli |
| Stars | 860 | — |
| License | MIT | — |
| 最新版本 | v0.13.1 (2025-05-06) | v1.0.0 |
| 活跃度 | ⭐⭐⭐ | ⭐⭐ |
| 数据来源 | 同花顺 iwencai.com | 同花顺 iwencai.com (Playwright) |
| 鉴权 | **需 Cookie** (2025+强制) | 无需登录（真实 Chrome） |
| 反爬风险 | 🔴 高（需 JS 加密 bypass） | 🟡 中（模拟真实浏览器） |
| 自动化 | ❌ 不适合 | ⚠️ 需 Chrome 运行 |
| 商业使用 | ❌ 作者不赞成商用 | ⚠️ 未声明 |

### A.8 efinance

| 属性 | 值 |
|------|-----|
| GitHub | https://github.com/Micro-sheep/efinance |
| Stars | 3.8k |
| License | MIT |
| 最新版本 | v0.5.5 (2025-03-15) |
| 活跃度 | ❌ 作者已宣布退役 (2025-06)，服务端计划未启动 |
| 数据来源 | 东方财富 |
| 反爬风险 | 🔴 极高（东财已针对性限流封禁） |
| 自动化 | ❌ 不可用 |
| 建议 | **不推荐接入** |

### A.9 easyquotation

| 属性 | 值 |
|------|-----|
| GitHub | https://github.com/shidenggui/easyquotation |
| Stars | 5.3k |
| License | MIT |
| 最新版本 | 199 commits, status unclear |
| 活跃度 | ⭐⭐⭐ 稳定但功能有限 |
| 数据来源 | 新浪/腾讯/集思录 |
| 鉴权 | 无需 |
| 反爬风险 | 低 |
| 自动化 | ⚠️ 仅实时快照，无历史数据 |
| 建议 | **仅限实时行情交叉验证，不做因子数据源** |

---

## Appendix B: Field Mapping Quick Reference

### B.1 Fund Flow Fields

```
AKShare stock_individual_fund_flow                → silver/fund_flow
──────────────────────────────────────────────────────────────────
日期                → date
主力净流入-净额      → main_net
超大单净流入-净额    → super_net
大单净流入-净额      → large_net
中单净流入-净额      → mid_net
小单净流入-净额      → small_net
主力净流入-净占比    → main_net_pct (新增)
收盘价               → close (新增，校验用)
涨跌幅               → pct_change (新增)
```

### B.2 Fundamental Fields

```
BaoStock query_profit_data                       → silver/fundamentals
──────────────────────────────────────────────────────────────────
code                → symbol
pubDate             → announce_date               ← PIT JOIN KEY
statDate            → period_end
roeAvg              → roe_avg_pct
netProfit           → net_profit
epsTTM              → eps_ttm
gpMargin            → gross_margin_pct
npMargin            → net_margin_pct
MBRevenue           → revenue
totalShare          → total_shares
liqaShare           → float_shares

BaoStock query_growth_data                       → silver/fundamentals
──────────────────────────────────────────────────────────────────
YOYNI               → net_profit_yoy_pct
YOYEquity           → equity_yoy_pct
YOYAsset            → asset_yoy_pct
```

### B.3 Lockup / Event Fields

```
a-stock-data 解禁日历                             → silver/lockup
──────────────────────────────────────────────────────────────────
股票代码             → symbol
解禁日期             → unlock_date
解禁类型             → lock_type
解禁数量(万股)       → shares_10k
解禁占总股本比例     → ratio_pct

a-stock-data 巨潮公告                             → silver/announcement_events
──────────────────────────────────────────────────────────────────
股票代码             → symbol
公告日期             → announce_date               ← PIT JOIN KEY
公告标题             → title
公告类别             → category
PDF链接              → pdf_url
```

---

## Appendix C: Anti-Crawl Mitigation Strategy

本项目已有 `_em_get()` 限流机制（≥1s per call + jitter），对所有东方财富接口生效。新增数据源按以下策略接入：

```
┌──────────────────────────────────────────────────────┐
│                   Collector Layer                     │
│                                                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐             │
│  │ flow    │  │fundamentals│ │ lockup   │ ...         │
│  │collector│  │collector  │  │collector │             │
│  └────┬────┘  └─────┬─────┘  └────┬─────┘             │
│       │              │              │                   │
│  ┌────▼──────────────▼──────────────▼──────────────┐  │
│  │              Source Adapter Layer                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │  │
│  │  │ Eastmoney│ │ BaoStock │ │ a-stock-data     │ │  │
│  │  │ (primary)│ │(fallback)│ │ (cninfo/解禁等)  │ │  │
│  │  └────┬─────┘ └────┬─────┘ └────────┬─────────┘ │  │
│  │       │             │               │            │  │
│  │  ┌────▼─────────────▼───────────────▼─────────┐  │  │
│  │  │         Rate Limiter (_em_get / _throttle) │  │  │
│  │  │         ≥1s delay + random jitter          │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Silver Parquet Store                 │  │
│  │  silver/fund_flow/    silver/fundamentals/        │  │
│  │  silver/lockup/       silver/announcement_events/ │  │
│  │  silver/financial_events/                         │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

规则:
1. 所有东财源共用 `_em_get()` 限流，禁止并行运行东财 collector
2. BaoStock / mootdx 无此限制，可独立运行
3. a-stock-data 的东财请求也经过 em_get，无需额外处理
4. 每个 collector 独立运行，失败不影响其他 collector
5. 所有数据写入前经过 schema enforcement（`store/schemas.py`）
