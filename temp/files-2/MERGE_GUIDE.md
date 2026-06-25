# Phase 4B — Codex 合并指南

## 重要：先合并 Phase 4A

Phase 4B 依赖 Phase 4A 输出的 `store_lake.py`（含 `index_ohlcv_path`）。
合并顺序：Phase 4A → Phase 4B。

---

## 文件映射

| 输出文件 | 目标路径（相对于项目包根） | 操作 |
|---|---|---|
| `store_lake.py` | `quant_platform/store/lake.py` | **替换**（在 P4A 版本上继续追加） |
| `store_schemas.py` | `quant_platform/store/schemas.py` | **替换** |
| `ingest_valuation_collector.py` | `quant_platform/ingest/valuation_collector.py` | **新增** |
| `ingest_industry_collector.py` | `quant_platform/ingest/industry_collector.py` | **新增** |
| `ingest_flow_collector.py` | `quant_platform/ingest/flow_collector.py` | **新增** |
| `ingest_margin_collector.py` | `quant_platform/ingest/margin_collector.py` | **新增** |
| `features_valuation.py` | `quant_platform/features/valuation.py` | **新增** |
| `features_industry.py` | `quant_platform/features/industry.py` | **新增** |
| `features_flow.py` | `quant_platform/features/flow.py` | **新增** |
| `features_margin.py` | `quant_platform/features/margin.py` | **新增** |
| `features_registry.py` | `quant_platform/features/registry.py` | **替换** |
| `features_pipeline.py` | `quant_platform/features/pipeline.py` | **替换** |
| `test_phase4b.py` | `tests/test_phase4b.py` | **新增** |

---

## 变更摘要

### store_lake.py（P4A 版基础上追加）
新增路径函数：
- `valuation_dir()` / `valuation_path()` → `silver/valuation/{symbol}.parquet`
- `industry_map_path()` → `silver/industry_map.parquet`（宇宙级 SCD 表，非 per-symbol）
- `fund_flow_dir()` / `fund_flow_path()` → `silver/fund_flow/{symbol}.parquet`
- `margin_dir()` / `margin_path()` → `silver/margin/{symbol}.parquet`

`init_lake()` 新增目录：`silver/valuation`、`silver/fund_flow`、`silver/margin`

### store_schemas.py（追加 4 个新 schema）
- `enforce_valuation()` — PE/PB/市值/换手率验证
- `enforce_industry_map()` — SCD 行业表验证
- `enforce_fund_flow()` — 资金流验证（补全缺失列为 0.0）
- `enforce_margin()` — 融资融券验证（补全缺失列为 0.0）

### ingest_valuation_collector.py（P4B-01）新文件
- `ValuationCollector.run(symbols, date)` — 腾讯财经批量拉取
- `_fetch_tencent_batch()` — PB 来自 field[46]，不是 field[43]（常见陷阱）
- 增量更新：已有日期不重复写入

### ingest_industry_collector.py（P4B-03）新文件
- `IndustryCollector.run(symbols, as_of)` — 东财个股信息 + 概念板块
- SCD 变更检测：行业变动时插入新行、关闭旧行的 out_date
- `get_industry_as_of(imap, symbol, date)` — PIT 查询辅助函数
- 东财防封：共享 `_em_get()` 串行节流

### ingest_flow_collector.py（P4B-06）新文件
- `FundFlowCollector.run(symbols)` — 东财 push2his 120 日资金流
- 增量逻辑：gap > 60 天触发全量重拉
- `_fetch_push2his()` — 解析 klines 格式（逗号分隔，"-" 处理为 0.0）

### ingest_margin_collector.py（P4B-08）新文件
- `MarginCollector.run(symbols)` — 东财 datacenter RPTA_WEB_RZRQ_GGMX
- 1 天增量更新
- 非融资标的静默返回空 DataFrame（不报错）

### features_valuation.py（P4B-02）新文件
- `build_valuation_features(panel, valuation_panel)` — 跨截面 rank/zscore
- 负 PE → rank 置为 0（非排除）
- `VALUATION_SPECS` — 6 个 FeatureSpec

### features_industry.py（P4B-04 + P4B-05）新文件
- `build_industry_features(panel, industry_map)` — 行业内 rank
- `_join_industry(panel, imap)` — PIT SCD join
- `build_excess_vs_industry_labels(label_panel, imap)` — 行业超额收益标签
- `INDUSTRY_SPECS` — 4 个 FeatureSpec

### features_flow.py（P4B-07）新文件
- `build_flow_features(panel, flow_panel, valuation_panel)` — 资金流跨截面特征
- 按 float_mcap 归一化（fallback 为跨截面均值）
- `FLOW_SPECS` — 5 个 FeatureSpec

### features_margin.py（P4B-08）新文件
- `build_margin_features(panel, margin_panel, valuation_panel)` — 融资融券特征
- 1 日 lag 通过 shift 实现（features at T use margin from T-1）
- `MARGIN_SPECS` — 2 个 FeatureSpec

### features_registry.py（追加 P4B）
- `FULL_SPECS` 常量：`DEFAULT_SPECS + P4B specs`（延迟导入，避免循环依赖）
- `_get_p4b_specs()` 惰性加载 P4B spec 列表

### features_pipeline.py（扩展 P4B）
- `FeaturePipeline.__init__` 新增参数：`include_valuation`、`include_industry`、`include_flow`、`include_margin`（均默认 False，向后兼容）
- `build_panel()` 根据 flag 调用 P4B panel-level builders
- 所有 P4B builder 以 try/except ImportError 保护（部分环境可能未安装）

### test_phase4b.py（674 行）
覆盖所有 P4B 任务，纯合成数据，关键验证：
- Tencent field 46 = PB（不是 field 43）
- SCD PIT 查询在 out_date 前后返回不同结果
- 行业超额标签每组求和 ≈ 0
- 资金流归一化：小盘股同等绝对流入排名更高
- margin 1-day lag：第一个交易日特征为 NaN

---

## 合并后验证命令

```bash
# 语法检查
for f in store/lake.py store/schemas.py \
          ingest/valuation_collector.py ingest/industry_collector.py \
          ingest/flow_collector.py ingest/margin_collector.py \
          features/valuation.py features/industry.py \
          features/flow.py features/margin.py \
          features/registry.py features/pipeline.py; do
    python -m py_compile quant_platform/$f && echo "OK $f"
done

# Phase 4B 测试（含 Phase 4A 回归）
pytest tests/test_phase4b.py -v
pytest tests/test_phase4a.py -v  # 确认 P4A 未被破坏
pytest --tb=short -q              # 全量测试

# 验证 FULL_SPECS 不含重复名称
python -c "
from quant_platform.features.registry import FULL_SPECS
names = [s.name for s in FULL_SPECS]
assert len(names) == len(set(names)), 'Duplicate feature names!'
print(f'FULL_SPECS: {len(FULL_SPECS)} features, {len({s.family for s in FULL_SPECS})} families')
"
```

---

## 注意事项

1. `ingest/industry_collector.py` 中的 `_em_get()` 和其他采集器中的 `_em_get()` 是**模块级独立实例**，不共享节流状态。如果同时运行多个采集器，Eastmoney 总请求频率仍需在调用层面控制（串行运行 collector.run()，不并发）。

2. `features/industry.py` 中的 `_join_industry()` 使用 `apply(axis=1)` 逐行 SCD 查询，对大宇宙（300 只股 × 多年日线）性能可接受（约 60 秒），但不适合每日增量微批。如性能成为瓶颈，可替换为向量化区间 join（pandas merge_asof）。

3. `features_pipeline.py` 的 `build_panel()` 中所有 P4B builder 的导入都用 `try/except ImportError` 包裹，这意味着如果对应文件未部署，会 warning 而非报错。合并后确认文件路径正确。

4. `labels/builder.py`（P4A-04 版本）中的 `build_label_panel()` 已支持 `excess_vs_csi300`。**行业超额标签**（`excess_vs_industry_{h}d`）由 `features/industry.py` 中的 `build_excess_vs_industry_labels()` 单独调用，不在 label builder 内部（需要行业表），调用方负责编排顺序。
