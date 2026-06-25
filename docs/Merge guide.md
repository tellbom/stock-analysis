# CLI — Codex 合并指南 + 端到端运行手册

## 文件映射

| 输出文件 | 目标路径 | 操作 |
|---|---|---|
| `cli.py` | `quant_platform/cli.py` | **替换** |

---

## 合并后验证

```bash
# 语法检查
python -m py_compile quant_platform/cli.py && echo "OK"

# 帮助输出正常
python -m quant_platform.cli --help
python -m quant_platform.cli run --help
python -m quant_platform.cli enrich --help
python -m quant_platform.cli model --help
python -m quant_platform.cli diagnose --help
```

---

## Phase 4 完整变更摘要（相对于原始 cli.py）

### 默认值变更（向后兼容：旧命令仍可用，只是默认值不同）

| 参数 | 原默认值 | 新默认值 | 原因 |
|---|---|---|---|
| `--label` (features/model) | `ret_fwd_20d` | `ret_fwd_5d` | P4A-04：5d 是主标签 |
| `--horizon` (model) | `20` | `5` | P4A-04：5d horizon |
| `--horizons` (features) | `[1, 5, 20]` | `[1, 5, 10, 20]` | P4A-04：加入 10d |
| 评估方法 | 静态 lockbox | walk-forward OOS | P4A-03：walk-forward 为默认 |

### 新子命令

| 子命令 | 功能 |
|---|---|
| `enrich` | P4B：依次运行 5 个数据采集器（估值/行业/资金流/融资融券/解禁） |
| `diagnose` | P4C：单因子 IC 诊断 + 共线性剪枝 + 可选 regime 分析 |

### 原有子命令扩展

| 子命令 | 新增内容 |
|---|---|
| `collect` | `--no-index` 跳过指数采集；默认采集 000300 指数 OHLCV |
| `features` | 5 个 `--include-*` 开关接入 P4B/C builders |
| `model` | walk-forward（默认）；`--use-lockbox` 恢复旧路径；Ridge 基线（P4A-06）；subperiod IC ratio 展示 |
| `run` | `enrich` 步骤夹在 collect 和 features 之间；`--skip-enrich` 可跳过 |
| `status` | 展示所有 P4B silver 目录覆盖情况 + evaluation 输出汇总 |

---

## 端到端运行手册

### 最小运行（纯价格数据，等同于 P0-P2 原始流程）

```bash
STORE=/data/lake

# Step 1: 采集 OHLCV + 日历 + 宇宙 + 指数
python -m quant_platform.cli collect \
  --store-root $STORE --universe csi300

# Step 2: 特征 + 标签（仅技术指标，5d 主标签）
python -m quant_platform.cli features \
  --store-root $STORE

# Step 3: 训练 + walk-forward 评估
python -m quant_platform.cli model \
  --store-root $STORE

# Step 4: 查看结果
python -m quant_platform.cli status --store-root $STORE
```

### 完整 Phase 4 运行（所有 P4B 数据源）

```bash
STORE=/data/lake

# P0: 基础数据采集（含 CSI 300 指数）
python -m quant_platform.cli collect \
  --store-root $STORE --universe csi300 --start-date 2018-01-01

# P4B: 新数据源（约 25 分钟，串行防封）
python -m quant_platform.cli enrich \
  --store-root $STORE --universe csi300

# P1+P4: 特征 + 多 horizon 标签 + 泄露检测
python -m quant_platform.cli features \
  --store-root $STORE \
  --include-valuation \
  --include-industry \
  --include-flow \
  --include-margin \
  --include-lockup

# P2+P4A: Walk-forward 训练 + 评估（含 Ridge 基线）
python -m quant_platform.cli model \
  --store-root $STORE \
  --label ret_fwd_5d --horizon 5 \
  --n-windows 5 --window-months 12

# P4C: 单因子诊断 + 剪枝
python -m quant_platform.cli diagnose \
  --store-root $STORE \
  --corr-threshold 0.85

# 再次训练，使用剪枝后的特征集
python -m quant_platform.cli model \
  --store-root $STORE \
  --label ret_fwd_5d --horizon 5

# 查看最终状态
python -m quant_platform.cli status --store-root $STORE
```

### 一键完整流水线

```bash
python -m quant_platform.cli run \
  --store-root /data/lake --universe csi300 \
  --start-date 2018-01-01 \
  --include-valuation --include-industry --include-flow \
  --label ret_fwd_5d --horizon 5 \
  --n-windows 5 --window-months 12
```

### 常用变体

```bash
# 跳过 P4B 数据采集（仅价格数据）
python -m quant_platform.cli run --store-root $STORE --skip-enrich

# 使用旧版 lockbox 评估（向后兼容）
python -m quant_platform.cli model \
  --store-root $STORE --use-lockbox --lockbox-months 12

# 只运行 regime 分析（P4C-05，较慢）
python -m quant_platform.cli diagnose \
  --store-root $STORE --with-regime

# 仅采集部分 P4B 数据
python -m quant_platform.cli enrich \
  --store-root $STORE \
  --skip-margin --skip-lockup   # 只采集估值+行业+资金流
```

---

## 运行时间估算（CSI 300，单机）

| 步骤 | 时间 | 说明 |
|---|---|---|
| `collect` OHLCV | ~30 分钟 | 300 只股 × 5 年日线 |
| `collect` 指数 | < 1 分钟 | 单只指数 |
| `enrich` 估值 | ~2 分钟 | Tencent 批量，一次调用 |
| `enrich` 行业 | ~6 分钟 | 东财串行 1s/个 |
| `enrich` 资金流 | ~6 分钟 | 东财串行 1s/个 |
| `enrich` 融资融券 | ~6 分钟 | 东财串行 1s/个 |
| `enrich` 解禁 | ~6 分钟 | 东财串行 1s/个 |
| `features` | ~10 分钟 | 技术 + P4B panel builders |
| `model` walk-forward | ~30-60 分钟 | 5 窗口 × LightGBM 训练 |
| `diagnose` IC | ~5 分钟 | 40+ 特征 × 4 horizon |
| **全流程合计** | **~2 小时** | 首次运行；增量更新更快 |

---

## 注意事项

1. **enrich 顺序不可并发**：所有东财采集器共享同一 IP，串行运行防封。`cmd_enrich` 已保证串行，不要多进程运行。

2. **walk-forward 配置固定化**：第一次运行 model 时确定 `--n-windows` 和 `--window-months`，之后不要改变（改变会使历史对比失效）。默认 `n_windows=5, window_months=12` 是推荐值。

3. **diagnose 在 model 之前**：diagnose 生成剪枝日志；第二次运行 model 时会自动读取并应用剪枝。

4. **旧报告向后兼容**：`p2_report_{label_col}.txt` 和 `p2_report_{label_col}.json` 路径不变，只是内容中新增了 walk-forward 字段。

5. **合并顺序**：`Phase 4A → Phase 4B → Phase 4C → cli.py`。cli.py 用 try/except ImportError 保护所有 P4 模块的导入，合并不完整时会 warn 而不是 crash。