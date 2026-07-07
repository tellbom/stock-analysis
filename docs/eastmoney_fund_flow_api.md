# Eastmoney H5 个股历史资金流 API

## 概述

- **端点**: `emdatah5.eastmoney.com`（非 `push2his.eastmoney.com`，后者有 TLS 兼容问题和 IP 限流）
- **协议**: HTTPS
- **方法**: GET
- **认证**: 无需 cookie / token / API key
- **反爬**: 需要 UA + Referer，无验证码
- **限流**: 未发现（批量 10 只 7 秒全成功）
- **浏览器环境**: **不需要** — 纯 HTTP GET，`urllib3` / `requests` 均可

---

## 1. 个股历史日频资金流

### 请求

```
GET https://emdatah5.eastmoney.com/dc/ZJLX/getDBHistoryData
```

**Query 参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `secid` | `1.600000` | `市场.代码`，市场: 1=沪, 0=深, 2=京 |
| `fields1` | `f1,f2,f3,f7` | 固定 |
| `fields2` | `f51,f52,...,f63` | 见下方字段表 |
| `ut` | `b2884a393a59ad64002292a3e90d46a5` | 固定 token |

**Headers**:

```
User-Agent: Mozilla/5.0
Referer: https://emdatah5.eastmoney.com/dc/zjlx/stock?fc={market}.{code}
```

### 市场映射

```python
# 沪市 → 1
prefixes_sh = ["600", "601", "603", "605", "688"]
# 深市 → 0
prefixes_sz = ["000", "001", "002", "003", "300", "301"]
# 京市 → 2
prefixes_bj = ["430", "830", "831", ..., "920"]
```

### 响应字段

```json
{
  "rc": 0,
  "rt": 22,
  "data": {
    "code": "600000",
    "market": 1,
    "name": "浦发银行",
    "klines": [
      "2026-01-05,88875050.0,-120033600.0,...,11.82,-4.98",
      "2026-01-06,...",
      ...
    ]
  }
}
```

**klines 字段映射** (逗号分隔，每行 13 列):

| 索引 | 字段名 | 含义 | 类型 | 示例 |
|------|--------|------|------|------|
| f51 | `trade_date` | 交易日期 | str | `2026-01-05` |
| f52 | `main_net` | 主力净流入(元) | float | `88875050.0` |
| f53 | `small_net` | 小单净流入(元) | float | `-120033600.0` |
| f54 | `medium_net` | 中单净流入(元) | float | `31158560.0` |
| f55 | `large_net` | 大单净流入(元) | float | `65471616.0` |
| f56 | `super_net` | 超大单净流入(元) | float | `23403434.0` |
| f57 | `main_net_rate` | 主力净流入占比(%) | float | `6.09` |
| f58 | `small_net_rate` | 小单净流入占比(%) | float | `-8.22` |
| f59 | `medium_net_rate` | 中单净流入占比(%) | float | `2.13` |
| f60 | `large_net_rate` | 大单净流入占比(%) | float | `4.48` |
| f61 | `super_net_rate` | 超大单净流入占比(%) | float | `1.60` |
| f62 | `close` | 收盘价 | float | `11.82` |
| f63 | `pct_change` | 涨跌幅(%) | float | `-4.98` |

### 返回范围

约 **120 个交易日**（~6 个月），当前覆盖 2026-01-05 至最新交易日。

### 调用示例

```python
import urllib3, json

http = urllib3.PoolManager(num_pools=1, maxsize=1, retries=0)

def get_fund_flow(code, mkt):
    url = (
        f"https://emdatah5.eastmoney.com/dc/ZJLX/getDBHistoryData"
        f"?secid={mkt}.{code}"
        f"&fields1=f1,f2,f3,f7"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63"
        f"&ut=b2884a393a59ad64002292a3e90d46a5"
    )
    r = http.request("GET", url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://emdatah5.eastmoney.com/dc/zjlx/stock?fc={mkt}.{code}"
        },
        timeout=15.0,
    )
    data = json.loads(r.data)
    records = []
    for kline in data["data"]["klines"]:
        parts = kline.split(",")
        records.append({
            "symbol": code,
            "trade_date": parts[0],
            "main_net": float(parts[1]),
            "small_net": float(parts[2]),
            "medium_net": float(parts[3]),
            "large_net": float(parts[4]),
            "super_net": float(parts[5]),
            "main_net_rate": float(parts[6]),
            "small_net_rate": float(parts[7]),
            "medium_net_rate": float(parts[8]),
            "large_net_rate": float(parts[9]),
            "super_net_rate": float(parts[10]),
            "close": float(parts[11]),
            "pct_change": float(parts[12]),
        })
    return records

# 示例
df = get_fund_flow("600000", 1)  # 浦发银行, 沪市
```

---

## 2. 多时间窗口资金流汇总

### 请求

```
GET https://emdatah5.eastmoney.com/dc/ZJLX/getZJLXData
  ?secid=1.600000
  &fields=f57,f58,f86,f135,f136,f137,f138,f139,f140,f141,f142,f143,f144,f145,f146,f147,f148,f149
  &ut=b2884a393a59ad64002292a3e90d46a5
```

### 字段映射

| 字段 | 含义 |
|------|------|
| `f57` | 股票代码 |
| `f58` | 股票名称 |
| `f86` | 时间戳 |
| `f135` | 今日主力净流入 |
| `f136` | 今日超大单净流入 |
| `f137` | 今日大单净流入 |
| `f138` | 3日主力净流入 |
| `f139` | 3日超大单净流入 |
| `f140` | 3日大单净流入 |
| `f141` | 5日主力净流入 |
| `f142` | 5日超大单净流入 |
| `f143` | 5日大单净流入 |
| `f144` | 10日主力净流入 |
| `f145` | 10日超大单净流入 |
| `f146` | 10日大单净流入 |
| `f147` | 20日主力净流入 |
| `f148` | 20日超大单净流入 |
| `f149` | 20日大单净流入 |

### 响应示例

```json
{
  "rc": 0,
  "data": {
    "f57": "600000",
    "f58": "浦发银行",
    "f135": 99980043.0,
    "f138": 23958481.0,
    "f141": 76021562.0,
    "f144": 102726121.0,
    "f147": 80461794.0
  }
}
```

---

## 对比总结

| | push2his (旧) | emdatah5 (新) |
|---|---|---|
| 子域名 | push2his.eastmoney.com | emdatah5.eastmoney.com |
| 协议 | HTTP 80 | HTTPS 443 |
| 本机 Python 3.13 | ❌ SSL 错误 | ✅ 稳定 |
| 限流 | 严重 (0-70%) | 未发现 (100%) |
| 字段 | f51-f65 (15列) | f51-f63 (13列) |
| 返回行数 | ~120 | ~120 |
| 需要浏览器 | 否 | **否** |
