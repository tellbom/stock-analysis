import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quant_platform.selection.ranker import TECH_SECTOR_INDUSTRIES

OUT = ROOT / "outputs/01a116ce-1f0c-7d41-8967-adcca880c1b8/2026-10-14_统一基线_D3_D5"
CORE = OUT / "comparison"
summary = json.loads((CORE / "training_summary.json").read_text(encoding="utf-8"))
picks = pd.read_csv(CORE / "all_picks.csv", dtype={"symbol": str})
attributes = pd.read_csv(CORE / "prediction_attributes.csv", dtype={"symbol": str}).set_index("symbol")
unions = {}
for h in (3, 5):
    day = picks[(picks.horizon == h) & ~picks.is_diagnostic]
    arms = {key: frame.set_index("symbol") for key, frame in day.groupby("strategy")}
    base = pd.read_csv(CORE / f"D{h}_original_scores.csv", dtype={"symbol": str}).set_index("symbol")
    recent = pd.read_csv(CORE / f"D{h}_recent_flow_scores.csv", dtype={"symbol": str}).set_index("symbol")
    union = sorted(set(day.symbol), key=lambda s: (arms["hybrid_flow"].loc[s, "recommendation_rank"] if s in arms["hybrid_flow"].index else 999, base.loc[s, "base_rank"], s))
    rows = []
    for symbol in union:
        a = attributes.loc[symbol]
        ranks = {key: int(frame.loc[symbol, "recommendation_rank"]) if symbol in frame.index else None for key, frame in arms.items()}
        in_equal = [name for key, name in (("ordinary", "普通"), ("hybrid_no_flow", "无流"), ("hybrid_flow", "有流")) if ranks[key] is not None and ranks[key] <= summary["horizons"][str(h)]["equal_count_n"]]
        rows.append({"symbol": symbol, "name": a["name"], "industry_name": a.industry_name,
                     "close": a.close, "ordinary_rank": ranks["ordinary"],
                     "no_flow_rank": ranks["hybrid_no_flow"], "flow_rank": ranks["hybrid_flow"],
                     "base_score": base.loc[symbol, "base_score"], "base_rank": int(base.loc[symbol, "base_rank"]),
                     "recent_rank": int(recent.loc[symbol, "recent_rank"]),
                     "fusion_score": 1/(60+base.loc[symbol, "base_rank"])+1/(60+recent.loc[symbol, "recent_rank"]),
                     "equal_membership": "/".join(in_equal), "is_tech": a.industry_name in TECH_SECTOR_INDUSTRIES})
    unions[str(h)] = rows

diagnostic = picks[picks.is_diagnostic].fillna("").to_dict("records")
membership = pd.read_parquet(ROOT / "models/data/universe/csi300/membership.parquet")
name_map = dict(zip(membership.symbol.astype(str).str.zfill(6), membership.name))
coverage = []
for symbol in sorted(name_map):
    price = pd.read_parquet(ROOT / f"models/data/silver/ohlcv/{symbol}.parquet").sort_values("date").iloc[-1]
    flow = pd.read_parquet(ROOT / f"models/data/silver/fund_flow/{symbol}.parquet").sort_values("date").iloc[-1]
    cap = pd.read_parquet(ROOT / f"models/data/silver/valuation/{symbol}.parquet").sort_values("date").iloc[-1]
    coverage.append({"symbol": symbol, "name": name_map[symbol], "price_date": str(price.date)[:10],
                     "flow_date": str(flow.date)[:10], "cap_date": str(cap.date)[:10], "close": float(price.close),
                     "main_net": float(flow.main_net), "small_net": float(flow.small_net),
                     "super_net": float(flow.super_net), "float_mcap_yi": float(cap.float_mcap_yi),
                     "status": "参与评分" if symbol in attributes.index else "排除：数据陈旧"})

importance = {str(h): dict(pd.read_csv(CORE / f"D{h}_recent_flow_importance.csv").values.tolist()) for h in (3,5)}
payload = {"summary": summary, "unions": unions, "diagnostic": diagnostic, "coverage": coverage,
           "importance": importance, "tech_industries": sorted(TECH_SECTOR_INDUSTRIES),
           "main_counts": {str(h): {key: len(frame) for key, frame in picks[(picks.horizon==h)&~picks.is_diagnostic].groupby("strategy")} for h in (3,5)}}
(OUT / "workbook_input.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=lambda v: v.item()), encoding="utf-8")
pd.DataFrame(coverage).to_csv(OUT / "data_coverage.csv", index=False, encoding="utf-8-sig")
print("Workbook inputs ready")
