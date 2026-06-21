"""
evaluation.backtest
===================
Cost-and-delay-aware signal backtest (T2.5).

Implements a simple long-short decile backtest:
  - On each date, rank all stocks by predicted score.
  - Go long top-decile, short bottom-decile.
  - Execute at T+1 (next day's open/close), not T.
  - Deduct transaction costs and turnover.

This backtest is the ARBITER, not the objective.
It confirms whether IC translates into a realizable spread.
Do not optimise against it.

Outputs
-------
  BacktestResult  — cumulative returns, Sharpe, max drawdown, net spread,
                    turnover, cost summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Summary of one signal backtest run."""
    # Performance
    annualised_return:  float = float("nan")
    annualised_vol:     float = float("nan")
    sharpe:             float = float("nan")
    max_drawdown:       float = float("nan")
    calmar:             float = float("nan")

    # Signal quality
    long_minus_short:   float = float("nan")   # gross spread (long - short avg return)
    net_long_minus_short: float = float("nan") # after costs

    # Turnover and costs
    avg_daily_turnover: float = float("nan")   # fraction of portfolio replaced per day
    total_cost:         float = float("nan")   # total cost as fraction of gross return

    # Time series
    daily_pnl:          pd.Series = field(default_factory=pd.Series)
    cumulative_return:  pd.Series = field(default_factory=pd.Series)

    # Config
    cost_bps:           float = 10.0          # one-way cost in basis points
    n_days:             int   = 0
    n_decile:           int   = 10

    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("SIGNAL BACKTEST")
        print("=" * 55)
        print(f"  Days:              {self.n_days}")
        print(f"  Ann. Return:       {self.annualised_return:+.2%}")
        print(f"  Ann. Volatility:   {self.annualised_vol:.2%}")
        print(f"  Sharpe:            {self.sharpe:+.2f}")
        print(f"  Max Drawdown:      {self.max_drawdown:.2%}")
        print(f"  Calmar:            {self.calmar:+.2f}")
        print(f"  Gross L-S spread:  {self.long_minus_short:+.4f}")
        print(f"  Net L-S spread:    {self.net_long_minus_short:+.4f}")
        print(f"  Avg Daily Turnover:{self.avg_daily_turnover:.2%}")
        print(f"  Total cost drag:   {self.total_cost:.2%}")
        print("=" * 55 + "\n")

    def summary_dict(self) -> dict:
        return {
            "annualised_return":    round(self.annualised_return, 6),
            "annualised_vol":       round(self.annualised_vol, 6),
            "sharpe":               round(self.sharpe, 4),
            "max_drawdown":         round(self.max_drawdown, 4),
            "gross_ls_spread":      round(self.long_minus_short, 6),
            "net_ls_spread":        round(self.net_long_minus_short, 6),
            "avg_daily_turnover":   round(self.avg_daily_turnover, 4),
            "cost_bps":             self.cost_bps,
            "n_days":               self.n_days,
        }


def run_backtest(
    panel: pd.DataFrame,
    pred_col:   str = "pred",
    return_col: str = "ret_fwd_1d",
    cost_bps:   float = 10.0,
    n_decile:   int   = 10,
    top_n:      int   = 1,     # number of deciles to go long  (top top_n)
    bot_n:      int   = 1,     # number of deciles to go short (bottom bot_n)
    trading_days_per_year: int = 252,
) -> BacktestResult:
    """
    Run a long-short decile backtest.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have: date, symbol, <pred_col>, <return_col>.
        Returns are T+1 returns (the label already embeds the execution lag).
    cost_bps : float
        One-way transaction cost in basis points.
    top_n / bot_n : int
        Number of deciles in the long / short leg.

    Returns
    -------
    BacktestResult
    """
    df = panel[[
        "date", "symbol", pred_col, return_col
    ]].dropna().copy()

    if df.empty:
        logger.warning("backtest: empty panel after dropna")
        return BacktestResult(cost_bps=cost_bps)

    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    cost_per_trade = cost_bps / 10_000.0   # basis points → fraction
    daily_pnl  = []
    long_port_prev: set  = set()
    short_port_prev: set = set()

    for date in dates:
        day = df[df["date"] == date].copy()
        if len(day) < n_decile:
            continue

        # Rank into deciles
        day["decile"] = pd.qcut(
            day[pred_col].rank(method="first"),
            n_decile, labels=False, duplicates="drop"
        )

        long_mask  = day["decile"] >= (n_decile - top_n)
        short_mask = day["decile"] < bot_n
        long_port  = set(day[long_mask]["symbol"])
        short_port = set(day[short_mask]["symbol"])

        # Gross returns
        long_ret  = day[long_mask][return_col].mean()
        short_ret = day[short_mask][return_col].mean()
        gross_pnl = (long_ret - short_ret) / 2   # 50/50 weight

        # Turnover cost: fraction of portfolio that changed
        long_new  = long_port  - long_port_prev
        short_new = short_port - short_port_prev
        n_long    = max(len(long_port), 1)
        n_short   = max(len(short_port), 1)
        turnover  = (len(long_new) / n_long + len(short_new) / n_short) / 2
        cost      = turnover * cost_per_trade

        daily_pnl.append({
            "date":     date,
            "gross":    gross_pnl,
            "cost":     cost,
            "net":      gross_pnl - cost,
            "turnover": turnover,
            "long_ret": long_ret,
            "short_ret": short_ret,
        })

        long_port_prev  = long_port
        short_port_prev = short_port

    if not daily_pnl:
        return BacktestResult(cost_bps=cost_bps)

    pnl_df = pd.DataFrame(daily_pnl).set_index("date")
    net    = pnl_df["net"]
    gross  = pnl_df["gross"]
    cum    = (1 + net).cumprod()

    ann_ret = float((1 + net.mean()) ** trading_days_per_year - 1)
    ann_vol = float(net.std(ddof=1) * np.sqrt(trading_days_per_year))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    # Max drawdown
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd   = float(drawdown.min())
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")

    return BacktestResult(
        annualised_return=ann_ret,
        annualised_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        long_minus_short=float(gross.sum()),
        net_long_minus_short=float(net.sum()),
        avg_daily_turnover=float(pnl_df["turnover"].mean()),
        total_cost=float(pnl_df["cost"].sum()),
        daily_pnl=net,
        cumulative_return=cum,
        cost_bps=cost_bps,
        n_days=len(pnl_df),
        n_decile=n_decile,
    )
