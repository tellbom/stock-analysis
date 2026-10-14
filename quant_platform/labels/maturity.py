"""
quant_platform.labels.maturity
================================
Time-convention helpers: feature cutoff vs. label maturity.

Project-wide T+1 close convention
----------------------------------
* Feature date ``d`` is a trading day (the day features are computed, using
  information available at ``d``'s close).
* Entry is at the close of the NEXT trading day (``d + 1`` trading day).
* The D+h label for feature date ``d`` is::

      ret_fwd_{h}d = close(d + h + 1) / close(d + 1) - 1

  i.e. buy at the close of d+1, sell at the close of d+h+1.
* The label **matures** (becomes observable) at the close of trading day
  ``d + h + 1``.

Two dates that must never be confused
--------------------------------------
* ``as_of`` (feature cutoff): the last feature date used. Model scores are
  produced for ``date == as_of``; entry is the first trading day after it.
* ``last_mature_feature_date(h, as_of)``: the last feature date whose D+h
  label is fully observable at ``as_of``. Training for horizon ``h`` must
  use only ``date <= last_mature_feature_date`` — never ``date <= as_of``,
  which would leak immature (NaN or future) labels.

Weekly cadence example (trading days only)
-------------------------------------------
as_of = Wed 2026-10-14  ->  entry = Thu 2026-10-15 (future[0])
D+3 expiry = Tue 2026-10-20 (future[3]); D+5 expiry = Thu 2026-10-22 (future[5])
D+3 last mature feature date at as_of = Thu 2026-10-08 (as_of - 4 trading days)
"""

from __future__ import annotations

import datetime as dt


def _sorted_days(trading_days) -> list[dt.date]:
    days = sorted(set(trading_days))
    if not days:
        raise ValueError("trading_days is empty")
    return days


def entry_date(as_of: dt.date, trading_days) -> dt.date:
    """First trading day strictly after ``as_of`` (T+1 entry)."""
    days = _sorted_days(trading_days)
    for d in days:
        if d > as_of:
            return d
    raise ValueError(f"No trading day after as_of={as_of}")


def expiry_date(as_of: dt.date, horizon: int, trading_days) -> dt.date:
    """Expiry (observation) date for a D+h position entered after ``as_of``.

    future = trading days after as_of; entry = future[0]; expiry = future[h].
    Matches the ``target_date = future[h]`` convention in both weekly entries.
    """
    if horizon not in (3, 5):
        raise ValueError(f"Only D+3/D+5 are supported, got {horizon}")
    days = _sorted_days(trading_days)
    future = [d for d in days if d > as_of]
    if len(future) <= horizon:
        raise ValueError(
            f"Trading calendar does not cover the D+{horizon} expiry after {as_of}"
        )
    return future[horizon]


def label_maturity_date(feature_date: dt.date, horizon: int, trading_days) -> dt.date:
    """Close date at which the D+h label for ``feature_date`` is observable."""
    if horizon not in (3, 5):
        raise ValueError(f"Only D+3/D+5 are supported, got {horizon}")
    days = _sorted_days(trading_days)
    if feature_date not in days:
        raise ValueError(f"feature_date={feature_date} is not a trading day")
    idx = days.index(feature_date) + horizon + 1
    if idx >= len(days):
        raise ValueError(
            f"D+{horizon} label for {feature_date} matures beyond the calendar"
        )
    return days[idx]


def last_mature_feature_date(as_of: dt.date, horizon: int, trading_days) -> dt.date:
    """Last feature date whose D+h label is fully observable at ``as_of``.

    Training for horizon ``h`` at cutoff ``as_of`` must use
    ``date <= last_mature_feature_date(...)`` — using ``date <= as_of``
    would include immature labels (NaN after the in-memory recompute, or
    worse, stale stored labels leaking the future).
    """
    days = _sorted_days(trading_days)
    mature = []
    for d in days:
        try:
            if label_maturity_date(d, horizon, days) <= as_of:
                mature.append(d)
        except ValueError:
            # 标签成熟日超出日历范围：该特征日本来就不成熟，跳过。
            continue
    if not mature:
        raise ValueError(f"No mature D+{horizon} feature date at as_of={as_of}")
    return max(mature)


def training_mask(dates, as_of: dt.date, horizon: int, trading_days) -> list[bool]:
    """Boolean mask: which feature dates may train the D+h model at ``as_of``."""
    cutoff = last_mature_feature_date(as_of, horizon, trading_days)
    return [d <= cutoff for d in dates]
