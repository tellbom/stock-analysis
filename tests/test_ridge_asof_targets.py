import datetime as dt

import numpy as np
import pandas as pd

from temp.latest_dual_horizon import restrict_targets_as_of


def test_historical_ridge_never_uses_later_rebuilt_labels():
    dates = [dt.date.fromisoformat(d) for d in (
        "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24",
        "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09",
    )]
    prices = pd.DataFrame({"symbol": "000001", "date": dates, "close": np.arange(10., 19.)})
    # The store now contains later outcomes, including at the signal date.
    panel = prices.assign(factor=1., ret_fwd_3d=999., ret_fwd_5d=999.)
    cutoff = dt.date(2026, 9, 30)
    actual = restrict_targets_as_of(panel, prices, cutoff)
    assert actual.loc[actual.ret_fwd_3d.notna(), "date"].max() == dt.date(2026, 9, 23)
    assert actual.loc[actual.ret_fwd_5d.notna(), "date"].max() == dt.date(2026, 9, 21)
    assert actual.iloc[0].ret_fwd_3d == 14. / 11. - 1
    assert actual.iloc[0].ret_fwd_5d == 16. / 11. - 1
    assert pd.isna(actual.loc[actual.date == cutoff, "ret_fwd_3d"]).all()
    assert not any(c.startswith("label_end_") for c in actual.columns)
    prices.loc[prices.date > cutoff, "close"] = 1e9
    pd.testing.assert_frame_equal(actual, restrict_targets_as_of(panel, prices, cutoff))
