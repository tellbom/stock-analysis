"""
ingest.index_proxy
==================
Offline CSI 300 index proxy builder.

When true 000300 index OHLCV is unavailable, this module computes a
temporary equal-weighted constituent-average proxy from the OHLCV files
already in the lake.  The proxy is clearly marked in ``_source`` so it is
not confused with real index data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.core.logging import get_logger
from quant_platform.store.lake import (
    index_ohlcv_path,
    init_lake,
    ohlcv_path,
)

logger = get_logger(__name__)

_PROXY_SYMBOL = "000300"
_PROXY_SOURCE = "equal_weighted_constituent_average"
_INITIAL_LEVEL = 3000.0


def build_index_proxy(
    store_root: Path | str,
    symbols: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Build an equal-weighted CSI 300 proxy from constituent OHLCV.

    If a non-proxy 000300 file already exists, it is kept unless
    ``overwrite=True`` is passed.
    """
    store_root = Path(store_root)
    init_lake(store_root)

    out_path = index_ohlcv_path(store_root, _PROXY_SYMBOL)
    if out_path.exists() and not overwrite:
        existing = pd.read_parquet(out_path)
        source = existing.get("_source", pd.Series([""])).iloc[0]
        if source != _PROXY_SOURCE:
            logger.info(
                "index_proxy: %s already exists with real data; skipping proxy build",
                out_path,
            )
            return out_path

    if symbols is None:
        symbols = _load_universe_symbols(store_root)
    if not symbols:
        raise ValueError("No symbols available to build the index proxy")

    logger.info(
        "index_proxy: building equal-weighted proxy from %d constituent OHLCV files",
        len(symbols),
    )

    close_frames: list[pd.DataFrame] = []
    for sym in symbols:
        path = ohlcv_path(store_root, sym)
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["date", "close"])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.rename(columns={"close": sym})
            close_frames.append(df.set_index("date"))
        except Exception as exc:
            logger.debug("index_proxy: could not read %s: %s", sym, exc)

    if not close_frames:
        raise ValueError("No OHLCV close data found in lake; run collect first")

    wide = pd.concat(close_frames, axis=1).sort_index()
    rets = wide.pct_change(fill_method=None)
    proxy_ret = rets.mean(axis=1, skipna=True)

    proxy_close = (1 + proxy_ret).cumprod() * _INITIAL_LEVEL
    proxy_close.iloc[0] = _INITIAL_LEVEL

    out_df = pd.DataFrame({
        "date": proxy_close.index,
        "open": proxy_close.values * 0.999,
        "high": proxy_close.values * 1.002,
        "low": proxy_close.values * 0.998,
        "close": proxy_close.values,
        "volume": np.nan,
        "amount": np.nan,
        "_source": _PROXY_SOURCE,
        "_n_constituents": wide.notna().sum(axis=1).values,
    })
    out_df = out_df.dropna(subset=["close"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    avg_constituents = out_df["_n_constituents"].mean()
    logger.info(
        "index_proxy: wrote %d rows -> %s (source=%s, constituents_avg=%.0f)",
        len(out_df),
        out_path,
        _PROXY_SOURCE,
        avg_constituents,
    )
    print(
        f"  [OK] CSI 300 proxy written: {len(out_df)} trading days, "
        f"avg {avg_constituents:.0f} constituents/day -> {out_path}"
    )
    return out_path


def _load_universe_symbols(store_root: Path) -> list[str]:
    """Load current CSI 300 universe symbols from the lake."""
    from quant_platform.store.lake import universe_root

    u_root = universe_root(store_root)
    if not u_root.exists():
        return []
    for directory in sorted(u_root.iterdir()):
        membership_path = directory / "membership.parquet"
        if not membership_path.exists():
            continue
        try:
            df = pd.read_parquet(membership_path)
            return df[df["out_date"].isna()]["symbol"].tolist()
        except Exception:
            continue
    return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build offline CSI 300 equal-weighted proxy from lake OHLCV"
    )
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    path = build_index_proxy(Path(args.store_root), overwrite=args.overwrite)
    print(f"Done: {path}")
    sys.exit(0)
