"""
store.duckdb_views
==================
DuckDB connection factory and gold-layer view definitions.

Design
------
- DuckDB runs in **in-process mode** (no server).  Connections are cheap;
  callers open one per query session and close it when done.
- ``get_connection(store_root)`` returns a DuckDB connection with all
  standard views pre-registered.  Callers run SQL against it directly.
- Views are defined over ``read_parquet(glob)`` so they always reflect
  the current state of the lake without manual refresh.
- No data is stored in DuckDB itself — it is a query engine only.

Standard views
--------------
  ohlcv         silver/ohlcv/*.parquet          — all symbols, all dates
  calendar      calendar/trading_calendar.parquet
  universe      universe/<key>/membership.parquet (one universe at a time)

Usage
-----
    from quant_platform.store.duckdb_views import get_connection
    con = get_connection('/path/to/lake', universe_key='csi300')
    df  = con.execute('SELECT * FROM ohlcv WHERE symbol = ? LIMIT 5',
                      ['600519']).fetchdf()
    con.close()
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from quant_platform.store.lake import (
    ohlcv_dir,
    calendar_path,
    universe_root,
)
from quant_platform.core.logging import get_logger

logger = get_logger(__name__)


def get_connection(
    store_root: Path | str,
    universe_key: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """
    Return an in-process DuckDB connection with standard views registered.

    Parameters
    ----------
    store_root : Path | str
        Root of the Parquet data lake.
    universe_key : str | None
        If given, registers a ``universe`` view pointing at
        ``universe/<universe_key>/membership.parquet``.
        If None, the ``universe`` view is not created.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Caller is responsible for calling ``.close()`` when done.

    Notes
    -----
    Views that reference non-existent paths are skipped with a warning
    rather than raising — the lake may be partially populated during
    initial ingestion runs.
    """
    root = Path(store_root)
    con  = duckdb.connect()

    _register_ohlcv_view(con, root)
    _register_calendar_view(con, root)
    if universe_key:
        _register_universe_view(con, root, universe_key)

    return con


# ---------------------------------------------------------------------------
# View registrations
# ---------------------------------------------------------------------------

def _register_ohlcv_view(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    """Register ``ohlcv`` view over all silver OHLCV Parquet files."""
    glob = str(ohlcv_dir(root) / "*.parquet")
    files = list(ohlcv_dir(root).glob("*.parquet"))

    if not files:
        logger.warning(
            "No OHLCV Parquet files found at %s — 'ohlcv' view will be empty. "
            "Run the OHLCV collector (T0.5) first.",
            ohlcv_dir(root),
        )
        # Register an empty view so queries don't crash with "view not found"
        con.execute(
            "CREATE VIEW ohlcv AS "
            "SELECT NULL::VARCHAR AS symbol, NULL::DATE AS date, "
            "NULL::DOUBLE AS open, NULL::DOUBLE AS high, "
            "NULL::DOUBLE AS low,  NULL::DOUBLE AS close, "
            "NULL::DOUBLE AS volume WHERE FALSE"
        )
        return

    con.execute(f"CREATE VIEW ohlcv AS SELECT * FROM read_parquet('{glob}')")
    logger.info("Registered view 'ohlcv' over %d files at %s", len(files), ohlcv_dir(root))


def _register_calendar_view(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    """Register ``calendar`` view over the trading calendar Parquet."""
    cal_path = calendar_path(root)

    if not cal_path.exists():
        logger.warning(
            "Calendar Parquet not found at %s — 'calendar' view will be empty. "
            "Run CalendarService.build_and_save() first.",
            cal_path,
        )
        con.execute(
            "CREATE VIEW calendar AS "
            "SELECT NULL::DATE AS date, NULL::BOOLEAN AS is_trading, "
            "NULL::VARCHAR AS source WHERE FALSE"
        )
        return

    con.execute(f"CREATE VIEW calendar AS SELECT * FROM read_parquet('{cal_path}')")
    logger.info("Registered view 'calendar' over %s", cal_path)


def _register_universe_view(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    universe_key: str,
) -> None:
    """Register ``universe`` view over a specific universe's membership Parquet."""
    membership_path = universe_root(root) / universe_key / "membership.parquet"

    if not membership_path.exists():
        logger.warning(
            "Universe membership Parquet not found at %s — 'universe' view will be empty. "
            "Run UniverseService.fetch_and_save() or load_from_csv() first.",
            membership_path,
        )
        con.execute(
            "CREATE VIEW universe AS "
            "SELECT NULL::VARCHAR AS symbol, NULL::DATE AS in_date, "
            "NULL::DATE AS out_date, NULL::VARCHAR AS name WHERE FALSE"
        )
        return

    con.execute(f"CREATE VIEW universe AS SELECT * FROM read_parquet('{membership_path}')")
    logger.info(
        "Registered view 'universe' for key '%s' over %s",
        universe_key, membership_path,
    )


# ---------------------------------------------------------------------------
# Convenience query helpers
# ---------------------------------------------------------------------------

def query(
    store_root: Path | str,
    sql: str,
    params: list | None = None,
    universe_key: str | None = None,
) -> "pd.DataFrame":
    """
    Open a connection, run *sql*, return a DataFrame, close the connection.
    Convenience wrapper for one-shot queries.

    Parameters
    ----------
    sql : str
        DuckDB SQL.  Use ``?`` placeholders for *params*.
    params : list | None
        Positional parameters for the query.
    universe_key : str | None
        If given, the ``universe`` view is available in the query.
    """
    con = get_connection(store_root, universe_key=universe_key)
    try:
        result = con.execute(sql, params or [])
        return result.fetchdf()
    finally:
        con.close()
