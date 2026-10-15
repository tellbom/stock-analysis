"""Lifecycle helpers for integration tests using temporary SQLite stores."""
from contextlib import contextmanager
from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def mlflow_tempdir():
    """Dispose temporary SQLAlchemy pools before Windows removes the DB file.

    Real MLflow operations still run. Only engines pointing inside this
    test's directory are closed; production tracking code is unchanged.
    """
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    @contextmanager
    def directory():
        with tempfile.TemporaryDirectory() as tmp:
            engines = set()
            root = Path(tmp).resolve()

            def remember_engine(connection):
                engine = connection.engine
                database = engine.url.database
                if database and engine.url.drivername.startswith("sqlite"):
                    if Path(database).resolve().is_relative_to(root):
                        engines.add(engine)

            event.listen(Engine, "engine_connect", remember_engine)
            try:
                yield tmp
            finally:
                event.remove(Engine, "engine_connect", remember_engine)
                for engine in engines:
                    engine.dispose()

    return directory
