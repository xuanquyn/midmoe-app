"""Engine / session factory and the FastAPI DB dependency."""
from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings


def make_engine(url: str, echo: bool = False) -> Engine:
    kwargs: dict = {"echo": echo, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_fk_on(dbapi_conn, _):  # SQLite ignores FKs unless asked
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


@lru_cache
def get_engine() -> Engine:
    s = get_settings()
    return make_engine(s.database_url, echo=s.sql_echo)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session_factory() -> sessionmaker[Session]:
    """FastAPI dependency for code that outlives the request scope (SSE streams open their own session)."""
    return get_sessionmaker()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
