from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from backend.db.session import get_db, get_session_factory, make_engine
from backend.main import create_app
from backend.routers.chat import get_llm

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False
    return cfg


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def migrated_url(db_url) -> str:
    command.upgrade(alembic_config(db_url), "head")
    return db_url


@pytest.fixture
def session_factory(migrated_url):
    engine = make_engine(migrated_url)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def db(session_factory):
    with session_factory() as s:
        yield s


@pytest.fixture
def llm_holder() -> dict:
    """Tests put a FakeLLM in llm_holder["llm"]; None simulates a missing API key."""
    return {"llm": None}


@pytest.fixture
def client(session_factory, llm_holder):
    app = create_app()

    def _get_db():
        with session_factory() as s:
            yield s

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_llm] = lambda: llm_holder["llm"]
    with TestClient(app) as c:
        yield c
