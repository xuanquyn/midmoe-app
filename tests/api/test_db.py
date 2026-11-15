import json
from datetime import date

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from backend.db.models import Base, ChatMessage, ChatSession, Explanation, News, Prediction, Ticker
from backend.db.seed import load_ticker_rows, seed_tickers
from backend.db.session import make_engine
from tests.api.conftest import alembic_config

TABLES = {"tickers", "prices", "news", "predictions", "explanations", "pipeline_runs", "chat_sessions", "chat_messages"}


def test_migration_matches_models(migrated_url):
    engine = make_engine(migrated_url)
    with engine.connect() as conn:
        assert TABLES <= set(inspect(conn).get_table_names())
        diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
    engine.dispose()
    assert diff == [], f"models and migration differ: {diff}"


def test_downgrade_to_base(migrated_url):
    command.downgrade(alembic_config(migrated_url), "base")
    engine = make_engine(migrated_url)
    with engine.connect() as conn:
        assert not (TABLES & set(inspect(conn).get_table_names()))
    engine.dispose()


def test_fk_and_unique_constraints(db):
    db.add(Ticker(ticker="FPT", sort_order=0))
    db.commit()
    db.add(Prediction(date=date(2026, 11, 13), ticker="XXX", score=0.1, score_std=0.0, rank=1, top5_votes=10, model_version="v1"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    db.add(News(ticker="FPT", date=date(2026, 11, 13), title="a", url="https://cafef.vn/1"))
    db.commit()
    db.add(News(ticker="FPT", date=date(2026, 11, 14), title="b", url="https://cafef.vn/1"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_json_payload_and_chat_cascade(db):
    db.add(Ticker(ticker="FPT", sort_order=0))
    db.flush()  # tickers must exist before FK rows (no ORM relationship to order the flush)
    db.add(Explanation(date=date(2026, 11, 13), ticker="FPT", payload={"rank": 2, "summary": "Xếp hạng 2/78"}))
    s = ChatSession()
    s.messages.append(ChatMessage(role="user", content="mã này thì sao?", tool_calls=[{"name": "x"}]))
    db.add(s)
    db.commit()
    assert db.get(Explanation, (date(2026, 11, 13), "FPT")).payload["summary"] == "Xếp hạng 2/78"
    db.delete(s)
    db.commit()
    assert db.query(ChatMessage).count() == 0


def test_seed_tickers_idempotent(db, tmp_path):
    (tmp_path / "vn_tickers_sorted.json").write_text(json.dumps(["VCB", "BID", "FPT"]))
    (tmp_path / "icb_industry.csv").write_text("symbol,organ_name,icb_name2\nVCB,Vietcombank,Ngân hàng\nFPT,FPT Corp,Công nghệ\n")
    rows = load_ticker_rows(tmp_path)
    assert seed_tickers(db, rows) == 3
    assert seed_tickers(db, rows) == 3
    assert db.query(Ticker).count() == 3
    fpt = db.get(Ticker, "FPT")
    assert (fpt.sort_order, fpt.name, fpt.icb_sector) == (2, "FPT Corp", "Công nghệ")
    assert db.get(Ticker, "BID").name is None
