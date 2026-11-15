import json
import uuid

import pytest

from assistant.agent import MAX_HISTORY_MESSAGES, MAX_TOOL_ROUNDS, collect, load_history, run_chat
from assistant.llm import FakeLLM, LLMClient, LLMError, TextDelta
from backend.db.models import ChatMessage
from backend.db.demo import seed_demo


@pytest.fixture
def demo(db):
    return seed_demo(db)


def test_tool_then_answer(db, demo):
    llm = FakeLLM([[("get_recommendations", {"k": 3})], "Top 3 hôm nay là ..."])
    events = list(run_chat(db, llm, "Top 3 hôm nay?"))
    types = [e["type"] for e in events]
    assert types[0] == "session" and types[-1] == "done"
    assert types.index("tool_start") < types.index("tool_end") < types.index("delta")
    tool_start = next(e for e in events if e["type"] == "tool_start")
    assert tool_start["label"] == "Đang lấy danh sách khuyến nghị"

    # 2nd request saw the tool result
    tool_msg = llm.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["items"][0]["rank"] == 1

    rows = db.query(ChatMessage).order_by(ChatMessage.id).all()
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[1].content == "Top 3 hôm nay là ..."
    assert rows[1].tool_calls["calls"][0]["name"] == "get_recommendations" and rows[1].tool_calls["calls"][0]["ok"]
    assert rows[1].tool_calls["first_token_ms"] is not None


def test_history_and_page_context(db, demo):
    t = demo["tickers"][2]
    llm = FakeLLM(["Trả lời 1", "Trả lời 2"])
    first = collect(run_chat(db, llm, "Câu 1", page_ticker=t.lower()))
    sid = uuid.UUID(first["session_id"])
    collect(run_chat(db, llm, "mã này thì sao?", session_id=sid, page_ticker=t))
    assert f"Mã người dùng đang xem: {t}" in llm.requests[1]["system"]
    assert [m["role"] for m in llm.requests[1]["messages"]] == ["user", "assistant", "user"]
    assert llm.requests[1]["messages"][1]["content"] == "Trả lời 1"


def test_history_is_capped(db, demo):
    llm = FakeLLM([f"a{i}" for i in range(15)])
    sid = None
    for i in range(15):
        sid = uuid.UUID(collect(run_chat(db, llm, f"q{i}", session_id=sid))["session_id"])
    hist = load_history(db, sid)
    assert len(hist) == MAX_HISTORY_MESSAGES and hist[0]["role"] == "user" and hist[-1]["content"] == "a14"


def test_tool_error_goes_back_to_model(db, demo):
    llm = FakeLLM([[("get_stock_explanation", {"ticker": "AAPL"})], "Mã AAPL không nằm trong danh sách."])
    out = collect(run_chat(db, llm, "AAPL thì sao?"))
    assert out["error"] is None and out["tools"] == ["get_stock_explanation"]
    assert "không nằm trong danh sách" in json.loads(llm.requests[1]["messages"][-1]["content"])["error"]


def test_too_many_tool_rounds(db, demo):
    llm = FakeLLM([[("get_model_info", {})]] * (MAX_TOOL_ROUNDS + 2))
    out = collect(run_chat(db, llm, "?"))
    assert "quá nhiều công cụ" in out["error"] and len(out["tools"]) == MAX_TOOL_ROUNDS


class BrokenLLM(LLMClient):
    provider = "broken"

    def stream(self, system, messages, tools):
        yield TextDelta("Đang ")
        raise LLMError("Trợ lý đang quá tải (giới hạn API), vui lòng thử lại sau ít phút.")


def test_llm_error_is_reported_and_partial_text_saved(db, demo):
    events = list(run_chat(db, BrokenLLM(), "?"))
    assert [e["type"] for e in events][-2:] == ["error", "done"]
    row = db.query(ChatMessage).filter_by(role="assistant").one()
    assert row.content == "Đang" and "quá tải" in row.tool_calls["error"]


def test_unknown_session_id_creates_session(db, demo):
    sid = uuid.uuid4()
    out = collect(run_chat(db, FakeLLM(["ok"]), "hi", session_id=sid))
    assert out["session_id"] == str(sid)


# ------------------------------------------------------------------ HTTP / SSE
def _events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_chat_endpoint_streams_sse(client, demo, llm_holder):
    llm_holder["llm"] = FakeLLM([[("get_recommendations", {})], "Đây là top 5."])
    r = client.post("/api/chat", json={"message": "Top 5 hôm nay?", "context": {"date": str(demo["pred_dates"][2])}})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = _events(r)
    assert evs[0]["type"] == "session" and evs[-1]["type"] == "done"
    assert "".join(e["text"] for e in evs if e["type"] == "delta") == "Đây là top 5."
    tool_result = json.loads(llm_holder["llm"].requests[1]["messages"][-1]["content"])
    assert tool_result["date"] == str(demo["pred_dates"][2])  # page date used as default

    sid = evs[0]["session_id"]
    hist = client.get(f"/api/chat/{sid}/messages").json()
    assert [m["role"] for m in hist] == ["user", "assistant"] and hist[1]["tools"] == ["get_recommendations"]
    assert client.get(f"/api/chat/{uuid.uuid4()}/messages").status_code == 404
