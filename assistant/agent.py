"""Agent loop for one chat turn: stream text, run tools, persist the conversation (M5.4, M5.5).

Yields UI events (dicts) that the /chat endpoint serialises as SSE:
    {"type": "session", "session_id": "..."}
    {"type": "delta", "text": "..."}
    {"type": "tool_start", "name": "...", "label": "Đang tra cứu ..."}
    {"type": "tool_end", "name": "...", "ok": true}
    {"type": "error", "message": "..."}
    {"type": "done", "message_id": 123}
"""
from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from collections.abc import Callable, Iterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from assistant.llm import LLMClient, LLMError, TextDelta, TurnEnd
from assistant.prompts import build_system_prompt
from assistant.tools import TOOLS, PageContext, dumps, execute_tool, tool_specs
from backend.db.models import ChatMessage, ChatSession
from backend.services.common import latest_prediction_date

log = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 20  # user + assistant messages kept as history (PLAN M5.5)
MAX_TOOL_ROUNDS = 6
TOOL_RESULT_LOG_CHARS = 4000  # tool results stored in chat_messages.tool_calls (truncated)


def _get_or_create_session(db: Session, session_id: uuid.UUID | None) -> ChatSession:
    if session_id is not None:
        s = db.get(ChatSession, session_id)
        if s is not None:
            return s
    s = ChatSession(id=session_id or uuid.uuid4())
    db.add(s)
    db.commit()
    return s


def load_history(db: Session, session_id: uuid.UUID, limit: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id, ChatMessage.role.in_(("user", "assistant")))
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    ).all()
    history = [{"role": r.role, "content": r.content} for r in reversed(rows) if r.content]
    while history and history[0]["role"] != "user":  # providers want the conversation to start with a user turn
        history.pop(0)
    return history


def run_chat(
    db: Session,
    llm: LLMClient,
    message: str,
    session_id: uuid.UUID | None = None,
    page_ticker: str | None = None,
    page_date: dt.date | None = None,
    today: dt.date | None = None,
    tool_observer: Callable[[str, dict, dict], None] | None = None,
) -> Iterator[dict]:
    """`tool_observer(name, arguments, full_result)` is called after every tool run (used by the eval)."""
    session = _get_or_create_session(db, session_id)
    yield {"type": "session", "session_id": str(session.id)}

    ctx = PageContext(ticker=page_ticker.upper() if page_ticker else None, date=page_date)
    system = build_system_prompt(ctx.ticker, ctx.date, latest_prediction_date(db), today)
    messages = load_history(db, session.id) + [{"role": "user", "content": message}]
    db.add(ChatMessage(session_id=session.id, role="user", content=message))
    db.commit()

    specs = tool_specs()
    answer_parts: list[str] = []
    trace: list[dict] = []
    started = time.perf_counter()
    first_token_ms: float | None = None
    error: str | None = None

    try:
        for _round in range(MAX_TOOL_ROUNDS + 1):
            end: TurnEnd | None = None
            for ev in llm.stream(system, messages, specs):
                if isinstance(ev, TextDelta):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    answer_parts.append(ev.text)
                    yield {"type": "delta", "text": ev.text}
                else:
                    end = ev
            if end is None or not end.tool_calls:
                break
            if _round == MAX_TOOL_ROUNDS:
                error = "Trợ lý gọi quá nhiều công cụ trong một câu hỏi, vui lòng hỏi cụ thể hơn."
                break
            if end.text and not end.text.endswith("\n"):
                answer_parts.append("\n\n")
                yield {"type": "delta", "text": "\n\n"}
            messages.append(llm.assistant_message(end))
            for call in end.tool_calls:
                tool = TOOLS.get(call.name)
                yield {"type": "tool_start", "name": call.name, "label": tool.label if tool else "Đang tra cứu dữ liệu"}
                if call.parse_error:
                    result, ok = {"error": f"Tham số không phải JSON hợp lệ: {call.parse_error}"}, False
                else:
                    result, ok = execute_tool(db, call.name, call.arguments, ctx)
                if tool_observer is not None:
                    tool_observer(call.name, call.arguments, result)
                content = dumps(result)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
                trace.append({"name": call.name, "arguments": call.arguments, "ok": ok, "result": content[:TOOL_RESULT_LOG_CHARS]})
                yield {"type": "tool_end", "name": call.name, "ok": ok}
    except LLMError as exc:
        error = str(exc)
    except Exception:  # never leave the stream hanging
        log.exception("chat turn failed (session=%s)", session.id)
        error = "Đã xảy ra lỗi khi tạo câu trả lời, vui lòng thử lại."

    answer = "".join(answer_parts).strip()
    if error:
        yield {"type": "error", "message": error}
    row = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=answer,
        tool_calls={
            "calls": trace,
            "first_token_ms": None if first_token_ms is None else round(first_token_ms),
            "total_ms": round((time.perf_counter() - started) * 1000),
            "error": error,
            "provider": llm.provider,
            "model": llm.model,
        },
    )
    db.add(row)
    db.commit()
    yield {"type": "done", "message_id": row.id}


def collect(events: Iterator[dict]) -> dict:
    """Consume run_chat events into {session_id, text, tools, error} (used by tests and the eval script)."""
    out: dict = {"session_id": None, "text": "", "tools": [], "error": None, "message_id": None}
    for ev in events:
        t = ev["type"]
        if t == "session":
            out["session_id"] = ev["session_id"]
        elif t == "delta":
            out["text"] += ev["text"]
        elif t == "tool_start":
            out["tools"].append(ev["name"])
        elif t == "error":
            out["error"] = ev["message"]
        elif t == "done":
            out["message_id"] = ev["message_id"]
    return out
