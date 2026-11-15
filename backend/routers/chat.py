"""POST /chat (SSE stream) and GET /chat/{session_id}/messages."""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from assistant.agent import run_chat
from assistant.llm import LLMClient, make_llm
from backend.config import get_settings
from backend.db.models import ChatMessage, ChatSession
from backend.db.session import get_db, get_session_factory
from backend.schemas import ChatMessageOut, ChatRequest

router = APIRouter(tags=["chat"])

NOT_CONFIGURED = "Trợ lý AI chưa được cấu hình (thiếu LLM_API_KEY)."


@lru_cache
def _cached_llm(provider: str, api_key: str, model: str) -> LLMClient:
    return make_llm(provider, api_key, model or None)


def get_llm() -> LLMClient | None:
    s = get_settings()
    if not s.llm_api_key and s.llm_provider != "fake":
        return None
    return _cached_llm(s.llm_provider, s.llm_api_key, s.llm_model)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post(
    "/chat",
    summary="Trợ lý AI (SSE stream)",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Luồng sự kiện SSE"}},
)
def chat(
    req: ChatRequest,
    factory: sessionmaker[Session] = Depends(get_session_factory),
    llm: LLMClient | None = Depends(get_llm),
) -> StreamingResponse:
    def events() -> Iterator[str]:
        if llm is None:
            yield _sse({"type": "error", "message": NOT_CONFIGURED})
            yield _sse({"type": "done", "message_id": None})
            return
        with factory() as db:
            for ev in run_chat(db, llm, req.message, req.session_id, req.context.ticker, req.context.date):
                yield _sse(ev)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/{session_id}/messages", response_model=list[ChatMessageOut], summary="Lịch sử một phiên chat")
def chat_messages(session_id: uuid.UUID, db: Session = Depends(get_db)) -> list[ChatMessageOut]:
    if db.get(ChatSession, session_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên trò chuyện.")
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id, ChatMessage.role.in_(("user", "assistant")))
        .order_by(ChatMessage.id)
    ).all()
    return [
        ChatMessageOut(
            id=r.id,
            role=r.role,
            content=r.content,
            tools=[c["name"] for c in (r.tool_calls or {}).get("calls", [])] if isinstance(r.tool_calls, dict) else [],
            created_at=r.created_at,
        )
        for r in rows
    ]
