"""Provider abstraction for chat + function calling + streaming (M5.1).

Provider-neutral message format used by the agent loop:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall...], "raw": <provider object>, "provider": str}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}   # JSON string

`raw` lets an adapter replay its own assistant turn unchanged inside one agent turn (Anthropic thinking blocks,
Gemini thought signatures must be sent back as-is). History from earlier turns is stored as plain text only.

`LLMClient.stream(system, messages, tools)` yields `TextDelta` events and ends with exactly one `TurnEnd`.
"""
from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "gemini": "gemini-flash-latest",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-opus-5-5",
}
REQUEST_TIMEOUT_S = 60.0


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema (type=object)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    parse_error: str | None = None  # arguments were not valid JSON


@dataclass
class TextDelta:
    text: str


@dataclass
class TurnEnd:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    stop_reason: str | None = None


StreamEvent = TextDelta | TurnEnd


class LLMError(Exception):
    """Provider failure, message is user-facing (Vietnamese)."""


class LLMClient(ABC):
    provider: str = "base"

    def __init__(self, model: str | None = None):
        self.model = model or DEFAULT_MODELS.get(self.provider, "")

    @abstractmethod
    def stream(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> Iterator[StreamEvent]: ...

    def assistant_message(self, end: TurnEnd) -> dict:
        return {
            "role": "assistant",
            "content": end.text,
            "tool_calls": end.tool_calls,
            "raw": end.raw,
            "provider": self.provider,
        }


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"


def _parse_args(s: str | None) -> tuple[dict, str | None]:
    if not s:
        return {}, None
    try:
        val = json.loads(s)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc}"
    return (val, None) if isinstance(val, dict) else ({}, "arguments must be a JSON object")


def _group_tool_messages(messages: list[dict]) -> list[list[dict] | dict]:
    """Collapse consecutive tool messages into one list (providers want all results of a turn together)."""
    out: list[list[dict] | dict] = []
    for m in messages:
        if m["role"] == "tool":
            if out and isinstance(out[-1], list):
                out[-1].append(m)
            else:
                out.append([m])
        else:
            out.append(m)
    return out


# ============================================================================ Anthropic
class AnthropicClient(LLMClient):
    provider = "anthropic"
    # Models that accept server-side refusal fallbacks ("default" routes by refusal category).
    FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")
    MAX_TOKENS = 16000

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(model)
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key or None, timeout=REQUEST_TIMEOUT_S)

    def _messages(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in _group_tool_messages(messages):
            if isinstance(m, list):
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": t["tool_call_id"], "content": t["content"]}
                            for t in m
                        ],
                    }
                )
            elif m["role"] == "assistant":
                if m.get("provider") == self.provider and m.get("raw") is not None:
                    out.append({"role": "assistant", "content": m["raw"]})  # replay unchanged (thinking blocks)
                else:
                    out.append({"role": "assistant", "content": m.get("content") or ""})
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    def stream(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> Iterator[StreamEvent]:
        anthropic = self._anthropic
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.MAX_TOKENS,
            "system": system,
            "messages": self._messages(messages),
            "tools": [
                {"name": t.name, "description": t.description, "input_schema": t.parameters, "eager_input_streaming": True}
                for t in tools
            ],
        }
        use_fallbacks = self.model.startswith(self.FALLBACK_MODELS)
        api = self.client.beta.messages if use_fallbacks else self.client.messages
        if use_fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

        for attempt in range(3):
            parts: list[str] = []
            try:
                with api.stream(**kwargs) as stream:
                    for event in stream:
                        if event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                            parts.append(event.delta.text)
                            yield TextDelta(event.delta.text)
                    final = stream.get_final_message()
                break
            except ValueError:
                # eager input streaming: tool input JSON the SDK could not parse at all -> re-issue the turn
                if attempt == 2 or parts:
                    raise LLMError("Mô hình trả về lời gọi công cụ không hợp lệ, vui lòng thử lại.") from None
            except anthropic.RateLimitError as exc:
                raise LLMError("Trợ lý đang quá tải (giới hạn API), vui lòng thử lại sau ít phút.") from exc
            except anthropic.AuthenticationError as exc:
                raise LLMError("Khoá API của trợ lý không hợp lệ.") from exc
            except anthropic.APIStatusError as exc:
                raise LLMError(f"Lỗi từ nhà cung cấp LLM ({exc.status_code}).") from exc
            except anthropic.APIConnectionError as exc:
                raise LLMError("Không kết nối được tới nhà cung cấp LLM.") from exc

        stop = final.stop_reason
        tool_calls = []
        if stop not in ("refusal", "max_tokens"):  # never run a truncated / refused tool call
            for b in final.content:
                if b.type == "tool_use":
                    args = b.input if isinstance(b.input, dict) else {}
                    tool_calls.append(ToolCall(id=b.id, name=b.name, arguments=args))
        text = "".join(parts)
        if stop == "refusal" and not text:
            text = "Xin lỗi, tôi không thể hỗ trợ yêu cầu này."
            yield TextDelta(text)
        yield TurnEnd(text=text, tool_calls=tool_calls, raw=final.content, stop_reason=stop)


# ============================================================================ OpenAI
class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(model)
        import openai

        self._openai = openai
        self.client = openai.OpenAI(api_key=api_key or None, timeout=REQUEST_TIMEOUT_S)

    def _messages(self, system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                        }
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    def stream(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> Iterator[StreamEvent]:
        openai = self._openai
        parts: list[str] = []
        calls: dict[int, dict] = {}
        finish = None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, messages),
                tools=[
                    {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                    for t in tools
                ],
                stream=True,
            )
            for chunk in resp:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    parts.append(delta.content)
                    yield TextDelta(delta.content)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
                finish = choice.finish_reason or finish
        except openai.RateLimitError as exc:
            raise LLMError("Trợ lý đang quá tải (giới hạn API), vui lòng thử lại sau ít phút.") from exc
        except openai.AuthenticationError as exc:
            raise LLMError("Khoá API của trợ lý không hợp lệ.") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"Lỗi từ nhà cung cấp LLM ({exc.status_code}).") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Không kết nối được tới nhà cung cấp LLM.") from exc

        tool_calls = []
        if finish != "length":
            for _, slot in sorted(calls.items()):
                args, err = _parse_args(slot["args"])
                tool_calls.append(ToolCall(id=slot["id"] or _new_id(), name=slot["name"], arguments=args, parse_error=err))
        yield TurnEnd(text="".join(parts), tool_calls=tool_calls, raw=None, stop_reason=finish)


# ============================================================================ Gemini
class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(model)
        from google import genai
        from google.genai import errors, types

        self._types = types
        self._errors = errors
        self.client = genai.Client(
            api_key=api_key or None, http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_S * 1000))
        )

    def _contents(self, messages: list[dict]) -> list:
        types = self._types
        out = []
        for m in _group_tool_messages(messages):
            if isinstance(m, list):
                parts = []
                for t in m:
                    try:
                        payload = json.loads(t["content"])
                    except json.JSONDecodeError:
                        payload = t["content"]
                    resp = payload if isinstance(payload, dict) else {"result": payload}
                    part = types.Part.from_function_response(name=t["name"], response=resp)
                    if part.function_response is not None:
                        part.function_response.id = t["tool_call_id"]
                    parts.append(part)
                out.append(types.Content(role="user", parts=parts))
            elif m["role"] == "assistant":
                if m.get("provider") == self.provider and m.get("raw") is not None:
                    out.append(m["raw"])  # keeps thought signatures
                else:
                    out.append(types.Content(role="model", parts=[types.Part(text=m.get("content") or "")]))
            else:
                out.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
        return out

    def stream(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> Iterator[StreamEvent]:
        types, errors = self._types, self._errors
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                        for t in tools
                    ]
                )
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        text_parts: list[str] = []
        all_parts: list = []
        tool_calls: list[ToolCall] = []
        finish = None
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model, contents=self._contents(messages), config=config
            ):
                if not chunk.candidates:
                    continue
                cand = chunk.candidates[0]
                finish = cand.finish_reason or finish
                for part in (cand.content.parts if cand.content and cand.content.parts else []):
                    all_parts.append(part)
                    if part.function_call:
                        fc = part.function_call
                        tool_calls.append(ToolCall(id=fc.id or _new_id(), name=fc.name or "", arguments=dict(fc.args or {})))
                    elif part.text and not part.thought:
                        text_parts.append(part.text)
                        yield TextDelta(part.text)
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code == 429:
                raise LLMError("Trợ lý đang quá tải (giới hạn API), vui lòng thử lại sau ít phút.") from exc
            if code in (401, 403):
                raise LLMError("Khoá API của trợ lý không hợp lệ.") from exc
            raise LLMError(f"Lỗi từ nhà cung cấp LLM ({code}).") from exc
        raw = types.Content(role="model", parts=all_parts) if all_parts else None
        yield TurnEnd(text="".join(text_parts), tool_calls=tool_calls, raw=raw, stop_reason=str(finish) if finish else None)


# ============================================================================ Fake (tests / offline demo)
class FakeLLM(LLMClient):
    """Scripted client. Each script item is either a str (final text answer) or a list of
    (tool_name, arguments) tuples (tool calls for that round). Records every request it receives."""

    provider = "fake"

    def __init__(self, script: list[str | list[tuple[str, dict]]] | None = None, chunk: int = 12):
        super().__init__("fake")
        self.script = list(script or [])
        self.chunk = chunk
        self.requests: list[dict] = []

    def stream(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> Iterator[StreamEvent]:
        self.requests.append({"system": system, "messages": [dict(m) for m in messages], "tools": [t.name for t in tools]})
        step = self.script.pop(0) if self.script else "Xin lỗi, tôi không có câu trả lời."
        if isinstance(step, str):
            for i in range(0, len(step), self.chunk):
                yield TextDelta(step[i : i + self.chunk])
            yield TurnEnd(text=step, stop_reason="stop")
        else:
            calls = [ToolCall(id=_new_id(), name=n, arguments=a) for n, a in step]
            yield TurnEnd(text="", tool_calls=calls, stop_reason="tool_calls")


PROVIDERS: dict[str, type[LLMClient]] = {"anthropic": AnthropicClient, "openai": OpenAIClient, "gemini": GeminiClient}


def make_llm(provider: str, api_key: str, model: str | None = None) -> LLMClient:
    provider = provider.lower().strip()
    if provider == "fake":
        return FakeLLM()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)} (got {provider!r})")
    return PROVIDERS[provider](api_key=api_key, model=model or None)
