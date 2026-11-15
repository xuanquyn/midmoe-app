"""Provider adapters tested against mocked SDK streams (no network, no API key)."""
from types import SimpleNamespace as NS

import pytest

from assistant.llm import (
    AnthropicClient,
    GeminiClient,
    LLMError,
    OpenAIClient,
    TextDelta,
    ToolCall,
    ToolSpec,
    TurnEnd,
    make_llm,
)

TOOLS = [ToolSpec("get_model_info", "info", {"type": "object", "properties": {}, "additionalProperties": False})]


def _run(client, messages=None):
    events = list(client.stream("SYS", messages or [{"role": "user", "content": "hi"}], TOOLS))
    assert isinstance(events[-1], TurnEnd) and all(isinstance(e, TextDelta) for e in events[:-1])
    return events


def _history(provider: str, raw):
    return [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall("c1", "get_model_info", {}), ToolCall("c2", "get_model_info", {})],
            "raw": raw,
            "provider": provider,
        },
        {"role": "tool", "tool_call_id": "c1", "name": "get_model_info", "content": '{"a": 1}'},
        {"role": "tool", "tool_call_id": "c2", "name": "get_model_info", "content": '{"b": 2}'},
    ]


# ------------------------------------------------------------------ Anthropic
class _FakeStream:
    def __init__(self, events, final, exc=None):
        self.events, self.final, self.exc = events, final, exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        yield from self.events
        if self.exc:
            raise self.exc

    def get_final_message(self):
        return self.final


def _anthropic(streams):
    c = AnthropicClient(api_key="test")
    calls = []

    def stream(**kw):
        calls.append(kw)
        return streams.pop(0)

    c.client = NS(beta=NS(messages=NS(stream=stream)), messages=NS(stream=stream))
    return c, calls


def _delta(text):
    return NS(type="content_block_delta", delta=NS(type="text_delta", text=text))


def test_anthropic_text_and_tool_use():
    final = NS(
        stop_reason="tool_use",
        content=[NS(type="text", text="Để tôi xem."), NS(type="tool_use", id="tu1", name="get_model_info", input={})],
    )
    c, calls = _anthropic([_FakeStream([_delta("Để tôi "), _delta("xem.")], final)])
    events = _run(c)
    end = events[-1]
    assert "".join(e.text for e in events[:-1]) == "Để tôi xem." == end.text
    assert end.tool_calls == [ToolCall("tu1", "get_model_info", {})] and end.raw is final.content
    kw = calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["tools"][0]["eager_input_streaming"] is True and kw["system"] == "SYS"


def test_anthropic_replays_raw_and_groups_tool_results():
    raw = [NS(type="thinking"), NS(type="tool_use")]
    c, _ = _anthropic([])
    msgs = c._messages(_history("anthropic", raw))
    assert msgs[1] == {"role": "assistant", "content": raw}
    assert msgs[2]["role"] == "user" and [b["tool_use_id"] for b in msgs[2]["content"]] == ["c1", "c2"]
    # a turn produced by another provider is sent as plain text
    assert c._messages(_history("gemini", object()))[1] == {"role": "assistant", "content": ""}


def test_anthropic_refusal_runs_no_tools():
    final = NS(stop_reason="refusal", content=[NS(type="tool_use", id="x", name="get_model_info", input={})])
    c, _ = _anthropic([_FakeStream([], final)])
    end = _run(c)[-1]
    assert end.tool_calls == [] and end.text.startswith("Xin lỗi")


def test_anthropic_retries_unparseable_tool_json_once():
    ok = NS(stop_reason="end_turn", content=[NS(type="text", text="ok")])
    c, calls = _anthropic([_FakeStream([], None, exc=ValueError("bad json")), _FakeStream([_delta("ok")], ok)])
    assert _run(c)[-1].text == "ok" and len(calls) == 2


def test_anthropic_non_fallback_model_uses_plain_endpoint():
    c, calls = _anthropic([_FakeStream([], NS(stop_reason="end_turn", content=[]))])
    c.model = "claude-haiku-5-5"
    _run(c)
    assert "fallbacks" not in calls[0] and "betas" not in calls[0]


# ------------------------------------------------------------------ OpenAI
def _chunk(content=None, tool_calls=None, finish=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls), finish_reason=finish)])


def _tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def _openai(chunks):
    c = OpenAIClient(api_key="test")
    calls = []

    def create(**kw):
        calls.append(kw)
        return iter(chunks)

    c.client = NS(chat=NS(completions=NS(create=create)))
    return c, calls


def test_openai_accumulates_tool_call_deltas():
    chunks = [
        _chunk("Xin "),
        _chunk("chào"),
        _chunk(tool_calls=[_tc(0, "c1", "get_model_info", '{"k"'), _tc(1, "c2", "get_model_info", "{")]),
        _chunk(tool_calls=[_tc(0, args=": 3}"), _tc(1, args="bad")], finish="tool_calls"),
    ]
    c, calls = _openai(chunks)
    end = _run(c)[-1]
    assert end.text == "Xin chào"
    assert end.tool_calls[0] == ToolCall("c1", "get_model_info", {"k": 3})
    assert end.tool_calls[1].arguments == {} and end.tool_calls[1].parse_error
    assert calls[0]["messages"][0] == {"role": "system", "content": "SYS"} and calls[0]["stream"] is True


def test_openai_message_conversion():
    c, _ = _openai([])
    msgs = c._messages("SYS", _history("openai", None))
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "get_model_info", "arguments": "{}"}
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool"]


def test_openai_truncated_turn_runs_no_tools():
    c, _ = _openai([_chunk(tool_calls=[_tc(0, "c1", "get_model_info", '{"k": 1}')], finish="length")])
    assert _run(c)[-1].tool_calls == []


# ------------------------------------------------------------------ Gemini
def _gemini(responses, raise_exc=None):
    c = GeminiClient(api_key="test")
    calls = []

    def gen(**kw):
        calls.append(kw)
        if raise_exc:
            raise raise_exc
        return iter(responses)

    c.client = NS(models=NS(generate_content_stream=gen))
    return c, calls


def test_gemini_stream_keeps_thought_signature():
    from google.genai import types

    def resp(*parts):
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))])

    fc = types.Part(function_call=types.FunctionCall(name="get_model_info", args={}), thought_signature=b"sig")
    c, calls = _gemini([resp(types.Part(text="ẩn", thought=True)), resp(types.Part(text="Xin chào")), resp(fc)])
    end = _run(c)[-1]
    assert end.text == "Xin chào"  # thought text is not shown
    assert end.tool_calls[0].name == "get_model_info" and end.tool_calls[0].id.startswith("call_")
    assert end.raw.parts[-1].thought_signature == b"sig"
    cfg = calls[0]["config"]
    assert cfg.automatic_function_calling.disable is True and cfg.system_instruction == "SYS"

    contents = c._contents(_history("gemini", end.raw))
    assert contents[1] is end.raw  # replayed unchanged
    assert contents[2].role == "user" and [p.function_response.response for p in contents[2].parts] == [{"a": 1}, {"b": 2}]


def test_gemini_rate_limit_maps_to_llm_error():
    from google.genai import errors

    c, _ = _gemini([], raise_exc=errors.ClientError(429, {"error": {"message": "quota"}}))
    with pytest.raises(LLMError, match="quá tải"):
        list(c.stream("SYS", [{"role": "user", "content": "hi"}], TOOLS))


def test_make_llm():
    assert isinstance(make_llm("gemini", "k"), GeminiClient)
    assert make_llm("OpenAI", "k", "gpt-x").model == "gpt-x"
    with pytest.raises(ValueError):
        make_llm("cohere", "k")
