"""OpenAI-compatible adapter: message/tool conversion and streaming-event
translation.

The underlying `AsyncOpenAI` client is replaced with a fake that yields
hand-built chat-completion chunks, so no network calls are made.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

import agent.models.openai_compat as openai_compat_mod
from agent.composition import _check_base_url_trusted  # pyright: ignore[reportPrivateUsage]
from agent.core.messages import Message, TextBlock, ToolResultBlock, ToolSpec, ToolUseBlock
from agent.models.base import StreamDone, StreamUsage, TextDelta, ToolCallComplete
from agent.models.openai_compat import (
    OpenAICompatModel,
    _to_openai_messages,  # pyright: ignore[reportPrivateUsage]
    _to_openai_tools,  # pyright: ignore[reportPrivateUsage]
)


def test_to_openai_messages_maps_roles_and_tool_calls() -> None:
    messages = [
        Message(role="system", content=[TextBlock(text="be nice")]),
        Message(role="user", content=[TextBlock(text="hi")]),
        Message(
            role="assistant",
            content=[
                TextBlock(text="calling tool"),
                ToolUseBlock(id="call_1", name="echo", input={"text": "hi"}),
            ],
        ),
        Message(
            role="tool",
            content=[
                ToolResultBlock(tool_use_id="call_1", content=[{"type": "text", "text": "hi"}])
            ],
        ),
    ]

    openai_messages = _to_openai_messages(messages, system="top-level system")

    assert openai_messages == [
        {"role": "system", "content": "top-level system"},
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hi"},
    ]


def test_to_openai_tools() -> None:
    tools = [ToolSpec(name="echo", description="Echo text", input_schema={"type": "object"})]
    assert _to_openai_tools(tools) == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text",
                "parameters": {"type": "object"},
            },
        }
    ]


async def test_generate_translates_stream_into_normalized_events() -> None:
    model = OpenAICompatModel("local-model")

    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello", tool_calls=None),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(name="echo", arguments='{"text": '),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments='"hi"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            ),
            choices=[],
        ),
    ]

    async def fake_create(**_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        async def _stream() -> AsyncIterator[SimpleNamespace]:
            for chunk in chunks:
                yield chunk

        return _stream()

    cast(Any, model)._client.chat.completions.create = fake_create

    events = [
        e
        async for e in model.generate(
            [Message(role="user", content=[TextBlock(text="echo hi")])],
            [ToolSpec(name="echo", description="Echo text", input_schema={"type": "object"})],
        )
    ]

    assert events[0] == TextDelta(text="Hello")
    assert events[1] == ToolCallComplete(
        block=ToolUseBlock(id="call_1", name="echo", input={"text": "hi"})
    )

    usage_event = events[2]
    assert isinstance(usage_event, StreamUsage)
    assert usage_event.usage.input_tokens == 10
    assert usage_event.usage.output_tokens == 5
    assert usage_event.usage.cache_read_tokens == 2

    assert events[3] == StreamDone(stop_reason="tool_use")


async def test_generate_estimates_usage_when_backend_reports_none() -> None:
    """Local llama.cpp/vLLM servers may not report `usage` even with
    `stream_options.include_usage` -- fall back to a rough estimate."""
    model = OpenAICompatModel("local-model")

    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello there", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        ),
    ]

    async def fake_create(**_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        async def _stream() -> AsyncIterator[SimpleNamespace]:
            for chunk in chunks:
                yield chunk

        return _stream()

    cast(Any, model)._client.chat.completions.create = fake_create

    events = [
        e async for e in model.generate([Message(role="user", content=[TextBlock(text="hi")])])
    ]

    assert events[0] == TextDelta(text="Hello there")

    usage_event = events[1]
    assert isinstance(usage_event, StreamUsage)
    assert usage_event.usage.estimated is True
    assert usage_event.usage.input_tokens > 0
    assert usage_event.usage.output_tokens > 0

    assert events[2] == StreamDone(stop_reason="end_turn")


# --- PR 1a: base_url allowlist ---


def test_trusted_base_url_check_passes_when_host_in_list() -> None:
    _check_base_url_trusted("http://localhost:8080/v1", ["localhost"])


def test_trusted_base_url_check_raises_when_host_not_in_list() -> None:
    with pytest.raises(ValueError, match="untrusted host"):
        _check_base_url_trusted("http://evil.example.com/v1", ["localhost"])


def test_trusted_base_url_check_uses_hostname_not_full_url() -> None:
    _check_base_url_trusted("http://trusted-llm:8080/v1", ["trusted-llm"])


# --- PR 1b: stream size cap ---


async def _make_model_with_text_stream(text: str) -> list[object]:
    model = OpenAICompatModel("local-model")

    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=text, tool_calls=None),
                    finish_reason="stop",
                )
            ],
        ),
    ]

    async def fake_create(**_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        async def _stream() -> AsyncIterator[SimpleNamespace]:
            for chunk in chunks:
                yield chunk

        return _stream()

    cast(Any, model)._client.chat.completions.create = fake_create
    return [e async for e in model.generate([Message(role="user", content=[TextBlock(text="hi")])])]


async def test_generate_aborts_when_text_stream_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_compat_mod, "_MAX_STREAM_CHARS", 5)
    with pytest.raises(RuntimeError, match="exceeded"):
        await _make_model_with_text_stream("123456")


async def test_generate_succeeds_when_text_stream_within_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_compat_mod, "_MAX_STREAM_CHARS", 5)
    events = await _make_model_with_text_stream("hi")
    assert events[0] == TextDelta(text="hi")


# --- PR 1c: malformed tool-call JSON ---


async def test_generate_tolerates_malformed_tool_args() -> None:
    model = OpenAICompatModel("local-model")

    chunks = [
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_bad",
                                function=SimpleNamespace(name="echo", arguments="{NOT JSON}"),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        ),
        SimpleNamespace(usage=None, choices=[]),
    ]

    async def fake_create(**_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        async def _stream() -> AsyncIterator[SimpleNamespace]:
            for chunk in chunks:
                yield chunk

        return _stream()

    cast(Any, model)._client.chat.completions.create = fake_create

    events = [
        e async for e in model.generate([Message(role="user", content=[TextBlock(text="hi")])])
    ]

    tool_event = next(e for e in events if isinstance(e, ToolCallComplete))
    assert tool_event.block.name == "echo"
    assert tool_event.block.input == {}
