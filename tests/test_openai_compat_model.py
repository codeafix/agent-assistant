"""OpenAI-compatible adapter: message/tool conversion and streaming-event
translation.

The underlying `AsyncOpenAI` client is replaced with a fake that yields
hand-built chat-completion chunks, so no network calls are made.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

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
