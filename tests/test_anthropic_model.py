"""Anthropic adapter: message/tool conversion and streaming-event translation.

The underlying `AsyncAnthropic` client is replaced with a fake that yields
hand-built raw stream events, so no network calls are made.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

from agent.core.messages import Message, TextBlock, ToolResultBlock, ToolSpec, ToolUseBlock
from agent.models.anthropic import (
    AnthropicModel,
    _to_anthropic_messages,  # pyright: ignore[reportPrivateUsage]
    _to_anthropic_tools,  # pyright: ignore[reportPrivateUsage]
)
from agent.models.base import StreamDone, StreamUsage, TextDelta, ToolCallComplete


def test_to_anthropic_messages_maps_roles_and_blocks() -> None:
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

    anthropic_messages, extra_system = _to_anthropic_messages(messages)

    assert extra_system == "be nice"
    assert anthropic_messages == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "calling tool"},
                {"type": "tool_use", "id": "call_1", "name": "echo", "input": {"text": "hi"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "hi",
                    "is_error": False,
                }
            ],
        },
    ]


def test_to_anthropic_tools() -> None:
    tools = [ToolSpec(name="echo", description="Echo text", input_schema={"type": "object"})]
    assert _to_anthropic_tools(tools) == [
        {"name": "echo", "description": "Echo text", "input_schema": {"type": "object"}}
    ]


async def test_generate_translates_stream_into_normalized_events() -> None:
    model = AnthropicModel("claude-sonnet-4-6", api_key="test")

    raw_events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=10, cache_read_input_tokens=2, cache_creation_input_tokens=0
                )
            ),
        ),
        SimpleNamespace(
            type="content_block_start", index=0, content_block=SimpleNamespace(type="text")
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Hello"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="call_1", name="echo"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"text": '),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='"hi"}'),
        ),
        SimpleNamespace(type="content_block_stop", index=1),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(
                output_tokens=5, cache_read_input_tokens=None, cache_creation_input_tokens=None
            ),
        ),
    ]

    async def fake_create(**_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        async def _stream() -> AsyncIterator[SimpleNamespace]:
            for event in raw_events:
                yield event

        return _stream()

    cast(Any, model)._client.messages.create = fake_create

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
    assert usage_event.usage.cache_write_tokens == 0

    assert events[3] == StreamDone(stop_reason="tool_use")


async def test_generate_applies_pricing() -> None:
    model = AnthropicModel(
        "claude-sonnet-4-6",
        api_key="test",
        price_per_input_token_usd=2.0,
        price_per_output_token_usd=10.0,
    )

    raw_events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=3, cache_read_input_tokens=0, cache_creation_input_tokens=0
                )
            ),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(
                output_tokens=4, cache_read_input_tokens=None, cache_creation_input_tokens=None
            ),
        ),
    ]

    async def fake_create(**_kwargs: object) -> AsyncIterator[SimpleNamespace]:
        async def _stream() -> AsyncIterator[SimpleNamespace]:
            for event in raw_events:
                yield event

        return _stream()

    cast(Any, model)._client.messages.create = fake_create

    events = [
        e async for e in model.generate([Message(role="user", content=[TextBlock(text="hi")])])
    ]

    usage_event = events[0]
    assert isinstance(usage_event, StreamUsage)
    # 3 input tokens * $2 + 4 output tokens * $10
    assert usage_event.usage.cost_usd == 46.0
