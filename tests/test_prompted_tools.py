"""Prompted-tool-calling shim: prompt augmentation and `<tool_call>` parsing."""

from __future__ import annotations

from collections.abc import AsyncIterator

from agent.core.messages import Message, TextBlock, ToolSpec
from agent.models.base import (
    StreamDone,
    StreamEvent,
    StreamUsage,
    TextDelta,
    ToolCallComplete,
    Usage,
)
from agent.models.prompted_tools import PromptedToolsModel

ECHO_TOOL = ToolSpec(name="echo", description="Echo text", input_schema={"type": "object"})


class FakeModel:
    """A `Model` that replays a fixed sequence of `StreamEvent`s, recording
    the `system` prompt it was called with."""

    name = "fake"

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events
        self.last_system: str | None = None

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[StreamEvent]:
        self.last_system = system
        for event in self._events:
            yield event


async def test_passthrough_when_no_tools() -> None:
    inner = FakeModel([TextDelta(text="hi"), StreamDone(stop_reason="end_turn")])
    model = PromptedToolsModel(inner)

    events = [
        e async for e in model.generate([Message(role="user", content=[TextBlock(text="hi")])])
    ]

    assert events == [TextDelta(text="hi"), StreamDone(stop_reason="end_turn")]
    assert inner.last_system is None


async def test_tool_call_is_parsed_out_of_text() -> None:
    text = (
        "Sure, calling the tool now.\n"
        '<tool_call>{"name": "echo", "arguments": {"text": "hi"}}</tool_call>'
    )
    inner = FakeModel(
        [
            TextDelta(text=text),
            StreamUsage(usage=Usage(input_tokens=1, output_tokens=2)),
            StreamDone(stop_reason="end_turn"),
        ]
    )
    model = PromptedToolsModel(inner)

    events = [
        e
        async for e in model.generate(
            [Message(role="user", content=[TextBlock(text="echo hi")])], [ECHO_TOOL]
        )
    ]

    # The system prompt is augmented with the tool catalog.
    assert inner.last_system is not None
    assert "echo" in inner.last_system
    assert "<tool_call>" in inner.last_system

    # StreamUsage is forwarded as soon as it's seen, before the buffered
    # text/tool-call events that are only known once the stream ends.
    usage_event = events[0]
    assert isinstance(usage_event, StreamUsage)
    assert usage_event.usage.input_tokens == 1

    assert events[1] == TextDelta(text="Sure, calling the tool now.")
    tool_call_event = events[2]
    assert isinstance(tool_call_event, ToolCallComplete)
    assert tool_call_event.block.name == "echo"
    assert tool_call_event.block.input == {"text": "hi"}
    assert events[3] == StreamDone(stop_reason="tool_use")


async def test_no_tool_call_keeps_inner_stop_reason() -> None:
    inner = FakeModel([TextDelta(text="just an answer"), StreamDone(stop_reason="end_turn")])
    model = PromptedToolsModel(inner)

    events = [
        e
        async for e in model.generate(
            [Message(role="user", content=[TextBlock(text="echo hi")])], [ECHO_TOOL]
        )
    ]

    assert events == [TextDelta(text="just an answer"), StreamDone(stop_reason="end_turn")]


async def test_malformed_tool_call_is_skipped() -> None:
    inner = FakeModel(
        [
            TextDelta(text="<tool_call>not json</tool_call>answer"),
            StreamDone(stop_reason="end_turn"),
        ]
    )
    model = PromptedToolsModel(inner)

    events = [
        e
        async for e in model.generate(
            [Message(role="user", content=[TextBlock(text="echo hi")])], [ECHO_TOOL]
        )
    ]

    # A malformed <tool_call> is dropped from the text but produces no
    # ToolCallComplete, so the inner model's stop_reason is preserved.
    assert events == [TextDelta(text="answer"), StreamDone(stop_reason="end_turn")]


async def test_system_prompt_is_combined_with_tool_preamble() -> None:
    inner = FakeModel([TextDelta(text="ok"), StreamDone(stop_reason="end_turn")])
    model = PromptedToolsModel(inner)

    async for _ in model.generate(
        [Message(role="user", content=[TextBlock(text="hi")])],
        [ECHO_TOOL],
        system="You are helpful.",
    ):
        pass

    assert inner.last_system is not None
    assert inner.last_system.startswith("You are helpful.")
    assert "echo" in inner.last_system
