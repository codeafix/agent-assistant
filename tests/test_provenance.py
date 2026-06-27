"""Unit tests for Phase 2: provenance tagging on the transcript.

Covers:
- ToolCallFinished carries TOOL_OUTPUT + source="tool:<name>"
- ModelTextDelta carries AGENT_REASONING
- UserTurnReceived carries USER_STATED
- _effective_provenance: high-water-mark-of-untrust across ToolCallFinished events,
  including transitive composition through nested sub-agents
"""

from __future__ import annotations

from pathlib import Path

from agent.agents.subagent_tools import (
    _effective_provenance,  # pyright: ignore[reportPrivateUsage]
)
from agent.core.entrypoint import run_agent
from agent.core.events import (
    ModelTextDelta,
    Provenance,
    ToolCallFinished,
    TranscriptEvent,
    UserTurnReceived,
)
from agent.core.messages import Message, TextBlock, ToolResultBlock, ToolSpec
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.memory.provider import EmptyMemoryProvider
from agent.models.replay import ReplayModel
from agent.observability.sink import InMemorySink
from agent.skills.registry import EmptySkillRegistry

TOOL_CALL_CASSETTE = Path(__file__).parent / "cassettes" / "single_tool_call.json"
TEXT_ONLY_CASSETTE = Path(__file__).parent / "cassettes" / "memory_recall.json"


class _SingleToolRegistry:
    """Handles exactly one tool: 'test_tool' on 'test-server'."""

    def list_tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="test_tool",
                description="A test tool.",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    def server_for_tool(self, tool_name: str) -> str:
        return "test-server"

    async def call_tool(
        self, server: str, tool: str, args: dict[str, object]
    ) -> tuple[ToolResultBlock, Provenance]:
        return ToolResultBlock(tool_use_id="", content="ok", is_error=False), Provenance.TOOL_OUTPUT


class _NullToolRegistry:
    def list_tool_specs(self) -> list[ToolSpec]:
        return []

    def server_for_tool(self, tool_name: str) -> str:
        return ""

    async def call_tool(
        self, server: str, tool: str, args: dict[str, object]
    ) -> tuple[ToolResultBlock, Provenance]:
        raise NotImplementedError


async def _run_text_only() -> InMemorySink:
    model = ReplayModel(TEXT_ONLY_CASSETTE)
    task = Task(
        id="test",
        messages=[Message(role="user", content=[TextBlock(text="What language?")])],
    )
    sink = InMemorySink()
    await run_agent(
        task,
        model=model,
        tools=_NullToolRegistry(),
        skills=EmptySkillRegistry(),
        permissions=AllowlistPolicy([]),
        sink=sink,
        memory_provider=EmptyMemoryProvider(),
    )
    return sink


async def _run_with_tool_call() -> InMemorySink:
    model = ReplayModel(TOOL_CALL_CASSETTE)
    task = Task(
        id="test",
        messages=[Message(role="user", content=[TextBlock(text="Call the tool.")])],
    )
    sink = InMemorySink()
    await run_agent(
        task,
        model=model,
        tools=_SingleToolRegistry(),
        skills=EmptySkillRegistry(),
        permissions=AllowlistPolicy([AllowRule(server="test-server", tool="test_tool")]),
        sink=sink,
        memory_provider=EmptyMemoryProvider(),
    )
    return sink


# ---------------------------------------------------------------------------
# Seam 1: tool results → TOOL_OUTPUT
# ---------------------------------------------------------------------------


async def test_tool_call_finished_has_tool_output_provenance() -> None:
    sink = await _run_with_tool_call()
    finished = [e for e in sink.events if isinstance(e, ToolCallFinished)]
    assert len(finished) == 1
    assert finished[0].provenance == Provenance.TOOL_OUTPUT


async def test_tool_call_finished_source_is_tool_name() -> None:
    sink = await _run_with_tool_call()
    finished = [e for e in sink.events if isinstance(e, ToolCallFinished)]
    assert finished[0].source == "tool:test_tool"


# ---------------------------------------------------------------------------
# Seam 2: model content → AGENT_REASONING
# ---------------------------------------------------------------------------


async def test_model_text_delta_has_agent_reasoning_provenance() -> None:
    sink = await _run_text_only()
    deltas = [e for e in sink.events if isinstance(e, ModelTextDelta)]
    assert len(deltas) > 0
    assert all(e.provenance == Provenance.AGENT_REASONING for e in deltas)


# ---------------------------------------------------------------------------
# Seam 3: user turn → USER_STATED
# ---------------------------------------------------------------------------


async def test_user_turn_received_is_emitted() -> None:
    sink = await _run_text_only()
    turns = [e for e in sink.events if isinstance(e, UserTurnReceived)]
    assert len(turns) == 1


async def test_user_turn_received_has_user_stated_provenance() -> None:
    sink = await _run_text_only()
    turns = [e for e in sink.events if isinstance(e, UserTurnReceived)]
    assert turns[0].provenance == Provenance.USER_STATED


async def test_user_turn_received_captures_message_content() -> None:
    sink = await _run_text_only()
    turns = [e for e in sink.events if isinstance(e, UserTurnReceived)]
    assert "What language" in turns[0].content


# ---------------------------------------------------------------------------
# Sub-agent effective provenance (_effective_provenance unit tests)
# ---------------------------------------------------------------------------


def _tcf(provenance: Provenance) -> ToolCallFinished:
    return ToolCallFinished(
        run_id="r",
        step_index=0,
        tool_use_id="id",
        result=ToolResultBlock(tool_use_id="id", content="x"),
        is_error=False,
        latency_ms=1.0,
        provenance=provenance,
        source="tool:x",
    )


def test_effective_provenance_empty_transcript_is_agent_reasoning() -> None:
    events: list[TranscriptEvent] = []
    assert _effective_provenance(events) == Provenance.AGENT_REASONING


def test_effective_provenance_mcp_tool_call_is_tool_output() -> None:
    events: list[TranscriptEvent] = [_tcf(Provenance.TOOL_OUTPUT)]
    assert _effective_provenance(events) == Provenance.TOOL_OUTPUT


def test_effective_provenance_purely_reasoning_subagent_is_agent_reasoning() -> None:
    # Sub-agent that did no tool calls returns AGENT_REASONING provenance;
    # the parent's ToolCallFinished carries that propagated value.
    events: list[TranscriptEvent] = [_tcf(Provenance.AGENT_REASONING)]
    assert _effective_provenance(events) == Provenance.AGENT_REASONING


def test_effective_provenance_high_water_mark_tool_output_wins() -> None:
    events: list[TranscriptEvent] = [
        _tcf(Provenance.AGENT_REASONING),
        _tcf(Provenance.TOOL_OUTPUT),
    ]
    assert _effective_provenance(events) == Provenance.TOOL_OUTPUT


def test_effective_provenance_composes_transitively() -> None:
    # Grandchild transcript: has an MCP tool call → effective = TOOL_OUTPUT.
    grandchild_events: list[TranscriptEvent] = [_tcf(Provenance.TOOL_OUTPUT)]
    grandchild_provenance = _effective_provenance(grandchild_events)
    assert grandchild_provenance == Provenance.TOOL_OUTPUT

    # Child transcript: one entry is the sub-agent call carrying the grandchild's
    # computed provenance. The child itself did no MCP calls.
    child_events: list[TranscriptEvent] = [_tcf(grandchild_provenance)]
    child_provenance = _effective_provenance(child_events)
    assert child_provenance == Provenance.TOOL_OUTPUT

    # Poison surfaces at every ancestor level — two levels of nesting.
    parent_events: list[TranscriptEvent] = [_tcf(child_provenance)]
    assert _effective_provenance(parent_events) == Provenance.TOOL_OUTPUT
