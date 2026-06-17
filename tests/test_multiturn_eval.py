"""Multi-turn conversation eval: message accumulation and per-turn scorers.

Tests the two-step flow bridge.py exercises:
  1. run_agent returns result.messages (full post-run history).
  2. Appending a user turn and calling run_agent again sees the accumulated
     context, exercises tools, and produces the right final answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.core.entrypoint import run_agent
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.models.replay import ReplayModel
from agent.observability.sink import InMemorySink
from agent.skills.registry import EmptySkillRegistry
from evals.mock_tools import MockToolRegistry
from evals.spec import ConversationTurn, EvalCase, ExpectedToolCall, MockToolResult

CASSETTES = Path(__file__).parent / "cassettes"

_GET_TIME_MOCK = MockToolResult(
    server="time",
    tool="get_time",
    content="2:30 PM PDT",
    description="Return the current time for a timezone.",
    input_schema={
        "type": "object",
        "properties": {"timezone": {"type": "string"}},
        "required": ["timezone"],
    },
)


async def test_run_agent_returns_full_message_history() -> None:
    """result.messages contains the initial user message plus every
    assistant/tool message appended during the run."""
    model = ReplayModel(CASSETTES / "multiturn_clarify.json")
    task = Task(
        id="clarify",
        messages=[Message(role="user", content=[TextBlock(text="What time is it?")])],
    )
    result = await run_agent(
        task,
        model=model,
        tools=MockToolRegistry([]),
        skills=EmptySkillRegistry(),
        permissions=AllowlistPolicy([]),
        sink=InMemorySink(),
    )

    assert result.stop_reason == "end_turn"
    # user + assistant = 2 messages
    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"
    assert "Which timezone?" in result.final_text()


async def test_multiturn_message_accumulation() -> None:
    """Second run_agent call sees the first turn's full conversation history,
    calls the right tool, and produces the expected final answer."""

    # Turn 1: agent asks clarifying question.
    model1 = ReplayModel(CASSETTES / "multiturn_clarify.json")
    task1 = Task(
        id="tz-clarify",
        messages=[
            Message(
                role="user",
                content=[TextBlock(text="I want to know the time in a different timezone.")],
            )
        ],
    )
    result1 = await run_agent(
        task1,
        model=model1,
        tools=MockToolRegistry([]),
        skills=EmptySkillRegistry(),
        permissions=AllowlistPolicy([]),
        sink=InMemorySink(),
    )
    assert result1.stop_reason == "end_turn"
    assert len(result1.messages) == 2  # user + assistant

    # Turn 2: user responds; agent calls get_time and answers.
    accumulated = list(result1.messages) + [
        Message(role="user", content=[TextBlock(text="Pacific Time")])
    ]
    model2 = ReplayModel(CASSETTES / "multiturn_respond.json")
    task2 = Task(id="tz-respond", messages=accumulated)
    result2 = await run_agent(
        task2,
        model=model2,
        tools=MockToolRegistry([_GET_TIME_MOCK]),
        skills=EmptySkillRegistry(),
        permissions=AllowlistPolicy([AllowRule(server="time", tool="get_time")]),
        sink=InMemorySink(),
    )

    assert result2.stop_reason == "end_turn"
    assert "2:30 PM" in result2.final_text()
    # user1 + asst1 + user2 + asst_with_tool + tool_result + final_asst = 6
    assert len(result2.messages) == 6


def test_eval_case_single_turn_validates() -> None:
    case = EvalCase.model_validate({"name": "t", "input": "hi", "cassette": "echo_clock.json"})
    assert case.cassette == "echo_clock.json"
    assert not case.turns


def test_eval_case_multi_turn_validates() -> None:
    case = EvalCase.model_validate(
        {
            "name": "tz",
            "turns": [
                {"user_message": "What time?", "cassette": "multiturn_clarify.json"},
                {
                    "user_message": "Pacific Time",
                    "cassette": "multiturn_respond.json",
                    "expected_tool_calls": [{"server": "time", "tool": "get_time"}],
                    "response_includes": "2:30 PM",
                },
            ],
        }
    )
    assert len(case.turns) == 2
    assert case.turns[1].response_includes == "2:30 PM"
    assert case.turns[1].expected_tool_calls == [ExpectedToolCall(server="time", tool="get_time")]


def test_eval_case_requires_cassette_or_turns() -> None:
    with pytest.raises(Exception, match="cassette.*turns|turns.*cassette"):
        EvalCase.model_validate({"name": "bad", "input": "hi"})


def test_conversation_turn_assertions() -> None:
    turn = ConversationTurn(
        user_message="Pacific Time",
        cassette="multiturn_respond.json",
        expected_tool_calls=[ExpectedToolCall(server="time", tool="get_time")],
        expected_stop_reason="end_turn",
        response_includes="14:30",
    )
    assert turn.expected_stop_reason == "end_turn"
    assert turn.expected_tool_calls[0].server == "time"
