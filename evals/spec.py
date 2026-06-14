"""Declarative eval-case schema. Each record in `evals/cases/*.jsonl` is one
`EvalCase`, loaded into an Inspect `Sample` (as `metadata`) by `evals.suite`.

A case is a scripted conversation (`cassette`) plus the ground truth the
scorers in `evals.scorers` check it against. Everything not specified by the
case -- skills, MCP servers, the default permission policy -- comes from the
real `agent.toml` configuration (see `evals.bridge`), so evals exercise the
same wiring as production.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.mcp.permissions import AllowRule


class ExpectedToolCall(BaseModel):
    """A (server, tool) call the agent is expected to make. `args`, if
    given, must be a subset of the actual call's arguments."""

    server: str
    tool: str
    args: dict[str, object] | None = None


class MockToolResult(BaseModel):
    """A canned tool result, used in place of a real MCP server call."""

    server: str
    tool: str
    content: str
    is_error: bool = False


class EvalCase(BaseModel):
    """One eval case: a scripted conversation plus the ground truth the
    scorers in `evals.scorers` check it against. Each ground-truth field is
    opt-in -- an empty/unset field is not checked."""

    model_config = ConfigDict(frozen=True)

    name: str
    input: str
    cassette: str
    system_prompt: str = ""

    # Overrides of the shared agent.toml configuration. Leave unset to use
    # the real skills/MCP servers/permissions.
    mock_tools: list[MockToolResult] = Field(default_factory=list[MockToolResult])
    permissions: list[AllowRule] | None = None

    # Ground truth, checked by evals.scorers.
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list[ExpectedToolCall])
    expected_skills: list[str] = Field(default_factory=list[str])
    denied_tools: list[ExpectedToolCall] = Field(default_factory=list[ExpectedToolCall])
    expected_stop_reason: str | None = None
    response_includes: str | None = None
