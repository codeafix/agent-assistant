"""Phase 1 checkpoint eval: replay model + echo-clock MCP server + allowlist,
driven through `run_agent` via `evals.bridge.run_agent_solver`.

Run with: `uv run inspect eval evals/tasks/echo_clock.py`
"""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes

from agent.config import MCPServerConfig
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.mcp.registry import MCPToolRegistry
from agent.models.replay import ReplayModel
from agent.skills.registry import EmptySkillRegistry
from evals.bridge import AgentDeps, run_agent_solver

CASSETTE = Path(__file__).parent.parent.parent / "tests" / "cassettes" / "echo_clock.json"

ECHO_CLOCK_SERVER = MCPServerConfig(
    name="echo-clock",
    transport="stdio",
    command=sys.executable,
    args=["-m", "mcp_servers.echo_clock.server"],
)


@asynccontextmanager
async def _build_deps() -> AsyncGenerator[AgentDeps]:
    model = ReplayModel(CASSETTE)
    permissions = AllowlistPolicy([AllowRule(server="echo-clock", tool="echo")])
    async with MCPToolRegistry([ECHO_CLOCK_SERVER]) as tools:
        yield model, tools, EmptySkillRegistry(), permissions


@task
def echo_clock() -> Task:
    return Task(
        dataset=[Sample(input="Please echo 'hello'.", target="hello")],
        solver=run_agent_solver(
            _build_deps,
            system_prompt="You are a helpful agent with access to an echo tool.",
        ),
        scorer=includes(),
    )
