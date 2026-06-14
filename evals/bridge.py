"""Bridges Inspect AI samples to `run_agent` for case-based evals (see
`evals.spec.EvalCase` and `evals.suite`).

Each sample carries an `EvalCase` as `Sample.metadata`, and may override the
tool registry (`MockToolRegistry`) and permission policy. Everything else --
skills, MCP servers, the default permission policy -- comes from the same
`agent.toml` used in production (via `agent.composition`), so evals exercise
the same wiring as production. This keeps a single code path between
production and evals: both call `run_agent`.

The assistant is either a scripted `ReplayModel` (the default `"replay"`,
deterministically replaying the case's cassette) or a real model from
`agent.toml`'s `[models]` registry, selected via `run_eval_case(model=...)`
the same way `python -m agent --model <key>` does.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from agent.composition import build_model, build_permissions, build_skills
from agent.config import AgentSettings
from agent.core.entrypoint import run_agent
from agent.core.interfaces import Model, PermissionPolicy, ToolRegistry
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy
from agent.mcp.registry import MCPToolRegistry
from agent.models.replay import ReplayModel
from agent.observability.sink import InMemorySink
from evals.mock_tools import MockToolRegistry
from evals.spec import EvalCase

CASSETTES_DIR = Path(__file__).parent.parent / "tests" / "cassettes"


@solver
def run_eval_case(model: str = "replay") -> Solver:
    """Run each sample's `EvalCase` through `run_agent`, reporting the final
    answer as `state.output` plus the full transcript and stop reason (via
    `state.store`) so `evals.scorers` can grade both the response and how the
    agent got there.

    `model` is `"replay"` (default) for deterministic cassette playback, or
    a `agent.toml` `[models]` registry key to run against a real model."""

    settings = AgentSettings()  # type: ignore[call-arg]  # agent.toml + env
    skills = build_skills(settings)
    default_permissions = build_permissions(settings)
    real_model: Model | None = (
        None if model == "replay" else build_model(settings.resolve_model(model))
    )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        case = state.metadata_as(EvalCase)
        active_model = real_model or ReplayModel(CASSETTES_DIR / case.cassette, name="replay")

        permissions: PermissionPolicy = (
            AllowlistPolicy(case.permissions)
            if case.permissions is not None
            else default_permissions
        )

        agent_task = Task(
            id=str(state.sample_id),
            system_prompt=case.system_prompt,
            messages=[Message(role="user", content=[TextBlock(text=state.input_text)])],
        )

        tools: ToolRegistry
        if case.mock_tools:
            tools = MockToolRegistry(case.mock_tools)
            result = await run_agent(
                agent_task,
                model=active_model,
                tools=tools,
                skills=skills,
                permissions=permissions,
                sink=InMemorySink(),
            )
        else:
            async with MCPToolRegistry(settings.mcp_servers) as tools:
                result = await run_agent(
                    agent_task,
                    model=active_model,
                    tools=tools,
                    skills=skills,
                    permissions=permissions,
                    sink=InMemorySink(),
                )

        final_text = result.final_text()
        state.output = ModelOutput.from_content(model=active_model.name, content=final_text)
        state.messages.append(ChatMessageAssistant(content=final_text))
        state.store.set("transcript", [e.model_dump(mode="json") for e in result.transcript])
        state.store.set("stop_reason", result.stop_reason)
        return state

    return solve
