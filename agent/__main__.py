"""Composition root CLI: `uv run python -m agent "<prompt>"`.

Loads `AgentSettings` (agent.toml + env), wires up a `Model`, MCP tools,
skills, permissions, and an `OtelSink`, runs one task through `run_agent`,
and prints the result.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent.composition import build_model, build_permissions, build_skills
from agent.config import AgentSettings
from agent.core.entrypoint import run_agent
from agent.core.loop import DEFAULT_SYSTEM_PROMPT
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.registry import MCPToolRegistry
from agent.observability.otel import build_tracer_provider
from agent.observability.sink import OtelSink


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent")
    parser.add_argument("prompt", help="user message to send to the agent")
    parser.add_argument("--task-id", default="cli-task")
    args = parser.parse_args(argv)

    settings = AgentSettings()  # type: ignore[call-arg]  # model comes from agent.toml/env

    model = build_model(settings.model)
    permissions = build_permissions(settings)
    skills = build_skills(settings)

    tracer_provider = build_tracer_provider(settings.otel)
    tracer = tracer_provider.get_tracer(settings.otel.service_name)
    sink = OtelSink(tracer)

    task = Task(
        id=args.task_id,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        messages=[Message(role="user", content=[TextBlock(text=args.prompt)])],
    )

    async with MCPToolRegistry(settings.mcp_servers) as tools:
        result = await run_agent(
            task,
            model=model,
            tools=tools,
            skills=skills,
            permissions=permissions,
            sink=sink,
            max_steps=settings.max_steps,
        )

    tracer_provider.shutdown()

    print(f"stop_reason: {result.stop_reason}")
    print(f"usage: {result.usage.model_dump()}")
    print("---")
    print(result.final_text())

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
