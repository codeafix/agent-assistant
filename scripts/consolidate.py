"""Consolidate episodic memories into semantic facts for a completed task.

Usage:
  uv run python scripts/consolidate.py --task-id <id> [--model <key>] [--dry-run]

Reads all episodic records for the given task, compares against existing
semantic facts, runs the consolidator agent to propose ops, applies the
deterministic promotion gate, then writes approved CREATE/UPDATE ops to
Claude/Memory/Semantic/ via obsidian-mcp-guard.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from agent.agents.registry import AgentRegistry
from agent.composition import build_model, memory_context
from agent.config import AgentSettings
from agent.core.entrypoint import run_agent
from agent.core.memory import MemoryRecord
from agent.core.messages import Message, TextBlock
from agent.core.state import Task
from agent.mcp.permissions import AllowlistPolicy
from agent.memory.consolidation import apply_promotion_gate, execute_ops, parse_ops
from agent.observability.sink import InMemorySink
from agent.skills.registry import EmptySkillRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger(__name__)


def _format_context(
    episodes: Sequence[MemoryRecord], semantic_facts: Sequence[MemoryRecord]
) -> str:
    """Format fetched records into the consolidator's user message."""
    lines: list[str] = []

    lines.append("## Episodic Records (this task)")
    if not episodes:
        lines.append("(none found)")
    else:
        for rec in episodes:
            lines.append(f"\n### Episode: {rec.id}")
            lines.append(f"provenance: {rec.provenance}")
            lines.append(rec.content)

    lines.append("\n## Existing Semantic Facts")
    if not semantic_facts:
        lines.append("(none yet)")
    else:
        for fact in semantic_facts:
            lines.append(f"\n### Fact: {fact.id}")
            lines.append(fact.content)

    return "\n".join(lines)


async def consolidate(
    task_id: str,
    *,
    model_key: str | None = None,
    dry_run: bool = False,
) -> None:
    settings = AgentSettings()  # type: ignore[call-arg]

    if not settings.memory.enabled:
        _log.error("memory.enabled is false in config — nothing to consolidate")
        sys.exit(1)

    async with memory_context(settings) as (_, store):
        if store is None:
            _log.error("memory store not configured")
            sys.exit(1)

        # Fetch episodic records scoped to this task.
        episodes = await store.search(
            settings.memory.episodic_folder,
            f"task {task_id}",
            top_k=20,
        )
        _log.info("found %d episodic records for task %s", len(episodes), task_id)

        # Fetch existing semantic facts for dedup context.
        semantic_facts = await store.search(
            settings.memory.semantic_folder,
            "all semantic facts",
            top_k=20,
        )
        _log.info("found %d existing semantic facts", len(semantic_facts))

        # Build episode provenance map for the gate.
        episode_provenances = {r.id: r.provenance for r in episodes}

        # Build context for the consolidator agent.
        context = _format_context(episodes, semantic_facts)

        # Load and run the consolidator agent.
        if settings.agents_dir is None:
            _log.error("agents_dir not set in config")
            sys.exit(1)

        async with AgentRegistry(settings.agents_dir, settings) as registry:
            runtime = await registry.get_runtime("consolidator")
            resolved_model = runtime.settings.resolve_model(model_key)
            model = build_model(resolved_model)

            task = Task(
                id=f"consolidate-{task_id}",
                task_id=task_id,
                system_prompt=runtime.settings.system_prompt,
                messages=[Message(role="user", content=[TextBlock(text=context)])],
            )
            result = await run_agent(
                task,
                model=model,
                tools=runtime.mcp_tools,
                skills=EmptySkillRegistry(),
                permissions=AllowlistPolicy([]),
                sink=InMemorySink(),
                max_steps=runtime.settings.max_steps,
            )

        raw_text = result.final_text()
        _log.debug("consolidator output:\n%s", raw_text)

        # Parse, gate, execute.
        ops = parse_ops(raw_text)
        _log.info("parsed %d ops", len(ops))

        gated = apply_promotion_gate(ops, episode_provenances)
        approved = [op for op in gated if op.op.value != "skip"]
        blocked = [op for op in gated if op.op.value == "skip"]
        _log.info("%d approved, %d blocked by gate", len(approved), len(blocked))
        for op in blocked:
            _log.info("  blocked: %s", op.blocked_reason or op.rationale)

        if dry_run:
            _log.info("dry-run: skipping writes")
            for op in approved:
                _log.info("  would write: op=%s content=%.80s", op.op, op.content)
            return

        written = await execute_ops(gated, store, settings.memory.semantic_folder)
        _log.info("wrote %d semantic facts: %s", len(written), written)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate episodic memories into semantic facts"
    )  # noqa: E501
    parser.add_argument("--task-id", required=True, help="Task ID whose episodes to consolidate")
    parser.add_argument("--model", default=None, help="Model key (default: default_model)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and gate but do not write")
    args = parser.parse_args()
    asyncio.run(consolidate(args.task_id, model_key=args.model, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
