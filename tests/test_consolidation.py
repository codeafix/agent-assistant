"""Unit tests for Phase 4 consolidation: parse_ops, apply_promotion_gate, execute_ops."""

from __future__ import annotations

import json

import pytest

from agent.core.events import Provenance
from agent.memory.consolidation import (
    MemoryOp,
    MemoryOpType,
    apply_promotion_gate,
    execute_ops,
    parse_ops,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AR = Provenance.AGENT_REASONING
US = Provenance.USER_STATED
TO = Provenance.TOOL_OUTPUT


def _create_op(
    *,
    episode_ids: list[str] | None = None,
    content: str = "A durable fact.",
    claimed_provenance: str = "agent_reasoning",
    confidence: float = 0.9,
) -> MemoryOp:
    return MemoryOp(
        op=MemoryOpType.CREATE,
        content=content,
        source_episode_ids=["ep-1"] if episode_ids is None else episode_ids,
        claimed_provenance=claimed_provenance,
        rationale="test",
        confidence=confidence,
    )


def _update_op(*, target_id: str = "fact-1", episode_ids: list[str] | None = None) -> MemoryOp:
    return MemoryOp(
        op=MemoryOpType.UPDATE,
        content="Updated fact.",
        target_fact_id=target_id,
        source_episode_ids=episode_ids or ["ep-1"],
        claimed_provenance="agent_reasoning",
        rationale="test",
        confidence=0.8,
    )


def _skip_op(rationale: str = "task-local") -> MemoryOp:
    return MemoryOp(op=MemoryOpType.SKIP, rationale=rationale)


class _CapturingStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    async def write(self, path: str, content: str) -> None:
        self.writes.append((path, content))


# ---------------------------------------------------------------------------
# parse_ops
# ---------------------------------------------------------------------------


def test_parse_ops_valid_json_array() -> None:
    raw = [{"op": "skip", "rationale": "nothing to do"}]
    result = parse_ops(json.dumps(raw))
    assert len(result) == 1
    assert result[0].op == MemoryOpType.SKIP
    assert result[0].rationale == "nothing to do"


def test_parse_ops_strips_json_fence() -> None:
    text = '```json\n[{"op": "skip", "rationale": "fenced"}]\n```'
    result = parse_ops(text)
    assert len(result) == 1
    assert result[0].op == MemoryOpType.SKIP


def test_parse_ops_strips_plain_fence() -> None:
    text = '```\n[{"op": "skip", "rationale": "plain fence"}]\n```'
    result = parse_ops(text)
    assert len(result) == 1
    assert result[0].op == MemoryOpType.SKIP


def test_parse_ops_ignores_surrounding_prose() -> None:
    text = 'Here are the ops:\n[{"op": "skip", "rationale": "done"}]\nEnd.'
    result = parse_ops(text)
    assert len(result) == 1
    assert result[0].op == MemoryOpType.SKIP


def test_parse_ops_no_array_returns_single_blocked_skip() -> None:
    result = parse_ops("Nothing here at all.")
    assert len(result) == 1
    assert result[0].op == MemoryOpType.SKIP
    assert result[0].blocked_reason is not None
    assert "no JSON array" in result[0].blocked_reason


def test_parse_ops_invalid_json_returns_blocked_skip() -> None:
    result = parse_ops("[{broken json")
    assert len(result) == 1
    assert result[0].op == MemoryOpType.SKIP
    assert result[0].blocked_reason is not None


def test_parse_ops_malformed_item_becomes_blocked_skip() -> None:
    raw = [
        {"op": "skip", "rationale": "fine"},
        {"op": "not_a_real_op_type"},  # invalid MemoryOpType
    ]
    result = parse_ops(json.dumps(raw))
    assert len(result) == 2
    assert result[0].op == MemoryOpType.SKIP
    assert result[0].blocked_reason is None  # valid SKIP from model
    assert result[1].op == MemoryOpType.SKIP
    assert result[1].blocked_reason is not None
    assert "malformed op at index 1" in result[1].blocked_reason


def test_parse_ops_parses_create_op() -> None:
    raw = [
        {
            "op": "create",
            "content": "Python is the project language.",
            "source_episode_ids": ["ep-abc"],
            "claimed_provenance": "agent_reasoning",
            "rationale": "stated explicitly",
            "confidence": 0.95,
        }
    ]
    result = parse_ops(json.dumps(raw))
    assert len(result) == 1
    op = result[0]
    assert op.op == MemoryOpType.CREATE
    assert op.content == "Python is the project language."
    assert op.source_episode_ids == ["ep-abc"]
    assert op.confidence == 0.95


def test_parse_ops_parses_update_op() -> None:
    raw = [
        {
            "op": "update",
            "target_fact_id": "fact-xyz",
            "content": "Updated fact.",
            "source_episode_ids": ["ep-2"],
            "claimed_provenance": "agent_reasoning",
            "rationale": "changed",
            "confidence": 0.8,
        }
    ]
    result = parse_ops(json.dumps(raw))
    assert result[0].op == MemoryOpType.UPDATE
    assert result[0].target_fact_id == "fact-xyz"


def test_parse_ops_non_list_json_returns_blocked_skip() -> None:
    result = parse_ops('{"op": "create"}')  # dict not list — no [ ] brackets
    assert result[0].op == MemoryOpType.SKIP
    assert result[0].blocked_reason is not None


# ---------------------------------------------------------------------------
# apply_promotion_gate
# ---------------------------------------------------------------------------


def test_gate_allows_create_with_agent_reasoning() -> None:
    ops = [_create_op(episode_ids=["ep-1"])]
    result = apply_promotion_gate(ops, {"ep-1": AR})
    assert result[0].op == MemoryOpType.CREATE
    assert result[0].blocked_reason is None


def test_gate_allows_create_with_user_stated() -> None:
    ops = [_create_op(episode_ids=["ep-1"])]
    result = apply_promotion_gate(ops, {"ep-1": US})
    assert result[0].op == MemoryOpType.CREATE


def test_gate_blocks_tool_only_provenance() -> None:
    ops = [_create_op(episode_ids=["ep-1"])]
    result = apply_promotion_gate(ops, {"ep-1": TO})
    assert result[0].op == MemoryOpType.SKIP
    assert result[0].blocked_reason is not None
    assert "tool-only" in result[0].blocked_reason


def test_gate_blocks_unresolvable_episode_id() -> None:
    ops = [_create_op(episode_ids=["ep-unknown"])]
    result = apply_promotion_gate(ops, {"ep-1": AR})
    assert result[0].op == MemoryOpType.SKIP
    assert "unresolvable" in (result[0].blocked_reason or "")


def test_gate_blocks_if_any_episode_unresolvable() -> None:
    """Fail-closed: even if one episode is trusted, an unresolvable one blocks."""
    ops = [_create_op(episode_ids=["ep-good", "ep-missing"])]
    result = apply_promotion_gate(ops, {"ep-good": AR})
    assert result[0].op == MemoryOpType.SKIP
    assert "unresolvable" in (result[0].blocked_reason or "")


def test_gate_blocks_empty_source_episode_ids() -> None:
    ops = [_create_op(episode_ids=[])]
    result = apply_promotion_gate(ops, {"ep-1": AR})
    assert result[0].op == MemoryOpType.SKIP
    assert "no source_episode_ids" in (result[0].blocked_reason or "")


def test_gate_passes_skip_through_unchanged() -> None:
    op = _skip_op("task-local state")
    result = apply_promotion_gate([op], {})
    assert result[0].op == MemoryOpType.SKIP
    assert result[0].rationale == "task-local state"
    assert result[0].blocked_reason is None


def test_gate_allows_update_and_preserves_target_fact_id() -> None:
    op = _update_op(target_id="fact-42", episode_ids=["ep-1"])
    result = apply_promotion_gate([op], {"ep-1": AR})
    assert result[0].op == MemoryOpType.UPDATE
    assert result[0].target_fact_id == "fact-42"
    assert result[0].blocked_reason is None


def test_gate_blocks_update_with_tool_only() -> None:
    op = _update_op(episode_ids=["ep-tool"])
    result = apply_promotion_gate([op], {"ep-tool": TO})
    assert result[0].op == MemoryOpType.SKIP


def test_gate_mixed_trusted_and_tool_allows_promotion() -> None:
    """At least one trusted episode is enough — the rest can be TOOL_OUTPUT."""
    ops = [_create_op(episode_ids=["ep-trust", "ep-tool"])]
    result = apply_promotion_gate(ops, {"ep-trust": AR, "ep-tool": TO})
    assert result[0].op == MemoryOpType.CREATE


def test_gate_preserves_order() -> None:
    ops = [_create_op(episode_ids=["ep-1"]), _skip_op(), _create_op(episode_ids=["ep-bad"])]
    episodes = {"ep-1": AR}
    result = apply_promotion_gate(ops, episodes)
    assert result[0].op == MemoryOpType.CREATE
    assert result[1].op == MemoryOpType.SKIP
    assert result[2].op == MemoryOpType.SKIP  # ep-bad unresolvable


# ---------------------------------------------------------------------------
# execute_ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_ops_writes_create_under_semantic_folder() -> None:
    store = _CapturingStore()
    ops = [_create_op()]
    written = await execute_ops(ops, store, "Claude/Memory/Semantic")
    assert len(written) == 1
    assert written[0].startswith("Claude/Memory/Semantic/")
    assert written[0].endswith(".md")


@pytest.mark.asyncio
async def test_execute_ops_skips_skip_ops() -> None:
    store = _CapturingStore()
    ops = [_skip_op(), _create_op()]
    await execute_ops(ops, store, "Claude/Memory/Semantic")
    assert len(store.writes) == 1


@pytest.mark.asyncio
async def test_execute_ops_update_uses_target_fact_id() -> None:
    store = _CapturingStore()
    op = _update_op(target_id="my-fact-id")
    await execute_ops([op], store, "Claude/Memory/Semantic")
    assert len(store.writes) == 1
    path, _ = store.writes[0]
    assert "my-fact-id" in path


@pytest.mark.asyncio
async def test_execute_ops_create_generates_uuid() -> None:
    store = _CapturingStore()
    op = _create_op()
    await execute_ops([op], store, "Claude/Memory/Semantic")
    path, _ = store.writes[0]
    # Path should be semantic_folder/<some-uuid>.md
    stem = path.split("/")[-1].replace(".md", "")
    # UUIDs are 36 chars
    assert len(stem) == 36


@pytest.mark.asyncio
async def test_execute_ops_content_in_written_file() -> None:
    store = _CapturingStore()
    op = _create_op(content="The sky is blue.")
    await execute_ops([op], store, "Claude/Memory/Semantic")
    _, content = store.writes[0]
    assert "The sky is blue." in content


@pytest.mark.asyncio
async def test_execute_ops_frontmatter_includes_source_episode_ids() -> None:
    store = _CapturingStore()
    op = _create_op(episode_ids=["ep-abc", "ep-def"])
    await execute_ops([op], store, "Claude/Memory/Semantic")
    _, content = store.writes[0]
    assert "ep-abc" in content
    assert "ep-def" in content


@pytest.mark.asyncio
async def test_execute_ops_returns_empty_for_all_skips() -> None:
    store = _CapturingStore()
    written = await execute_ops([_skip_op(), _skip_op()], store, "Claude/Memory/Semantic")
    assert written == []
    assert store.writes == []
