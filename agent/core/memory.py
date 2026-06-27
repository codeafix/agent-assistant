"""Provider-agnostic memory types for the recall protocol.

These are the only memory shapes agent/core sees. All concrete persistence
and retrieval live in agent/memory/ and are never imported by agent/core.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from agent.core.events import Provenance


class MemoryKind(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryRecord(BaseModel, frozen=True):
    id: str
    kind: MemoryKind
    content: str
    provenance: Provenance
    source: str = ""
    task_id: str | None = None
    score: float = 0.0
