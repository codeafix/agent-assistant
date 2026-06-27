"""Phase 3 write-path types: richer record shapes for episodic + semantic storage.

Phase 1 recall only needs MemoryRecord (from agent.core.memory). These types
are used by the Phase 3 sink to write new memories after each run and are
provided here so the full data model is in place from the start.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agent.core.events import Provenance


class EpisodicRecord(BaseModel, frozen=True):
    """Distilled summary of one run, stored in the episodic collection."""

    id: str
    task_id: str
    run_id: str
    timestamp: datetime
    summary: str
    provenance: Provenance
    entities: list[str] = Field(default_factory=list[str])
    tags: list[str] = Field(default_factory=list[str])


class SemanticFact(BaseModel, frozen=True):
    """Durable synthesised fact derived from one or more episodes."""

    id: str
    content: str
    provenance: Provenance
    source_episode_ids: list[str] = Field(default_factory=list[str])
    entities: list[str] = Field(default_factory=list[str])
    tags: list[str] = Field(default_factory=list[str])
    timestamp: datetime
