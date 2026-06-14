"""Deterministic replay model adapter.

Reads a hand-written JSON cassette of pre-recorded `StreamEvent` sequences,
one sequence ("turn") per `generate()` call. Lets the loop, tools, and
permissions be exercised end-to-end with zero network/model dependency, and
underpins the record/replay regression eval suite (Phase 4).

Cassette format::

    {
      "turns": [
        [ {"type": "text_delta", "text": "..."}, {"type": "done", "stop_reason": "end_turn"} ],
        [ ... ]
      ]
    }
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import TypeAdapter

from agent.core.messages import Message, ToolSpec
from agent.models.base import StreamEvent

_event_adapter: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)


class CassetteExhausted(RuntimeError):
    """Raised when `generate()` is called more times than the cassette has turns."""


class ReplayModel:
    """A `Model` that replays pre-recorded `StreamEvent` turns in order."""

    name: str

    def __init__(self, cassette_path: Path, name: str = "replay") -> None:
        self.name = name
        data = json.loads(Path(cassette_path).read_text())
        self._turns: list[list[StreamEvent]] = [
            [_event_adapter.validate_python(e) for e in turn] for turn in data["turns"]
        ]
        self._next_turn = 0

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[StreamEvent]:
        if self._next_turn >= len(self._turns):
            raise CassetteExhausted(
                f"cassette '{self.name}' has only {len(self._turns)} turn(s), "
                f"but generate() was called a {self._next_turn + 1}th time"
            )
        turn = self._turns[self._next_turn]
        self._next_turn += 1
        for event in turn:
            yield event
