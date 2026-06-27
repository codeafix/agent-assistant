"""MemoryStore: Protocol + MCP-backed implementation.

MemoryStore is the read-side interface: retrieve records from a named
collection by semantic similarity to a query string.

McpMemoryStore wraps the markdown-rag MCP server, reusing the same
MCPServerConnection pattern as MCPToolRegistry. The caller manages the
lifecycle via async context manager.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import Protocol, cast, runtime_checkable

from agent.config import MCPServerConfig
from agent.core.events import Provenance
from agent.core.memory import MemoryKind, MemoryRecord
from agent.mcp.client import MCPServerConnection


@runtime_checkable
class MemoryStore(Protocol):
    async def search(
        self,
        collection: str,
        query: str,
        *,
        where: dict[str, str] | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]: ...


def _str_from(mapping: dict[str, object], key: str, default: str = "") -> str:
    val = mapping.get(key)
    return str(val) if val is not None else default


def _parse_record(raw: dict[str, object]) -> MemoryRecord:
    raw_meta = raw.get("metadata")
    metadata: dict[str, object] = (
        cast(dict[str, object], raw_meta) if isinstance(raw_meta, dict) else {}
    )

    distance = raw.get("distance", 0.0)
    score = max(0.0, 1.0 - float(distance)) if isinstance(distance, (int, float)) else 0.0

    kind_str = _str_from(metadata, "kind", "semantic")
    try:
        kind = MemoryKind(kind_str)
    except ValueError:
        kind = MemoryKind.SEMANTIC

    prov_str = _str_from(metadata, "provenance", "tool_output")
    try:
        provenance = Provenance(prov_str)
    except ValueError:
        provenance = Provenance.TOOL_OUTPUT

    content_val = raw.get("document") or raw.get("content") or ""
    task_id_val = metadata.get("task_id")

    return MemoryRecord(
        id=_str_from(raw, "id"),
        kind=kind,
        content=str(content_val),
        provenance=provenance,
        source=_str_from(metadata, "source"),
        task_id=str(task_id_val) if task_id_val is not None else None,
        score=score,
    )


class McpMemoryStore:
    """Calls the markdown-rag MCP server search tool.

    Usage::

        async with McpMemoryStore(config) as store:
            records = await store.search("semantic", "query string", top_k=5)
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        search_tool: str = "search",
    ) -> None:
        self._connection = MCPServerConnection(config)
        self._search_tool = search_tool

    async def __aenter__(self) -> McpMemoryStore:
        await self._connection.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._connection.close()

    async def search(
        self,
        collection: str,
        query: str,
        *,
        where: dict[str, str] | None = None,
        top_k: int = 10,
    ) -> list[MemoryRecord]:
        args: dict[str, object] = {
            "collection": collection,
            "query": query,
            "n_results": top_k,
        }
        if where:
            args["where"] = where

        result = await self._connection.call_tool(self._search_tool, args)
        if result.isError:
            return []

        records: list[MemoryRecord] = []
        for item in result.content:
            item_dict = item.model_dump(mode="json")
            if item_dict.get("type") != "text":
                continue
            text = item_dict.get("text", "")
            if not isinstance(text, str):
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                items: list[object] = cast(list[object], data)
                for elem in items:
                    if isinstance(elem, dict):
                        try:
                            records.append(_parse_record(cast(dict[str, object], elem)))
                        except Exception:  # noqa: BLE001
                            pass
            elif isinstance(data, dict):
                try:
                    records.append(_parse_record(cast(dict[str, object], data)))
                except Exception:  # noqa: BLE001
                    pass

        return records
