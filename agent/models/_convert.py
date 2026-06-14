"""Shared content-block helpers for model adapters."""

from __future__ import annotations

import json


def tool_result_to_text(content: str | list[dict[str, object]]) -> str:
    """Flatten a `ToolResultBlock.content` payload to plain text.

    MCP tool results are lists of content-part dicts (usually
    `{"type": "text", "text": ...}`); chat APIs that take normalized
    tool-result messages want a single string or text block.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        else:
            parts.append(json.dumps(item))
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough ~4-chars-per-token estimate for providers that report no usage.

    Used only to populate `Usage(..., estimated=True)` for local endpoints
    that don't report token counts; not a substitute for a real tokenizer.
    """
    return (len(text) + 3) // 4 if text else 0
