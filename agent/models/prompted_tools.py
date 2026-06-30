"""Prompted-tool-calling shim for models without native function calling.

Wraps a `Model` that only produces plain text: encodes `ToolSpec`s into the
system prompt and parses a `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
convention out of the model's text output into `ToolCallComplete` events.

Use for local models that don't support the OpenAI/Anthropic tool-calling
APIs (set `native_tool_calling = false` in `agent.toml`). Because tool calls
can only be recognized once the full response is parsed, this shim buffers
the inner model's text and re-emits it as a single `TextDelta` rather than
streaming token-by-token.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import cast

from agent.core.interfaces import Model
from agent.core.messages import Message, ToolSpec, ToolUseBlock
from agent.models.base import StreamDone, StreamEvent, StreamUsage, TextDelta, ToolCallComplete

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_MAX_STREAM_CHARS = 2_000_000

_TOOLS_PREAMBLE = """
You can call tools by writing a line in this exact format (one call per line):
<tool_call>{{"name": "<tool name>", "arguments": <json object>}}</tool_call>

Available tools:
{tool_descriptions}

If you don't need a tool, just answer normally without using <tool_call> tags.
""".strip()


def _format_tools(tools: list[ToolSpec]) -> str:
    return "\n".join(
        f"- {tool.name}: {tool.description} (arguments schema: {json.dumps(tool.input_schema)})"
        for tool in tools
    )


class PromptedToolsModel:
    """Wraps `inner` to provide tool calling via prompting instead of a native API."""

    name: str

    def __init__(self, inner: Model) -> None:
        self.name = inner.name
        self._inner = inner

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[StreamEvent]:
        if not tools:
            async for event in self._inner.generate(
                messages, system=system, max_tokens=max_tokens, temperature=temperature
            ):
                yield event
            return

        preamble = _TOOLS_PREAMBLE.format(tool_descriptions=_format_tools(tools))
        augmented_system = f"{system}\n\n{preamble}" if system else preamble

        text = ""
        stop_reason = "end_turn"
        async for event in self._inner.generate(
            messages, system=augmented_system, max_tokens=max_tokens, temperature=temperature
        ):
            if isinstance(event, TextDelta):
                if len(text) + len(event.text) > _MAX_STREAM_CHARS:
                    raise RuntimeError(
                        f"model stream exceeded {_MAX_STREAM_CHARS:,} chars; aborting"
                    )
                text += event.text
            elif isinstance(event, StreamUsage):
                yield event
            elif isinstance(event, StreamDone):
                stop_reason = event.stop_reason
            else:
                pass

        matches = list(_TOOL_CALL_RE.finditer(text))
        remaining_text = _TOOL_CALL_RE.sub("", text).strip()
        if remaining_text:
            yield TextDelta(text=remaining_text)

        tool_calls_emitted = 0
        for match in matches:
            try:
                call: dict[str, object] = json.loads(match.group(1).strip())
                name = str(call["name"])
                arguments_raw = call.get("arguments", {})
                arguments = (
                    cast(dict[str, object], arguments_raw)
                    if isinstance(arguments_raw, dict)
                    else {}
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            tool_calls_emitted += 1
            yield ToolCallComplete(
                block=ToolUseBlock(id=f"call_{uuid.uuid4().hex[:8]}", name=name, input=arguments)
            )

        if tool_calls_emitted:
            stop_reason = "tool_use"
        yield StreamDone(stop_reason=stop_reason)
