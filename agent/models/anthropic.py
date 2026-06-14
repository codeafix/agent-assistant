"""Anthropic adapter: native tool calling, streaming, and usage/cache tokens.

Translates the normalized `Message`/`ToolSpec` types to Anthropic's Messages
API and translates Anthropic's streaming events back to `StreamEvent`s.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Literal

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam
from anthropic.types.text_block_param import TextBlockParam
from anthropic.types.tool_result_block_param import ToolResultBlockParam
from anthropic.types.tool_use_block_param import ToolUseBlockParam

from agent.core.messages import (
    ContentBlock,
    Message,
    TextBlock,
    ToolSpec,
    ToolUseBlock,
)
from agent.models._convert import tool_result_to_text
from agent.models._pricing import priced_usage
from agent.models.base import (
    StreamDone,
    StreamEvent,
    StreamUsage,
    TextDelta,
    ToolCallComplete,
    Usage,
)


class AnthropicModel:
    """`Model` adapter for the Anthropic Messages API."""

    name: str

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        price_per_input_token_usd: float | None = None,
        price_per_output_token_usd: float | None = None,
    ) -> None:
        self.name = model
        self._client = AsyncAnthropic(api_key=api_key)
        self._price_in = price_per_input_token_usd
        self._price_out = price_per_output_token_usd

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> AsyncIterator[StreamEvent]:
        anthropic_messages, extra_system = _to_anthropic_messages(messages)
        full_system = "\n\n".join(part for part in (system, extra_system) if part)
        anthropic_tools = _to_anthropic_tools(tools) if tools else []

        input_tokens = 0
        output_tokens = 0
        cache_read = 0
        cache_write = 0
        stop_reason = "end_turn"

        tool_use_blocks: dict[int, tuple[str, str]] = {}
        tool_input_parts: dict[int, list[str]] = {}

        stream = await self._client.messages.create(
            model=self.name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=full_system,
            messages=anthropic_messages,
            tools=anthropic_tools,
            stream=True,
        )
        async for event in stream:
            if event.type == "message_start":
                usage = event.message.usage
                input_tokens = usage.input_tokens
                cache_read = usage.cache_read_input_tokens or 0
                cache_write = usage.cache_creation_input_tokens or 0
            elif event.type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    tool_use_blocks[event.index] = (block.id, block.name)
                    tool_input_parts[event.index] = []
            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    yield TextDelta(text=delta.text)
                elif delta.type == "input_json_delta":
                    tool_input_parts[event.index].append(delta.partial_json)
            elif event.type == "content_block_stop":
                if event.index in tool_use_blocks:
                    tool_id, tool_name = tool_use_blocks.pop(event.index)
                    raw_json = "".join(tool_input_parts.pop(event.index))
                    tool_input: dict[str, object] = json.loads(raw_json) if raw_json else {}
                    yield ToolCallComplete(
                        block=ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)
                    )
            elif event.type == "message_delta":
                if event.delta.stop_reason is not None:
                    stop_reason = event.delta.stop_reason
                output_tokens = event.usage.output_tokens
                cache_read = event.usage.cache_read_input_tokens or cache_read
                cache_write = event.usage.cache_creation_input_tokens or cache_write

        usage_out = priced_usage(
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            ),
            price_per_input_token_usd=self._price_in,
            price_per_output_token_usd=self._price_out,
        )
        yield StreamUsage(usage=usage_out)
        yield StreamDone(stop_reason=stop_reason)


def _to_anthropic_messages(messages: list[Message]) -> tuple[list[MessageParam], str]:
    """Translate normalized messages to Anthropic's `MessageParam` list.

    `role="system"` messages have no place in Anthropic's `messages` list
    (system instructions are a separate top-level parameter), so their text
    is concatenated and returned for the caller to merge into `system`.
    `role="tool"` messages become `user` messages containing `tool_result`
    blocks.
    """
    anthropic_messages: list[MessageParam] = []
    system_parts: list[str] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(
                "".join(block.text for block in message.content if isinstance(block, TextBlock))
            )
            continue
        role: Literal["user", "assistant"] = (
            "user" if message.role in ("user", "tool") else "assistant"
        )
        anthropic_messages.append(
            {
                "role": role,
                "content": [_to_anthropic_block(block) for block in message.content],
            }
        )
    return anthropic_messages, "\n\n".join(part for part in system_parts if part)


def _to_anthropic_block(
    block: ContentBlock,
) -> TextBlockParam | ToolUseBlockParam | ToolResultBlockParam:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": tool_result_to_text(block.content),
        "is_error": block.is_error,
    }


def _to_anthropic_tools(tools: list[ToolSpec]) -> list[ToolParam]:
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in tools
    ]
