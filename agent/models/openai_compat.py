"""OpenAI-compatible chat-completions adapter.

Targets the OpenAI API itself as well as local OpenAI-compatible servers
(llama.cpp's `llama-server`, vLLM, etc.) via `base_url`. Uses native
function-calling; for backends without tool support, wrap this adapter with
`agent/models/prompted_tools.py`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from agent.core.messages import Message, TextBlock, ToolResultBlock, ToolSpec, ToolUseBlock
from agent.models._convert import estimate_tokens, tool_result_to_text
from agent.models._pricing import priced_usage
from agent.models.base import (
    StreamDone,
    StreamEvent,
    StreamUsage,
    TextDelta,
    ToolCallComplete,
    Usage,
)

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}


class OpenAICompatModel:
    """`Model` adapter for the OpenAI chat-completions API and compatibles."""

    name: str

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        price_per_input_token_usd: float | None = None,
        price_per_output_token_usd: float | None = None,
    ) -> None:
        self.name = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")
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
        openai_messages = _to_openai_messages(messages, system)
        openai_tools = _to_openai_tools(tools) if tools else []

        stop_reason = "end_turn"
        usage_out: Usage | None = None
        tool_calls: dict[int, tuple[str, str, list[str]]] = {}
        output_text_parts: list[str] = []

        stream = await self._client.chat.completions.create(
            model=self.name,
            messages=openai_messages,
            tools=openai_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                cached = 0
                if chunk.usage.prompt_tokens_details is not None:
                    cached = chunk.usage.prompt_tokens_details.cached_tokens or 0
                usage_out = priced_usage(
                    Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        cache_read_tokens=cached,
                    ),
                    price_per_input_token_usd=self._price_in,
                    price_per_output_token_usd=self._price_out,
                )
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                output_text_parts.append(delta.content)
                yield TextDelta(text=delta.content)
            for tool_call_delta in delta.tool_calls or []:
                tool_id, tool_name, args_parts = tool_calls.get(tool_call_delta.index, ("", "", []))
                if tool_call_delta.id:
                    tool_id = tool_call_delta.id
                if tool_call_delta.function is not None:
                    if tool_call_delta.function.name:
                        tool_name = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        args_parts.append(tool_call_delta.function.arguments)
                tool_calls[tool_call_delta.index] = (tool_id, tool_name, args_parts)
            if choice.finish_reason:
                stop_reason = _FINISH_REASON_MAP.get(choice.finish_reason, choice.finish_reason)

        for tool_id, tool_name, args_parts in tool_calls.values():
            raw_json = "".join(args_parts)
            tool_input: dict[str, object] = json.loads(raw_json) if raw_json else {}
            output_text_parts.append(raw_json)
            yield ToolCallComplete(block=ToolUseBlock(id=tool_id, name=tool_name, input=tool_input))

        if usage_out is None:
            # This backend didn't report usage (common for local llama.cpp/vLLM
            # servers) -- estimate from the request/response text instead.
            input_text = _request_text(messages, system, tools)
            usage_out = priced_usage(
                Usage(
                    input_tokens=estimate_tokens(input_text),
                    output_tokens=estimate_tokens("".join(output_text_parts)),
                    estimated=True,
                ),
                price_per_input_token_usd=self._price_in,
                price_per_output_token_usd=self._price_out,
            )

        yield StreamUsage(usage=usage_out)
        yield StreamDone(stop_reason=stop_reason)


def _request_text(messages: list[Message], system: str | None, tools: list[ToolSpec] | None) -> str:
    """Flatten a request to plain text for `estimate_tokens`."""
    parts: list[str] = [system] if system else []
    for message in messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(json.dumps(block.input))
            else:
                parts.append(tool_result_to_text(block.content))
    for tool in tools or []:
        parts.append(tool.description)
        parts.append(json.dumps(tool.input_schema))
    return "\n".join(parts)


def _to_openai_messages(
    messages: list[Message], system: str | None
) -> list[ChatCompletionMessageParam]:
    openai_messages: list[ChatCompletionMessageParam] = []
    if system:
        openai_messages.append({"role": "system", "content": system})
    for message in messages:
        if message.role == "tool":
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": tool_result_to_text(block.content),
                        }
                    )
            continue
        if message.role == "assistant":
            text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
            tool_use_blocks = [
                block for block in message.content if isinstance(block, ToolUseBlock)
            ]
            if tool_use_blocks:
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            }
                            for block in tool_use_blocks
                        ],
                    }
                )
            else:
                openai_messages.append({"role": "assistant", "content": text})
            continue
        text = "".join(block.text for block in message.content if isinstance(block, TextBlock))
        if message.role == "system":
            openai_messages.append({"role": "system", "content": text})
        else:
            openai_messages.append({"role": "user", "content": text})
    return openai_messages


def _to_openai_tools(tools: list[ToolSpec]) -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]
