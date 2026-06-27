"""Composition root: builds concrete `Model`/`ToolRegistry`/etc.
implementations from `AgentSettings`. Nothing in `agent/core` imports this
module -- only `agent/__main__.py` (and, eventually, other entry points like
an HTTP server) do.
"""

from __future__ import annotations

import os

from agent.config import AgentSettings, MCPServerConfig, ModelConfig
from agent.core.interfaces import MemoryProvider, Model, PermissionPolicy, SkillRegistry
from agent.mcp.permissions import AllowlistPolicy, AllowRule
from agent.memory.provider import EmptyMemoryProvider, RagMemoryProvider
from agent.memory.store import McpMemoryStore
from agent.models.anthropic import AnthropicModel
from agent.models.openai_compat import OpenAICompatModel
from agent.models.prompted_tools import PromptedToolsModel
from agent.models.replay import ReplayModel
from agent.skills.registry import EmptySkillRegistry, FileSystemSkillRegistry


def build_model(config: ModelConfig) -> Model:
    model = _build_base_model(config)
    if not config.native_tool_calling:
        model = PromptedToolsModel(model)
    return model


def _build_base_model(config: ModelConfig) -> Model:
    if config.provider == "replay":
        if config.cassette_path is None:
            raise ValueError("model.provider='replay' requires model.cassette_path")
        return ReplayModel(config.cassette_path, name=config.name)

    api_key = os.environ[config.api_key_env] if config.api_key_env else None

    if config.provider == "anthropic":
        return AnthropicModel(
            config.name,
            api_key=api_key,
            price_per_input_token_usd=config.price_per_input_token_usd,
            price_per_output_token_usd=config.price_per_output_token_usd,
        )
    if config.provider == "openai_compat":
        return OpenAICompatModel(
            config.name,
            base_url=config.base_url,
            api_key=api_key,
            price_per_input_token_usd=config.price_per_input_token_usd,
            price_per_output_token_usd=config.price_per_output_token_usd,
        )
    raise NotImplementedError(f"model.provider='{config.provider}' is not implemented yet")


def build_permissions(settings: AgentSettings) -> PermissionPolicy:
    return build_permissions_from_rules(settings.permissions)


def build_permissions_from_rules(rules: list[AllowRule] | None = None) -> PermissionPolicy:
    return AllowlistPolicy(rules)


def build_skills(settings: AgentSettings) -> SkillRegistry:
    if settings.skills_dir is None:
        return EmptySkillRegistry()
    return FileSystemSkillRegistry(settings.skills_dir)


def build_memory_provider(settings: AgentSettings) -> MemoryProvider:
    """Returns EmptyMemoryProvider when memory is disabled (the default).

    When memory.enabled is True, call build_memory_store() instead: it returns
    an async context manager wrapping McpMemoryStore + RagMemoryProvider so the
    MCP connection lifecycle is managed correctly.
    """
    if not settings.memory.enabled:
        return EmptyMemoryProvider()
    raise RuntimeError(
        "memory.enabled=true requires an async context manager for McpMemoryStore. "
        "Use build_memory_store() and enter it with `async with` in your entrypoint."
    )


def build_memory_store(settings: AgentSettings) -> McpMemoryStore:
    """Build a McpMemoryStore from settings. The caller must use it as an async
    context manager and pass the connected store to RagMemoryProvider."""
    ms = settings.memory.server
    server_config = MCPServerConfig(
        name=ms.name,
        transport=ms.transport,
        url=ms.url,
        command=ms.command,
        args=ms.args,
    )
    return McpMemoryStore(server_config, search_tool=ms.search_tool)


def build_rag_provider(settings: AgentSettings, store: McpMemoryStore) -> RagMemoryProvider:
    """Pair a connected store with the settings-driven collection names and budget."""
    mem = settings.memory
    return RagMemoryProvider(
        store,
        episodic_collection=mem.episodic_collection,
        semantic_collection=mem.semantic_collection,
        top_k=mem.top_k,
        token_budget=mem.token_budget,
    )
