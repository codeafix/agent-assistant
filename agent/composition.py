"""Composition root: builds concrete `Model`/`ToolRegistry`/etc.
implementations from `AgentSettings`. Nothing in `agent/core` imports this
module -- only `agent/__main__.py` (and, eventually, other entry points like
an HTTP server) do.
"""

from __future__ import annotations

from agent.config import AgentSettings, ModelConfig
from agent.core.interfaces import Model, PermissionPolicy, SkillRegistry
from agent.mcp.permissions import AllowlistPolicy
from agent.models.replay import ReplayModel
from agent.skills.registry import EmptySkillRegistry


def build_model(config: ModelConfig) -> Model:
    if config.provider == "replay":
        if config.cassette_path is None:
            raise ValueError("model.provider='replay' requires model.cassette_path")
        return ReplayModel(config.cassette_path, name=config.name)
    raise NotImplementedError(
        f"model.provider='{config.provider}' is not implemented yet (Phase 2)"
    )


def build_permissions(settings: AgentSettings) -> PermissionPolicy:
    return AllowlistPolicy(settings.permissions)


def build_skills(settings: AgentSettings) -> SkillRegistry:
    # Phase 3 will load `SKILL.md` packages from `settings.skills_dir`.
    return EmptySkillRegistry()
