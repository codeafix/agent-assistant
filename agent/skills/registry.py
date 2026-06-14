"""Skill registry implementations of `agent.core.interfaces.SkillRegistry`."""

from __future__ import annotations

from agent.skills.base import Skill


class EmptySkillRegistry:
    """No skills registered. The default until Phase 3 wires up
    `FileSystemSkillRegistry`."""

    def list_skills(self) -> list[Skill]:
        return []

    def get_skill(self, name: str) -> Skill | None:
        return None
