"""Skill registry implementations of `agent.core.interfaces.SkillRegistry`."""

from __future__ import annotations

from pathlib import Path

from agent.skills.base import Skill


class EmptySkillRegistry:
    """No skills registered."""

    def list_skills(self) -> list[Skill]:
        return []

    def get_skill(self, name: str) -> Skill | None:
        return None


class FileSystemSkillRegistry:
    """Discovers `<root>/*/SKILL.md` packages at construction time.

    Each skill's frontmatter (name/description/when_to_use) is parsed
    eagerly to build the system-prompt index; the body is read on demand via
    `Skill.load_body()` (progressive disclosure). Dropping a new
    `<root>/<name>/SKILL.md` extends the agent with no code changes.
    """

    def __init__(self, root: Path) -> None:
        skills = (Skill.from_skill_md(p) for p in sorted(root.glob("*/SKILL.md")))
        self._skills: dict[str, Skill] = {skill.name: skill for skill in skills}

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)
