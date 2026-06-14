"""Skill model: a local progressive-disclosure package.

A skill is a directory containing a `SKILL.md` whose frontmatter declares
`name`, `description`, and `when_to_use`. The agent's system prompt is
composed from the *index* (name + description + when_to_use) of every
registered skill; the full body is loaded into context only when a skill is
selected (see `agent/skills/registry.py`).

Skills extend behaviour/knowledge in-process. They never execute capability
themselves -- that always flows through the MCP layer + permission
interceptor (`agent/mcp/`).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

_FRONTMATTER_DELIM = "---"


class Skill(BaseModel):
    name: str
    description: str
    when_to_use: str
    path: Path

    @classmethod
    def from_skill_md(cls, path: Path) -> Skill:
        """Parse a `SKILL.md` file's YAML frontmatter into a `Skill` index
        entry. The body is not read here -- only on `load_body()`."""
        frontmatter, _ = _split_frontmatter(path.read_text())
        data: dict[str, str] = yaml.safe_load(frontmatter) or {}
        return cls(
            name=data["name"],
            description=data["description"],
            when_to_use=data["when_to_use"],
            path=path,
        )

    def load_body(self) -> str:
        """Read the full SKILL.md body on demand (progressive disclosure),
        with the YAML frontmatter stripped."""
        _, body = _split_frontmatter(self.path.read_text())
        return body


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split `---\\n<yaml>\\n---\\n<body>` into `(yaml, body)`. If there is no
    frontmatter delimiter, treat the whole file as the body."""
    if not text.startswith(_FRONTMATTER_DELIM):
        return "", text
    _, _, rest = text.partition("\n")
    frontmatter, sep, body = rest.partition(f"\n{_FRONTMATTER_DELIM}\n")
    if not sep:
        return "", text
    return frontmatter, body.lstrip("\n")
