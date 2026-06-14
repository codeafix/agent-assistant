"""Phase 3 checkpoint: `FileSystemSkillRegistry` loads `SKILL.md` packages
with progressive disclosure, and the loop composes them into the system
prompt and synthesizes a tool spec per skill."""

from pathlib import Path

from agent.core.loop import (
    _skill_tool_specs,  # pyright: ignore[reportPrivateUsage]
    compose_system_prompt,
)
from agent.skills.base import Skill
from agent.skills.registry import EmptySkillRegistry, FileSystemSkillRegistry

REPO_SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _write_skill(root: Path, dirname: str, body: str) -> None:
    skill_dir = root / dirname
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(body)


def test_from_skill_md_parses_frontmatter_and_strips_it_from_the_body(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "greeting",
        "---\n"
        "name: greeting\n"
        "description: Greet the user warmly.\n"
        "when_to_use: The user says hello.\n"
        "---\n"
        "# Greeting\n"
        "Say hi back.\n",
    )

    skill = Skill.from_skill_md(tmp_path / "greeting" / "SKILL.md")

    assert skill.name == "greeting"
    assert skill.description == "Greet the user warmly."
    assert skill.when_to_use == "The user says hello."
    assert skill.load_body() == "# Greeting\nSay hi back.\n"


def test_registry_discovers_skills_by_dropping_in_a_folder(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "alpha",
        "---\nname: alpha\ndescription: Alpha skill.\nwhen_to_use: For alpha.\n---\nAlpha body\n",
    )
    _write_skill(
        tmp_path,
        "beta",
        "---\nname: beta\ndescription: Beta skill.\nwhen_to_use: For beta.\n---\nBeta body\n",
    )

    registry = FileSystemSkillRegistry(tmp_path)

    assert {s.name for s in registry.list_skills()} == {"alpha", "beta"}
    beta = registry.get_skill("beta")
    assert beta is not None
    assert beta.load_body() == "Beta body\n"
    assert registry.get_skill("missing") is None


def test_empty_registry_has_no_skills() -> None:
    registry = EmptySkillRegistry()
    assert registry.list_skills() == []
    assert registry.get_skill("anything") is None


def test_compose_system_prompt_indexes_skills(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "alpha",
        "---\nname: alpha\ndescription: Alpha skill.\nwhen_to_use: For alpha things.\n---\nbody\n",
    )
    registry = FileSystemSkillRegistry(tmp_path)

    prompt = compose_system_prompt("Base instructions.", registry)

    assert "Base instructions." in prompt
    assert "alpha: Alpha skill. (use when: For alpha things.)" in prompt


def test_compose_system_prompt_unchanged_with_no_skills() -> None:
    assert compose_system_prompt("Base instructions.", EmptySkillRegistry()) == "Base instructions."


def test_skill_tool_specs_one_per_skill(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "alpha",
        "---\nname: alpha\ndescription: Alpha skill.\nwhen_to_use: For alpha things.\n---\nbody\n",
    )
    registry = FileSystemSkillRegistry(tmp_path)

    specs = _skill_tool_specs(registry)

    assert len(specs) == 1
    assert specs[0].name == "alpha"
    assert "Alpha skill." in specs[0].description


def test_shipped_timestamping_skill_is_discoverable() -> None:
    registry = FileSystemSkillRegistry(REPO_SKILLS_DIR)

    skill = registry.get_skill("timestamping")

    assert skill is not None
    assert "clock" in skill.load_body()
    assert "UTC" in skill.description
