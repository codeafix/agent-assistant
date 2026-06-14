"""Case-based eval suite: one Inspect task per `evals/cases/*.jsonl` file.

Each file is a family of `evals.spec.EvalCase` records -- ground truth (input,
scripted cassette, expected tool calls/skills/response/stop reason) -- run
through `run_agent` with the real `agent.toml` configuration (skills, MCP
servers, permissions) unless a case overrides them. See `evals.suite`.

Run with: `uv run inspect eval evals/tasks/cases.py`

By default the assistant is a scripted `ReplayModel` that deterministically
replays each case's cassette. Pass `-T model=<key>` (a registry key from
`agent.toml`'s `[models]`, e.g. `granite-local` or `anthropic`) to run the
same ground truth against a real model instead: `uv run inspect eval
evals/tasks/ -T model=granite-local`.
"""

from __future__ import annotations

from inspect_ai import Task, task

from evals.suite import case_task


@task
def tool_choice(model: str = "replay") -> Task:
    """Basic tool-choice ground truth: the agent calls the right tool with
    the right arguments and reports the result."""
    return case_task("tool_choice.jsonl", model)


@task
def skills(model: str = "replay") -> Task:
    """Skill usage ground truth: a skill is loaded and its instructions
    (call `clock`, format the result) are followed."""
    return case_task("skills.jsonl", model)


@task
def regression(model: str = "replay") -> Task:
    """Record/replay regression suite: a second MCP server dispatched
    correctly, and the loop-detection guard rail stopping a repeating model."""
    return case_task("regression.jsonl", model)


@task
def prompt_injection(model: str = "replay") -> Task:
    """Adversarial: a tool result coerces the model into requesting a
    disallowed tool, which must be denied and never executed."""
    return case_task("prompt_injection.jsonl", model)
