"""Shared cost computation for model adapters with known per-token pricing."""

from __future__ import annotations

from agent.models.base import Usage


def priced_usage(
    usage: Usage,
    *,
    price_per_input_token_usd: float | None,
    price_per_output_token_usd: float | None,
) -> Usage:
    """Return `usage` with `cost_usd` set from per-token prices, if given."""
    if price_per_input_token_usd is None and price_per_output_token_usd is None:
        return usage
    cost = usage.input_tokens * (price_per_input_token_usd or 0.0)
    cost += usage.output_tokens * (price_per_output_token_usd or 0.0)
    return usage.model_copy(update={"cost_usd": cost})
