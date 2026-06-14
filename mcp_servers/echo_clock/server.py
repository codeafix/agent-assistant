"""A trivial MCP server with two tools: `echo` and `clock`.

This is the example server for the Phase-1 vertical slice and for the
permission-policy / loop-detection evals (Phase 4). It runs over stdio and,
in `deploy/compose.yaml`, as its own isolated service.

Run directly: `uv run python -m mcp_servers.echo_clock.server`
"""

from __future__ import annotations

from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-clock")


@mcp.tool()
def echo(text: str) -> str:
    """Return the given text unchanged."""
    return text


@mcp.tool()
def clock() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    mcp.run()
