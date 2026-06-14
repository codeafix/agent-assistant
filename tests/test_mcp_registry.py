"""Phase 3 checkpoint: `MCPToolRegistry` discovers and dispatches tools
across *multiple* configured servers, merging them into one namespace --
adding a server is a config-only change (see `agent.toml`)."""

import sys

from agent.config import MCPServerConfig
from agent.mcp.registry import MCPToolRegistry

ECHO_CLOCK_SERVER = MCPServerConfig(
    name="echo-clock",
    transport="stdio",
    command=sys.executable,
    args=["-m", "mcp_servers.echo_clock.server"],
)

WORDCOUNT_SERVER = MCPServerConfig(
    name="wordcount",
    transport="stdio",
    command=sys.executable,
    args=["-m", "mcp_servers.wordcount.server"],
)


async def test_registry_merges_tools_from_multiple_servers() -> None:
    async with MCPToolRegistry([ECHO_CLOCK_SERVER, WORDCOUNT_SERVER]) as tools:
        names = {spec.name for spec in tools.list_tool_specs()}
        assert names == {"echo", "clock", "count_words"}

        assert tools.server_for_tool("echo") == "echo-clock"
        assert tools.server_for_tool("clock") == "echo-clock"
        assert tools.server_for_tool("count_words") == "wordcount"

        result = await tools.call_tool("wordcount", "count_words", {"text": "one two three"})
        assert result.is_error is False
        assert isinstance(result.content, list)
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "3"
