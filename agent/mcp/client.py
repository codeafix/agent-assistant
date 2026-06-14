"""MCP client connection management.

Each `MCPServerConnection` owns one transport + `ClientSession` to a single
(process-isolated) MCP server. `agent/mcp/registry.py` builds the
`ToolRegistry` on top of one or more connections.
"""

from __future__ import annotations

from contextlib import AsyncExitStack

from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent.config import MCPServerConfig


class MCPServerConnection:
    """A connected session to one MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.name = config.name
        self._config = config
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        if self._config.transport != "stdio":
            raise NotImplementedError(
                f"MCP transport '{self._config.transport}' is not yet supported "
                "(only 'stdio' is implemented)"
            )
        if self._config.command is None:
            raise ValueError(f"MCP server '{self.name}': stdio transport requires 'command'")

        params = StdioServerParameters(command=self._config.command, args=self._config.args)
        read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._session = session

    async def close(self) -> None:
        await self._exit_stack.aclose()
        self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(f"MCP server '{self.name}' is not connected")
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, tool: str, arguments: dict[str, object]) -> types.CallToolResult:
        return await self.session.call_tool(tool, arguments)
