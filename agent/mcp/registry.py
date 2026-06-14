"""MCP tool registry: discovers tools across configured servers and
dispatches calls. Implements `agent.core.interfaces.ToolRegistry` -- this is
the extension point for adding MCP servers without touching `agent/core`."""

from __future__ import annotations

from types import TracebackType

from agent.config import MCPServerConfig
from agent.core.messages import ToolResultBlock, ToolSpec
from agent.mcp.client import MCPServerConnection


class MCPToolRegistry:
    """Connects to every configured MCP server and merges their tools into
    one namespace, remembering which server backs each tool name."""

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._connections: dict[str, MCPServerConnection] = {
            c.name: MCPServerConnection(c) for c in configs
        }
        self._tool_specs: list[ToolSpec] = []
        self._tool_to_server: dict[str, str] = {}

    async def connect(self) -> None:
        for connection in self._connections.values():
            await connection.connect()
            for tool in await connection.list_tools():
                spec = ToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                )
                self._tool_specs.append(spec)
                self._tool_to_server[tool.name] = connection.name

    async def close(self) -> None:
        for connection in self._connections.values():
            await connection.close()

    async def __aenter__(self) -> MCPToolRegistry:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    def list_tool_specs(self) -> list[ToolSpec]:
        return self._tool_specs

    def server_for_tool(self, tool_name: str) -> str:
        try:
            return self._tool_to_server[tool_name]
        except KeyError:
            raise KeyError(f"no MCP server provides tool '{tool_name}'") from None

    async def call_tool(self, server: str, tool: str, args: dict[str, object]) -> ToolResultBlock:
        connection = self._connections[server]
        result = await connection.call_tool(tool, args)
        content = [item.model_dump(mode="json") for item in result.content]
        return ToolResultBlock(tool_use_id="", content=content, is_error=result.isError)
