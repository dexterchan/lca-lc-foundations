# Personal notes

## MCP: the manual style and FastMCP

Noted on 14 September 2026.

I first worked with MCP in 2025 Q2. I recall a harder, more manual way to build a server. The exact library and version from that time have not been checked.

The time server installed in this project uses `Server` from the Python MCP SDK (the library for building MCP apps). Our small server uses `FastMCP` from that same SDK.

Both use decorators—the `@...` lines above functions—but they register different things.

| Job | Lower-level `Server` | `FastMCP` |
|---|---|---|
| Describe tool inputs | Write `inputSchema`, the input rules, by hand | Read the function's Python types |
| List tools | Write a `@server.list_tools()` handler | Build the list from registered functions |
| Run a tool by name | Write a `@server.call_tool()` handler | Find and run the matching function |
| Format the reply | Build MCP reply objects | Convert the function's result |

With `@mcp.tool()`, FastMCP registers the function and reads its name, docstring, and input types. It handles the MCP `tools/list` and `tools/call` messages for us.

FastMCP means less setup code to write and maintain. That gives me fewer places to make mistakes. I still need to check my own logic and handle failures, such as a web search failing.

For small servers like these examples, I would use FastMCP. The time server could use it too; its time-zone logic would still be needed.

Files checked, with paths from the project root:

```text
notebooks/module-2/resources/2.1_mcp_server.py
.venv/lib/python3.12/site-packages/mcp_server_time/server.py
.venv/lib/python3.12/site-packages/mcp/server/fastmcp/server.py
```
