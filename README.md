# example_agent

An example **Agent Node** with a full agentic loop that calls MCP tools via Streamable HTTP.

## What it does

1. Connects to the Tagentacle Daemon as `agent_node`
2. Discovers MCP servers via `/mcp/directory` Topic subscription (or `MCP_SERVER_URL` env var)
3. Subscribes to `/chat/input` → receives user messages
4. Calls `/inference/chat` Service for LLM completions
5. Parses `tool_calls` → executes via MCP HTTP session → backfills results → re-infers
6. Publishes final reply to `/chat/output` and memory to `/memory/latest`

## Prerequisites

- Tagentacle Daemon running (`tagentacle daemon`)
- `example_mcp_server` node running (provides the `get_weather` tool)
- An inference service (or `example-inference` package)

## Run

```bash
# Via CLI (recommended)
tagentacle run --pkg .

# With explicit MCP server URL
MCP_SERVER_URL=http://127.0.0.1:8200/mcp tagentacle run --pkg .

# Via Bringup (auto-starts all dependencies)
# Clone example-bringup and run from there
```

## Key Concepts

- **Native MCP HTTP Client**: Uses `mcp.client.streamable_http.streamable_http_client()` — direct HTTP connection to MCP servers, no bus transport.
- **Auto-Discovery**: Subscribes to `/mcp/directory` Topic; when a server with the matching `_target_server` appears, connects automatically.
- **`MCP_SERVER_URL`**: Bypass discovery — provide a direct URL to the MCP server.
