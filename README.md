# example_agent

An example **Agent Node** that calls MCP tools through the Tagentacle message bus.

## What it does

1. Connects to the Tagentacle Daemon as `weather_agent_node`
2. Opens an MCP session to `mcp_server_node` via bus transport
3. Lists available tools, then calls `get_weather` for Shenzhen and Tokyo
4. Prints the results and exits

## Prerequisites

- Tagentacle Daemon running (`tagentacle daemon`)
- `example_mcp_server` node running (provides the `get_weather` tool)

## Run

```bash
# Via CLI (recommended)
tagentacle run --pkg .

# Via Bringup (auto-starts all dependencies)
# Clone example-bringup and run from there
```

## Key Concepts

- **`tagentacle_client_transport`**: Bridges MCP `ClientSession` over Tagentacle Service calls, so the agent uses the standard MCP SDK without modification.
- **Zero network config**: The agent doesn't need the MCP server's address — it routes through the bus by `node_id`.
