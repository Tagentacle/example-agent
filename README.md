# example_agent

An example **Agent Node** with an inbox-as-resource agentic loop.

## Architecture

The agent uses an **inbox-as-resource** pattern: the LLM context never
contains directly injected user messages.  Instead, each inference cycle
receives only a **trigger reason** (e.g. "new message on /chat/input").
The LLM autonomously decides whether to read the inbox via built-in
mailbox tools (`poll_messages`, `subscribe_topic`, etc.).

**Key components:**
- **InboxMCP** (from `tagentacle-py-mcp`): In-process mailbox with dual
  Python/MCP access.  No HTTP server — `FastMCP` is used only for tool
  schema registration.
- **InferenceMux**: IDLE/BUSY state machine controlling when to run
  inference cycles.
- **MCP Client**: Connects to external MCP servers for tool execution.

## What it does

1. Connects to the Tagentacle Daemon as `agent_node`
2. Subscribes to `/chat/input` → buffers messages in InboxMCP mailbox
3. Discovers external MCP servers via `/mcp/directory` (or `MCP_SERVER_URL` env var)
4. On trigger: builds context with trigger reason → calls `/inference/chat`
5. LLM calls `poll_messages` to read inbox → calls external tools via MCP → responds
6. Publishes final reply to `/chat/output` and cycle log to `/memory/latest`

## Built-in Mailbox Tools

The LLM has access to these locally-routed tools (no HTTP round-trip):

| Tool | Description |
|------|-------------|
| `poll_messages(topic?, limit?)` | Read & drain buffered messages |
| `subscribe_topic(topic, level?)` | Subscribe to a new bus topic |
| `set_subscription_level(topic, level)` | Change trigger/silent level |
| `unsubscribe_topic(topic)` | Remove subscription |

## Prerequisites

- Tagentacle Daemon running (`tagentacle daemon`)
- An inference service (or `example-inference` package)
- (Optional) `example_mcp_server` for external tools like `get_weather`

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

- **Inbox-as-Resource**: User messages are never injected into LLM context. The LLM reads them as a resource via `poll_messages`.
- **InboxMCP (Mode 2 — no HTTP)**: `FastMCP` instance registers tool schemas only. Mailbox tools are locally intercepted in `_call_tool()` and routed to the Python API.
- **Native MCP HTTP Client**: Uses `mcp.client.streamable_http.streamable_http_client()` for external MCP servers — direct HTTP connection, no bus transport.
- **Auto-Discovery**: Subscribes to `/mcp/directory` Topic; when a matching server appears, connects automatically.
