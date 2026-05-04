"""
Tagentacle AgentNode — inbox-as-resource agentic loop.

Architecture:
  InboxMCP (tagentacle-py-mcp): per-topic message buffer with dual access —
    MCP tools for the LLM, Python API for in-process consumers.
    No HTTP server; FastMCP is used only for tool schema registration.
  InferenceMux (inferencemux): IDLE/BUSY state machine + followup queue
  MCP Client: connects to external MCP servers for tool execution
  build_context(): pure function — system prompt + trigger reason → LLM context
  _run_inference_cycle(): the sole orchestrator

Data flow:
  bus message → subscribe callback → mailbox.push()
    → mux.trigger(detail=topic) → _run_inference_cycle()
    → build_context(trigger_detail) → call_service(/inference/chat)
    → LLM calls poll_messages (local) / external tools (MCP)
    → publish(/chat/output)

Key design:
  The LLM context contains ONLY the trigger reason (e.g. "new message on
  /chat/input"). The LLM autonomously decides whether to read the inbox
  via the poll_messages tool. User messages are never injected directly
  into the context — the agent reads them as a resource.

MCP Server discovery:
  Subscribes to /mcp/directory (infrastructure, not in inbox).
  Connects to servers via native MCP SDK Streamable HTTP client.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from tagentacle_py_core import LifecycleNode
from tagentacle_py_inferencemux import InferenceMux, TriggerSignal
from tagentacle_py_mcp import InboxMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ErrorData,
    TextContent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration defaults ---
DEFAULT_MODEL = os.environ.get("INFERENCE_MODEL", "moonshotai/kimi-k2.5")
# Hard wall-clock cap on a single inference cycle (seconds). Prevents
# the mux from staying BUSY indefinitely if a tool/LLM hangs.
INFERENCE_CYCLE_TIMEOUT_S = float(os.environ.get("INFERENCE_CYCLE_TIMEOUT_S", "180"))
SAMPLING_TIMEOUT_S = float(os.environ.get("SAMPLING_TIMEOUT_S", "120"))
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant running inside a Tagentacle agent node.\n\n"
    "## Mailbox\n"
    "You have a mailbox that buffers messages from bus topics. Each inference "
    "cycle, you receive a trigger reason telling you WHY you were woken up "
    "(e.g. a new message arrived on a topic, or a timer fired). Your mailbox "
    "tools let you inspect and consume messages:\n"
    "- **poll_messages**: Read and drain buffered messages from a topic (or all topics).\n"
    "- **subscribe_topic**: Subscribe to a new bus topic.\n"
    "- **set_subscription_level**: Change a topic between 'trigger' and 'silent'.\n"
    "- **unsubscribe_topic**: Unsubscribe from a topic.\n\n"
    "When triggered, first call poll_messages to read what arrived, then decide "
    "how to respond. Always respond in the same language as the user."
)
MAX_TOOL_ROUNDS = 10


# --- Mailbox tool schemas (OpenAI function-calling format) ---

_MAILBOX_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "poll_messages",
            "description": (
                "Read and drain buffered messages from the mailbox. "
                "Returns up to `limit` messages and removes them from the buffer. "
                "If topic is omitted, polls all subscribed topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic path to poll (e.g. '/chat/input'). Omit to poll all.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to return (default 50).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subscribe_topic",
            "description": (
                "Subscribe to a Tagentacle bus topic and start buffering "
                "incoming messages. Use poll_messages to read buffered messages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic path, e.g. '/sensor/data'.",
                    },
                    "level": {
                        "type": "string",
                        "description": "'trigger' (notify on message) or 'silent' (buffer only). Default: 'trigger'.",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_subscription_level",
            "description": (
                "Change subscription level for an already-subscribed topic. "
                "'trigger' sends notifications on new messages; 'silent' buffers only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic path.",
                    },
                    "level": {
                        "type": "string",
                        "description": "'trigger' or 'silent'.",
                    },
                },
                "required": ["topic", "level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unsubscribe_topic",
            "description": "Unsubscribe from a topic and clear its message buffer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic path to unsubscribe from.",
                    },
                },
                "required": ["topic"],
            },
        },
    },
]

_MAILBOX_TOOL_NAMES: set[str] = {t["function"]["name"] for t in _MAILBOX_TOOLS}


# --- Utilities ---


def mcp_tools_to_openai_schema(mcp_tools) -> list[dict]:
    """Convert MCP Tool objects to OpenAI function-calling tool schema."""
    result = []
    for tool in mcp_tools:
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        if "properties" not in schema:
            schema["properties"] = {}
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return result


def build_context(
    system_prompt: str,
    cycle_messages: list[dict],
    trigger_detail: str,
) -> list[dict]:
    """Pure function: render LLM context from system prompt + trigger reason.

    The LLM receives only the trigger reason — it must use mailbox tools
    (poll_messages) to read actual message content autonomously.
    """
    ctx = [{"role": "system", "content": system_prompt}]
    if trigger_detail:
        ctx.append(
            {
                "role": "system",
                "content": f"Inference triggered by: {trigger_detail}",
            }
        )
    ctx.extend(cycle_messages)
    return ctx


# --- AgentNode ---


class AgentNode(LifecycleNode):
    """Inbox-as-resource agent with InferenceMux trigger control.

    The agent NEVER writes business logic in bus callbacks.
    All callbacks do one thing: push to mailbox.  InferenceMux controls
    when to run an inference cycle.

    The LLM context contains only the trigger reason.  The LLM reads
    the inbox autonomously via built-in mailbox tools (poll_messages, etc.).
    InboxMCP provides dual access: MCP tool schemas for the LLM,
    Python API for in-process consumers — no HTTP server needed.

    Supports multiple instances via config:
      node_id, input_topic, output_topic, system_prompt
    """

    def __init__(self, node_id: str = "agent_node"):
        super().__init__(node_id)
        # InboxMCP: in-process mailbox with MCP tool schemas (no HTTP)
        self._fastmcp = FastMCP(name=node_id)
        self.mailbox = InboxMCP(self, self._fastmcp)
        self.mux = InferenceMux()

        # Session state (no cross-cycle conversation history)
        self.session_id: str = str(uuid.uuid4())[:8]
        self.model: str = DEFAULT_MODEL
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT

        # Topics (configurable for multi-agent)
        self._input_topic: str = "/chat/input"
        self._output_topic: str = "/chat/output"
        self._extra_subscribe: list[str] = []

        # MCP client state (external servers)
        self.openai_tools: list[dict] = []
        self._mcp_sessions: dict[str, ClientSession] = {}
        # tool_name -> server_id  (built so we route each tool to exactly
        # one MCP server instead of fan-out / first-match).
        self._tool_to_server: dict[str, str] = {}
        self._mcp_tasks: dict[str, asyncio.Task] = {}
        self._server_urls: dict[str, str] = {}
        self._target_server: str | None = None
        self._inference_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_configure(self, config: dict):
        self.model = config.get("model", DEFAULT_MODEL)
        if "system_prompt" in config:
            self.system_prompt = config["system_prompt"]
        if "input_topic" in config:
            self._input_topic = config["input_topic"]
        if "output_topic" in config:
            self._output_topic = config["output_topic"]
        if "extra_subscribe" in config:
            self._extra_subscribe = config["extra_subscribe"]
        self._target_server = config.get("mcp_server_id")
        logger.info(
            "Agent configured: model=%s session=%s input=%s output=%s",
            self.model,
            self.session_id,
            self._input_topic,
            self._output_topic,
        )

    async def on_activate(self):
        # Mailbox subscription — callback only pushes to mailbox
        self._subscribe_mailbox(self._input_topic, "trigger")

        # Extra subscriptions (e.g. /agent/b/output for cross-agent communication)
        for topic in self._extra_subscribe:
            self._subscribe_mailbox(topic, "trigger")

        # Infrastructure subscription — not in inbox, handled directly
        @self.subscribe("/mcp/directory")
        async def _on_directory(msg):
            await self._handle_directory(msg)

        # Auto-connect from env (fallback for simple setups)
        server_url = os.environ.get("MCP_SERVER_URL")
        if server_url:
            self._start_mcp_session("env_server", server_url)

        # Active discovery: query existing MCP servers via their services
        asyncio.create_task(self._discover_existing_servers())

        # Ensure mailbox tools are always available (even before MCP connects)
        self.openai_tools = list(_MAILBOX_TOOLS)

        # Start inference loop
        self._inference_task = asyncio.create_task(self._inference_loop())
        logger.info("AgentNode activated — waiting for messages")

    async def on_shutdown(self):
        if self._inference_task:
            self._inference_task.cancel()
        for t in self._mcp_tasks.values():
            t.cancel()
        self.mux.reset()
        logger.info("AgentNode shut down.")

    # ------------------------------------------------------------------
    # Mailbox wiring
    # ------------------------------------------------------------------

    def _subscribe_mailbox(self, topic: str, level: str = "trigger"):
        """Subscribe to a bus topic using the mailbox pattern.

        The callback does exactly ONE thing: push to mailbox.
        If level is 'trigger', it also triggers InferenceMux with
        the topic name as detail.
        """
        self.mailbox.register(topic, level)

        @self.subscribe(topic)
        async def _on_message(msg, _topic=topic):
            should_trigger = self.mailbox.push(_topic, msg)
            if should_trigger:
                await self.mux.trigger(TriggerSignal(topic=_topic, detail=_topic))

    # ------------------------------------------------------------------
    # MCP server discovery + connection
    # ------------------------------------------------------------------

    async def _handle_directory(self, msg: dict):
        """Process /mcp/directory for auto-discovery."""
        payload = msg.get("payload", {})
        server_id = payload.get("server_id")
        url = payload.get("url")
        status = payload.get("status")

        if not server_id:
            return
        if status == "available" and url and server_id not in self._mcp_sessions:
            if self._target_server is None or server_id == self._target_server:
                self._server_urls[server_id] = url
                self._start_mcp_session(server_id, url)
                logger.info("Discovered MCP server: %s at %s", server_id, url)
        elif status == "unavailable":
            self._server_urls.pop(server_id, None)
            self._mcp_sessions.pop(server_id, None)
            task = self._mcp_tasks.pop(server_id, None)
            if task and not task.done():
                task.cancel()
            await self._refresh_tools()

    async def _discover_existing_servers(self):
        """Pull-based discovery: query gateway for all known MCP servers."""
        try:
            resp = await self.call_service(
                "/mcp/gateway/list_servers",
                {},
                timeout=10.0,
            )
            servers = resp.get("servers", []) if isinstance(resp, dict) else []
            for entry in servers:
                server_id = entry.get("server_id")
                url = entry.get("url")
                if not server_id or not url:
                    continue
                if entry.get("status") not in ("available", None):
                    continue
                if self._target_server is not None and server_id != self._target_server:
                    continue
                if server_id not in self._mcp_sessions:
                    self._server_urls[server_id] = url
                    self._start_mcp_session(server_id, url)
                    logger.info(
                        "Discovered (pull) MCP server: %s at %s", server_id, url
                    )
        except Exception as e:
            logger.info("MCP pull-discovery skipped (gateway not ready): %s", e)

    def _start_mcp_session(self, server_id: str, url: str):
        """Start an MCP client session in the background with retry."""
        if server_id in self._mcp_sessions or server_id in self._mcp_tasks:
            return  # already connected or connecting

        async def _loop():
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    async with streamable_http_client(url) as (r, w, _):
                        async with ClientSession(
                            r,
                            w,
                            sampling_callback=self._handle_sampling,
                        ) as session:
                            await session.initialize()
                            self._mcp_sessions[server_id] = session
                            await self._refresh_tools()
                            logger.info(
                                "MCP ready: %s (%d tools)",
                                server_id,
                                len(self.openai_tools),
                            )
                            await asyncio.Future()  # keep alive
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    self._mcp_sessions.pop(server_id, None)
                    await self._refresh_tools()
                    if attempt < max_retries - 1:
                        delay = min(2**attempt, 10)
                        logger.warning(
                            "MCP connect failed (%s), retry %d/%d in %ds: %s",
                            server_id,
                            attempt + 1,
                            max_retries,
                            delay,
                            e,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "MCP connect gave up (%s) after %d retries: %s",
                            server_id,
                            max_retries,
                            e,
                        )
                finally:
                    self._mcp_sessions.pop(server_id, None)
                    await self._refresh_tools()

        self._mcp_tasks[server_id] = asyncio.create_task(_loop())

    async def _refresh_tools(self):
        """Collect tools from all connected MCP sessions + built-in mailbox tools.

        Also rebuilds ``self._tool_to_server`` so :meth:`_call_tool` can
        deterministically route a tool name to a specific server.
        Last-writer wins on duplicate tool names; a warning is logged so
        operators notice the collision.
        """
        all_tools: list[dict] = []
        seen: dict[str, str] = {}  # tool_name -> server_id
        for server_id, session in list(self._mcp_sessions.items()):
            try:
                result = await session.list_tools()
            except Exception as e:
                logger.warning("MCP list_tools failed (%s): %s", server_id, e)
                continue
            for schema in mcp_tools_to_openai_schema(result.tools):
                name = schema["function"]["name"]
                if name in seen and seen[name] != server_id:
                    logger.warning(
                        "Duplicate MCP tool '%s' on servers %s and %s; routing to %s",
                        name,
                        seen[name],
                        server_id,
                        server_id,
                    )
                seen[name] = server_id
                all_tools.append(schema)
        all_tools.extend(_MAILBOX_TOOLS)
        self.openai_tools = all_tools
        self._tool_to_server = seen

    # ------------------------------------------------------------------
    # MCP sampling — Server-initiated LLM requests routed to /inference/chat
    # ------------------------------------------------------------------

    async def _handle_sampling(
        self,
        ctx: RequestContext[ClientSession, Any],
        params: CreateMessageRequestParams,
    ) -> CreateMessageResult | ErrorData:
        """Sampling callback for MCP servers (e.g. shell-mcp).

        Translates ``CreateMessageRequestParams`` to the OpenAI chat
        format expected by ``/inference/chat`` and returns the assistant
        reply as a ``CreateMessageResult``. If the bus service fails,
        an ``ErrorData`` is returned instead of raising — per MCP spec
        the server expects either branch.
        """
        try:
            messages: list[dict] = []
            if params.systemPrompt:
                messages.append({"role": "system", "content": params.systemPrompt})
            for sm in params.messages:
                content = sm.content
                if isinstance(content, TextContent):
                    text = content.text
                else:
                    # Non-text sampling content (image/audio) is not
                    # supported by the bus inference service today.
                    text = getattr(content, "text", "") or ""
                messages.append({"role": sm.role, "content": text})

            request: dict[str, Any] = {"model": self.model, "messages": messages}
            if params.maxTokens:
                request["max_tokens"] = params.maxTokens
            if params.temperature is not None:
                request["temperature"] = params.temperature
            if params.stopSequences:
                request["stop"] = list(params.stopSequences)

            result = await self.call_service(
                "/inference/chat",
                request,
                timeout=SAMPLING_TIMEOUT_S,
            )
            if "error" in result:
                return ErrorData(code=-32000, message=str(result["error"]))

            choice = result["choices"][0]["message"]
            text = choice.get("content") or ""
            stop_reason = result["choices"][0].get("finish_reason") or "endTurn"
            model = result.get("model", self.model)
            return CreateMessageResult(
                role="assistant",
                content=TextContent(type="text", text=text),
                model=model,
                stopReason=stop_reason,
            )
        except Exception as e:
            logger.exception("Sampling handler failed")
            return ErrorData(code=-32000, message=f"sampling failed: {e}")

    # ------------------------------------------------------------------
    # Inference loop — the core of the agent
    # ------------------------------------------------------------------

    async def _inference_loop(self):
        """Wait for mux trigger → run inference cycle → release → repeat."""
        while True:
            signal = await self.mux.wait()
            try:
                await asyncio.wait_for(
                    self._run_inference_cycle(signal),
                    timeout=INFERENCE_CYCLE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Inference cycle exceeded wall-clock timeout (%.0fs)",
                    INFERENCE_CYCLE_TIMEOUT_S,
                )
                await self.publish(
                    self._output_topic,
                    {
                        "text": "⚠️ Inference timed out.",
                        "session_id": self.session_id,
                    },
                )
            except Exception as e:
                logger.error("Inference cycle error: %s", e, exc_info=True)
                await self.publish(
                    self._output_topic,
                    {
                        "text": f"⚠️ Error: {e}",
                        "session_id": self.session_id,
                    },
                )
            finally:
                self.mux.release()

    async def _run_inference_cycle(self, trigger: TriggerSignal):
        """One complete inference cycle — the sole orchestrator.

        Phase 1: Build trigger context (why was the agent woken up?)
        Phase 2: Inference + tool loop (LLM reads inbox via tools)
        Phase 3: Publish final reply
        """
        # --- Phase 1: trigger context ---
        trigger_detail = trigger.detail or "timer"
        cycle_messages: list[dict] = []

        # --- Phase 2: inference + tool loop ---
        for round_num in range(MAX_TOOL_ROUNDS):
            ctx = build_context(
                self.system_prompt,
                cycle_messages,
                trigger_detail if round_num == 0 else "",
            )

            request = {"model": self.model, "messages": ctx}
            if self.openai_tools:
                request["tools"] = self.openai_tools

            logger.info("Inference round %d...", round_num + 1)
            result = await self.call_service(
                "/inference/chat",
                request,
                timeout=120,
            )
            if "error" in result:
                raise RuntimeError(f"Inference: {result['error']}")

            assistant_msg = result["choices"][0]["message"]
            cycle_messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                # --- Phase 3: final reply ---
                content = assistant_msg.get("content", "")
                logger.info("Reply (round %d): %.80s...", round_num + 1, content)
                await self.publish(
                    self._output_topic,
                    {
                        "text": content,
                        "session_id": self.session_id,
                    },
                )
                await self.publish(
                    "/memory/latest",
                    {
                        "session_id": self.session_id,
                        "messages": cycle_messages,
                    },
                )
                return

            # Execute tool calls (mailbox tools local, MCP tools remote)
            logger.info("Tools: %s", [tc["function"]["name"] for tc in tool_calls])
            for tc in tool_calls:
                name = tc["function"]["name"]
                args_str = tc["function"].get("arguments", "{}")
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                tool_result = await self._call_tool(name, args)
                cycle_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                )

        # Safety: exceeded max rounds
        await self.publish(
            self._output_topic,
            {
                "text": "⚠️ Exceeded maximum tool call rounds.",
                "session_id": self.session_id,
            },
        )

    async def _call_tool(self, name: str, arguments: dict) -> str:
        """Route tool call: mailbox tools local, MCP tools by routing map."""
        if name in _MAILBOX_TOOL_NAMES:
            return await self._call_mailbox_tool(name, arguments)
        server_id = self._tool_to_server.get(name)
        session = self._mcp_sessions.get(server_id) if server_id else None
        if session is None:
            return f"Error: no MCP server can execute tool '{name}'"
        try:
            result = await session.call_tool(name, arguments=arguments)
        except Exception as e:
            return f"Error: tool '{name}' on server '{server_id}' failed: {e}"
        return "\n".join(
            b.text if hasattr(b, "text") else str(b) for b in result.content
        )

    async def _call_mailbox_tool(self, name: str, arguments: dict) -> str:
        """Execute a built-in mailbox tool via InboxMCP Python API."""
        if name == "poll_messages":
            topic = arguments.get("topic", "")
            limit = arguments.get("limit", 50)
            msgs = self.mailbox.drain(topic, limit=limit)
            return json.dumps(msgs, ensure_ascii=False, default=str)

        if name == "subscribe_topic":
            topic = arguments.get("topic", "")
            level = arguments.get("level", "trigger")
            if not topic:
                return json.dumps({"error": "missing 'topic'"})
            if self.mailbox.is_subscribed(topic):
                pending = self.mailbox.pending_for(topic)
                return (
                    f"Already subscribed to '{topic}'. {pending} buffered message(s)."
                )
            self._subscribe_mailbox(topic, level)
            return f"Subscribed to '{topic}' (level={level})."

        if name == "set_subscription_level":
            topic = arguments.get("topic", "")
            level = arguments.get("level", "trigger")
            if not self.mailbox.is_subscribed(topic):
                return f"Not subscribed to '{topic}'. Subscribe first."
            try:
                old = self.mailbox.set_level(topic, level)
            except ValueError as e:
                return f"Error: {e}"
            return f"Subscription level for '{topic}': {old} → {level}"

        if name == "unsubscribe_topic":
            topic = arguments.get("topic", "")
            if not self.mailbox.is_subscribed(topic):
                return f"Not subscribed to '{topic}'."
            count = self.mailbox.forget(topic)
            self.subscribers.pop(topic, None)
            return f"Unsubscribed from '{topic}'. Cleared {count} buffered message(s)."

        return f"Error: unknown mailbox tool '{name}'"


async def main():
    node_id = os.environ.get("AGENT_ID", "agent_node")
    agent = AgentNode(node_id=node_id)

    # Build config from environment for multi-agent support
    config: dict = {}
    if os.environ.get("AGENT_ROLE"):
        role = os.environ["AGENT_ROLE"]
        if role == "executor":
            config.setdefault("input_topic", "/agent/b/input")
            config.setdefault("output_topic", "/agent/b/output")
            config.setdefault("extra_subscribe", ["/agent/a/output"])
            config.setdefault(
                "system_prompt",
                "You are Agent B, an executor. You receive tasks from Agent A "
                "and execute them using available tools. Report results back. "
                "Always respond in the same language as the input.",
            )
        elif role == "coordinator":
            config.setdefault("extra_subscribe", ["/agent/b/output"])
            config.setdefault(
                "system_prompt",
                "You are Agent A, a coordinator. You receive user requests and "
                "can delegate tasks to Agent B by publishing to /agent/b/input. "
                "You have access to tools. Always respond in the same language "
                "as the user.",
            )

    await agent.bringup(config)
    await agent.spin()


if __name__ == "__main__":
    asyncio.run(main())
