"""
Tagentacle AgentNode — mailbox-based agentic loop.

Architecture (new_agent_node.md / Q27 邮箱模型):
  Inbox (core): per-topic message buffer — callbacks only push, no logic
  InferenceMux (inferencemux): IDLE/BUSY state machine + followup queue
  MCP Client: connects to external MCP servers for tool execution
  build_context(): pure function — messages + notifications → LLM context
  _run_inference_cycle(): the sole orchestrator

Data flow:
  bus message → subscribe callback → inbox.push()
    → mux.trigger() → _run_inference_cycle()
    → inbox.drain() → build_context() → call_service(/inference/chat)
    → tool_calls (MCP) → publish(/chat/output)

MCP Server discovery:
  Subscribes to /mcp/directory (infrastructure, not in inbox).
  Connects to servers via native MCP SDK Streamable HTTP client.
"""

import asyncio
import json
import logging
import os
import uuid

from tagentacle_py_core import LifecycleNode, Inbox, TopicMode
from tagentacle_py_inferencemux import InferenceMux, TriggerSignal
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration defaults ---
DEFAULT_MODEL = os.environ.get("INFERENCE_MODEL", "moonshotai/kimi-k2.5")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. You have access to tools that you can use "
    "to help answer questions. When a user asks about the weather, use the "
    "get_weather tool. Always respond in the same language as the user."
)
MAX_TOOL_ROUNDS = 10


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
    messages: list[dict],
    notifications: list[dict],
) -> list[dict]:
    """Pure function: render LLM context from conversation + notifications.

    Notifications (new inbox items since last cycle) are injected as a system
    message so the LLM sees ambient information without explicit user input.
    """
    ctx = [{"role": "system", "content": system_prompt}]
    ctx.extend(messages)
    if notifications:
        parts = []
        for n in notifications:
            topic = n.get("topic", "?")
            payload = {k: v for k, v in n.items() if k not in ("topic", "ts")}
            parts.append(f"[{topic}] {json.dumps(payload, ensure_ascii=False)}")
        ctx.append(
            {
                "role": "system",
                "content": "New bus notifications:\n" + "\n".join(parts),
            }
        )
    return ctx


# --- AgentNode ---


class AgentNode(LifecycleNode):
    """Mailbox-based agent with InferenceMux trigger control.

    The agent NEVER writes business logic in bus callbacks.
    All callbacks do one thing: push to inbox.  InferenceMux controls
    when to drain the inbox and run an inference cycle.

    Supports multiple instances via config:
      node_id, input_topic, output_topic, system_prompt
    """

    def __init__(self, node_id: str = "agent_node"):
        super().__init__(node_id)
        self.inbox = Inbox()
        self.mux = InferenceMux()

        # Conversation state
        self.messages: list[dict] = []
        self.session_id: str = str(uuid.uuid4())[:8]
        self.model: str = DEFAULT_MODEL
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT

        # Topics (configurable for multi-agent)
        self._input_topic: str = "/chat/input"
        self._output_topic: str = "/chat/output"
        self._extra_subscribe: list[str] = []

        # MCP client state
        self.openai_tools: list[dict] = []
        self._mcp_sessions: dict[str, ClientSession] = {}
        self._mcp_tasks: list[asyncio.Task] = []
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
        # Mailbox subscription — callback only pushes to inbox
        self._subscribe_mailbox(self._input_topic, TopicMode.FOLLOWUP)

        # Extra subscriptions (e.g. /agent/b/output for cross-agent communication)
        for topic in self._extra_subscribe:
            self._subscribe_mailbox(topic, TopicMode.FOLLOWUP)

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

        # Start inference loop
        self._inference_task = asyncio.create_task(self._inference_loop())
        logger.info("AgentNode activated — waiting for messages")

    async def on_shutdown(self):
        if self._inference_task:
            self._inference_task.cancel()
        for t in self._mcp_tasks:
            t.cancel()
        logger.info("AgentNode shut down.")

    # ------------------------------------------------------------------
    # Mailbox wiring
    # ------------------------------------------------------------------

    def _subscribe_mailbox(self, topic: str, mode: TopicMode):
        """Subscribe to a bus topic using the mailbox pattern.

        The callback does exactly ONE thing: push to inbox.
        If mode is FOLLOWUP, it also triggers InferenceMux.
        """
        self.inbox.set_mode(topic, mode)

        @self.subscribe(topic)
        async def _on_message(msg, _topic=topic):
            should_trigger = self.inbox.push(_topic, msg)
            if should_trigger:
                await self.mux.trigger(TriggerSignal(topic=_topic))

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
        if server_id in self._mcp_sessions:
            return  # already connected

        async def _loop():
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    async with streamable_http_client(url) as (r, w, _):
                        async with ClientSession(r, w) as session:
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

        self._mcp_tasks.append(asyncio.create_task(_loop()))

    async def _refresh_tools(self):
        """Collect tools from all connected MCP sessions."""
        all_tools = []
        for session in self._mcp_sessions.values():
            try:
                result = await session.list_tools()
                all_tools.extend(mcp_tools_to_openai_schema(result.tools))
            except Exception:
                pass
        self.openai_tools = all_tools

    # ------------------------------------------------------------------
    # Inference loop — the core of the agent
    # ------------------------------------------------------------------

    async def _inference_loop(self):
        """Wait for mux trigger → run inference cycle → release → repeat."""
        while True:
            signal = await self.mux.wait()
            try:
                await self._run_inference_cycle(signal)
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

        Phase 1: Drain inbox (take all buffered messages)
        Phase 2: Extract user input → append to conversation
        Phase 3: Inference + tool loop
        Phase 4: Publish final reply
        """
        # --- Phase 1: drain inbox ---
        notifications = self.inbox.drain()

        # --- Phase 2: extract user messages ---
        other_notifs = []
        for n in notifications:
            if n.get("topic") == self._input_topic:
                text = n.get("payload", {}).get("text", "").strip()
                if text:
                    self.messages.append({"role": "user", "content": text})
                sid = n.get("payload", {}).get("session_id")
                if sid:
                    self.session_id = sid
            else:
                other_notifs.append(n)

        if not self.messages:
            return

        # --- Phase 3: inference + tool loop ---
        for round_num in range(MAX_TOOL_ROUNDS):
            ctx = build_context(
                self.system_prompt,
                self.messages,
                other_notifs if round_num == 0 else [],
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
            self.messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                # --- Phase 4: final reply ---
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
                        "messages": self.messages,
                    },
                )
                return

            # Execute tool calls via MCP
            logger.info("Tools: %s", [tc["function"]["name"] for tc in tool_calls])
            for tc in tool_calls:
                name = tc["function"]["name"]
                args_str = tc["function"].get("arguments", "{}")
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                tool_result = await self._call_tool(name, args)
                self.messages.append(
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
        """Execute a tool via the first MCP session that can handle it."""
        for session in self._mcp_sessions.values():
            try:
                result = await session.call_tool(name, arguments=arguments)
                return "\n".join(
                    b.text if hasattr(b, "text") else str(b) for b in result.content
                )
            except Exception:
                continue
        return f"Error: no MCP server can execute tool '{name}'"


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
