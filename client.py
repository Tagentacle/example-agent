"""
Tagentacle Chatbot Agent: Full Agentic Loop with MCP Tool Use.

This Agent Node owns the complete agentic loop:
  1. Subscribe /chat/input → receive user messages
  2. Manage context window (messages array)
  3. Call /inference/chat Service for LLM completion
  4. Parse tool_calls → execute via MCP Transport → backfill results → re-infer
  5. Publish final reply to /chat/output
  6. Publish memory state to /memory/latest on every update

Data flow:
    /chat/input ──▶ Agent (agentic loop)
                      ├─ call_service("/inference/chat") ──▶ Inference Node
                      ├─ MCP session.call_tool()         ──▶ MCP Server Node
                      ├─ publish /chat/output             ──▶ Frontend Node
                      └─ publish /memory/latest           ──▶ Memory Node + Frontend
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from tagentacle_py_core import LifecycleNode
from mcp import ClientSession
from tagentacle_py_mcp.transport import tagentacle_client_transport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Default configuration ---
DEFAULT_MODEL = os.environ.get("INFERENCE_MODEL", "moonshotai/kimi-k2.5")
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. You have access to tools that you can use "
    "to help answer questions. When a user asks about the weather, use the "
    "get_weather tool. Always respond in the same language as the user."
)
MAX_TOOL_ROUNDS = 10  # Safety: max consecutive tool-call rounds


def mcp_tools_to_openai_schema(mcp_tools) -> list[dict]:
    """Convert MCP Tool objects to OpenAI function-calling tool schema.

    MCP's inputSchema IS JSON Schema — same format as OpenAI's function
    parameters. This conversion is trivial.

    Args:
        mcp_tools: List of mcp.types.Tool objects from session.list_tools()

    Returns:
        List of OpenAI tool dicts: [{"type": "function", "function": {...}}, ...]
    """
    openai_tools = []
    for tool in mcp_tools:
        schema = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
        # Ensure 'properties' key exists (OpenAI requires it)
        if "properties" not in schema:
            schema["properties"] = {}
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            }
        })
    return openai_tools


class ChatbotAgent(LifecycleNode):
    """Full agentic loop chatbot with MCP tool integration."""

    def __init__(self):
        super().__init__("agent_node")
        self.messages: list[dict] = []
        self.session_id: str = str(uuid.uuid4())[:8]
        self.model: str = DEFAULT_MODEL
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.openai_tools: list[dict] = []
        self._mcp_session: ClientSession | None = None
        self._mcp_task: asyncio.Task | None = None
        self._mcp_ready = asyncio.Event()
        self._processing = False  # Guard against concurrent agentic loops

    def on_configure(self, config: dict):
        """Initialize agent configuration."""
        self.model = config.get("model", DEFAULT_MODEL)
        if "system_prompt" in config:
            self.system_prompt = config["system_prompt"]
        # Initialize with system message
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logger.info(f"Agent configured: model={self.model}, session={self.session_id}")

    async def on_activate(self):
        """Register subscriptions and establish MCP session."""
        # Subscribe to user input
        @self.subscribe("/chat/input")
        async def on_user_input(msg: dict):
            await self._on_user_message(msg)

        # Start MCP session in background task (keeps context manager alive)
        self._mcp_task = asyncio.create_task(self._run_mcp_session())

        # Wait for MCP session to be ready
        try:
            await asyncio.wait_for(self._mcp_ready.wait(), timeout=15)
            logger.info(f"Agent activated. Tools: {[t['function']['name'] for t in self.openai_tools]}")
        except asyncio.TimeoutError:
            logger.warning("MCP session not ready within 15s — agent will work without tools")

    async def _run_mcp_session(self):
        """Maintain MCP client session for the lifetime of the node."""
        try:
            async with tagentacle_client_transport(self, server_node_id="mcp_server_node") as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._mcp_session = session

                    # Discover tools and convert to OpenAI schema
                    tools_result = await session.list_tools()
                    self.openai_tools = mcp_tools_to_openai_schema(tools_result.tools)
                    logger.info(f"MCP tools discovered: {[t.name for t in tools_result.tools]}")

                    # Signal ready
                    self._mcp_ready.set()

                    # Keep session alive until cancelled
                    try:
                        await asyncio.Future()  # Block forever
                    except asyncio.CancelledError:
                        pass
        except Exception as e:
            logger.error(f"MCP session error: {e}")
            self._mcp_ready.set()  # Unblock activation even on failure

    async def _on_user_message(self, msg: dict):
        """Handle incoming user message from /chat/input."""
        payload = msg.get("payload", {})
        user_text = payload.get("text", "").strip()
        session_id = payload.get("session_id")

        if not user_text:
            return

        if self._processing:
            logger.warning("Already processing a message — ignoring concurrent input")
            return

        # Update session_id if provided by frontend
        if session_id:
            self.session_id = session_id

        self._processing = True
        try:
            await self._agentic_loop(user_text)
        except Exception as e:
            logger.error(f"Agentic loop error: {e}", exc_info=True)
            # Send error message to frontend
            await self.publish("/chat/output", {
                "text": f"⚠️ Error: {e}",
                "session_id": self.session_id,
            })
        finally:
            self._processing = False

    async def _agentic_loop(self, user_text: str):
        """
        Core agentic loop:
        1. Append user message to context
        2. Call inference → get completion
        3. If tool_calls: execute tools → append results → re-infer (loop)
        4. If no tool_calls: publish final reply → done
        """
        # Step 1: Append user message
        self.messages.append({"role": "user", "content": user_text})
        await self._publish_memory()

        # Step 2-4: Inference loop
        for round_num in range(MAX_TOOL_ROUNDS):
            # Call Inference Node
            inference_payload = {
                "model": self.model,
                "messages": self.messages,
            }
            if self.openai_tools:
                inference_payload["tools"] = self.openai_tools

            logger.info(f"Calling inference (round {round_num + 1})...")
            result = await self.call_service("/inference/chat", inference_payload, timeout=120)

            if "error" in result:
                raise RuntimeError(f"Inference error: {result['error']}")

            # Extract assistant message
            choice = result["choices"][0]
            assistant_msg = choice["message"]

            # Append assistant message to context
            self.messages.append(assistant_msg)
            await self._publish_memory()

            # Check for tool calls
            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                # No tool calls — we have the final answer
                content = assistant_msg.get("content", "")
                logger.info(f"Final reply (round {round_num + 1}): {content[:80]}...")
                await self.publish("/chat/output", {
                    "text": content,
                    "session_id": self.session_id,
                })
                return

            # Execute tool calls
            logger.info(f"Tool calls in round {round_num + 1}: "
                        f"{[tc['function']['name'] for tc in tool_calls]}")

            for tool_call in tool_calls:
                tc_id = tool_call["id"]
                func_name = tool_call["function"]["name"]
                func_args_str = tool_call["function"].get("arguments", "{}")

                # Parse arguments
                try:
                    func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                except json.JSONDecodeError:
                    func_args = {}

                # Execute via MCP
                tool_result_text = await self._execute_tool(func_name, func_args)

                # Append tool result to context
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result_text,
                })

            await self._publish_memory()
            # Loop continues → re-infer with tool results

        # Safety: exceeded max rounds
        logger.warning(f"Exceeded max tool rounds ({MAX_TOOL_ROUNDS})")
        await self.publish("/chat/output", {
            "text": "⚠️ Exceeded maximum tool call rounds.",
            "session_id": self.session_id,
        })

    async def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool via the MCP session."""
        if not self._mcp_session:
            return f"Error: MCP session not available, cannot call tool '{name}'"

        try:
            logger.info(f"Executing tool: {name}({arguments})")
            result = await self._mcp_session.call_tool(name, arguments=arguments)

            # Extract text from result content
            texts = []
            for content_block in result.content:
                if hasattr(content_block, "text"):
                    texts.append(content_block.text)
                else:
                    texts.append(str(content_block))

            tool_output = "\n".join(texts)
            logger.info(f"Tool result: {tool_output[:100]}...")
            return tool_output

        except Exception as e:
            logger.error(f"Tool execution error ({name}): {e}")
            return f"Error executing tool '{name}': {e}"

    async def _publish_memory(self):
        """Publish current conversation state to /memory/latest."""
        await self.publish("/memory/latest", {
            "session_id": self.session_id,
            "messages": self.messages,
        })

    async def on_shutdown(self):
        """Clean up MCP session."""
        if self._mcp_task and not self._mcp_task.done():
            self._mcp_task.cancel()
            try:
                await self._mcp_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent shut down.")


async def main():
    agent = ChatbotAgent()
    await agent.connect()

    # Start spin BEFORE lifecycle transitions so that
    # on_activate()'s MCP session can exchange messages via the bus.
    spin_task = asyncio.create_task(agent.spin())

    await agent.configure()
    await agent.activate()

    # spin() is already running — just await it
    await spin_task


if __name__ == "__main__":
    asyncio.run(main())
