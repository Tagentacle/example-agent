"""
Example Agent Client: Calls MCP tools through the Tagentacle bus.

This demonstrates how an Agent node uses tagentacle_client_transport to
call MCP tools on a remote server node via the bus, using the standard
MCP ClientSession API without modification.
"""

import asyncio
import logging

from tagentacle_py_core import Node
from mcp import ClientSession
from tagentacle_py_mcp.transport import tagentacle_client_transport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # 1. Initialize Tagentacle Node
    node = Node("weather_agent_node")
    await node.connect()
    spin_task = asyncio.create_task(node.spin())

    # 2. Open MCP session over Tagentacle bus transport
    try:
        async with tagentacle_client_transport(node, server_node_id="mcp_server_node") as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                logger.info("Connecting to Weather Server over bus...")
                await session.initialize()

                # List tools
                tools = await session.list_tools()
                logger.info(f"Agent sees tools: {[t.name for t in tools.tools]}")

                # Call get_weather
                logger.info("Calling get_weather for Shenzhen...")
                result = await session.call_tool("get_weather", arguments={"city": "Shenzhen"})
                logger.info(f"Agent received result: {result.content[0].text}")

                # Call for another city
                logger.info("Calling get_weather for Tokyo...")
                result = await session.call_tool("get_weather", arguments={"city": "Tokyo"})
                logger.info(f"Agent received result: {result.content[0].text}")
    except Exception as e:
        logger.error(f"Agent session error: {e}")
    finally:
        spin_task.cancel()
        try:
            await spin_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
