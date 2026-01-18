"""
Lab 4 - Using our new MCP server!

This script runs outside of Jupyter to avoid the stderr fileno issue on Windows.
"""

import asyncio
import sys
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    trace,
    ModelSettings,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)
from agents.mcp import MCPServerStdio

load_dotenv(override=True)

# Set Windows event loop policy for async subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Define your settings (this is for the context/hardware)
settings = ModelSettings(
    tool_choice="auto",
    temperature=0,
    max_completion_tokens=1024,  # Output tokens
)

# Create LLM
external_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
LLM = OpenAIChatCompletionsModel(model="gpt-oss:20b-lab", openai_client=external_client)


async def list_mcp_tools():
    """List the tools available from the MCP server."""
    params = {"command": "uv", "args": ["run", "questions_mcp_server.py"]}

    async with MCPServerStdio(
        params=params, client_session_timeout_seconds=30
    ) as server:
        mcp_tools = await server.session.list_tools()
        return mcp_tools


async def run_agent_task():
    """Run the agent with the MCP server to answer questions."""
    params = {"command": "uv", "args": ["run", "questions_mcp_server.py"]}

    with trace("First MCP server"):
        async with MCPServerStdio(
            params=params, client_session_timeout_seconds=30
        ) as server:
            agent = Agent("Twin", model=LLM, mcp_servers=[server])
            task = (
                "What are the questions for which you have an official recorded answer?"
            )
            response = await Runner.run(agent, task)
            print("\n" + "=" * 60)
            print("Agent Response:")
            print("=" * 60)
            print(response.final_output)
            print("=" * 60)


async def main():
    """Main function to run the lab exercises."""
    print("Lab 4 - Using our new MCP server!")
    print("\n" + "=" * 60)

    # First, list the available tools
    print("\n1. Listing MCP server tools...")
    try:
        mcp_tools = await list_mcp_tools()
        print(f"Available tools: {mcp_tools}")
    except Exception as e:
        print(f"Error listing tools: {e}")

    # Then run the agent task
    print("\n2. Running agent task...")
    try:
        await run_agent_task()
    except Exception as e:
        print(f"Error running agent task: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
