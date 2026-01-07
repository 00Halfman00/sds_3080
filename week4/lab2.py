from csv import excel_tab
from dotenv import load_dotenv
from agents import Agent, Runner, trace, ModelSettings, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.mcp import MCPServerStdio
import os
import sys
import asyncio
load_dotenv(override=True)

# Define your settings (this is for the context/hardware)
settings = ModelSettings(
    tool_choice="auto",
    temperature=0,
    max_completion_tokens=1024,  # Output tokens
)

external_client = AsyncOpenAI(base_url = "http://localhost11434/v1", api_key="ollama")
LLM = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=external_client)

# An MCP Server is defined by Parameters

async def main():
    try:
        fetch_params = {"command": "uvx", "args": ["mcp-server-fetch"]}

        async with MCPServerStdio(params=fetch_params, client_session_timeout_seconds=30) as fetch:
            tools = await fetch.session.list_tools()
            print("tools: ", tools.tools)
    except Exception as e:
        print(f"An error occurred: {e}")



        
if __name__ == "__main__":
    # 4. Entry point to run the async loop
    asyncio.run(main())