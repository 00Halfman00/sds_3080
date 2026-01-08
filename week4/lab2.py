from csv import excel_tab
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

external_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
LLM = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=external_client)

# An MCP Server is defined by Parameters


async def main():
    try:
        ##################    An MCP Server is defined by Parameters   ################################
        # --- MCP fetch parameters ---
        fetch_params = {"command": "uvx", "args": ["mcp-server-fetch"]}

        ###################   Print file tools    ####################################################
        sandbox_path = os.path.abspath(os.path.join(os.getcwd(), "sandbox"))
        # --- MCP filesystem parameters ---
        # --- MCP playwright parameters ---
        playwright_params = {
            "command": "npx",  # Added .cmd for Windows
            "args": [
                "-y",
                "@playwright/mcp@latest",
            ],  # Added -y to skip prompts
        }

        # --- MCP filesystem parameters ---
        files_params = {
            "command": "npx",  # Added .cmd for Windows
            "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_path],
        }

        # 1. Be extremely literal in instructions
        # Be very specific about tool names
        # Use the correct Microsoft tool names
        # Be extremely strict to prevent hallucinations
        # Force a clean output format
        instructions = (
            f"You are a file-writing bot. Use 'browser_navigate' to open a site, "
            "DO NOT explain your reasoning. DO NOT output internal thoughts. DO NOT use other tools."
            f"then use 'browser_snapshot' to read it. "
            f"Extract the text of the main headline, write a short summary, "
            f"and save it to {sandbox_path} directory using 'write_file'.\n"
            "Your final response to the user must be: 'Task Complete: [Headline Name] saved to news.md'. Do not provide any other text."
        )

        input_task = "Navigate to https://www.bbc.com/news, look for the largest text heading, summarize the article, and write to file."

        with trace("News"):
            async with MCPServerStdio(params=playwright_params) as browser:
                async with MCPServerStdio(params=files_params) as filesystem:
                    agent = Agent(
                        name="News",
                        instructions=instructions,
                        model=LLM,
                        mcp_servers=[browser, filesystem],
                    )
                    result = await Runner.run(agent, input_task, max_turns=30)
                    print(result.final_output)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # 4. Entry point to run the async loop
    asyncio.run(main())
