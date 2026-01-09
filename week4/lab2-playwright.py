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
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# 1. Define our targets in a LIST (the correct way to iterate!)
news_sites = ["https://www.bbc.com/news"]


async def main():
    try:
        ##################    An MCP Server is defined by Parameters   ################################

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
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_path],
        }

        # 2. Give it one specific target
        input_task = "Fetch a news site, summarize the top story, write to news.md"

        with trace("News"):
            async with MCPServerStdio(
                params=playwright_params, client_session_timeout_seconds=30
            ) as browser:
                async with MCPServerStdio(
                    params=files_params, client_session_timeout_seconds=30
                ) as filesystem:
                    agent = Agent(
                        name="NewsBot",
                        instructions=(
                            "If you need a tool, ONLY use browser and filesystem. DO NOT make up any tools."
                            "Use 'browser_navigate' tool from browser to open a site."
                            "Use browser_snapshot tool from browser to find the main article."
                            "If the snapshot is too complex, use browser_evaluate from browser with () => document.querySelector('article').innerText to get just the story."
                            "Summarize it in a few sentences in markdown. "
                            f"Use filesystem to save the summary to {sandbox_path}/news.md.\n"
                        ),
                        model=LLM,
                        mcp_servers=[browser, filesystem],
                    )

                    # 2. Iterate over the LIST of sites, not an integer
                    for site in news_sites:
                        print(f"--- Processing {site} ---")
                        task = (
                            f"Summarize the top story from {site} and write to news.md"
                        )
                        await Runner.run(agent, task)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # 4. Entry point to run the async loop
    asyncio.run(main())
