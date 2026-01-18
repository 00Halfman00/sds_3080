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
import psutil

p = psutil.Process(os.getpid())
# P-cores are usually lower-numbered CPUs (those 8 cores are your high-performance P-Cores.)
p.cpu_affinity(list(range(0, 8)))
# Prevent CPU thread explosion (important on i9-12900H)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

load_dotenv(override=True)

# Define your settings (this is for the context/hardware)
settings = ModelSettings(
    tool_choice="auto",
    temperature=0,
    max_completion_tokens=512,  # Tools don’t need long outputs
)

# Define Model
external_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
LLM = OpenAIChatCompletionsModel(
    model="gpt-oss:20b-lab",
    openai_client=external_client,
)


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

SEM = asyncio.Semaphore(1)  # one agent run at a time

# 1. Define our targets in a LIST (the correct way to iterate!)
news_sites = ["https://www.reuters.com"]


async def main():
    try:
        ##################    An MCP Server is defined by Parameters   ################################

        sandbox_path = os.path.abspath(os.path.join(os.getcwd(), "sandbox"))

        playwright_params = {
            "command": "npx",
            "args": [
                "-y",
                "@playwright/mcp@latest",
            ],  # Added -y to skip prompts
        }

        files_params = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_path],
        }

        # 2. Give it one specific target
        input_task = "Fetch a news site, summarize the top story, write to news.md"

        # instructions = (
        #     "Use ONLY browser and filesystem tools. "
        #     "Browser tools available: browser_navigate, browser_evaluate, browser_snapshot. "
        #     "Navigate to the site using browser_navigate. "
        #     "Extract article text FIRST using browser_evaluate with:\n"
        #     "() => document.querySelector('article')?.innerText\n"
        #     "Only if this fails, use browser_snapshot. "
        #     "Extract only the main article text. "
        #     "Exclude menus, headers, footers, ads, and links. "
        #     "Summarize in markdown. "
        #     f"Save to {sandbox_path}/news.md using filesystem."
        # )

        # Updated Instructions: Minimal and Direct
        instructions = (
            "You are a test bot. Use ONLY these tools: browser_navigate, browser_evaluate, write_file. "
            "DO NOT summarize. DO NOT get body text. "
            "1. Navigate to the URL. "
            "2. Use browser_evaluate with exactly: () => { return document.title; } "
            f"3. Use write_file to write the title and the URL to {sandbox_path}/news.md."
        )

        with trace("News"):
            async with MCPServerStdio(
                params=playwright_params, client_session_timeout_seconds=60
            ) as browser:
                async with MCPServerStdio(
                    params=files_params, client_session_timeout_seconds=60
                ) as filesystem:
                    agent = Agent(
                        name="NewsBot",
                        instructions=instructions,
                        model=LLM,
                        mcp_servers=[browser, filesystem],
                    )

                    # 2. Iterate over the LIST of sites, not an integer
                    for site in news_sites:
                        print(f"--- Processing {site} ---")
                        # Task: Be extremely literal
                        task = (
                            f"1. Call browser_navigate('{site}')\n"
                            "2. IMMEDIATELY call browser_evaluate('() => document.title')\n"
                            "3. Take the result of step 2 and use write_file to save it."
                        )
                        async with SEM:
                            await Runner.run(agent, task)

    except Exception as e:
        print(f"An error occurred: {e}")

        # After the loop finishes
        import httpx

        print("Cleaning up Ollama VRAM...")
        unload_payload = {"model": "gpt-oss:20b-lab", "keep_alive": 0}

        # We use standard httpx to talk to Ollama's native API
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://localhost:11434/api/generate", json=unload_payload
            )

        print("VRAM cleared. Script finished.")


if __name__ == "__main__":
    # 4. Entry point to run the async loop
    asyncio.run(main())
