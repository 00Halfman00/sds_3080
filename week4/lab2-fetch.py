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
# P-cores are usually lower-numbered CPUs
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


async def main():
    try:
        ##################    An MCP Server is defined by Parameters   ################################
        fetch_params = {"command": "uvx", "args": ["mcp-server-fetch"]}

        sandbox_path = os.path.abspath(os.path.join(os.getcwd(), "sandbox"))

        files_params = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_path],
        }

        # 1. Be extremely literal in instructions
        instructions = (
            f"You are a file-writing assistant. Your only goal is to:\n"
            f"1. Use a fetch tool to read the text from a URL.\n"
            f"2. Summarize it in a few sentences using markdown.\n"
            f"3. Use a filesystem tool to save that summary to {sandbox_path}/summary.md.\n"
            f"DO NOT write code or HTML. DO NOT use a tool that is not provided."
        )

        # 2. Give it one specific target
        input_task = "Fetch any 2 different news sites, summarize the top story of each, write to summary.md"

        with trace("News"):
            async with MCPServerStdio(
                params=fetch_params, client_session_timeout_seconds=30
            ) as fetch:
                async with MCPServerStdio(
                    params=files_params, client_session_timeout_seconds=30
                ) as filesystem:
                    agent = Agent(
                        name="News",
                        instructions=instructions,
                        model=LLM,
                        mcp_servers=[fetch, filesystem],
                    )
                    async with SEM:
                        result = await Runner.run(agent, input_task, max_turns=30)
                        print(result.final_output)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # 4. Entry point to run the async loop
    asyncio.run(main())
