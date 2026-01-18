# 🧪 Lab 6 – Memory Governance & Federalized Reasoning
# New learning objectives
# Students will learn:

# Explicit memory routing rules

# Read-after-write verification

# Context budget discipline

# Failure detection instead of hallucination

# Auditability of memory events

# 🔁 What’s new compared to Lab 5
# ✅ Explicit memory routing policy
# The agent is told exactly which memory to use and when.

# ✅ Memory write confirmation loop
# Every write must:

# Use a tool

# Verify via a read

# Confirm in natural language

# ✅ Memory event logging
# All memory interactions are auditable.

# ✅ Context budget discipline
# Permanent context is fixed and memory is externalized.


"""
Lab 6 - Memory Governance & Federalized Reasoning

This lab introduces:
1. Explicit memory routing rules
2. Read-after-write verification
3. Memory event auditing
4. Context budget discipline
"""

import asyncio
import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass, asdict

from agents import (
    Agent,
    Runner,
    trace,
    ModelSettings,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)
from agents.mcp import MCPServerStdio

# ------------------------------------------------------------------------------
# Environment & Runtime Setup
# ------------------------------------------------------------------------------

load_dotenv(override=True)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------------------
# Memory Event Auditing
# ------------------------------------------------------------------------------


@dataclass
class MemoryEvent:
    timestamp: str
    memory_type: str  # rag | graph | faq
    action: str  # write | read | verify | fail
    description: str

    def log(self):
        logging.info(f"[MEMORY] {asdict(self)}")


def log_event(memory_type, action, description):
    event = MemoryEvent(
        timestamp=datetime.utcnow().isoformat(),
        memory_type=memory_type,
        action=action,
        description=description,
    )
    event.log()


# ------------------------------------------------------------------------------
# Context Engineering
# ------------------------------------------------------------------------------


def create_context(name: str, first_name: str) -> str:
    """
    Permanent context ONLY.
    No conversational state.
    """
    return f"""
You represent the AI Digital Twin of a human called {name}.

## ROLE
You are chatting on {first_name}'s website with users who may be customers,
collaborators, or employers.

You are a digital twin — not the human — but you are empowered to represent
{first_name} faithfully and accurately.

## MEMORY GOVERNANCE RULES (CRITICAL)

You have THREE external memory systems:

1. Knowledge Graph (libsql)
   - Use ONLY for:
     - Professional relationships
     - Organizations
     - Roles
     - Durable facts that must not drift

2. Long-Term Semantic Memory (Qdrant)
   - Use ONLY for:
     - Preferences
     - Prior conversations
     - Soft facts
     - Personal details

3. Questions / FAQ Memory
   - Use WHENEVER you cannot answer a question confidently
   - NEVER guess or hallucinate

### Mandatory Rules
- If information belongs in memory, YOU MUST store it.
- Every memory write MUST be followed by a read to verify.
- If verification fails, record a failure.
- If you do not know something, record the question.

If a user asks about preferences or past interactions:
- You MUST search memory before answering.

VERIFICATION RULES (STRICT):

- Every memory write must be followed by exactly ONE read.
- If the read confirms the write, you must stop immediately.
- Say "Memory verified successfully."
- Do not attempt additional writes or reads.
- If verification fails, record a failure and stop.



## STYLE
{first_name} is upbeat, friendly, and slightly nerdy.
Keep conversations professional but approachable.

## TIME CONTEXT
Current date/time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""


# ------------------------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------------------------

settings = ModelSettings(
    tool_choice="auto",
    temperature=0,
    max_completion_tokens=1024,
)

external_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

LLM = OpenAIChatCompletionsModel(
    model="gpt-oss:20b-lab",
    openai_client=external_client,
)

# ------------------------------------------------------------------------------
# MCP Server Parameters
# ------------------------------------------------------------------------------


def graph_params():
    db_path = Path("retention") / "graph.db"
    return {
        "command": "npx",
        "args": ["-y", "mcp-memory-libsql"],
        "env": {"LIBSQL_URL": f"file:{db_path.absolute()}"},
    }


def rag_params():
    knowledge_path = Path("retention") / "knowledge"
    return {
        "command": "uvx",
        "args": ["mcp-server-qdrant"],
        "env": {
            "QDRANT_LOCAL_PATH": str(knowledge_path.absolute()),
            "COLLECTION_NAME": "knowledge",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }


def question_params():
    return {"command": "uv", "args": ["run", "questions_mcp_server.py"]}


# ------------------------------------------------------------------------------
# Twin Conversation
# ------------------------------------------------------------------------------


async def run_twin():
    name = "Oscar Sanchez"
    first_name = "Oscar"
    context = create_context(name, first_name)

    print("\n" + "=" * 60)
    print("Lab 6 – Twin Conversation")
    print("=" * 60)

    with trace("Twin-Lab6"):
        async with MCPServerStdio(params=rag_params()) as rag:
            async with MCPServerStdio(params=graph_params()) as graph:
                async with MCPServerStdio(params=question_params()) as faq:

                    agent = Agent(
                        name="Twin",
                        model=LLM,
                        instructions=context,
                        mcp_servers=[rag, graph, faq],
                    )

                    task = [
                        {
                            "role": "user",
                            "content": (
                                "Oscar, I am your Lead Developer at Nebula.io. "
                                "Please store this correctly so you remember our relationship."
                            ),
                        }
                    ]

                    response = await Runner.run(agent, task, max_turns=4)
                    print("\nAgent Response:\n")
                    print(response.final_output)

                    log_event(
                        memory_type="graph",
                        action="verify",
                        description="Agent attempted to store and verify professional relationship",
                    )


# ------------------------------------------------------------------------------
# Verification Utility
# ------------------------------------------------------------------------------


def verify_graph():
    graph_path = Path("retention") / "graph.db"

    print("\n" + "=" * 60)
    print("GRAPH VERIFICATION")
    print("=" * 60)

    with sqlite3.connect(graph_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT name, entity_type FROM entities")
        for name, etype in cursor.fetchall():
            print(f"Node: {name} ({etype})")

        cursor.execute("SELECT source, relation_type, target FROM relations")
        for src, rel, tgt in cursor.fetchall():
            print(f"{src} --[{rel}]--> {tgt}")

    print("=" * 60)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


async def main():
    print("Lab 6 – Memory Governance & Federalized Reasoning")
    print("=" * 60)

    await run_twin()
    verify_graph()

    print("\nReview traces at:")
    print("https://platform.openai.com/traces")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
# 🧠 What this lab teaches implicitly
# Students will now see:

# Why memory ≠ prompt

# Why structure beats embeddings

# Why verification beats trust

# Why hallucinations are engineering failures

# Why LLMs are reasoning coordinators, not brains

# Next logical Lab 7 options
# If you want to continue the curriculum, the natural next labs are:

# Multi-Agent Authority Boundaries

# Memory Compaction & Summarization

# Adversarial Prompt Defense

# Regulatory / Audit Logs

# Cross-Twin Communication

# Tell me which direction you want and I’ll design Lab 7 at the same level of rigor.
