"""
Lab 5 - Context Engineering

This script runs outside of Jupyter to avoid the stderr fileno issue on Windows.

We want to develop several types of Memory:
1. Long Term Memory - graph: A knowledge graph as a persistent store of entities
2. Long Term Memory - knowledge: A RAG database of Q&A and any other useful information
3. Permanent context: Summary and linkedin profile included in everything
4. FAQ: A list of questions and answers
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
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


async def list_memory_graph_tools():
    """List tools from the Knowledge Graph MCP server (libsql)."""
    file_path = Path("memory") / Path("graph.db")
    url = f"file:{file_path.absolute()}"

    memory_graph_params = {
        "command": "npx",
        "args": ["-y", "mcp-memory-libsql"],
        "env": {"LIBSQL_URL": url},
    }

    async with MCPServerStdio(
        params=memory_graph_params, client_session_timeout_seconds=30
    ) as memory_graph:
        memory_graph_tools = await memory_graph.session.list_tools()
        return memory_graph_tools.tools


async def list_memory_rag_tools():
    """List tools from the Vector Store RAG memory MCP server (Qdrant)."""
    long_term_path = Path("memory") / Path("knowledge")

    memory_rag_params = {
        "command": "uvx",
        "args": ["mcp-server-qdrant"],
        "env": {
            "QDRANT_LOCAL_PATH": str(long_term_path.absolute()),
            "COLLECTION_NAME": "knowledge",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }

    async with MCPServerStdio(
        params=memory_rag_params, client_session_timeout_seconds=30
    ) as memory_rag:
        memory_rag_tools = await memory_rag.session.list_tools()
        return memory_rag_tools.tools


async def list_question_tools():
    """List tools from the questions MCP server."""
    question_params = {"command": "uv", "args": ["run", "questions_mcp_server.py"]}

    async with MCPServerStdio(
        params=question_params, client_session_timeout_seconds=30
    ) as question_server:
        question_tools = await question_server.session.list_tools()
        return question_tools.tools


def create_context(name: str) -> str:
    return f"""
You are an AI Digital researcher representing {name}. Your job is to fetch information that exist and record information if it doesn't exist.

## MEMORY GOVERNANCE (STRICT)
You have three memory systems. Follow this logic tree for EVERY turn:

1. KNOWLEDGE GRAPH (Professional Facts):
   - ACTION: Before any write, call `search_nodes`.
   - CONSTRAINT: If the search results show the entity or relationship already exists, you are EXPLICITLY FORBIDDEN from calling `create_entities` or `create_relations` for that data.
   - ATOMICITY: Professional roles require a connection. Ensure both the Person and Company nodes exist, then link them.

2. RAG MEMORY (Personal/Preferences):
   - Use `qdrant-find` to check context.
   - Use `qdrant-store` for soft facts (preferences, stories).

3. QUESTIONS SERVER (The Gap):
   - If searches fail, call `record_question_with_no_answer`.
   - Do not guess. Check `get_questions_with_answer` first.

## CRITICAL EXECUTION
- You may NOT claim to have recorded, logged, saved, or noted anything
  unless the corresponding tool has returned success in THIS TURN.

## ACKNOWLEDGEMENT GATE (MANDATORY)
- The phrases:
  "I have recorded",
  "I logged",
  "I saved",
  "I noted"
  are STRICTLY FORBIDDEN
  unless `record_question_with_no_answer` was successfully called.
- If the tool was NOT called or did NOT return success:
  - You MUST say:
    "I do not currently have this information."

## IDEMPOTENCY RULE (MANDATORY)
- If `search_nodes` returns an entity or relationship that satisfies the request:
  - You MUST NOT call any create tools.
  - You MUST immediately stop and respond with:
    "I checked my records and I already have that mapped!"


## EVIDENCE RULE (MANDATORY)
- You may NOT answer from memory without searching it first.
- When answering:
  - Knowledge Graph → "This comes from my knowledge graph."
  - RAG → "I found this in my long-term memory."

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""


settings = ModelSettings(
    tool_choice="auto",
    temperature=0,
    max_completion_tokens=1024,  # Output tokens
)

# Create LLM
external_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
LLM = OpenAIChatCompletionsModel(model="gpt-oss:20b-lab", openai_client=external_client)


async def run_twin_conversation():
    """Run a conversation with the Twin agent using all three MCP servers."""

    # Set up parameters for all three MCP servers
    file_path = Path("memory") / Path("graph.db")
    url = f"file:{file_path.absolute()}"

    memory_graph_params = {
        "command": "npx",
        "args": ["-y", "mcp-memory-libsql"],
        "env": {"LIBSQL_URL": url},
    }

    long_term_path = Path("memory") / Path("knowledge")
    memory_rag_params = {
        "command": "uvx",
        "args": ["mcp-server-qdrant"],
        "env": {
            "QDRANT_LOCAL_PATH": str(long_term_path.absolute()),
            "COLLECTION_NAME": "knowledge",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }

    question_params = {"command": "uv", "args": ["run", "questions_mcp_server.py"]}

    # Create context
    name = "Oscar Sanchez"
    context = create_context(name)

    print("=" * 60)
    print("Running Twin Conversation")
    print("=" * 60)
    # print(f"\nContext:\n{context}\n")
    print("=" * 60)

    # Temporary debug logging
    import logging

    logging.basicConfig(level=logging.INFO)

    with trace("Twin"):
        async with MCPServerStdio(
            params=memory_rag_params, client_session_timeout_seconds=30
        ) as long_term_memory:
            async with MCPServerStdio(
                params=memory_graph_params, client_session_timeout_seconds=30
            ) as medium_term_memory:
                async with MCPServerStdio(
                    params=question_params, client_session_timeout_seconds=30
                ) as question_server:
                    agent = Agent(
                        "Twin",
                        model=LLM,
                        instructions=context,
                        mcp_servers=[
                            long_term_memory,
                            medium_term_memory,
                            question_server,
                        ],
                    )
                    # task = [
                    #     {
                    #         "role": "user",
                    #         "content": "Hello, I'm a potential customer. Does Oscar have a space ship?",
                    #     }
                    # ]
                    # ---   store something to RAG
                    # task = [
                    #     {
                    #         "role": "user",
                    #         "content": "My favorite programming language is Rust.",
                    #     }
                    # ]
                    # task = [
                    #     {
                    #         "role": "user",
                    #         "content": "I have a dog that is a dalmation named Pongo.",
                    #     }
                    # ]
                    # ---  test RAG to see if it stored favorite programming language: rust
                    # task = [
                    #     {
                    #         "role": "user",
                    #         "content": "Hey Oscar, do you remember what my favorite programming language is?",
                    #     }
                    # ]
                    # task = [
                    #     {
                    #         "role": "user",
                    #         "content": "Hey Oscar, do you remember what my dog is named?",
                    #     }
                    # ]
                    #  ---- store something in graph
                    # task = [
                    #     {
                    #         "role": "user",
                    #         "content": "Oscar, I want you to know that I am your lead developer here at Nebula.io. Please record this relationship in your knowledge graph so you don't forget our professional connection.",
                    #     }
                    # ]
                    task = [
                        {
                            "role": "user",
                            "content": "Hello. My name is John Rambo. I'm the Lead Developer at Nebula.io.",
                        }
                    ]

                    print("task: ", task)

                    response = await Runner.run(agent, task)
                    print("\n" + "=" * 60)
                    print("Agent Response:")
                    print("=" * 60)
                    print(response.final_output)
                    print("=" * 60)


# test graph for successful entries
import sqlite3


def peek_at_graph():
    graph_path = Path("memory") / "graph.db"

    print("\n" + "=" * 60)
    print("DATABASE PEEK: Current Graph Connections")
    print("=" * 60)

    try:
        with sqlite3.connect(graph_path) as conn:
            cursor = conn.cursor()

            print("--- Entities (Nodes) ---")
            cursor.execute("SELECT name, entity_type FROM entities")
            entities = cursor.fetchall()
            if not entities:
                print("No entities found in database.")
            for name, e_type in entities:
                print(f"Node: {name} (Type: {e_type})")

            print("\n--- Relationships (Edges) ---")
            # Using 'source' and 'target' as identified by your PRAGMA check
            query = """
                SELECT r.source, r.relation_type, r.target 
                FROM relations r
            """
            cursor.execute(query)
            relations = cursor.fetchall()
            if not relations:
                print("No relationships found in database.")
            for src, rel, target in relations:
                print(f"{src} --[{rel}]--> {target}")

    except Exception as e:
        print(f"Error reading graph.db: {e}")
    print("=" * 60 + "\n")


async def main():
    """Main function to run the lab exercises."""
    print("Lab 5 - Context Engineering")
    print("\n" + "=" * 60)
    print("MCP Servers used:")
    print(
        "- Knowledge Graph (libsql): https://glama.ai/mcp/servers/@joleyline/mcp-memory-libsql"
    )
    print(
        "- Vector Store RAG (Qdrant): https://glama.ai/mcp/servers/@qdrant/mcp-server-qdrant"
    )
    print("- Questions Server: local questions_mcp_server.py")
    print("=" * 60)

    # # List available tools from each server
    print("\n1. Listing Memory Graph tools...")
    try:
        graph_tools = await list_memory_graph_tools()
        print(f"Found {len(graph_tools)} tools")
        for tool in graph_tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"Error listing graph tools: {e}")
        import traceback

        traceback.print_exc()

    print("\n2. Listing Memory RAG tools...")
    try:
        rag_tools = await list_memory_rag_tools()
        print(f"Found {len(rag_tools)} tools")
        for tool in rag_tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"Error listing RAG tools: {e}")
        import traceback

        traceback.print_exc()

    print("\n3. Listing Question tools...")
    try:
        question_tools = await list_question_tools()
        print(f"Found {len(question_tools)} tools")
        for tool in question_tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"Error listing question tools: {e}")
        import traceback

        traceback.print_exc()

    # Run the twin conversation
    print("\n4. Running Twin conversation...")
    try:
        await run_twin_conversation()
    except Exception as e:
        print(f"Error running conversation: {e}")
        import traceback

        traceback.print_exc()
    print("\n4. Taking a look at records in graph database...")
    peek_at_graph()

    print("\n" + "=" * 60)
    print("Check traces at: https://platform.openai.com/traces")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
