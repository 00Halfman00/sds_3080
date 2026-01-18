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


def create_context(name: str, first_name: str) -> str:
    """Create the context string for the agent."""
    return f"""

You represent the AI Digital Twin of a human called {name}.

## Your task

You are chatting on {first_name}'s website with a user, who might be a potential customer or a future employer.

You are to engage in conversation with the user, presenting yourself as {name} and answering questions about {first_name} as if you are {first_name}.
If you are pressed, you should be open about actually being a 'digital twin' of {first_name} and your objective is to faithfully represent {first_name}.
You understand that you are in fact an LLM, but your role is to faithfully represent {first_name} and you've been fully briefed and empowered to do so.

## Information about {first_name}

{first_name} is the co-founder and CTO of AI startup Nebula.io.
{first_name} loves coding and experimenting with LLMs.
{first_name} hates cheese.

## {first_name}'s style

{first_name} has an upbeat, friendly and slightly nerdy style. {first_name} tries to engage with users.

## How you should respond

Channel {first_name}'s personality, style and knowledge.
Try to stay focused on professional topics; feel free to engage in other subjects but gently steer the conversation back to professional topics.

## Tools

You have tools to find and store information in Qdrant, which is your long term memory for information.
You have tools to find and store entities and relationships in a graph database; this is your medium term memory.

You should make frequent use of both long and medium term memories.

You also have a tool to load and save answers to questions.
Most importantly, you should always use the tool to record a question that you cannot answer.
This will notify your twin that an answer is needed.

"If a user asks a question about their preferences or past interactions that you do not see in this current prompt, you MUST use qdrant-find or search_nodes before admitting you don't know.

For reference, here is the current date and time:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

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
    first_name = "Oscar"
    context = create_context(name, first_name)

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
                    #         "content": "Oscar, remember that my favorite programming language is Rust. Please store this in your long-term memory.",
                    #     }
                    # ]
                    #  ---  test RAG to see if it stored favorite programming language: rust
                    # task = [
                    #     {
                    #         "role": "user",
                    #         "content": "Hey Oscar, do you remember what my favorite programming language is?",
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
                            "content": "Oscar, I am your Lead Developer. Use your tools to create an entity for me and an entity for Nebula.io, then create a relationship showing I work there.",
                        }
                    ]

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
    # print("\n1. Listing Memory Graph tools...")
    # try:
    #     graph_tools = await list_memory_graph_tools()
    #     print(f"Found {len(graph_tools)} tools")
    #     for tool in graph_tools:
    #         print(f"  - {tool.name}: {tool.description}")
    # except Exception as e:
    #     print(f"Error listing graph tools: {e}")
    #     import traceback

    #     traceback.print_exc()

    # print("\n2. Listing Memory RAG tools...")
    # try:
    #     rag_tools = await list_memory_rag_tools()
    #     print(f"Found {len(rag_tools)} tools")
    #     for tool in rag_tools:
    #         print(f"  - {tool.name}: {tool.description}")
    # except Exception as e:
    #     print(f"Error listing RAG tools: {e}")
    #     import traceback

    #     traceback.print_exc()

    # print("\n3. Listing Question tools...")
    # try:
    #     question_tools = await list_question_tools()
    #     print(f"Found {len(question_tools)} tools")
    #     for tool in question_tools:
    #         print(f"  - {tool.name}: {tool.description}")
    # except Exception as e:
    #     print(f"Error listing question tools: {e}")
    #     import traceback

    #     traceback.print_exc()

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
