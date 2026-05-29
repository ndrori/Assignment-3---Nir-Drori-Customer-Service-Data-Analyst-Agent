# Bitext Dataset MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server built with **FastMCP** that exposes the Bitext customer-support dataset as callable tools. Any MCP-compatible client — Claude Desktop, Cursor, or a custom Python client — can call these tools directly.

---

## Tools exposed

| Tool | Description |
|---|---|
| `search_intents` | Search intent and category names by keyword — always call this first |
| `count_rows` | Count rows for a specific intent |
| `get_examples` | Paginated utterance examples for an intent |
| `list_categories` | All unique category names |
| `list_intents` | All unique intent names |
| `category_distribution` | Intent breakdown inside a category |
| `get_examples_by_category` | Paginated examples for a whole category |
| `count_by_category` | Row count for a category |

---

## Installation

```bash
pip install fastmcp pandas datasets python-dotenv
```

Create a `.env` file in the same folder (only needed if you also run the main agent):

```
NEBIUS_API_KEY=your_key_here
```

Make sure `bitext_dataset.csv` is present, or let the server download it on first run.

---

## Starting the server

### stdio transport — for Claude Desktop / MCP clients

```bash
python mcp_server.py
```

This is the standard transport for local MCP integrations. The server communicates over stdin/stdout.

### SSE transport — for HTTP clients

```bash
python mcp_server.py --transport sse
# custom port:
python mcp_server.py --transport sse --port 9000
```

The server starts at `http://0.0.0.0:8000` (or your chosen port).

---

## Connecting a client

### Python client (SSE transport)

Start the server first:
```bash
python mcp_server.py --transport sse --port 8000
```

Then in a separate terminal or script:

```python
import asyncio
from fastmcp import Client

async def main():
    # Connect to the running SSE server
    async with Client("http://localhost:8000/sse") as client:

        # 1. List all available tools
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        # 2. Search for intents related to 'refund'
        result = await client.call_tool("search_intents", {"keyword": "refund"})
        print("Search result:", result)

        # 3. Count rows for a specific intent
        count = await client.call_tool("count_rows", {"intent": "track_refund"})
        print("Row count:", count)

        # 4. Fetch 3 example utterances
        examples = await client.call_tool(
            "get_examples",
            {"intent": "track_refund", "limit": 3, "offset": 0}
        )
        print("Examples:", examples)

asyncio.run(main())
```

### 
## Example tool calls

```python
# Find everything related to 'shipping'
await client.call_tool("search_intents", {"keyword": "shipping"})
# → {"matched_intents": ["track_order", ...], "matched_categories": ["SHIPPING"]}

# How many rows in the REFUND category?
await client.call_tool("count_by_category", {"category": "REFUND"})
# → 450

# What intents exist inside REFUND and how many rows each?
await client.call_tool("category_distribution", {"category": "REFUND"})
# → {"get_refund_status": 90, "track_refund": 90, ...}

# Get 5 example utterances from SHIPPING (page 2)
await client.call_tool("get_examples_by_category",
    {"category": "SHIPPING", "limit": 5, "offset": 5})
```

---

## File structure

```
mcp_server.py          ← the MCP server (this file)
main.py                ← the LangGraph agent (uses the same dataset)
bitext_dataset.csv     ← auto-downloaded on first run
profiles/              ← per-user profile JSON files
agent_sessions.db      ← LangGraph SQLite checkpoints
README_MCP.md          ← this file
```
