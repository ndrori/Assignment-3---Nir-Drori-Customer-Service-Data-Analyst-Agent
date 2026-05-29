# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:02:33 2026

@author: IMOE001
"""

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