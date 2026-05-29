# -*- coding: utf-8 -*-
"""
Bitext Dataset MCP Server
=========================
Exposes Bitext customer-support dataset tools over the Model Context Protocol
using FastMCP.  Supports both stdio (default) and SSE transports.

Usage:
    python mcp_server.py                  # stdio transport (for Claude Desktop / MCP clients)
    python mcp_server.py --transport sse  # SSE transport  (HTTP, port 8000)
    python mcp_server.py --transport sse --port 9000  # custom port
"""

import argparse
import json
import os
from typing import List

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from fastmcp import FastMCP

# ============================================================
# ENV & DATA
# ============================================================

load_dotenv()

# Load (or re-use cached CSV) --------------------------------
DATA_PATH = "bitext_dataset.csv"

if not os.path.exists(DATA_PATH):
    print("[MCP] Downloading Bitext dataset …")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    dataset["train"].to_pandas().to_csv(DATA_PATH, index=False)
    print("[MCP] Dataset saved to", DATA_PATH)

df = pd.read_csv(DATA_PATH)
print(f"[MCP] Dataset loaded — {len(df):,} rows, {df['intent'].nunique()} intents, "
      f"{df['category'].nunique()} categories.")

# ============================================================
# FastMCP SERVER
# ============================================================

mcp = FastMCP(
    name="bitext-dataset-server",
    instructions=(
        "Tools for exploring the Bitext customer-support dataset. "
        "Always call search_intents before using an intent name in other tools."
    ),
)


# ── Tool 1 ──────────────────────────────────────────────────
@mcp.tool()
def search_intents(keyword: str) -> dict:
    """
    Search for intent names and category names that contain a given keyword.

    Tries the full keyword, each individual word, and 5-char prefixes.
    Always call this first when a user mentions a topic — it returns the
    exact intent/category names to pass to other tools.

    Args:
        keyword: Topic to search for (e.g. 'refund', 'cancel order').

    Returns:
        Dict with keys 'matched_intents' and 'matched_categories' (sorted lists).
    """
    all_intents    = df["intent"].unique().tolist()
    all_categories = df["category"].unique().tolist()

    words = keyword.lower().split()
    terms = set(words + [keyword.lower()] + [w[:5] for w in words if len(w) >= 5])

    matched_intents = sorted(
        {i for i in all_intents for t in terms if t in i.lower()}
    )
    matched_categories = sorted(
        {c for c in all_categories for t in terms if t in c.lower()}
    )

    result: dict = {
        "matched_intents":    matched_intents,
        "matched_categories": matched_categories,
    }

    if not matched_intents and not matched_categories:
        result["suggestion"] = (
            f"No matches found for '{keyword}'. "
            "Call list_intents or list_categories to see all available names."
        )

    return result


# ── Tool 2 ──────────────────────────────────────────────────
@mcp.tool()
def count_rows(intent: str) -> str:
    """
    Count how many dataset rows belong to a specific intent.

    Use an exact intent name (get one from search_intents first).

    Args:
        intent: Exact intent name, e.g. 'track_refund'.

    Returns:
        The row count as a string, or a hint if the intent is not found.
    """
    count = int(len(df[df["intent"].str.lower() == intent.lower()]))
    if count == 0:
        similar = [i for i in df["intent"].unique() if intent.lower()[:4] in i.lower()]
        hint = (
            f" Similar intents: {similar[:5]}" if similar
            else " Call search_intents to explore."
        )
        return f"0 rows found for intent='{intent}'.{hint}"
    return str(count)


# ── Tool 3 ──────────────────────────────────────────────────
@mcp.tool()
def get_examples(intent: str, limit: int = 5, offset: int = 0) -> List[dict]:
    """
    Return example utterances (instruction + response) for a given intent.

    Supports pagination via offset for 'show more' requests.

    Args:
        intent: Exact intent name (e.g. 'cancel_order').
        limit:  Number of examples to return (default 5).
        offset: Number of rows to skip for pagination (default 0).

    Returns:
        List of dicts with 'instruction' and 'response' keys.
    """
    filtered = df[df["intent"].str.lower() == intent.lower()]
    page = filtered[["instruction", "response"]].iloc[offset: offset + limit]
    return page.to_dict(orient="records")


# ── Tool 4 ──────────────────────────────────────────────────
@mcp.tool()
def list_categories() -> List[str]:
    """
    Return all unique category names in the dataset (sorted).

    Categories are broad groups like REFUND, SHIPPING, ORDER, CANCELLATION.
    Use a category name with get_examples_by_category or count_by_category.

    Returns:
        Sorted list of category name strings.
    """
    return sorted(df["category"].unique().tolist())


# ── Tool 5 ──────────────────────────────────────────────────
@mcp.tool()
def list_intents() -> List[str]:
    """
    Return all unique intent names in the dataset (sorted).

    Returns:
        Sorted list of intent name strings.
    """
    return sorted(df["intent"].unique().tolist())


# ── Tool 6 ──────────────────────────────────────────────────
@mcp.tool()
def category_distribution(category: str) -> dict:
    """
    Return how many rows each intent has inside a given category.

    Useful for understanding which intents dominate a category.

    Args:
        category: Exact category name (e.g. 'REFUND').

    Returns:
        Dict mapping intent name → row count, sorted by count descending.
    """
    filtered = df[df["category"].str.lower() == category.lower()]
    distribution = filtered["intent"].value_counts().to_dict()
    return distribution


# ── Tool 7 ──────────────────────────────────────────────────
@mcp.tool()
def get_examples_by_category(
    category: str, limit: int = 5, offset: int = 0
) -> List[dict]:
    """
    Return example utterances for a given CATEGORY (not intent), with pagination.

    Use this instead of get_examples when the user refers to a category like
    REFUND, SHIPPING, ORDER, or CANCELLATION.

    Args:
        category: Exact category name (e.g. 'SHIPPING').
        limit:    Number of examples to return (default 5).
        offset:   Number of rows to skip for pagination (default 0).

    Returns:
        List of dicts with 'intent', 'instruction', and 'response' keys.
    """
    filtered = df[df["category"].str.upper() == category.upper()]
    page = filtered[["intent", "instruction", "response"]].iloc[offset: offset + limit]
    return page.to_dict(orient="records")


# ── Tool 8 ──────────────────────────────────────────────────
@mcp.tool()
def count_by_category(category: str) -> int:
    """
    Count how many rows belong to a given CATEGORY.

    Args:
        category: Exact category name (e.g. 'CANCELLATION').

    Returns:
        Integer row count.
    """
    return int(len(df[df["category"].str.upper() == category.upper()]))


# ============================================================
# ENTRY POINT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Bitext MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport to use: 'stdio' (default) or 'sse' (HTTP)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for SSE transport (default: 8000)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.transport == "sse":
        print(f"[MCP] Starting SSE server on http://{args.host}:{args.port}")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        print("[MCP] Starting stdio server …")
        mcp.run(transport="stdio")
