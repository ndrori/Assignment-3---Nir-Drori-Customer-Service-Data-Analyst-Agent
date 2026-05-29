# -*- coding: utf-8 -*-
"""
LangGraph-Based ReAct Agent for the Bitext Dataset
With SQLite-backed conversation persistence and session ID support.

Usage:
    python main.py                          # generates a new session ID
    python main.py --session my_session     # restores or starts a named session
    python main.py --list-sessions          # list all saved sessions
"""

import os
import json
import argparse
import uuid
from pathlib import Path
from typing import Annotated, TypedDict, List

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver   # ← persistent checkpointer


# ============================================================
# ENV
# ============================================================

load_dotenv()


# ============================================================
# LOAD DATA
# ============================================================

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)
df = dataset["train"].to_pandas()
df.to_csv("bitext_dataset.csv", index=False)

DATA_PATH = "bitext_dataset.csv"
df = pd.read_csv(DATA_PATH)


# ============================================================
# LLM
# ============================================================


# ============================================================
# NEBIUS-COMPATIBLE CLIENT
#
# Llama 3.x on Nebius rejects any request body containing
# `parallel_tool_calls`. LangChain's ChatOpenAI always injects
# this field when bind_tools() is used.
#
# The fix is `disabled_params={"parallel_tool_calls": None}`,
# a built-in ChatOpenAI parameter that tells LangChain to
# unconditionally exclude that key from every request payload.
# Confirmed fix: github.com/langchain-ai/langchain/issues/25357
# ============================================================

from langchain_openai import ChatOpenAI as _BaseChatOpenAI

client = _BaseChatOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    #model="meta-llama/Llama-3.3-70B-Instruct",
    model ="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    temperature=0,
    disabled_params={"parallel_tool_calls": None},
)

key = os.environ.get("NEBIUS_API_KEY")
if not key:
    raise ValueError("NEBIUS_API_KEY not found")

print("Key loaded successfully")


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    route: str
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    final_answer: str
    # Tracks values from previous turns so the agent can reference them
    # e.g. {"last_intent": "refund", "last_count": 42, "last_examples_offset": 3}
    context: dict
    # Persistent user profile: distilled facts about the user (not raw messages)
    # e.g. {"name": "Alice", "topics": ["refund", "shipping"], "preferences": [...]}
    user_profile: dict


# ============================================================
# ROUTER
# ============================================================

ROUTER_PROMPT = """
You are a query router.

Classify the user query into one of:

1. structured
   - factual
   - counting
   - filtering
   - listing
   - aggregation

2. unstructured
   - summarization
   - qualitative analysis
   - trends
   - style analysis

3. out_of_scope
   - unrelated to the Bitext dataset
   - world knowledge
   - creative writing

Return ONLY valid JSON:

{
  "route": "structured"
}
"""


def router_node(state: AgentState):
    question = state["question"]

    response = client.invoke([
        HumanMessage(content=ROUTER_PROMPT + f"\n\nQuestion: {question}")
    ])

    try:
        parsed = json.loads(response.content)
        route = parsed["route"]
    except (json.JSONDecodeError, KeyError):
        route = "structured"  # safe default

    print(f"\n[ROUTER] -> {route}\n")

    return {**state, "route": route}


# ============================================================
# TOOL SCHEMAS
# ============================================================

class IntentInput(BaseModel):
    intent: str = Field(description="Intent name to filter by")


class CategoryInput(BaseModel):
    category: str = Field(description="Dataset category name")


class IntentAndLimitInput(BaseModel):
    intent: str = Field(description="Intent name")
    limit: int = Field(description="Number of examples to return")


class IntentLimitOffsetInput(BaseModel):
    intent: str = Field(description="Intent name")
    limit: int = Field(description="Number of examples to return")
    offset: int = Field(default=0, description="Number of rows to skip (for pagination)")


class CategoryLimitOffsetInput(BaseModel):
    category: str = Field(description="Dataset category name (e.g. REFUND, SHIPPING, ORDER)")
    limit: int = Field(description="Number of examples to return")
    offset: int = Field(default=0, description="Number of rows to skip (for pagination)")


class SummarizeInput(BaseModel):
    category: str = Field(description="Category to summarize")


class ResponsePatternInput(BaseModel):
    intent: str = Field(description="Intent to analyze responses for")


# ============================================================
# TOOLS
# ============================================================

@tool(args_schema=IntentInput)
def filter_by_intent(intent: str) -> str:
    """
    Filter dataset rows by intent.

    Use this tool whenever the user asks about:
    - a specific intent
    - counting requests of an intent
    - examples for an intent
    - response patterns for an intent

    Returns matching rows serialized as JSON.
    """
    filtered = df[df["intent"].str.lower() == intent.lower()]
    return filtered.to_json(orient="records")


@tool
def list_categories() -> List[str]:
    """
    Return all unique dataset categories.

    Use this tool when the user asks:
    - what categories exist
    - list categories
    - available categories
    """
    return sorted(df["category"].unique().tolist())


@tool
def list_intents() -> List[str]:
    """
    Return all unique intents in the dataset.

    Use this tool when the user asks:
    - available intents
    - intent names
    - what intents exist
    """
    return sorted(df["intent"].unique().tolist())


class KeywordInput(BaseModel):
    keyword: str = Field(description="Keyword to search for in intent names")


@tool(args_schema=KeywordInput)
def search_intents(keyword: str) -> dict:
    """
    Search for intent names and category names that contain a given keyword.

    ALWAYS use this tool FIRST when the user mentions any topic before
    calling count_rows, get_examples, or filter_by_intent.

    Automatically tries:
    - the full keyword
    - each individual word in the keyword
    - the first 5 characters of the keyword

    Returns a dict with matching intents and matching categories.

    Examples:
    - search_intents('refund')         → intents: ['get_refund_status', 'track_refund']
    - search_intents('refund request') → same result (splits into 'refund', 'request')
    - search_intents('cancel')         → intents: ['cancel_order', ...], categories: ['CANCELLATION']
    """
    all_intents    = df["intent"].unique().tolist()
    all_categories = df["category"].unique().tolist()

    # Build a set of search terms: full phrase + each individual word + 5-char prefix
    words = keyword.lower().split()
    terms = set(words + [keyword.lower()] + [w[:5] for w in words if len(w) >= 5])

    matched_intents = set()
    for term in terms:
        matched_intents.update(i for i in all_intents if term in i.lower())

    matched_categories = set()
    for term in terms:
        matched_categories.update(c for c in all_categories if term in c.lower())

    result = {
        "matched_intents":    sorted(matched_intents),
        "matched_categories": sorted(matched_categories),
    }

    if not matched_intents and not matched_categories:
        result["suggestion"] = (
            f"No matches found for '{keyword}'. "
            f"Call list_intents() or list_categories() to see all available names."
        )

    return result


@tool(args_schema=IntentInput)
def count_rows(intent: str) -> int:
    """
    Count how many rows belong to a given intent.

    IMPORTANT: Only call this with an exact intent name from the dataset.
    If you are not sure of the exact intent name, call search_intents first.

    Use this tool for quantitative questions such as:
    - how many refund requests
    - count shipping queries
    - number of cancellation requests
    """
    count = int(len(df[df["intent"].str.lower() == intent.lower()]))
    if count == 0:
        available = [i for i in df["intent"].unique() if intent.lower()[:4] in i.lower()]
        hint = f" Hint — similar intents found: {available[:5]}" if available else \
               " Hint — no similar intents found; call search_intents to explore."
        return f"0 rows found for intent='{intent}'.{hint}"
    return count


@tool(args_schema=IntentLimitOffsetInput)
def get_examples(intent: str, limit: int, offset: int = 0) -> List[dict]:
    """
    Return example utterances for a given intent, with optional offset for pagination.

    Use this tool when the user asks:
    - show examples
    - give sample utterances
    - provide sample requests
    - show more (use offset = previous limit to get the next page)
    """
    filtered = df[df["intent"].str.lower() == intent.lower()]
    examples = filtered[["instruction", "response"]].iloc[offset: offset + limit]
    return examples.to_dict(orient="records")


@tool(args_schema=CategoryLimitOffsetInput)
def get_examples_by_category(category: str, limit: int, offset: int = 0) -> List[dict]:
    """
    Return example utterances for a given CATEGORY (not intent), with pagination.

    IMPORTANT: Use this tool — NOT get_examples — when the user refers to a
    category name such as REFUND, SHIPPING, ORDER, CANCELLATION, etc.
    Categories group multiple intents together.

    Use this tool when the user asks:
    - show examples from the REFUND category
    - give me samples from SHIPPING
    - show more (when the prior question was about a category)

    Parameters:
    - category: the category name (case-insensitive)
    - limit: how many examples to return
    - offset: how many to skip for pagination (default 0)
    """
    filtered = df[df["category"].str.upper() == category.upper()]
    examples = filtered[["intent", "instruction", "response"]].iloc[offset: offset + limit]
    return examples.to_dict(orient="records")


@tool(args_schema=CategoryInput)
def count_by_category(category: str) -> int:
    """
    Count how many rows belong to a given CATEGORY.

    Use this tool when the user asks:
    - how many rows in the REFUND category
    - count of SHIPPING entries
    - total complaints (map to the relevant category)
    """
    return int(len(df[df["category"].str.upper() == category.upper()]))


@tool(args_schema=CategoryInput)
def category_distribution(category: str) -> dict:
    """
    Return intent distribution inside a category.

    Use this tool for aggregation questions such as:
    - distribution of intents
    - breakdown by intent
    - intent statistics in a category
    """
    filtered = df[df["category"].str.lower() == category.lower()]
    distribution = filtered["intent"].value_counts().to_dict()
    return distribution


@tool(args_schema=SummarizeInput)
def summarize_category(category: str) -> str:
    """
    Summarize the themes and content of a dataset category.

    Use this tool for qualitative or summarization questions.
    """
    filtered = df[df["category"].str.lower() == category.lower()]
    texts = filtered["instruction"].head(50).tolist()

    prompt = f"""
    Summarize the following customer requests from the {category} category.

    Requests:
    {texts}
    """
    response = client.invoke(prompt)
    return response.content


@tool(args_schema=ResponsePatternInput)
def analyze_responses(intent: str) -> str:
    """
    Analyze how customer service representatives respond to a specific intent.

    Use this tool for questions asking:
    - how agents respond
    - response style
    - typical support responses
    """
    filtered = df[df["intent"].str.lower() == intent.lower()]
    responses = filtered["response"].head(50).tolist()

    prompt = f"""
    Analyze the following customer service responses.

    Describe:
    - common patterns
    - tone
    - style
    - recurring actions

    Responses:
    {responses}
    """
    result = client.invoke(prompt)
    return result.content


TOOLS = [
    filter_by_intent,
    list_categories,
    list_intents,
    search_intents,             # ← always use this before guessing an intent name
    count_rows,
    get_examples,
    get_examples_by_category,
    count_by_category,
    category_distribution,
    summarize_category,
    analyze_responses,
]

TOOL_MAP = {t.name: t for t in TOOLS}


# ============================================================
# REACT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a dataset QA agent working ONLY on the Bitext customer-support dataset.

DATASET STRUCTURE:
- `category`: broad group (e.g. REFUND, SHIPPING, ORDER, CANCELLATION)
- `intent`: specific action within a category (e.g. track_refund, cancel_order)

CRITICAL RULES — follow these strictly:

1. NEVER GUESS AN INTENT NAME.
   Before calling count_rows, get_examples, filter_by_intent, or analyze_responses
   you MUST first call search_intents(keyword) to find the exact intent name.
   Example: user asks "how many refund requests" →
     Step 1: search_intents('refund')  → finds ['get_refund_status', 'track_refund']
     Step 2: count_rows for each match or sum them up

2. CATEGORY vs INTENT:
   - User says "REFUND category", "examples from SHIPPING"
     → use get_examples_by_category or count_by_category
   - User says a specific intent name
     → use get_examples / count_rows (but still verify with search_intents first)

3. IF count_rows RETURNS 0:
   The intent name was wrong. Immediately call search_intents to find the correct name.
   Never report 0 as the final answer without first verifying the intent exists.

4. CONVERSATION MEMORY:
   - For "show me more" → reuse the same category/intent from prior turns,
     increment offset by the previous limit.
   - For "total of the last two counts" → read the numbers from prior messages
     and compute directly, no tool needed.

Always think step-by-step. Never invent data.
If the query is unrelated to the dataset, politely refuse.
"""


# ============================================================
# AGENT NODE
# ============================================================

MAX_ITERATIONS = 12


def agent_node(state: AgentState):
    iterations = state["iterations"] + 1

    if iterations > MAX_ITERATIONS:
        return {
            **state,
            "final_answer": (
                "I could not complete the reasoning process within "
                "the allowed iteration limit."
            )
        }

    messages = state["messages"]
    response = client.bind_tools(TOOLS).invoke(messages)

    print("\n[AGENT]")
    print(response.content)

    messages.append(response)

    return {**state, "messages": messages, "iterations": iterations}


# ============================================================
# TOOL NODE
# ============================================================

def tool_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # Schema lookup for type coercion (LLMs sometimes pass ints as strings)
    INT_FIELDS = {"limit", "offset"}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = dict(tool_call["args"])  # make a mutable copy

        # Coerce any int fields that arrived as strings
        for field in INT_FIELDS:
            if field in tool_args and isinstance(tool_args[field], str):
                try:
                    tool_args[field] = int(tool_args[field])
                except ValueError:
                    pass

        print(f"\n[TOOL CALL] {tool_name}")
        print(f"ARGS: {tool_args}")

        tool_result = TOOL_MAP[tool_name].invoke(tool_args)

        print(f"[OBSERVATION]\n{tool_result}\n")

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

    return {**state, "messages": messages}


# ============================================================
# ROUTING LOGIC
# ============================================================

def router_condition(state: AgentState):
    if state["route"] == "out_of_scope":
        return "reject"
    return "agent"


def tools_condition(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "final"


# ============================================================
# REJECTION & FINAL NODES
# ============================================================

def reject_node(state: AgentState):
    return {
        **state,
        "final_answer": "I can only answer questions related to the Bitext dataset."
    }


def final_node(state: AgentState):
    last_message = state["messages"][-1]
    return {**state, "final_answer": last_message.content}


# ============================================================
# USER PROFILE HELPERS
# ============================================================

PROFILE_UPDATE_PROMPT = """
You maintain a structured profile for a user based on their conversations.

CURRENT PROFILE (JSON):
{current_profile}

RECENT CONVERSATION (last few turns):
{recent_messages}

Your task: Update the profile by extracting any NEW facts from the conversation.
Only add information that was clearly stated — never invent or guess.

Profile fields to maintain:
- "name":        The user's name if mentioned (string or null)
- "topics":      List of dataset topics/intents/categories the user frequently asks about
- "preferences": Any stated preferences (e.g. "prefers examples", "wants concise answers")
- "facts":       Any other notable facts the user shared about themselves or their work

Rules:
- MERGE with the current profile; do not discard existing facts unless contradicted
- Keep "topics" as a deduplicated list of strings
- If nothing new was learned, return the current profile unchanged
- Return ONLY valid JSON, no explanation, no markdown fences

Example output:
{{"name": "Alice", "topics": ["refund", "shipping"], "preferences": ["concise answers"], "facts": ["works in QA"]}}
"""


def load_profile_from_file(session_id: str) -> dict:
    """Load a user profile from its per-session JSON file, if it exists."""
    path = PROFILES_DIR / f"{session_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_profile_to_file(session_id: str, profile: dict) -> None:
    """Persist the profile to a per-session JSON file."""
    path = PROFILES_DIR / f"{session_id}.json"
    path.write_text(json.dumps(profile, indent=2))


def profile_node(state: AgentState):
    """
    After each agent turn, distill new facts from the conversation into the
    persistent user_profile dict.  This node runs last in the graph so it
    always sees the completed exchange.
    """
    current_profile = state.get("user_profile") or {}

    # Summarise only the last 6 messages to keep the prompt short
    messages = state.get("messages", [])
    recent   = messages[-6:]
    recent_text = "\n".join(
        f"{type(m).__name__.replace('Message', '')}: {str(m.content)[:400]}"
        for m in recent
        if hasattr(m, "content") and m.content
    )

    prompt = PROFILE_UPDATE_PROMPT.format(
        current_profile=json.dumps(current_profile, indent=2),
        recent_messages=recent_text or "(no messages yet)"
    )

    try:
        response = client.invoke([HumanMessage(content=prompt)])
        raw      = response.content.strip()
        # Strip accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        updated_profile = json.loads(raw)
    except Exception as e:
        print(f"[PROFILE] Could not update profile: {e}")
        updated_profile = current_profile

    print(f"\n[PROFILE] {updated_profile}\n")
    return {**state, "user_profile": updated_profile}


def format_profile_for_prompt(profile: dict) -> str:
    """Render the user profile as a readable block for injection into the system prompt."""
    if not profile:
        return "(No profile yet — this is the first session or no facts known.)"
    lines = []
    if profile.get("name"):
        lines.append(f"- Name: {profile['name']}")
    if profile.get("topics"):
        lines.append(f"- Frequent topics: {', '.join(profile['topics'])}")
    if profile.get("preferences"):
        lines.append(f"- Preferences: {', '.join(profile['preferences'])}")
    if profile.get("facts"):
        lines.append(f"- Other facts: {', '.join(profile['facts'])}")
    return "\n".join(lines) if lines else "(profile exists but is empty)"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(checkpointer):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reject", reject_node)
    workflow.add_node("final", final_node)
    workflow.add_node("profile", profile_node)   # ← always runs last

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        router_condition,
        {"reject": "reject", "agent": "agent"}
    )

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "final": "final"}
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("reject", "profile")   # profile runs even on rejection
    workflow.add_edge("final", "profile")
    workflow.add_edge("profile", END)

    return workflow.compile(checkpointer=checkpointer)


# ============================================================
# SESSION HELPERS
# ============================================================

DB_PATH      = "agent_sessions.db"
PROFILES_DIR = Path("profiles")
PROFILES_DIR.mkdir(exist_ok=True)


def list_sessions(checkpointer):
    """Print all session IDs stored in the SQLite database."""
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        rows = cur.fetchall()
        conn.close()
        if rows:
            print("\nSaved sessions:")
            for (thread_id,) in rows:
                print(f"  • {thread_id}")
        else:
            print("\nNo saved sessions found.")
    except Exception as e:
        print(f"\nCould not read sessions: {e}")


def get_or_create_session_id(requested_id: str | None) -> str:
    """Return requested session ID or generate a new one."""
    if requested_id:
        return requested_id
    new_id = str(uuid.uuid4())[:8]
    print(f"\n[SESSION] No session ID provided. Starting new session: {new_id}")
    return new_id


# ============================================================
# AGENT RUNNER
# ============================================================

def run_agent(app, question: str, session_id: str, turn_number: int):
    """
    Invoke the agent for one user turn.

    LangGraph's checkpointer automatically:
    - saves state after each run
    - restores prior state (including full message history) on the next call
      with the same thread_id

    We only need to pass the NEW question; the graph reads prior messages
    from the checkpoint and appends to them via the `add_messages` reducer.

    The user_profile is loaded from the checkpoint (or from the profile file
    if this is the first turn of a restored session) and injected into the
    system prompt so the agent can answer "what do you remember about me?".
    After the run, the updated profile is mirrored to profiles/<session_id>.json.
    """
    config = {"configurable": {"thread_id": session_id}}

    # Retrieve existing state so we can append to its message list
    existing_state = app.get_state(config)

    if existing_state.values:
        # Session exists — append the new user question to the existing messages
        prior_messages  = existing_state.values.get("messages", [])
        prior_context   = existing_state.values.get("context", {})
        # Prefer in-checkpoint profile; fall back to file (handles cross-session case)
        prior_profile   = (
            existing_state.values.get("user_profile") or
            load_profile_from_file(session_id)
        )
        new_messages = prior_messages + [HumanMessage(content=question)]
        initial_state = {
            **existing_state.values,
            "question":     question,
            "route":        "",
            "messages":     new_messages,
            "iterations":   0,
            "final_answer": "",
            "context":      prior_context,
            "user_profile": prior_profile,
        }
    else:
        # Brand-new session — try to load a profile file (e.g. same user, new session)
        prior_profile = load_profile_from_file(session_id)

        # Build the system prompt with the profile pre-loaded
        profile_block = format_profile_for_prompt(prior_profile)
        system_prompt_with_profile = (
            SYSTEM_PROMPT +
            f"\n\nUSER PROFILE (what you know about this user so far):\n{profile_block}"
        )

        initial_state = {
            "question":     question,
            "route":        "",
            "messages":     [HumanMessage(content=system_prompt_with_profile + f"\n\nQuestion: {question}")],
            "iterations":   0,
            "final_answer": "",
            "context":      {},
            "user_profile": prior_profile,
        }

    result = app.invoke(initial_state, config=config)

    # Mirror the updated profile to disk for cross-session portability
    updated_profile = result.get("user_profile", {})
    if updated_profile:
        save_profile_to_file(session_id, updated_profile)

    return result["final_answer"]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bitext LangGraph Agent with persistent sessions"
    )
    parser.add_argument(
        "--session", "-s",
        type=str,
        default=None,
        help="Session ID to restore or create (default: auto-generated)"
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List all saved session IDs and exit"
    )
    parser.add_argument(
        "--show-profile",
        action="store_true",
        help="Print the saved profile for the given session and exit"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # SqliteSaver persists to disk — survives restarts
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:

        if args.list_sessions:
            list_sessions(checkpointer)

        elif args.show_profile:
            sid = get_or_create_session_id(args.session)
            profile = load_profile_from_file(sid)
            if profile:
                print(f"\nProfile for session '{sid}':")
                print(json.dumps(profile, indent=2))
            else:
                print(f"\nNo profile found for session '{sid}'.")

        else:
            session_id = get_or_create_session_id(args.session)
            app = build_graph(checkpointer)

            print(f"\nBitext LangGraph Agent  |  session: {session_id}")
            print("Type 'exit' to quit, 'history' to print conversation, 'profile' to show your profile.\n")

            # Show existing profile on session restore
            existing_profile = load_profile_from_file(session_id)
            if existing_profile:
                print("[PROFILE] Loaded existing profile:")
                print(format_profile_for_prompt(existing_profile))
                print()

            turn = 0
            while True:
                question = input("User> ").strip()

                if not question:
                    continue

                if question.lower() in ("exit", "quit"):
                    print(f"\nSession '{session_id}' saved. Resume with:\n"
                          f"  python main.py --session {session_id}\n")
                    break

                if question.lower() == "history":
                    config = {"configurable": {"thread_id": session_id}}
                    state  = app.get_state(config)
                    msgs   = state.values.get("messages", []) if state.values else []
                    print(f"\n--- Conversation history ({len(msgs)} messages) ---")
                    for i, m in enumerate(msgs):
                        role = type(m).__name__.replace("Message", "")
                        snippet = str(m.content)[:200].replace("\n", " ")
                        print(f"  [{i}] {role}: {snippet}")
                    print("---\n")
                    continue

                if question.lower() == "profile":
                    config  = {"configurable": {"thread_id": session_id}}
                    state   = app.get_state(config)
                    profile = (
                        (state.values.get("user_profile") if state.values else None)
                        or load_profile_from_file(session_id)
                        or {}
                    )
                    print("\n--- Your profile ---")
                    print(json.dumps(profile, indent=2) if profile else "(empty — chat a bit first!)")
                    print("---\n")
                    continue

                turn += 1
                answer = run_agent(app, question, session_id, turn)

                print("\nFINAL ANSWER:")
                print(answer)
                print("\n" + "=" * 60 + "\n")
