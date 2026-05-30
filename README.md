# Bitext Dataset Agent — `agentUserProfile.py`

This is a conversational AI agent built with **LangGraph** and **LangChain** that explores the [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) The agent presented in the code is answer to assignement 1, 2 and 2b. The agent remembers the conversation across restarts and builds a persistent profile of the user over time.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the agent](#running-the-agent)
7. [CLI reference](#cli-reference)
8. [In-conversation commands](#in-conversation-commands)
9. [Example session](#example-session)
10. [User profiles explained](#user-profiles-explained)
11. [Dataset structure](#dataset-structure)
12. [Available tools](#available-tools)
13. [File structure](#file-structure)
14. [Troubleshooting](#troubleshooting)

---

## What it does

- Answers natural-language questions about the Bitext customer-support dataset (counts, examples, distributions, summaries, response-style analysis)
- Routes queries intelligently — structured questions go to data tools; qualitative questions go to the LLM; out-of-scope questions are politely refused
- **Persists conversation history** across restarts using a SQLite checkpoint database
- **Builds a user profile** automatically as you chat — name, frequent topics, preferences, and other facts — and injects it into every session so the agent remembers you

---

## Architecture overview

```
User input
    │
    ▼
┌─────────┐     out_of_scope    ┌──────────┐
│ Router  │ ─────────────────▶  │  Reject  │
└─────────┘                     └──────────┘
    │ structured / unstructured       │
    ▼                                 │
┌─────────┐   tool calls    ┌──────────────┐
│  Agent  │ ◀────────────▶  │  Tool Node   │
└─────────┘                 └──────────────┘
    │ no tool calls
    ▼
┌─────────┐
│  Final  │
└─────────┘
    │
    ▼
┌─────────┐   ← always runs last, updates user profile
│ Profile │
└─────────┘
    │
   END
```

**Key components:**

| Component | Role |
|---|---|
| **Router node** | Classifies each query as `structured`, `unstructured`, or `out_of_scope` |
| **Agent node** | ReAct loop — reasons about the query and decides which tools to call |
| **Tool node** | Executes the chosen tools against the dataset |
| **Final node** | Extracts the agent's last message as the answer |
| **Profile node** | LLM-powered — distils new facts from the conversation and updates the user profile |
| **SqliteSaver** | LangGraph checkpointer that persists the full graph state to `agent_sessions.db` |

---

## Prerequisites

- Python **3.10 or later**
- A **Nebius API key** (the agent uses Nebius-hosted Llama / Nemotron models)
- Internet access on first run (to download the Bitext dataset from Hugging Face)

---

## Installation

### 1 — Clone or copy the file

Place `agentUserProfile.py` in a folder of your choice, e.g. `C:\personal\`.

### 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

Or with conda (as used in development):

```bash
conda create -n bitex_agent python=3.11
conda activate bitex_agent
```

### 3 — Install dependencies

```bash
pip install langchain langchain-openai langchain-core
pip install langgraph
pip install datasets pandas
pip install python-dotenv
pip install pydantic
```

Or install everything in one command:

```bash
pip install langchain langchain-openai langchain-core langgraph datasets pandas python-dotenv pydantic
```

---

## Configuration

Create a file called **`.env`** in the same folder as `agentUserProfile.py`:

```
NEBIUS_API_KEY=your_nebius_api_key_here
```

To get a Nebius API key:
1. Sign up at [studio.nebius.com](https://studio.nebius.com)
2. Go to **API Keys** and create a new key
3. Paste it into `.env` as shown above

The agent will raise `ValueError: NEBIUS_API_KEY not found` if the key is missing.

### Switching models

The model is set near the top of the file:

```python
client = _BaseChatOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",   # ← change this
    temperature=0,
    disabled_params={"parallel_tool_calls": None},    # required for Llama/Nemotron
)
```

Other available Nebius models (comment/uncomment as needed):

```python
model="meta-llama/Llama-3.3-70B-Instruct"
model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
```

> **Note:** `disabled_params={"parallel_tool_calls": None}` is required for all Llama and Nemotron models on Nebius. Do not remove it.

---

## Running the agent

### Start a new session (auto-generated ID)

```bash
python agentUserProfile.py
```

The agent generates a short random session ID (e.g. `a3f92b1c`) and starts the chat loop.

### Start or resume a named session

```bash
python agentUserProfile.py --session nir
```

If a session named `nir` already exists in the database, its full conversation history and profile are restored. If not, a new session with that name is created.

### List all saved sessions

```bash
python agentUserProfile.py --list-sessions
```

Output example:
```
Saved sessions:
  • nir
  • alice
  • a3f92b1c
```

### View a saved user profile without starting a chat

```bash
python agentUserProfile.py --session nir --show-profile
```

Output example:
```json
{
  "name": "Nir",
  "topics": ["refund", "cancellation", "shipping"],
  "preferences": ["detailed examples"],
  "facts": ["works in QA"]
}
```

> **First run only:** the Bitext dataset (~26 MB) is downloaded from Hugging Face and saved as `bitext_dataset.csv`. This takes 10–30 seconds depending on your connection. Subsequent runs load from the CSV instantly.

---

## CLI reference

| Flag | Short | Description |
|---|---|---|
| `--session SESSION_ID` | `-s SESSION_ID` | Named session to restore or create |
| `--list-sessions` | — | Print all saved session IDs and exit |
| `--show-profile` | — | Print the profile for `--session` and exit |

---

## In-conversation commands

While the agent is running, type any of these special commands instead of a question:

| Command | What it does |
|---|---|
| `profile` | Print your current profile as JSON |
| `history` | Print a numbered list of all messages in this session |
| `exit` or `quit` | Save and exit (session is preserved for next time) |

---

## Example session

```
python agentUserProfile.py --session nir

Bitext LangGraph Agent  |  session: nir
Type 'exit' to quit, 'history' to print conversation, 'profile' to show your profile.

User> My name is Nir and I work in QA. I mostly ask about refund and cancellation intents.

FINAL ANSWER:
Got it, Nir! I'll keep that in mind. Feel free to ask anything about the Bitext dataset.

============================================================

User> How many refund-related rows are there?

[ROUTER] -> structured
[TOOL CALL] search_intents  ARGS: {'keyword': 'refund'}
[TOOL CALL] count_rows      ARGS: {'intent': 'track_refund'}
[TOOL CALL] count_rows      ARGS: {'intent': 'get_refund_status'}

FINAL ANSWER:
There are 180 refund-related rows in total:
- track_refund: 90 rows
- get_refund_status: 90 rows

============================================================

User> Show me 3 examples of cancellation requests

[ROUTER] -> structured
[TOOL CALL] search_intents        ARGS: {'keyword': 'cancellation'}
[TOOL CALL] get_examples_by_category  ARGS: {'category': 'CANCELLATION', 'limit': 3, 'offset': 0}

FINAL ANSWER:
Here are 3 examples from the CANCELLATION category:
1. "I need to cancel my order, it hasn't shipped yet."
2. "Can I cancel the subscription I just signed up for?"
3. "Please cancel order #84729, I changed my mind."

============================================================

User> profile

--- Your profile ---
{
  "name": "Nir",
  "topics": ["refund", "cancellation"],
  "preferences": [],
  "facts": ["works in QA"]
}
---

User> exit

Session 'nir' saved. Resume with:
  python agentUserProfile.py --session nir
```

---

## User profiles explained

The profile is built and updated automatically after every message — you never need to manage it manually.

### What gets captured

```json
{
  "name":        "Nir",
  "topics":      ["refund", "cancellation", "shipping"],
  "preferences": ["prefers detailed examples", "wants concise answers"],
  "facts":       ["works in QA", "team lead"]
}
```

### Where profiles are stored

Two layers of persistence:

| Layer | Location | Purpose |
|---|---|---|
| LangGraph checkpoint | `agent_sessions.db` | Full graph state, auto-managed |
| Profile file | `profiles/<session_id>.json` | Human-readable, portable across sessions |

### Seeding your profile quickly

Just tell the agent about yourself in natural language at the start of a session:

```
User> My name is Nir, I work in QA and I mostly ask about refund and shipping intents.
User> I prefer to see examples rather than just counts.
```

After two or three messages, `profile` will show these facts captured.

### Profile across sessions

If you start a new session (`--session nir2`) and a profile file already exists for `nir`, you can copy it:

```bash
copy profiles\nir.json profiles\nir2.json    # Windows
cp profiles/nir.json profiles/nir2.json      # macOS / Linux
```

The new session will load it automatically.

---

## Dataset structure

The Bitext dataset has three main columns the agent works with:

| Column | Description | Example |
|---|---|---|
| `category` | Broad topic group | `REFUND`, `SHIPPING`, `ORDER`, `CANCELLATION` |
| `intent` | Specific action within a category | `track_refund`, `cancel_order` |
| `instruction` | Customer utterance (input) | `"Where is my refund?"` |
| `response` | Agent reply (output) | `"Your refund was processed on …"` |

**Rule of thumb:**
- Ask about a **category** → the agent uses `count_by_category` / `get_examples_by_category`
- Ask about a specific **intent** → the agent uses `count_rows` / `get_examples`

---

## Available tools

The agent selects from these tools automatically based on your question:

| Tool | When the agent uses it |
|---|---|
| `search_intents(keyword)` | **Always called first** — finds exact intent/category names matching a keyword |
| `count_rows(intent)` | "How many `<intent>` rows?" |
| `count_by_category(category)` | "How many rows in the `<CATEGORY>` category?" |
| `get_examples(intent, limit, offset)` | "Show me examples of `<intent>`" |
| `get_examples_by_category(category, limit, offset)` | "Show examples from the `<CATEGORY>` category" |
| `list_intents()` | "What intents exist?" |
| `list_categories()` | "What categories are there?" |
| `filter_by_intent(intent)` | "Give me all rows for `<intent>`" |
| `category_distribution(category)` | "Breakdown of intents in `<CATEGORY>`" |
| `summarize_category(category)` | "Summarize the `<CATEGORY>` category" |
| `analyze_responses(intent)` | "How do agents respond to `<intent>` requests?" |

---

## File structure

After the first run, your folder will look like this:

```
agentUserProfile.py       ← the agent script
.env                      ← your Nebius API key (create this)
bitext_dataset.csv        ← auto-downloaded on first run (~26 MB)
agent_sessions.db         ← SQLite database with all session checkpoints
profiles/
    nir.json              ← one JSON file per session ID
    alice.json
```

---

## Troubleshooting

### `NEBIUS_API_KEY not found`
Create a `.env` file in the same directory as the script with:
```
NEBIUS_API_KEY=your_key_here
```

### `No profile found for session 'X'`
This is normal — the profile is created after your **first message** in a session. Run the agent normally (`python agentUserProfile.py --session X`), chat a bit, then `--show-profile` will work.

### `Error code: 400 — This model only supports single tool-calls at once`
This means `disabled_params={"parallel_tool_calls": None}` was removed from the client setup. Restore it — it is required for all Llama and Nemotron models on Nebius.

### Dataset download fails
The script downloads from Hugging Face on first run. If it fails, download manually:
```python
from datasets import load_dataset
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
ds["train"].to_pandas().to_csv("bitext_dataset.csv", index=False)
```
Then re-run the agent — it will find the CSV and skip the download.

### Agent hits iteration limit
The agent allows up to 12 reasoning steps per turn (`MAX_ITERATIONS = 12`). If it hits this, rephrase your question more specifically, or break it into smaller sub-questions.

### Resuming a session shows no history
Make sure you pass the exact same session ID used originally:
```bash
python agentUserProfile.py --session nir   # ← same ID each time
```
Run `--list-sessions` to see all saved session IDs.
