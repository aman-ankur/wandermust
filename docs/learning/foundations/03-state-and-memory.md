# State & Memory — How Agents Remember

A function call is stateless. You pass inputs, get outputs, everything is forgotten. An agent is different — it accumulates information across multiple steps. The flights it found in step 1 need to be available when it compares hotels in step 3. The user preference it learned ten minutes ago should influence the recommendation it makes now.

How does this work? It's less mysterious than you might think, but the design choices you make here have outsized impact on how well your agent performs.

---

## Three Types of Agent Memory

When we talk about "agent memory," we're actually talking about three distinct things that serve different purposes and have different lifetimes. Conflating them is a common source of confusion.

### Short-Term Memory: The Conversation History

This is the most fundamental form of memory. It's the `messages` array that you send to the LLM on every call. Every message the user sent, every response the assistant generated, every tool call and its result — it's all there, and the LLM sees all of it on every iteration.

```python
messages = [
    {"role": "system", "content": "You are a travel planning assistant."},
    {"role": "user", "content": "Find flights to Tokyo for next month"},
    {"role": "assistant", "content": "I'll search for flights to Tokyo..."},
    {"role": "assistant", "content": '{"tool_call": "search_flights", "args": {"destination": "Tokyo"}}'},
    {"role": "tool", "content": '{"flights": [{"airline": "JAL", "price": 500}, {"airline": "ANA", "price": 480}]}'},
    {"role": "assistant", "content": "I found two flights: JAL at $500 and ANA at $480. Would you like me to search for hotels too?"},
    {"role": "user", "content": "Yes, find hotels near Shibuya"},
]
# Every new LLM call includes ALL of these messages
# The LLM can "remember" the flights because they're right there in the context
```

This is how an LLM "remembers" things within a conversation. It's not actually remembering — it's re-reading the entire conversation history on every call. This has a critical implication: **short-term memory is bounded by the context window**. For a model with 128K tokens of context, you can fit a lot of conversation. For a model with 8K tokens, you'll run out fast.

When conversations get long, you have to decide what to drop. Common strategies:

```python
# Strategy 1: Sliding window — keep the last N messages
messages = messages[-20:]

# Strategy 2: Summarize old messages — compress history
if len(messages) > 50:
    summary = llm.invoke("Summarize this conversation so far: " + str(messages[:40]))
    messages = [{"role": "system", "content": f"Previous conversation summary: {summary}"}] + messages[40:]

# Strategy 3: Keep system prompt + first user message + recent messages
# Preserves the original task context even as the conversation grows
messages = [messages[0], messages[1]] + messages[-15:]
```

Each strategy has tradeoffs. Sliding window is simple but loses early context. Summarization preserves meaning but costs an extra LLM call and can lose details. The hybrid approach is a pragmatic middle ground.

### Working Memory: Structured Task State

Short-term memory (the messages array) is unstructured — it's a log of everything that happened. Working memory is different. It's a structured data object that represents the current state of the task the agent is working on.

```python
state = {
    "task": "Plan a trip to Tokyo",
    "budget": 2000,
    "dates": {"start": "2026-04-01", "end": "2026-04-08"},
    "flights_found": [
        {"airline": "JAL", "price": 500, "departure": "2026-04-01 10:00"},
        {"airline": "ANA", "price": 480, "departure": "2026-04-01 14:30"},
    ],
    "selected_flight": None,           # not yet decided
    "hotels_found": [],                  # not yet searched
    "selected_hotel": None,              # not yet decided
    "activities": [],                    # not yet planned
    "total_estimated_cost": None,        # will be calculated
    "recommendation": None,              # final output
}
```

This is different from the messages array in important ways:

1. **It's structured** — you know exactly what fields exist and what they mean
2. **It's queryable** — you can check `state["flights_found"]` without parsing natural language
3. **It accumulates** — each agent step reads from and writes to the state
4. **It's the source of truth** — when you want to know what the agent has found, you check the state, not the conversation log

Working memory is what lets agents make progress on multi-step tasks. Without it, every step would need to re-extract information from the conversation history. With it, information flows cleanly from one step to the next.

### Long-Term Memory: Persistent Storage

Both short-term and working memory disappear when the conversation ends. Long-term memory is anything that persists beyond a single session:

```python
# Examples of long-term memory

# 1. User preferences stored in a database
user_prefs = db.get_user_prefs(user_id)
# {"preferred_airlines": ["JAL"], "seat_preference": "window", "dietary": "vegetarian"}

# 2. Past interactions stored for personalization
past_trips = db.get_past_trips(user_id)
# [{"destination": "Osaka", "date": "2025-09", "rating": 5}]

# 3. Vector store for semantic search over documents
relevant_docs = vector_store.search("Tokyo travel tips", top_k=5)
# Returns the 5 most semantically similar documents

# 4. File-based memory (like Claude Code's CLAUDE.md)
# Stores project-specific context that persists across sessions
```

Long-term memory is what makes an agent feel "smart" over time. The first time you use a travel agent, it knows nothing about you. The tenth time, it remembers you prefer window seats, like Japanese food, and always want direct flights. That's long-term memory at work.

OpenAI's agent guide discusses this under "context" — the idea that an agent's instructions and available information shape its behavior. Anthropic frames it as part of the "augmented LLM," where retrieval systems feed relevant long-term information into the model's context at query time.

## Why State Management Matters

You might wonder — if the LLM sees the whole conversation history anyway, why bother with explicit state? Three reasons:

**1. Reliability.** Asking an LLM to extract "what flights have we found so far?" from a long conversation is error-prone. Reading `state["flights_found"]` is deterministic.

**2. Efficiency.** A structured state object is compact. The conversation history that produced it might be 10x larger. When passing context to the LLM, you can include a summary of the state instead of the entire history.

**3. Inter-agent communication.** This is the big one. When multiple agents or steps need to share information, state is the communication channel.

## State as Communication Channel

This is a key architectural insight that becomes critical when you move to multi-agent systems. Agents don't talk to each other directly — they read from and write to shared state.

```python
# Agent A: Flight Search Agent
# Writes flight results to shared state
def flight_agent(state):
    flights = search_flights(state["destination"], state["dates"])
    state["flights_found"] = flights
    state["cheapest_flight"] = min(flights, key=lambda f: f["price"])
    return state

# Agent B: Hotel Search Agent
# Reads destination from state (written by user), writes hotel results
def hotel_agent(state):
    hotels = search_hotels(state["destination"], state["dates"])
    state["hotels_found"] = hotels
    state["cheapest_hotel"] = min(hotels, key=lambda h: h["price"])
    return state

# Agent C: Budget Analyzer
# Reads from both Agent A and Agent B's results — doesn't know they exist
def budget_agent(state):
    flight_cost = state["cheapest_flight"]["price"]
    hotel_cost = state["cheapest_hotel"]["price"] * state["dates"]["nights"]
    state["total_estimated_cost"] = flight_cost + hotel_cost
    state["within_budget"] = state["total_estimated_cost"] <= state["budget"]
    return state
```

Notice that the budget agent doesn't import the flight agent or call it. It just reads `state["cheapest_flight"]` — it doesn't care where that data came from. This decoupling is powerful. You can swap out the flight agent's implementation, add new agents, or change the execution order, all without modifying the budget agent.

This is the exact pattern that LangGraph uses. Each "node" in a LangGraph graph is a function that takes state and returns updated state. The graph defines the execution order, and state is the communication channel.

## TypedDict and Pydantic for State

When your state is a plain dictionary, mistakes are silent. Misspell a key? You get `None` instead of an error. Forget to initialize a field? The next agent crashes with a confusing `KeyError`.

Python gives you two good options for adding structure.

### TypedDict: Lightweight, Used by LangGraph

```python
from typing import TypedDict, Optional

class TravelState(TypedDict):
    """State schema for the travel planning agent."""
    destination: str
    dates: dict
    budget: float
    flights_found: list
    selected_flight: Optional[dict]
    hotels_found: list
    selected_hotel: Optional[dict]
    total_cost: Optional[float]
    recommendation: Optional[str]

# Usage — your editor will autocomplete field names and flag typos
state: TravelState = {
    "destination": "Tokyo",
    "dates": {"start": "2026-04-01", "end": "2026-04-08"},
    "budget": 2000.0,
    "flights_found": [],
    "selected_flight": None,
    "hotels_found": [],
    "selected_hotel": None,
    "total_cost": None,
    "recommendation": None,
}
```

TypedDict is a type hint, not runtime enforcement. Your IDE will catch `state["destnation"]` (typo) at edit time, but Python won't raise an error at runtime. This is fine for many use cases, and it's why LangGraph chose it — minimal overhead, good developer experience.

### Pydantic: Validation, Serialization, More Features

```python
from pydantic import BaseModel, Field
from typing import Optional

class Flight(BaseModel):
    airline: str
    price: float
    departure: str

class TravelState(BaseModel):
    destination: str
    budget: float = Field(gt=0, description="Must be positive")
    flights_found: list[Flight] = []
    selected_flight: Optional[Flight] = None
    total_cost: Optional[float] = None

    # Pydantic validates at runtime
    # TravelState(destination="Tokyo", budget=-100) → raises ValidationError

# Serialization for free
state = TravelState(destination="Tokyo", budget=2000)
state_json = state.model_dump_json()     # serialize to JSON string
state_back = TravelState.model_validate_json(state_json)  # deserialize
```

Pydantic gives you runtime validation (catch bad data immediately), serialization (save/load state easily), and nested model support (a `Flight` inside `TravelState`). It's heavier than TypedDict but worth it when state correctness is critical — like when state is being persisted to a database or passed between services.

The rule of thumb: use TypedDict for prototypes and simple agents, Pydantic for production systems and anything that persists state.

## Checkpointing

Long-running agents can fail mid-way. A network error, an API timeout, a rate limit — any of these can interrupt an agent that's been working for minutes. Without checkpointing, you start over from scratch. With it, you resume from the last good state.

```python
import json
from pathlib import Path

class CheckpointedAgent:
    """
    An agent that saves its state after each step.
    If it crashes and restarts, it picks up where it left off.
    """

    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, state, step_number):
        """Save state to disk after a successful step."""
        path = self.checkpoint_dir / f"step_{step_number}.json"
        path.write_text(json.dumps(state, indent=2))
        print(f"  [Checkpoint saved: step {step_number}]")

    def load_latest_checkpoint(self):
        """Find and load the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.json"))
        if checkpoints:
            latest = checkpoints[-1]
            state = json.loads(latest.read_text())
            step = int(latest.stem.split("_")[1])
            print(f"  [Resuming from checkpoint: step {step}]")
            return state, step
        return None, 0
```

LangGraph has built-in checkpointing with multiple backends (memory, SQLite, Postgres). This is one of its most practical features — when you add `checkpointer=MemorySaver()` to a LangGraph graph, every node execution automatically saves state. If the graph crashes, you can resume from the last checkpoint.

Checkpointing also enables **human-in-the-loop patterns**. The agent runs to a certain point, saves a checkpoint, and pauses for human review. The human approves or modifies the state, and the agent resumes. This is how many production agents handle high-stakes decisions — the LLM does the research, a human makes the final call.

## Industry References

**OpenAI** discusses memory primarily through the lens of "context" — the information available to the model at each step. Their agent guide emphasizes that carefully curated context (what the model sees) is more important than raw context length (how much the model can see). A well-structured state object passed as context beats a 100K-token conversation dump.

**Anthropic's "Building Effective Agents"** paper highlights state management as a key differentiator between simple chains and sophisticated workflows. Their prompt chaining pattern (the simplest orchestration pattern) passes the output of one LLM call as input to the next. This is state management in its most basic form — each step's output becomes the next step's context.

Both perspectives reinforce the same lesson: **how you manage information flow between steps is often more important than the sophistication of any individual step**.

---

## Runnable Example: A Stateful Research Agent

Let's build an agent that demonstrates all three types of memory working together. This agent researches a topic by searching, evaluating whether it has enough information, and searching again if needed. The state grows at each step, and we print it to make the accumulation visible.

```python
# Complete runnable example — a stateful research agent
#
# Demonstrates:
#   - Short-term memory (messages array)
#   - Working memory (structured state dict)
#   - State-driven decisions (agent checks state to decide next action)
#   - State accumulation across multiple steps
#
# The agent:
#   1. Receives a research question
#   2. Searches for information (mock tool)
#   3. Evaluates if it has enough info (checks state)
#   4. Searches again with a refined query if needed
#   5. Synthesizes a final answer from accumulated state

import json
from datetime import datetime


# ─── Structured State ────────────────────────────────────────────────

def create_initial_state(question):
    """
    Creates the initial working memory for a research task.
    Every field is explicitly defined — no surprises later.
    """
    return {
        "question": question,
        "search_queries": [],          # queries we've run
        "search_results": [],          # all results accumulated
        "sources_count": 0,            # how many sources we've found
        "sufficient_info": False,      # do we have enough to answer?
        "synthesis": None,             # the final answer
        "iterations": 0,              # how many search rounds
        "started_at": datetime.now().isoformat(),
    }


# ─── Mock Tools ──────────────────────────────────────────────────────

def search(query):
    """
    Mock search tool. Returns different results based on the query.
    In production: Tavily, Brave Search, Google Custom Search, etc.
    """
    mock_database = {
        "AI agents overview": [
            {"title": "What Are AI Agents?", "snippet": "AI agents are systems that use LLMs to autonomously accomplish tasks, using tools and reasoning in a loop.", "source": "anthropic.com"},
            {"title": "The Rise of AI Agents", "snippet": "The agent paradigm shifts from single-turn Q&A to multi-step task completion with tool use.", "source": "openai.com"},
        ],
        "AI agents tools and function calling": [
            {"title": "Function Calling in LLMs", "snippet": "Tool use allows LLMs to interact with external systems by generating structured function calls that the application executes.", "source": "docs.anthropic.com"},
            {"title": "Building Tool-Using Agents", "snippet": "Effective tool design — clear names, detailed descriptions, minimal parameters — is critical for reliable agent behavior.", "source": "cookbook.openai.com"},
        ],
        "AI agents state management patterns": [
            {"title": "State in Multi-Agent Systems", "snippet": "Shared state objects serve as communication channels between agents, enabling decoupled architectures where agents read/write without direct interaction.", "source": "langchain.com"},
            {"title": "Checkpointing for Reliability", "snippet": "Saving agent state at each step enables crash recovery, human-in-the-loop review, and debugging of multi-step workflows.", "source": "blog.langchain.dev"},
        ],
        "AI agents real world applications 2026": [
            {"title": "Agents in Production", "snippet": "Companies are deploying agents for customer support, code generation, data analysis, and research automation, with careful guardrails.", "source": "reuters.com"},
        ],
    }

    # Find the best matching results
    results = []
    for key, entries in mock_database.items():
        # Simple keyword overlap matching
        query_words = set(query.lower().split())
        key_words = set(key.lower().split())
        if len(query_words & key_words) >= 2:
            results.extend(entries)

    if not results:
        results = [{"title": "General AI Info", "snippet": "AI continues to evolve rapidly with new agent architectures.", "source": "techreview.com"}]

    return results


# ─── Mock LLM ────────────────────────────────────────────────────────

def mock_llm_decide(state, messages):
    """
    Simulates LLM decision-making based on the current state.

    The LLM checks:
    - How many sources have we gathered?
    - Have we searched from multiple angles?
    - Is it time to synthesize?

    A real LLM would read the state + messages and reason about this.
    Our mock uses simple rules that mirror that reasoning.
    """
    iterations = state["iterations"]
    sources_count = state["sources_count"]

    if iterations == 0:
        # First search: broad query based on the question
        return {
            "action": "search",
            "query": "AI agents overview",
            "reasoning": "Starting with a broad search to understand the landscape."
        }

    elif iterations == 1 and sources_count < 5:
        # Second search: more specific angle
        return {
            "action": "search",
            "query": "AI agents tools and function calling",
            "reasoning": f"Found {sources_count} sources so far. Need more depth on tools and function calling."
        }

    elif iterations == 2 and sources_count < 7:
        # Third search: another angle
        return {
            "action": "search",
            "query": "AI agents state management patterns",
            "reasoning": f"Have {sources_count} sources. Adding information about state management."
        }

    else:
        # Enough info — time to synthesize
        return {
            "action": "synthesize",
            "reasoning": f"Gathered {sources_count} sources across {iterations} searches. Sufficient to answer the question."
        }


def mock_llm_synthesize(state):
    """
    Simulates LLM synthesizing a final answer from accumulated state.
    A real LLM would read all the search results and write a coherent summary.
    """
    sources = state["search_results"]
    snippets = [r["snippet"] for r in sources]

    # Combine the snippets into a "synthesized" answer
    synthesis = (
        f"Based on {len(sources)} sources, here is what I found about '{state['question']}':\n\n"
    )
    for i, result in enumerate(sources, 1):
        synthesis += f"  {i}. {result['snippet']} (Source: {result['source']})\n"

    synthesis += (
        f"\nIn summary: AI agents combine LLMs with tools and state management to autonomously "
        f"accomplish multi-step tasks. The key components are the reasoning loop, tool integration, "
        f"and structured state that flows between steps."
    )

    return synthesis


# ─── Agent Loop ──────────────────────────────────────────────────────

def print_state(state, label=""):
    """Pretty-print the current state for visibility."""
    print(f"\n  {'─'*50}")
    print(f"  State {label}:")
    print(f"    Queries run:    {state['search_queries']}")
    print(f"    Sources found:  {state['sources_count']}")
    print(f"    Iterations:     {state['iterations']}")
    print(f"    Sufficient:     {state['sufficient_info']}")
    if state['synthesis']:
        print(f"    Synthesis:      (generated, {len(state['synthesis'])} chars)")
    print(f"  {'─'*50}")


def run_research_agent(question, max_iterations=5):
    """
    A research agent that accumulates information across multiple search steps.

    Key pattern to observe:
    - State is created at the start and grows with each step
    - The LLM checks state to decide what to do next
    - Each search adds to state["search_results"] (accumulation, not replacement)
    - The synthesis step reads from the accumulated state
    """
    print(f"\n{'='*60}")
    print(f"  Research Question: {question}")
    print(f"{'='*60}")

    # Initialize working memory
    state = create_initial_state(question)

    # Initialize short-term memory (conversation history)
    messages = [
        {"role": "system", "content": "You are a research assistant. Search for information, evaluate completeness, and synthesize findings."},
        {"role": "user", "content": question},
    ]

    print_state(state, "(initial)")

    for iteration in range(1, max_iterations + 1):
        print(f"\n  === Step {iteration} ===")

        # LLM decides what to do based on current state
        decision = mock_llm_decide(state, messages)
        print(f"  Reasoning: {decision['reasoning']}")

        if decision["action"] == "search":
            query = decision["query"]
            print(f"  Action: search(\"{query}\")")

            # Execute the search
            results = search(query)
            print(f"  Found: {len(results)} result(s)")
            for r in results:
                print(f"    - {r['title']} ({r['source']})")

            # ── Update state (this is the critical part) ──
            # Results ACCUMULATE — we append, not replace
            state["search_queries"].append(query)
            state["search_results"].extend(results)  # extend, not assign
            state["sources_count"] = len(state["search_results"])
            state["iterations"] += 1

            # Add to conversation history (short-term memory)
            messages.append({
                "role": "assistant",
                "content": f"Searched for: {query}"
            })
            messages.append({
                "role": "tool",
                "content": json.dumps(results)
            })

            print_state(state, "(after search)")

        elif decision["action"] == "synthesize":
            print(f"  Action: synthesize from {state['sources_count']} accumulated sources")

            # Generate final answer from accumulated state
            synthesis = mock_llm_synthesize(state)

            # Update state with final result
            state["synthesis"] = synthesis
            state["sufficient_info"] = True

            print_state(state, "(final)")
            print(f"\n  Final Answer:")
            print(f"  {'-'*50}")
            for line in synthesis.split('\n'):
                print(f"  {line}")
            print(f"  {'-'*50}")
            print(f"\n  Research completed in {iteration} step(s)")
            print(f"  Total sources consulted: {state['sources_count']}")
            print(f"  Search queries used: {state['search_queries']}")

            return state

    print(f"\n  Hit max iterations without completing research.")
    return state


# ─── Run the Agent ───────────────────────────────────────────────────

# The agent will:
#   Step 1: Search broadly → find 2 sources → state grows
#   Step 2: Search for tools/function calling → find 2 more → state grows
#   Step 3: Search for state management → find 2 more → state grows
#   Step 4: Decide it has enough → synthesize from all 6 sources
#
# Watch the state accumulate at each step!

final_state = run_research_agent("What are AI agents and how do they work?")


# ─── Bonus: Inspecting the Final State ──────────────────────────────
# This shows why structured state is valuable — you can programmatically
# inspect exactly what the agent found and did.

print(f"\n{'='*60}")
print(f"  Post-Run State Inspection")
print(f"{'='*60}")
print(f"  Question:       {final_state['question']}")
print(f"  Searches:       {len(final_state['search_queries'])}")
print(f"  Total sources:  {final_state['sources_count']}")
print(f"  Unique domains: {set(r['source'] for r in final_state['search_results'])}")
print(f"  Has answer:     {final_state['sufficient_info']}")
print(f"  Answer length:  {len(final_state['synthesis'])} characters")
```

When you run this, you'll see state accumulating at each step — the `sources_count` grows from 0 to 2 to 4 to 6, and the final synthesis draws from all accumulated sources. That's the core pattern: state as accumulated context that enables multi-step reasoning.

---

## Key Takeaways

- **Short-term memory is the messages array.** The LLM re-reads the entire conversation on every call. It's powerful but bounded by the context window.
- **Working memory is structured state.** A dictionary or typed object that accumulates information across agent steps. More reliable than extracting data from conversation history.
- **Long-term memory is persistent storage.** Databases, files, vector stores — anything that survives beyond a single session.
- **State is the communication channel between agents.** Agents don't call each other — they read from and write to shared state. This decoupling is what makes multi-agent systems composable.
- **Use TypedDict for prototypes, Pydantic for production.** Types catch bugs early and make state self-documenting.
- **Checkpointing enables reliability and human-in-the-loop.** Save state at key points so you can resume after failures or pause for human review.

---

## What's Next

You understand the building blocks: LLMs, tools, and state. Now let's look at how to wire them together into architectures. Chapter 4 covers single-agent patterns — the common ways to structure an agent's control flow, from simple prompt chaining to fully autonomous loops.
