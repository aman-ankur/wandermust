# Chapter 6: Graphs & Orchestration Frameworks

You could wire agents together with `if/else` and function calls. And for simple cases, you should. But as your agent system grows — parallel execution, conditional routing, error recovery, state management — you need a framework. The most natural abstraction? Graphs.

Think about it: every agent workflow you've built so far has been a series of steps with connections between them. A chatbot is one step. A ReAct agent is a loop. A multi-agent pipeline is a branching tree. These are all graphs. The moment you start drawing your system on a whiteboard, you're drawing nodes and edges. Frameworks just formalize what you're already thinking.

This chapter explores why graphs are the dominant abstraction for agent orchestration, takes a deep dive into LangGraph (the most popular graph-based framework), compares it to alternatives like CrewAI, and helps you decide when a framework is worth the overhead — and when it isn't.

---

## Why Graphs for Agent Orchestration?

Every agent system you build has the same fundamental structure: steps connected by transitions. A step might call an LLM, hit an API, run some computation, or delegate to another agent. A transition decides what happens next. That's a graph.

Here's the progression from simple to complex:

```
Chatbot:     [LLM] → done

ReAct:       [LLM] → [Tool] → [LLM] → [Tool] → ... (cycle)

Multi-agent: [Supervisor] → [Agent A] ─┐
                            [Agent B] ─┤→ [Scorer] → [Output]
                            [Agent C] ─┘
```

A chatbot is a single node with one edge to "done." A ReAct agent is a cycle — the LLM calls a tool, reads the result, decides whether to call another tool or respond, and loops. A multi-agent system is a directed graph where a supervisor fans out to multiple specialist agents, their results fan back in to a scorer, and the scorer produces final output.

Once you see your system as a graph, you get several things for free:

- **Visualization.** You can literally draw the execution flow. LangGraph can even render your graph as a diagram. When your system has 12 agents and 20 transitions, this matters enormously for debugging.
- **Defined execution order.** The graph structure guarantees that node B only runs after node A finishes. No race conditions to think about (the framework handles it).
- **Parallel branches.** If two nodes have no dependency between them, the framework can run them simultaneously. You declare the structure; it handles the concurrency.
- **Conditional routing.** Instead of nested if/else chains, you declare a routing function on an edge. The graph engine evaluates it at runtime and sends execution down the right path.

Anthropic's "Building Effective Agents" paper describes the progression from simple prompt-chaining to orchestrator-worker patterns. Each pattern they describe maps naturally to a graph topology. The prompt chain is a linear graph. The routing pattern is a graph with conditional edges. The orchestrator-worker pattern is a fan-out/fan-in graph. The framework just makes these topologies explicit and executable.

---

## Graph Vocabulary

Before diving into code, let's nail down the terms. These apply across frameworks, not just LangGraph.

**Node:** A function that does something. It might call an LLM, fetch data from an API, compute a score, or format output. In code, it's literally a Python function that takes state and returns updates. Nodes are the "workers" in your system.

**Edge:** A connection between two nodes that controls execution order. If there's an edge from A to B, then B runs after A completes. Edges are the "wiring."

**State:** The data that flows through the graph, modified by each node as it passes through. Think of it as a shared dictionary. Node A writes `{"weather": data}` into state. Node B reads `state["weather"]` to do its work. State is the communication channel between nodes — they never call each other directly.

**Entry point:** The node where execution begins. When you invoke the graph, this is the first node that runs.

**END:** A special marker indicating execution is complete. When a node's edge points to END, the graph stops and returns the final state.

**Conditional edge:** An edge that picks the next node based on the current state. Instead of always going from A to B, a conditional edge runs a function that inspects state and returns the name of the next node. This is how you implement branching logic — "if the user asked about flights, go to the flights agent; if hotels, go to the hotels agent."

These six concepts are enough to model any agent workflow. Everything else is syntactic sugar.

---

## LangGraph Deep Dive

LangGraph is built on top of LangChain, but don't let that scare you. You don't need to understand (or even use) the full LangChain ecosystem. LangGraph's core is remarkably simple: define a state type, add nodes, add edges, compile, run.

The core class is `StateGraph`. Here's the absolute minimum — a graph with one node:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. Define state — this is the data flowing through the graph.
#    Every node receives the full state and returns a partial update.
class MyState(TypedDict):
    input: str
    result: str

# 2. Define nodes — plain Python functions.
#    Takes state, returns a dict with ONLY the fields to update.
def process(state: MyState) -> dict:
    return {"result": state["input"].upper()}

# 3. Build the graph
graph = StateGraph(MyState)
graph.add_node("process", process)       # Register the node
graph.set_entry_point("process")         # Execution starts here
graph.add_edge("process", END)           # After processing, stop

# 4. Compile and run
app = graph.compile()
output = app.invoke({"input": "hello", "result": ""})
print(output["result"])  # "HELLO"
```

That's it. Define state, define nodes, wire them up, compile, invoke. Every LangGraph application follows this pattern, no matter how complex.

---

## Nodes in Detail

A node is just a Python function with a specific signature: it takes the full state as input and returns a dictionary containing only the fields it wants to update. The framework handles merging the update back into the state.

This is a critical design choice. Nodes don't know about each other. They don't call each other. They communicate exclusively through state. This means you can add, remove, or reorder nodes without changing their internal code — as long as the state contract is maintained.

```python
def weather_node(state):
    # I only care about "destination" from state.
    # I only update "weather_data".
    # I have no idea what other nodes exist.
    weather = fetch_weather(state["destination"])
    return {"weather_data": weather}

def scorer_node(state):
    # I read "weather_data" (set by weather_node) and update "ranked_windows".
    # I don't know or care who set weather_data.
    scores = calculate_scores(state["weather_data"])
    return {"ranked_windows": scores}
```

This separation is powerful. If you later add a `climate_node` that also contributes to `weather_data`, the scorer doesn't change. If you swap `weather_node` for a `mock_weather_node` during testing, everything else still works. Nodes are pure functions over state — composable, testable, replaceable.

One subtlety: when a node returns `{"weather_data": data}`, the framework *merges* this into the existing state. It doesn't replace the entire state. Fields you don't mention in the return dict are preserved unchanged. This means each node only needs to worry about its own outputs.

---

## Edges — Normal and Conditional

Normal edges are straightforward — they always route execution from one node to another:

```python
# After node_a finishes, always run node_b
graph.add_edge("node_a", "node_b")
```

Conditional edges are where things get interesting. A conditional edge calls a routing function that inspects the current state and returns the name of the next node to execute:

```python
# After check_node finishes, decide where to go based on state
def route(state):
    if state["needs_more_data"]:
        return "fetch_more"
    return "summarize"

graph.add_conditional_edges("check_node", route, {
    "fetch_more": "data_fetcher",    # If route returns "fetch_more", go here
    "summarize": "summarizer",       # If route returns "summarize", go here
})
```

The third argument is a mapping from return values to node names. It's optional — if omitted, the return value of `route` is used directly as the node name — but it's good practice for clarity and validation.

Conditional edges are how you build loops (route back to a previous node), implement retries (route back to the same node with updated state), and create branching pipelines (route to different specialists based on the query type).

Here's a classic ReAct loop expressed as a conditional edge:

```python
def should_continue(state):
    # If the LLM's last message included a tool call, keep looping.
    # If it produced a final answer, stop.
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

graph.add_conditional_edges("llm", should_continue, {
    "tools": "tool_executor",
    "end": END,
})
graph.add_edge("tool_executor", "llm")  # After tools, go back to LLM
```

That's the entire ReAct loop: LLM generates, check if it wants a tool, execute the tool, feed results back to LLM. Two nodes, two edges (one conditional), done.

---

## Parallel Execution (Fan-Out / Fan-In)

This is one of LangGraph's most powerful features. When multiple edges leave from the same node, the target nodes execute in parallel:

```python
# supervisor fans out to three nodes — all 3 run simultaneously
graph.add_edge("supervisor", "weather")
graph.add_edge("supervisor", "flights")
graph.add_edge("supervisor", "hotels")

# All three fan back in to scorer — scorer waits for all 3 to finish
graph.add_edge("weather", "scorer")
graph.add_edge("flights", "scorer")
graph.add_edge("hotels", "scorer")
```

LangGraph handles the parallelism automatically. When `supervisor` finishes, `weather`, `flights`, and `hotels` all start at the same time. Each parallel node receives a copy of the state at that point. When all three complete, their state updates are merged, and `scorer` runs with the combined result.

This is a massive win for performance. If each API call takes 2 seconds, running them sequentially costs 6 seconds. Running them in parallel costs 2 seconds. In an agent system with 5-10 external calls, parallelism can cut response time by 70-80%.

The merge behavior is worth understanding: if `weather` returns `{"weather_data": ...}` and `flights` returns `{"flight_data": ...}`, the merged state has both fields. But if two parallel nodes update the *same* field, you need to be careful. LangGraph uses "reducers" for this — you can annotate state fields with a merge strategy (like appending to a list). For now, the simple rule is: design parallel nodes to write to different state fields.

---

## Framework Comparison

LangGraph isn't the only option. Here's how it compares to CrewAI (a popular role-based framework) and plain Python:

| Feature | LangGraph | CrewAI | Raw Python |
|---------|-----------|--------|------------|
| Abstraction | Graph (nodes/edges) | Roles (agents/tasks) | Your own |
| Transparency | High — you see every node | Medium — some magic | Total |
| Parallel execution | Built-in | Limited | DIY with asyncio |
| State management | Built-in TypedDict | Internal | DIY |
| Conditional routing | Built-in | Limited | if/else |
| Learning curve | Medium | Easy start, hard to customize | Low start, high ceiling |
| Model flexibility | Any via LangChain | Any via LiteLLM | Any |
| Checkpointing | Built-in | No | DIY |

CrewAI takes a fundamentally different approach. Instead of nodes and edges, you define *agents* with roles and *tasks* they perform:

```python
# CrewAI is role-based, not graph-based.
# You describe WHO does WHAT, not the execution flow.
from crewai import Agent, Crew, Task

researcher = Agent(
    role="Researcher",
    goal="Find information about the destination",
    llm=llm
)
writer = Agent(
    role="Writer",
    goal="Write a compelling travel summary",
    llm=llm
)

research_task = Task(description="Research Tokyo travel info", agent=researcher)
write_task = Task(description="Write a summary from research", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
```

CrewAI is great for quick prototypes where you want to define agents by their roles and let the framework figure out coordination. But when you need fine-grained control over execution flow, conditional routing, or parallel execution with specific merge behavior, LangGraph gives you more power.

---

## When to Use a Framework vs. Not

This is the most important decision in this chapter, and the industry leaders are clear about the tradeoffs.

Anthropic's "Building Effective Agents" puts it directly: *"Frameworks can help with standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. But they also create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug."* Their recommendation: start with direct API calls, and only add framework complexity when you genuinely need it.

OpenAI's "Practical Guide to Building Agents" echoes this: start with a single agent, add tools, and only split into multi-agent when a single agent's tool set gets too large or its responsibilities too broad.

Here are practical guidelines:

**Use LangGraph when:** you need parallel execution across multiple data sources, conditional routing based on intermediate results, complex state management across many nodes, or built-in checkpointing for long-running workflows. If your whiteboard diagram has more than 5 nodes and includes branches or loops, LangGraph will save you time.

**Use CrewAI when:** you want a quick prototype with role-based delegation, your agents map cleanly to human roles (researcher, writer, reviewer), and you don't need fine-grained control over execution order. CrewAI gets you to a working demo faster, but customization is harder.

**Use raw Python when:** your flow is linear (step A, then B, then C), you're learning the fundamentals and want to understand every line, or the framework overhead (dependency management, version conflicts, debugging through abstraction layers) isn't worth it. A simple `async def` pipeline with `asyncio.gather` for parallelism is often enough.

The honest truth: most agent systems start as raw Python and graduate to a framework when the complexity demands it. Don't reach for LangGraph on day one. Reach for it when you're drowning in state management and conditional logic.

---

## Runnable Example: Parallel Processing with LangGraph

Let's build a complete 4-node graph that demonstrates parallel execution. The scenario: we receive raw text, parse it into structured data, run two different processors on it simultaneously, then combine their results.

```python
"""
Parallel Processing Graph with LangGraph
=========================================

This script builds a 4-node graph:
  1. input_parser   — takes raw text, extracts structured data
  2. processor_a    — transforms data one way (runs in parallel)
  3. processor_b    — transforms data another way (runs in parallel)
  4. combiner       — merges results from both processors

Requires: pip install langgraph langchain-core

Graph topology:
  input_parser → processor_a ─┐
                               ├→ combiner → END
               → processor_b ─┘
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import json


# ──────────────────────────────────────────────
# Step 1: Define the state schema.
# This TypedDict describes ALL data flowing through the graph.
# Every node reads from this state and returns partial updates.
# ──────────────────────────────────────────────
class PipelineState(TypedDict):
    # Input fields
    raw_text: str

    # Set by input_parser
    parsed_words: list[str]
    word_count: int

    # Set by processor_a (uppercase transformation)
    uppercase_result: str

    # Set by processor_b (statistics computation)
    stats_result: str

    # Set by combiner (final merged output)
    final_output: str


# ──────────────────────────────────────────────
# Step 2: Define each node as a plain function.
# Each function takes the full state, returns ONLY
# the fields it wants to update.
# ──────────────────────────────────────────────

def input_parser(state: PipelineState) -> dict:
    """
    Node 1: Parse raw text into structured data.
    Reads: raw_text
    Writes: parsed_words, word_count
    """
    raw = state["raw_text"]
    words = raw.strip().split()
    print(f"[input_parser] Parsed {len(words)} words from input")
    print(f"[input_parser] State after: parsed_words={words}, word_count={len(words)}")
    return {
        "parsed_words": words,
        "word_count": len(words),
    }


def processor_a(state: PipelineState) -> dict:
    """
    Node 2a: Uppercase transformation (runs in PARALLEL with processor_b).
    Reads: parsed_words
    Writes: uppercase_result
    """
    words = state["parsed_words"]
    uppercased = [w.upper() for w in words]
    result = " ".join(uppercased)
    print(f"[processor_a] Uppercased: {result}")
    return {"uppercase_result": result}


def processor_b(state: PipelineState) -> dict:
    """
    Node 2b: Statistics computation (runs in PARALLEL with processor_a).
    Reads: parsed_words, word_count
    Writes: stats_result
    """
    words = state["parsed_words"]
    count = state["word_count"]

    # Compute some basic statistics about the text
    avg_length = sum(len(w) for w in words) / max(count, 1)
    longest = max(words, key=len) if words else ""
    shortest = min(words, key=len) if words else ""

    stats = (
        f"Words: {count}, "
        f"Avg length: {avg_length:.1f}, "
        f"Longest: '{longest}', "
        f"Shortest: '{shortest}'"
    )
    print(f"[processor_b] Stats: {stats}")
    return {"stats_result": stats}


def combiner(state: PipelineState) -> dict:
    """
    Node 3: Merge results from both parallel processors.
    Reads: uppercase_result, stats_result, word_count
    Writes: final_output

    This node only runs AFTER both processor_a and processor_b finish.
    LangGraph handles the synchronization automatically.
    """
    combined = (
        f"=== Processing Complete ===\n"
        f"Transformed text: {state['uppercase_result']}\n"
        f"Analysis: {state['stats_result']}\n"
        f"Total words processed: {state['word_count']}"
    )
    print(f"[combiner] Final output assembled")
    return {"final_output": combined}


# ──────────────────────────────────────────────
# Step 3: Build the graph.
# Add nodes, set entry point, wire edges.
# ──────────────────────────────────────────────

# Create a new graph with our state type
graph = StateGraph(PipelineState)

# Add all four nodes
graph.add_node("input_parser", input_parser)
graph.add_node("processor_a", processor_a)
graph.add_node("processor_b", processor_b)
graph.add_node("combiner", combiner)

# Set the entry point — execution starts here
graph.set_entry_point("input_parser")

# Fan-out: input_parser → processor_a AND processor_b (parallel)
graph.add_edge("input_parser", "processor_a")
graph.add_edge("input_parser", "processor_b")

# Fan-in: both processors → combiner (combiner waits for both)
graph.add_edge("processor_a", "combiner")
graph.add_edge("processor_b", "combiner")

# Combiner → END (stop execution)
graph.add_edge("combiner", END)


# ──────────────────────────────────────────────
# Step 4: Compile and run.
# .compile() validates the graph and returns a runnable.
# .invoke() executes the graph with the given initial state.
# ──────────────────────────────────────────────

# Compile the graph into a runnable application
app = graph.compile()

# Prepare initial state — we only need to provide the input fields.
# All other fields will be populated by nodes as they execute.
initial_state = {
    "raw_text": "the quick brown fox jumps over the lazy dog",
    "parsed_words": [],
    "word_count": 0,
    "uppercase_result": "",
    "stats_result": "",
    "final_output": "",
}

# Run the graph
print("=" * 50)
print("Starting graph execution...")
print("=" * 50)
result = app.invoke(initial_state)

# Print the final output
print()
print(result["final_output"])

# ──────────────────────────────────────────────
# Expected output:
#
# ==================================================
# Starting graph execution...
# ==================================================
# [input_parser] Parsed 9 words from input
# [input_parser] State after: parsed_words=[...], word_count=9
# [processor_a] Uppercased: THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
# [processor_b] Stats: Words: 9, Avg length: 3.7, Longest: 'jumps', Shortest: 'the'
# [combiner] Final output assembled
#
# === Processing Complete ===
# Transformed text: THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
# Analysis: Words: 9, Avg length: 3.7, Longest: 'jumps', Shortest: 'the'
# Total words processed: 9
# ──────────────────────────────────────────────
```

Notice what's happening: `processor_a` and `processor_b` both depend on `input_parser`'s output, but they don't depend on each other. LangGraph detects this from the edge structure and runs them in parallel. The `combiner` node has incoming edges from both processors, so it waits until both finish before executing. You declared the structure; the framework handled the scheduling.

---

## Key Takeaways

1. **Your agent workflow is already a graph.** Nodes are steps, edges are transitions. Frameworks just make this explicit and executable.
2. **State is the communication channel.** Nodes never call each other directly. They read from and write to shared state. This makes them composable and testable.
3. **Conditional edges replace if/else chains.** Instead of nesting logic, declare a routing function. The graph engine handles it.
4. **Parallel execution is free.** Declare fan-out edges and the framework runs nodes concurrently. This is the single biggest performance win in multi-agent systems.
5. **Start without a framework.** Build with raw Python first. Graduate to LangGraph when you need parallel execution, conditional routing, or checkpointing. Don't add complexity until it earns its keep.

---

## What's Next

You know how to build agents and wire them together. But what happens when things go wrong? The API goes down. The LLM hallucinates. Your rate limit gets hit. Chapter 7 covers making agents reliable — error handling, fallbacks, retries, caching, and guardrails that keep your system running when the real world gets messy.
