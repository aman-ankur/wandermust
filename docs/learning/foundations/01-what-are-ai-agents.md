# What Are AI Agents?

You've called an LLM API. You sent a prompt, got a response. But what if the LLM could decide on its own to search the web, check a database, and retry if something failed — all without you writing the control flow?

That's the shift from "using an LLM" to "building an agent." It sounds like a small distinction, but it changes everything about how you architect systems. Let's break it down.

---

## Chatbot vs Agent — The Fundamental Difference

Here's the simplest possible chatbot interaction:

```python
response = llm.invoke("What's the capital of France?")
print(response)  # "Paris"
```

One call. One response. Done. The LLM has no ability to go look something up, no ability to retry, no ability to decide it needs more information. It's a function call — input in, output out.

Now here's the mental model for an agent:

```python
done = False
while not done:
    thought = llm.reason(observation)    # What should I do next?
    action = llm.decide(thought)          # Pick a tool or respond
    observation = execute(action)          # Run the tool, get result
    done = llm.should_stop(observation)   # Are we finished?
```

See the difference? The chatbot is a single exchange. The agent is a **loop**. The LLM keeps running until it decides it has accomplished the task. It thinks, acts, observes the result, and thinks again. This loop is the beating heart of every agent system you'll encounter.

This isn't just a theoretical distinction. When you use Claude Code to fix a bug, it doesn't just suggest a fix — it reads your code, proposes a change, applies the edit, runs the tests, sees they fail, reads the error, tries a different approach, and runs the tests again. That's the loop in action.

## The Agent Loop

Let's make this concrete with a diagram. Every agent, regardless of framework or complexity, follows some version of this cycle:

```
┌─────────────┐
│   Perceive   │ ← get input / observe tool results
└──────┬───────┘
       ▼
┌─────────────┐
│   Reason     │ ← LLM decides what to do next
└──────┬───────┘
       ▼
┌─────────────┐
│     Act      │ ← call a tool, generate output, or stop
└──────┬───────┘
       ▼
┌─────────────┐
│   Observe    │ ← check the result
└──────┬───────┘
       │
       └──→ back to Reason (loop until done)
```

Let's walk through each step with a concrete example. Imagine you ask an agent: "Find the cheapest flight from NYC to Tokyo next month."

1. **Perceive**: The agent receives your request. This is the initial input.
2. **Reason**: The LLM thinks — "I need to search for flights. I have a `search_flights` tool. I should use it with origin=NYC, destination=Tokyo, date=next month."
3. **Act**: The agent calls `search_flights(origin="NYC", destination="Tokyo", date="2026-04")`.
4. **Observe**: The tool returns 15 flights. The agent now has data.
5. **Reason** (again): The LLM looks at the results — "The cheapest is JAL at $480, but the user might want to know about layovers. Let me also check direct flights specifically."
6. **Act**: Calls `search_flights(origin="NYC", destination="Tokyo", date="2026-04", direct_only=True)`.
7. **Observe**: Gets 3 direct flights.
8. **Reason**: "Now I have enough information to answer the user's question comprehensively."
9. **Act**: Generates a final response summarizing the options.
10. **Done**: The loop exits.

Notice something important: **you didn't code this control flow**. You didn't write "first search all flights, then search direct flights." The LLM decided on its own that a second search would be helpful. That's the essence of agentic behavior.

## The Autonomy Spectrum

Not every system that uses an LLM is an agent. It's more useful to think of a spectrum:

**Level 0 — Chatbot (no tools, no loop)**

```python
response = llm.invoke(prompt)
# That's it. No tools, no iteration.
```

This is your basic ChatGPT conversation, a customer support bot reading from a script, or any system where the LLM generates one response and stops. Perfectly useful for many tasks, but not an agent.

**Level 1 — Tool-use agent (LLM picks tools, human-defined flow)**

```python
# You define WHEN tools can be called, the LLM decides WHICH tool
response = llm.invoke(prompt, tools=[search, calculator])
if response.has_tool_call:
    result = execute_tool(response.tool_call)
    final = llm.invoke(prompt + result)
```

The LLM chooses which tool to use, but you've hard-coded that there's exactly one round of tool use. Many production systems operate at this level — it's predictable, debuggable, and often sufficient.

**Level 2 — Autonomous agent (LLM controls the flow, decides when to stop)**

```python
while not done:
    response = llm.invoke(messages, tools=tools)
    if response.wants_to_stop:
        done = True
    else:
        result = execute_tool(response.tool_call)
        messages.append(result)
```

Now the LLM is in the driver's seat. It decides how many iterations to run, which tools to call in what order, and when the task is complete. Claude Code, Cursor's agent mode, and most "AI coding assistants" operate here.

**Level 3 — Multi-agent system (multiple LLMs coordinating)**

```python
researcher = Agent(role="research", tools=[web_search, arxiv])
analyst = Agent(role="analysis", tools=[calculator, chart])
writer = Agent(role="writing", tools=[editor])

# Agents pass results to each other
research_results = researcher.run("Find data on X")
analysis = analyst.run(f"Analyze: {research_results}")
report = writer.run(f"Write report from: {analysis}")
```

Multiple specialized agents, each with their own tools and instructions, working together on a larger task. This is where frameworks like LangGraph, CrewAI, and AutoGen live.

Most real systems don't fit neatly into one level. A coding agent might be Level 2 for file editing but Level 1 for running tests (always run tests after edits, no LLM decision needed). That's fine — the spectrum is a mental model, not a rigid taxonomy.

## Agents vs Workflows — Anthropic's Key Distinction

Anthropic's "Building Effective Agents" paper draws an important line between **workflows** and **agents** that's worth internalizing:

> "Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage."

Here's what that means in practice. A **workflow** looks like this:

```python
# Workflow: YOU define the steps
def process_support_ticket(ticket):
    category = llm.classify(ticket)           # Step 1: always classify
    if category == "refund":
        order = db.lookup_order(ticket.id)    # Step 2: always look up order
        response = llm.draft_refund(order)    # Step 3: always draft refund
    elif category == "technical":
        docs = search_kb(ticket.text)         # Step 2: always search KB
        response = llm.draft_tech_help(docs)  # Step 3: always draft help
    return response
```

The LLM is doing useful work (classifying, drafting), but the control flow is deterministic. You wrote the if/else. You decided what happens after classification. The LLM is a component in your pipeline, not the orchestrator.

An **agent** looks like this:

```python
# Agent: THE LLM defines the steps
def handle_support_ticket(ticket):
    messages = [{"role": "user", "content": ticket}]
    while True:
        response = llm.invoke(messages, tools=[
            classify_ticket, lookup_order, search_kb,
            check_inventory, draft_response, escalate
        ])
        if response.is_final:
            return response.text
        result = execute(response.tool_call)
        messages.append(result)
```

Same tools available, but the LLM decides the order. Maybe it classifies first. Maybe it checks the order before classifying. Maybe it searches the knowledge base, realizes it needs more context, checks inventory, and then drafts a response. The path isn't predetermined.

Anthropic's practical advice: **start with workflows**. They're more predictable, easier to test, and cheaper to run. Move to agents only when the task genuinely requires dynamic decision-making — when you can't predict the right sequence of steps in advance.

## OpenAI's Definition

OpenAI's "Practical Guide to Building Agents" offers a complementary framing. They define an agent as:

> "A system that independently accomplishes tasks on your behalf."

Their three components are straightforward:

1. **Model** — The LLM that does the reasoning
2. **Tools** — Functions the model can call (APIs, databases, code execution)
3. **Instructions** — The system prompt that tells the model its role and constraints

This is a useful simplification. When you're designing an agent, you're really making three decisions: which model, what tools, and what instructions. Everything else — the loop, the state management, the error handling — is infrastructure.

## Real-World Examples

Let's ground this with agents you've probably already used:

**Coding Agent (Claude Code, Cursor, Windsurf)**

The agent receives a task like "fix the failing test in auth.py." It reads the test file, reads the source code, understands the bug, edits the file, runs the tests, sees if they pass, and iterates if they don't. The tools: file read, file write, shell execution, search. The loop: keep going until tests pass or it's stuck.

**Research Agent**

Given "What are the latest advances in battery technology?", the agent searches the web, reads several articles, follows citations, checks for contradictions, and synthesizes a report. Tools: web search, URL fetching, note-taking. The loop: search → read → decide if more info needed → search again → synthesize.

**Customer Support Agent**

A customer says "I was charged twice for order #12345." The agent checks the order database, sees two charges, checks the refund policy, determines the customer is eligible, initiates a refund, and confirms with the customer. Tools: order lookup, refund API, policy search. The loop: investigate → determine action → execute → confirm.

In each case, the pattern is the same: perceive, reason, act, observe, repeat. The tools change, the domain changes, but the architecture is consistent.

---

## Runnable Example: A Minimal Agent Loop

Let's build this from scratch. No frameworks, no dependencies — just Python. This agent can answer questions directly or decide to use a "search" tool when it doesn't know something.

```python
# Complete runnable example — a minimal agent loop
# This shows the core concept: LLM decides what to do, we execute, repeat
#
# Since we don't want to require API keys for a learning example,
# we simulate the LLM with a deterministic function. In production,
# you'd replace mock_llm() with a real API call.

import json


def mock_llm(messages):
    """
    Simulates an LLM that can decide to use tools or respond directly.

    In a real system, this would be:
        response = client.chat.completions.create(model="gpt-4", messages=messages)
    or:
        response = anthropic.messages.create(model="claude-sonnet-4-20250514", messages=messages)

    Our mock LLM follows simple rules:
    - If the last message asks about a city's population → use search tool
    - If the last message contains search results → synthesize an answer
    - Otherwise → respond directly
    """
    last_message = messages[-1]["content"].lower()

    # If we just got tool results back, synthesize an answer
    if messages[-1]["role"] == "tool":
        tool_result = messages[-1]["content"]
        return {
            "type": "response",
            "content": f"Based on my research: {tool_result} — this makes it one of the most populous cities in the world."
        }

    # If the user asks about population, decide to search
    if "population" in last_message:
        # Extract the city name (simplified logic)
        if "tokyo" in last_message:
            city = "Tokyo"
        elif "paris" in last_message:
            city = "Paris"
        else:
            city = "that city"

        return {
            "type": "tool_call",
            "tool": "search",
            "args": {"query": f"population of {city} 2026"},
            "thought": f"The user is asking about {city}'s population. I should search for current data rather than relying on my training data."
        }

    # For simple questions, respond directly (no tool needed)
    return {
        "type": "response",
        "content": "I can help you with that! Feel free to ask me about city populations or other topics."
    }


def search_tool(query):
    """
    A mock search tool. In production, this would call a search API
    like Google, Brave, or Tavily.

    Returns a string result, just like a real search would.
    """
    mock_results = {
        "population of Tokyo 2026": "Tokyo's metropolitan area has approximately 37.4 million residents as of 2026",
        "population of Paris 2026": "Paris has approximately 2.1 million residents in the city proper, 12.2 million in the metro area",
    }
    # Return a result if we have one, otherwise a generic response
    return mock_results.get(query, f"Search results for '{query}': No specific data found.")


# Registry of available tools — maps tool names to their functions
TOOLS = {
    "search": search_tool,
}


def run_agent(user_input, max_iterations=5):
    """
    The core agent loop.

    This is the pattern every agent follows:
    1. Send messages to the LLM
    2. If LLM wants to use a tool → execute it, add result to messages, go to 1
    3. If LLM wants to respond → return the response
    4. Safety valve: stop after max_iterations to prevent infinite loops
    """
    # Initialize the conversation with the user's input
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to a search tool. Use it when you need current or factual data."},
        {"role": "user", "content": user_input},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_input}")
    print(f"{'='*60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Step 1: Ask the LLM what to do
        llm_output = mock_llm(messages)

        if llm_output["type"] == "tool_call":
            # Step 2a: LLM wants to use a tool
            tool_name = llm_output["tool"]
            tool_args = llm_output["args"]
            thought = llm_output.get("thought", "")

            print(f"LLM thinks: {thought}")
            print(f"LLM action: calling '{tool_name}' with args {json.dumps(tool_args)}")

            # Execute the tool
            tool_fn = TOOLS[tool_name]
            result = tool_fn(**tool_args)
            print(f"Tool result: {result}")

            # Add the tool call and result to the conversation
            # This is how the LLM "remembers" what tools returned
            messages.append({"role": "assistant", "content": f"[Calling {tool_name}({json.dumps(tool_args)})]"})
            messages.append({"role": "tool", "content": result})

        elif llm_output["type"] == "response":
            # Step 2b: LLM wants to respond to the user — we're done
            final_response = llm_output["content"]
            print(f"LLM responds: {final_response}")
            print(f"\nAgent completed in {iteration} iteration(s)")
            return final_response

    # Safety valve: if we hit max iterations, return what we have
    print(f"\nAgent hit max iterations ({max_iterations}), stopping.")
    return "I wasn't able to complete the task within the iteration limit."


# --- Run the agent with different inputs ---

# Example 1: A question that requires the search tool
# The agent should: reason → call search → observe result → respond
run_agent("What's the population of Tokyo?")

# Example 2: A simple question that doesn't need tools
# The agent should: reason → respond directly (1 iteration)
run_agent("Hello, what can you help me with?")

# Example 3: Another tool-use case
# The agent should: reason → call search → observe result → respond
run_agent("Tell me about the population of Paris")
```

When you run this, you'll see the agent loop in action:

```
============================================================
User: What's the population of Tokyo?
============================================================

--- Iteration 1 ---
LLM thinks: The user is asking about Tokyo's population. I should search for current data rather than relying on my training data.
LLM action: calling 'search' with args {"query": "population of Tokyo 2026"}
Tool result: Tokyo's metropolitan area has approximately 37.4 million residents as of 2026

--- Iteration 2 ---
LLM responds: Based on my research: Tokyo's metropolitan area has approximately 37.4 million residents as of 2026 — this makes it one of the most populous cities in the world.

Agent completed in 2 iteration(s)
```

Two iterations for the tool-use case, one iteration for the direct response. That's the agent loop — simple, powerful, and the foundation of everything that follows.

---

## Key Takeaways

- **An agent is an LLM in a loop.** It perceives, reasons, acts, observes, and repeats until the task is done.
- **The chatbot-to-agent spectrum is gradual.** Most production systems aren't fully autonomous — they blend deterministic workflows with agentic decision-making.
- **Anthropic distinguishes workflows (predefined paths) from agents (dynamic control).** Start with workflows, add agency where the task demands it.
- **OpenAI's model is simple: model + tools + instructions.** Get those three right and the loop takes care of itself.
- **The agent loop is framework-agnostic.** Whether you use LangChain, LangGraph, CrewAI, or raw API calls, the underlying pattern is the same.
- **Always have a stopping condition.** An agent without a max iteration limit is a recipe for infinite loops and runaway API costs.

---

## What's Next

Now you know what an agent is. But how does it actually call tools? That `search_tool` function in our example was a mock — in real life, how does an LLM say "I want to call this function with these arguments"? Chapter 2 digs into the mechanics of tool use, the protocol that makes it work, and how to design tools that LLMs can use effectively.
