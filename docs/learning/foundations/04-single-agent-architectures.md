# Chapter 4: Single Agent Architectures

You have an LLM with tools. But how does it decide what to do? There's more than one way to wire the decision loop, and the architecture you choose determines how your agent thinks.

In the last chapter we gave our agent tools — the ability to *do things* in the world. But we glossed over something important: the control flow. When the agent gets a task, what happens inside that `while` loop? Does it plan first? Does it reason out loud? Does it just start calling tools and hope for the best?

These aren't academic questions. The architecture you choose for your agent's decision loop has real consequences: how many tokens it burns, how debuggable it is, how often it gets stuck in loops, and whether it can handle genuinely complex tasks or only simple ones.

Let's walk through the major patterns, from simplest to most sophisticated. By the end, you'll know which one to reach for and — just as importantly — which ones to avoid for your use case.

---

## 1. The ReAct Pattern (Reasoning + Acting)

ReAct is the most widely discussed single-agent architecture. It comes from a 2022 paper by Yao et al. ("ReAct: Synergizing Reasoning and Acting in Language Models") and the core idea is beautifully simple: make the LLM *think out loud* before it acts.

The loop looks like this:

**Thought** → **Action** → **Observation** → **Thought** → **Action** → **Observation** → ... → **Final Answer**

At each step, the LLM explicitly writes out its reasoning ("I need to search for flights from Tokyo to Osaka"), then picks a tool to call, then observes the result, then reasons again about what to do next. The key word is *explicitly* — the reasoning is part of the output, not hidden inside the model's weights.

Here's the skeleton:

```python
while not done:
    # Thought: LLM reasons about what to do next
    thought = llm.invoke(f"Given {observations}, what should I do next?")

    # Action: LLM picks a tool and arguments
    action = llm.invoke(f"Based on: {thought}, which tool should I call?")

    # Observation: Execute the tool, get the result
    observation = execute_tool(action)

    # Check: Are we done?
    done = llm.invoke(f"Given {observation}, do I have enough info to answer?")
```

In practice, you don't usually make four separate LLM calls per iteration. Most implementations use a single prompt that asks the LLM to produce its thought *and* action in one go, formatted in a predictable way (often with `Thought:` and `Action:` prefixes). But conceptually, these are the four stages.

**Why is this useful?** Because the explicit reasoning trace gives you two things. First, debuggability — when the agent does something wrong, you can read its thoughts and see *where* the reasoning went off the rails. Second, better performance on complex tasks — by forcing the model to "show its work," you get the same benefits as chain-of-thought prompting, but integrated into the action loop.

**The downsides are real, though.** ReAct is verbose. All those `Thought:` blocks cost tokens, and on simple tasks ("what's the weather in Paris?"), the overhead is wasteful. The model already knows it should call the weather tool — forcing it to write a paragraph explaining *why* is busywork.

Worse, ReAct agents can get into reasoning loops. The model writes "I should search for X," searches, gets a result, then writes "Hmm, I'm not sure about this, let me search for X again" — and you're burning tokens going in circles. You need a max-iterations guard or the model will happily reason its way into your cloud bill.

**When to use ReAct:** Complex multi-step research tasks, where the model genuinely needs to reason about which information to gather next. Tasks where auditability matters — you want a log of *why* each action was taken.

**When to avoid ReAct:** Simple tool calls, real-time applications, or cost-sensitive workloads where implicit reasoning is sufficient.

---

## 2. Tool-Use Agent (The "Just Call Tools" Pattern)

This is what most production agents actually use. No explicit reasoning step — the LLM directly decides whether to call a tool or respond with text. The "reasoning" is implicit, happening inside the model's forward pass rather than written out as text.

If you've used OpenAI's function calling or Anthropic's tool use, you've already built this pattern. The API handles the structure:

```python
while not done:
    response = llm.invoke(messages, tools=available_tools)
    if response.has_tool_call:
        result = execute_tool(response.tool_call)
        messages.append(tool_result(result))
    else:
        done = True  # LLM responded with text — it's done
```

That's it. The loop is: send messages and tool definitions to the LLM. If it responds with a tool call, execute it, append the result, and loop. If it responds with text, you're done. The model figures out the "when" and "what" of tool usage entirely on its own.

This is elegant because it uses native API features. The tool schemas are passed as structured definitions (usually JSON Schema), and the model returns structured tool calls — not freeform text you need to parse. No regex extraction, no brittle parsing of `Action: search("flights to tokyo")` strings.

```python
# Tools are defined as structured schemas
tools = [
    {
        "name": "search_flights",
        "description": "Search for available flights",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string", "format": "date"}
            },
            "required": ["origin", "destination", "date"]
        }
    }
]

# The LLM returns structured tool calls, not freeform text
response = llm.invoke(messages, tools=tools)
# response.tool_calls = [{"name": "search_flights", "arguments": {"origin": "NRT", ...}}]
```

**The tradeoff:** You lose the reasoning trace. When the agent calls `search_flights` instead of `search_hotels`, you can't see *why* from the output alone. For debugging, you're left staring at a sequence of tool calls trying to reverse-engineer the model's logic. This is fine for simple workflows but frustrating for complex ones.

Some teams add a lightweight reasoning step by including an instruction like "before calling a tool, briefly explain your reasoning" in the system prompt. This gives you a hybrid — mostly tool-use agent, but with optional reasoning. It's a pragmatic middle ground that many production systems use.

**When to use:** API integrations, clear tool boundaries, production systems where you want simplicity and native API features. This is your default starting point.

**When to avoid:** Tasks where you need an explicit audit trail of reasoning, or where the model frequently makes wrong tool choices and you need to debug why.

---

## 3. Plan-then-Execute

Sometimes the agent shouldn't be figuring out what to do *while doing it*. Some tasks have a clear structure: search for flights, search for hotels, compare prices, recommend the best option. Why let the agent discover this step-by-step when you (or the agent) could plan the whole sequence upfront?

Plan-then-Execute separates thinking from doing:

```python
# Phase 1: Planning — the LLM creates a full plan
plan = llm.invoke("Create a step-by-step plan to: {task}")
# plan = ["1. Search flights NYC→London", "2. Search hotels in London",
#         "3. Compare total costs", "4. Recommend best option"]

# Phase 2: Execution — work through the plan
results = {}
for step in plan:
    results[step] = execute_step(step)  # may use tools, may use LLM
```

The key difference from ReAct: the plan is made *upfront*, before any tools are called. The model thinks about the whole task, decides on a sequence, and then executes it. In ReAct, each step is decided only after seeing the result of the previous step.

Anthropic's "Building Effective Agents" blog calls this **prompt chaining** when the steps are predefined (you hard-code the sequence) and reserves more dynamic planning for when the LLM generates the steps. The distinction matters:

```python
# Prompt chaining (predefined steps — you wrote the plan)
def handle_travel_request(request):
    flights = search_flights(request)        # Step 1: always search flights
    hotels = search_hotels(request)          # Step 2: always search hotels
    comparison = compare_options(flights, hotels)  # Step 3: always compare
    return format_recommendation(comparison)  # Step 4: always format

# Plan-then-Execute (LLM writes the plan)
def handle_request(request):
    plan = llm.invoke(f"Plan how to handle: {request}")
    for step in plan:
        execute(step)  # LLM-generated steps, less predictable
```

Prompt chaining is simpler and more reliable — you control the flow. True plan-then-execute gives the LLM more autonomy at the cost of predictability.

**The big weakness:** the plan might be wrong. The agent plans to search for flights, but what if the search returns zero results and the plan doesn't account for fallbacks? ReAct adapts naturally ("no flights found, let me try different dates"), but plan-then-execute charges ahead with a potentially broken plan.

Some implementations add re-planning: after each step, check if the plan still makes sense and revise if needed. This makes the pattern more robust but also more complex — and at some point, you're basically reinventing ReAct with extra steps.

**When to use:** Well-understood tasks with clear structure. Workflows where you want predictable execution. Tasks where the steps are largely independent (so a wrong step doesn't cascade).

**When to avoid:** Exploratory tasks, tasks where the next step depends heavily on previous results, or tasks where you can't predict the structure upfront.

---

## 4. Reflection / Self-Correction

This isn't a standalone architecture — it's a *meta-pattern* you layer on top of any of the above. The idea: after the agent produces output, have an LLM (possibly the same one, possibly a different one) critique the output and retry if it's not good enough.

```python
output = agent.run(task)
critique = llm.invoke(f"Is this output correct and complete? {output}")
if "no" in critique.lower():
    output = agent.run(task, feedback=critique)  # retry with feedback
```

Anthropic calls the formalized version of this the **evaluator-optimizer** pattern: one LLM generates output, another evaluates it, and the cycle repeats until the evaluator is satisfied (or you hit a max iteration count).

```python
max_retries = 3
for attempt in range(max_retries):
    output = generator.invoke(task, previous_feedback=feedback)
    evaluation = evaluator.invoke(f"Rate this output 1-10 and explain issues: {output}")

    if evaluation.score >= 8:
        break  # Good enough
    feedback = evaluation.issues  # Feed issues back to generator
```

The pattern is powerful because LLMs are often better at *judging* output than *producing* it on the first try. Think about how you write code: your first draft works, but a review catches edge cases you missed. Same principle.

**The critical gotcha:** without a bounded retry count, reflection can loop forever. The evaluator says "not quite right," the generator tweaks something, the evaluator says "better but still not right," and you're stuck in an infinite improvement loop while your token bill climbs. Always set `max_retries`. Three is a good starting point — if three attempts don't produce good output, the problem is likely in your prompt or tool design, not in the number of retries.

**When to use:** Quality-critical outputs — code generation, customer-facing text, anything where getting it wrong has high cost. Tasks where the output is easy to evaluate programmatically (code that needs to pass tests, JSON that needs to match a schema).

**When to avoid:** Cost-sensitive applications (reflection roughly doubles your token usage), real-time applications where latency matters, or tasks where the evaluation criteria are vague.

---

## 5. Routing — The Dispatcher Pattern

Routing isn't a full architecture — it's a decision point. An LLM classifies the input and sends it to the appropriate handler. It's the "receptionist" pattern: figure out who should handle this request, then hand it off.

```python
category = llm.invoke(
    "Classify this request into one category: billing, technical, general\n"
    f"Request: {user_input}"
)

if category == "billing":
    handle_billing(user_input)
elif category == "technical":
    handle_technical(user_input)
else:
    handle_general(user_input)
```

Anthropic identifies routing as one of the fundamental building blocks in their "Building Effective Agents" blog. It's simple, cheap (one short LLM call for classification), and lets you build specialized handlers for each category without stuffing everything into one mega-prompt.

The handlers themselves can be anything — another LLM call with a specialized prompt, a traditional function, another agent with its own tools. Routing is about *decomposition at the entry point*.

**A subtlety:** routing doesn't have to be LLM-based. For well-defined categories, a simple keyword match or embedding similarity search might work better and cost zero tokens. Use the LLM for routing only when the classification is genuinely ambiguous and requires understanding.

```python
# Sometimes you don't need an LLM for routing
def simple_router(user_input):
    if any(word in user_input.lower() for word in ["bill", "charge", "payment", "refund"]):
        return "billing"
    elif any(word in user_input.lower() for word in ["error", "bug", "crash", "broken"]):
        return "technical"
    return "general"
```

**When to use:** Multi-domain systems where different inputs need fundamentally different handling. When you want to keep each handler's prompt focused and small.

**When to avoid:** Single-purpose agents that only do one thing. If everything goes to the same handler, the routing step is pure overhead.

---

## 6. Deciding Which Pattern to Use

Here's the decision framework. Memorize this table — it'll save you from over-engineering.

| Pattern | Best For | Avoid When |
|---------|----------|------------|
| **ReAct** | Complex exploration, research, multi-step reasoning | Simple lookups, cost-sensitive apps |
| **Tool-use** | API integrations, clear tool boundaries, production systems | Tasks needing explicit reasoning traces |
| **Plan-Execute** | Multi-step known workflows, predictable tasks | Unpredictable tasks, tasks needing adaptation |
| **Reflection** | Quality-critical outputs, code generation | Cost-sensitive, real-time applications |
| **Routing** | Multi-domain input handling, large systems | Single-purpose agents |

Both OpenAI and Anthropic converge on the same advice. OpenAI's "Practical Guide to Building Agents" says it plainly: **"Start with a single agent. Add complexity only when you have clear evidence it's needed."** Anthropic echoes this in "Building Effective Agents": **"When building agents, try the simplest solution first."** They recommend prompt chaining and routing as the simplest building blocks, graduating to more complex patterns only when simpler ones fail.

The implication is clear: your default should be a tool-use agent. When that's not enough, add explicit reasoning (ReAct). When that's not enough, add planning. When that's not enough, add reflection. Don't start at the complex end.

One more thing worth noting: these patterns compose. A ReAct agent can use routing to dispatch sub-tasks. A plan-then-execute agent can use reflection to validate each step's output. A tool-use agent can have a reflection layer on its final answer. The patterns aren't mutually exclusive — they're building blocks.

---

## Runnable Example: A ReAct-Style Research Agent

Let's put it all together. Here's a complete, runnable ReAct agent that answers multi-part questions by searching (with a mock search tool), reasoning about what it found, and deciding when it has enough information to synthesize a final answer.

```python
"""
ReAct-style Research Agent
==========================
Demonstrates the Thought → Action → Observation loop.
Uses a mock search tool (no API keys needed).

Run: python 04_react_agent.py
"""

import json
import re

# ---------------------------------------------------------------------------
# Mock search tool — simulates a web search with canned results.
# In a real agent, this would call a search API (Tavily, SerpAPI, etc.).
# ---------------------------------------------------------------------------
MOCK_SEARCH_DB = {
    "largest cities in japan": (
        "The three largest cities in Japan by population are: "
        "Tokyo (approx. 13.96 million), Yokohama (approx. 3.75 million), "
        "and Osaka (approx. 2.75 million). These figures refer to city proper "
        "population, not metro area."
    ),
    "population of tokyo": (
        "Tokyo has a population of approximately 13.96 million people "
        "(city proper, 2023 estimate). The Greater Tokyo Area has about "
        "37.4 million, making it the most populous metropolitan area in the world."
    ),
    "population of yokohama": (
        "Yokohama has a population of approximately 3.75 million people "
        "(2023 estimate). It is located in Kanagawa Prefecture, just south of Tokyo."
    ),
    "population of osaka": (
        "Osaka has a population of approximately 2.75 million people "
        "(city proper, 2023 estimate). The Osaka metropolitan area (Keihanshin) "
        "has about 19 million."
    ),
}


def mock_search(query: str) -> str:
    """
    Simulate a web search. Looks for partial matches in our mock database.
    Returns the best matching result, or a 'no results' message.
    """
    query_lower = query.lower()
    for key, value in MOCK_SEARCH_DB.items():
        if key in query_lower or query_lower in key:
            return value
    # Try partial word matching as a fallback
    for key, value in MOCK_SEARCH_DB.items():
        if any(word in query_lower for word in key.split()):
            return value
    return f"No results found for: {query}"


# ---------------------------------------------------------------------------
# Simulated LLM — a rule-based stand-in for a real language model.
# This lets us demonstrate the ReAct loop without needing an API key.
# In production, replace these functions with actual LLM calls.
# ---------------------------------------------------------------------------
class MockReActLLM:
    """
    Simulates an LLM that follows the ReAct pattern.

    In a real system, each method here would be a call to an LLM API
    (e.g., openai.chat.completions.create or anthropic.messages.create).
    The LLM would receive the full conversation history and produce
    its thought, action, or final answer.
    """

    def __init__(self):
        # Track the agent's internal state across steps
        self.step = 0
        self.cities_found = False
        self.populations = {}

    def think(self, question: str, observations: list[dict]) -> str:
        """
        Generate a 'Thought' — the agent's reasoning about what to do next.

        In a real implementation:
            thought = llm.invoke(
                f"Question: {question}\n"
                f"Previous observations: {observations}\n"
                "What should I do next? Think step by step."
            )
        """
        self.step += 1

        if self.step == 1:
            return (
                "I need to find the 3 largest cities in Japan first, "
                "then get the population of each one. Let me start by "
                "searching for the largest cities."
            )
        elif self.step == 2:
            return (
                "I found the three largest cities: Tokyo, Yokohama, and Osaka. "
                "I already have their populations from the search result. "
                "Let me verify Tokyo's population with a dedicated search."
            )
        elif self.step == 3:
            return (
                "Good, Tokyo's population is confirmed at ~13.96 million. "
                "Let me also verify Yokohama's population."
            )
        elif self.step == 4:
            return (
                "Yokohama confirmed at ~3.75 million. Let me verify Osaka too "
                "for completeness."
            )
        else:
            return (
                "I now have verified populations for all three cities. "
                "I have enough information to provide a final answer."
            )

    def decide_action(self, thought: str) -> dict | None:
        """
        Based on the thought, decide whether to call a tool or finish.

        Returns a tool call dict, or None if the agent should produce
        a final answer.

        In a real implementation:
            response = llm.invoke(
                messages=[...],
                tools=[{"name": "search", "parameters": {...}}]
            )
            if response.has_tool_call:
                return {"tool": response.tool_call.name, "args": response.tool_call.args}
            return None
        """
        if self.step == 1:
            return {"tool": "search", "query": "largest cities in Japan"}
        elif self.step == 2:
            return {"tool": "search", "query": "population of Tokyo"}
        elif self.step == 3:
            return {"tool": "search", "query": "population of Yokohama"}
        elif self.step == 4:
            return {"tool": "search", "query": "population of Osaka"}
        else:
            return None  # No more tool calls — time for the final answer

    def synthesize(self, question: str, observations: list[dict]) -> str:
        """
        Produce the final answer from all gathered observations.

        In a real implementation:
            final = llm.invoke(
                f"Based on these observations, answer the question.\n"
                f"Question: {question}\n"
                f"Observations: {observations}"
            )
        """
        return (
            "The three largest cities in Japan and their populations are:\n"
            "1. Tokyo — approximately 13.96 million people\n"
            "2. Yokohama — approximately 3.75 million people\n"
            "3. Osaka — approximately 2.75 million people\n\n"
            "These are city proper populations (2023 estimates). "
            "Note that metropolitan area populations are significantly larger — "
            "Greater Tokyo alone has about 37.4 million people."
        )


# ---------------------------------------------------------------------------
# The ReAct loop — this is the core of the pattern.
# ---------------------------------------------------------------------------
def react_agent(question: str, max_steps: int = 10) -> str:
    """
    Run a ReAct-style agent loop.

    The loop follows this cycle:
        1. THOUGHT  — the agent reasons about what to do
        2. ACTION   — the agent picks a tool to call (or decides it's done)
        3. OBSERVATION — the tool result is recorded

    The loop continues until the agent decides it has enough information
    (returns no action) or we hit max_steps (a safety limit).

    Args:
        question: The user's question to answer.
        max_steps: Maximum number of think-act-observe cycles (prevents infinite loops).

    Returns:
        The agent's final synthesized answer.
    """
    llm = MockReActLLM()
    observations = []  # Stores all (action, result) pairs

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    for step_num in range(1, max_steps + 1):
        # ---- THOUGHT ----
        # The agent examines the question and all observations so far,
        # then reasons about what to do next.
        thought = llm.think(question, observations)
        print(f"\n--- Step {step_num} ---")
        print(f"THOUGHT: {thought}")

        # ---- ACTION ----
        # Based on its reasoning, the agent either picks a tool to call
        # or decides it has enough information to answer.
        action = llm.decide_action(thought)

        if action is None:
            # The agent is done gathering information.
            print("ACTION: None (ready to synthesize final answer)")
            break

        print(f"ACTION: search(\"{action['query']}\")")

        # ---- OBSERVATION ----
        # Execute the chosen tool and record the result.
        # The observation is added to the agent's memory for future steps.
        result = mock_search(action["query"])
        observations.append({
            "step": step_num,
            "action": action,
            "result": result
        })
        print(f"OBSERVATION: {result}")

    else:
        # We hit max_steps without the agent finishing.
        # This is the safety net against infinite reasoning loops.
        print(f"\nWARNING: Hit max steps ({max_steps}). Forcing synthesis.")

    # ---- SYNTHESIS ----
    # Combine all observations into a final answer.
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(f"{'='*60}")
    answer = llm.synthesize(question, observations)
    print(answer)
    return answer


# ---------------------------------------------------------------------------
# Run it!
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    react_agent("What's the population of the 3 largest cities in Japan?")
```

Run this and you'll see the full ReAct trace: the agent's reasoning at each step, the tool calls it makes, the observations it collects, and the final synthesized answer. In a real system, the `MockReActLLM` class would be replaced with actual LLM API calls, but the loop structure — think, act, observe, repeat — stays exactly the same.

---

## Key Takeaways

1. **ReAct** gives you explicit reasoning traces but costs more tokens. Use it for complex research and debugging-heavy workflows.

2. **Tool-use agents** are the pragmatic default. Native API support, simple loop, no parsing headaches. Start here.

3. **Plan-then-Execute** works when you know the task structure upfront. Great for workflows, fragile for exploration.

4. **Reflection** is a meta-pattern — layer it on top of anything when quality matters more than cost.

5. **Routing** is the entry point for multi-domain systems. Cheap, effective, often overlooked.

6. **Start simple, add complexity with evidence.** Both OpenAI and Anthropic agree: a single tool-use agent should be your first attempt. Graduate to fancier patterns only when the simple one demonstrably fails.

7. **These patterns compose.** A production agent might use routing at the entry, plan-then-execute for structured tasks, ReAct for exploratory sub-tasks, and reflection for quality-critical outputs. Don't think of them as mutually exclusive choices.

---

## What's Next

Single agents are powerful but have limits. What happens when one agent needs to handle too many tools, too many domains, or tasks that naturally decompose into parallel work streams? Chapter 5 explores what happens when you need multiple agents working together — and the architectural patterns that make that work without descending into chaos.
