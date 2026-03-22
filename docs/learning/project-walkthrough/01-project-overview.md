# Project Overview — Why Multi-Agent?

We're building a travel optimizer — a tool that finds the best time to visit a destination by analyzing weather, flights, and hotels. We could build this as one script. Why use agents at all?

This chapter answers that question by working through the alternatives and explaining why the architecture we chose makes sense for this problem.

---

## The Problem

A user provides a destination (say "Tokyo, Japan"), a flexible date range ("June through September 2026"), and a trip duration (7 days). The system needs to:

1. Generate candidate travel windows across that range
2. Fetch weather forecasts for each window
3. Fetch flight prices for each window
4. Fetch hotel prices for each window
5. Normalize and rank the windows across all three dimensions
6. Explain the results in natural language

That is six distinct steps pulling from three external APIs (Open-Meteo for weather, Amadeus for flights and hotels) and one LLM (for the final explanation). The question is how to wire them together.

---

## Why NOT a Single Script

The most obvious approach is a Python script that runs steps 1 through 6 sequentially. Something like:

```python
windows = generate_windows(start, end, duration=7)
weather = [fetch_weather(w) for w in windows]
flights = [fetch_flights(origin, dest, w) for w in windows]
hotels  = [fetch_hotels(dest, w) for w in windows]
ranked  = score_and_rank(weather, flights, hotels)
print(summarize(ranked))
```

This would work. For a personal script you run once, it is fine. But it has real limitations:

**Partial failure kills everything.** If Amadeus is down (it has a 500-call/month free tier, so quota exhaustion is a real scenario), the entire script fails. You get nothing — not even the weather data that was perfectly available.

**Adding a data source means rewriting the pipeline.** Want to add a "local events" data source? You need to touch the main pipeline, the scoring function, and the output formatting. Every new source is a cross-cutting change.

**No parallelism without effort.** Weather, flights, and hotels are independent — they don't depend on each other's results. In the sequential script above, you wait for weather to finish before starting flights. That is wasted time. You could add `asyncio` or `concurrent.futures`, but then you are manually managing concurrency in your pipeline code.

A multi-agent architecture addresses all three. Each data source is an independent unit. If one fails, the others still complete. Adding a new source means adding one node and two edges. And the framework handles parallel execution for you.

---

## Why NOT a Single LLM Agent with Tools

The second alternative is giving a single LLM (like Claude or GPT-4) three tools — `fetch_weather`, `fetch_flights`, `fetch_hotels` — and letting it orchestrate the work. This is the pattern from [Foundations Chapter 02](../foundations/02-tools-giving-llms-hands.md) and [Chapter 04](../foundations/04-single-agent-architectures.md).

It would look something like:

```python
agent = create_agent(
    llm=claude,
    tools=[fetch_weather, fetch_flights, fetch_hotels, score_results],
    instructions="Find the best travel window..."
)
agent.run("Tokyo, June to September")
```

This approach has three problems for our use case:

**Sequential tool calls.** Most LLM agent loops call one tool at a time, wait for the result, then decide what to do next. That means weather, then flights, then hotels — serially. Our three API calls are independent and could run in parallel, saving 10-20 seconds per search.

**Expensive reasoning for deterministic work.** The LLM would use tokens deciding "I should call the weather tool next" and "now I need to call the flights tool." That reasoning is predictable. We know the execution order at design time. Paying an LLM to re-derive it at runtime is wasteful.

**Unpredictable execution.** The LLM might skip a data source, call one twice, or hallucinate results instead of calling the tool. For a system where we want consistent, reproducible searches, non-deterministic orchestration is a liability.

Our approach: use an LLM only where it adds value — the final synthesis step where natural language explanation genuinely benefits from language understanding. Everything else is deterministic Python.

---

## The Architecture Choice — Orchestrator with Fan-Out

The pattern we use maps directly to two patterns from [Foundations Chapter 05](../foundations/05-multi-agent-architectures.md): **orchestrator-workers** and **parallelization**.

A supervisor generates work (candidate windows). Three workers fetch data in parallel. A scorer ranks the results deterministically. A synthesizer explains the top picks using an LLM. Here is the full graph:

```
[User Input via Streamlit]
     │
[Supervisor] → generates 16 candidate windows
     │
     ├── [Weather Agent]  ── Open-Meteo (free, no LLM)
     ├── [Flight Agent]   ── Amadeus API (no LLM)
     └── [Hotel Agent]    ── Amadeus API (no LLM)
                               │
                        (parallel, ~10-30 seconds)
                               │
                     [Scorer] → deterministic ranking (no LLM)
                               │
                     [Synthesizer] → LLM explains top picks
                               │
                     [Streamlit UI]
```

This is implemented as a LangGraph `StateGraph` — a directed graph where nodes are Python functions and edges define execution order. When LangGraph sees three edges leaving the supervisor node, it runs all three target nodes concurrently. When it sees three edges entering the scorer node, it waits for all three to complete before running the scorer. The framework handles the fan-out and fan-in mechanics. We just declare the topology.

We will read this graph line by line in [Chapter 02](./02-the-graph.md).

---

## Key Design Decision: Minimize LLM Usage

Of the six nodes in our graph, only one uses an LLM — the synthesizer. This is deliberate.

| Node | Uses LLM? | Why / Why Not |
|------|-----------|---------------|
| Supervisor | No | Generates date windows with `timedelta` arithmetic. No ambiguity. |
| Weather Agent | No | Calls Open-Meteo API, scores with a formula. Deterministic. |
| Flight Agent | No | Calls Amadeus API, parses JSON prices. Deterministic. |
| Hotel Agent | No | Calls Amadeus API, parses JSON prices. Deterministic. |
| Scorer | No | Min-max normalization + weighted sum. Pure math. |
| Synthesizer | Yes | Turns ranked data into a human-readable recommendation. Language generation is what LLMs are for. |

This principle — "if you can write a function for it, don't use an LLM" — comes from [Foundations Chapter 07](../foundations/07-reliability-and-production.md). The result is a system that is cheaper (one LLM call per search instead of dozens), faster (no waiting for LLM reasoning between steps), and more predictable (same inputs produce same rankings every time).

The synthesizer itself has a fallback: if the LLM call fails, it returns the raw ranked data as formatted text. The user still gets results. This is the reliability principle in action — degrade gracefully, never fail silently.

---

## Key Takeaways

1. **Multi-agent is not always the answer.** A single script works for simple pipelines. We use agents here because we need partial failure tolerance, parallelism, and extensibility.

2. **A single LLM agent with tools is the wrong pattern when execution order is known at design time.** Reserve LLM orchestration for problems where the agent genuinely needs to decide what to do next.

3. **Minimize LLM usage.** Our system has 6 nodes and only 1 uses an LLM. Deterministic code is faster, cheaper, and more predictable.

4. **The orchestrator-with-fan-out pattern** (supervisor generates work, workers execute in parallel, results merge) is a natural fit for multi-source data aggregation problems.

5. **Graceful degradation is a first-class design goal.** Every agent handles its own failures. The system returns partial results rather than crashing.

Next: [Chapter 02 — The Graph](./02-the-graph.md), where we read `graph.py` line by line and trace state through the entire execution.
