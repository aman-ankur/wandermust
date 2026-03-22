# Chapter 7: Reliability & Production Patterns

Your agent works perfectly in your demo. The API responds instantly, the LLM never hallucinates, and everything runs in under 2 seconds. Now deploy it. The API goes down. The LLM returns garbage. Your rate limit is hit. Welcome to production.

The gap between "works on my laptop" and "works reliably for real users" is enormous in agent systems. Traditional software has well-understood failure modes: network timeouts, database errors, input validation failures. Agents inherit all of those and add new ones: LLM hallucinations, tool misuse, cascading failures across multiple agents, and nondeterministic behavior that makes bugs almost impossible to reproduce.

This chapter covers the patterns that bridge that gap. These aren't theoretical — they're the exact patterns you'll need the first time your agent fails in a way you didn't anticipate. And it will.

---

## Error Handling — Graceful Degradation

The most important principle in production agent systems: **every component can fail independently without bringing down the whole pipeline.**

In a traditional application, an unhandled exception crashes the request. In a multi-agent system, a crash in one agent can cascade — downstream agents receive no data, produce garbage outputs, and the entire pipeline returns nothing useful. The fix is surprisingly simple: each node catches its own errors and flags them in state.

```python
def flights_node(state):
    """
    Fetch flight data. If the API fails, record the error
    and return empty data so downstream nodes can still work.
    """
    # Preserve existing errors — don't overwrite them
    errors = list(state.get("errors", []))
    try:
        data = fetch_flights(state["destination"])
        return {"flight_data": data, "errors": errors}
    except Exception as e:
        # Log the error, but don't crash
        errors.append(f"Flights unavailable: {e}")
        return {"flight_data": [], "errors": errors}
    # The scorer still works with weather + hotels, even if flights failed
```

The key insight is that `flight_data: []` is a valid state. The scorer node downstream doesn't crash when it encounters an empty list — it just produces a result without flight information. The user gets a recommendation based on weather and hotel data, plus a message saying "flight data was temporarily unavailable." That's infinitely better than a 500 error.

This pattern — catch, log, return empty — should be your default for every node that touches an external system. APIs go down. Rate limits get hit. DNS fails. Your agent should handle all of these without a stack trace.

Anthropic's "Building Effective Agents" emphasizes this point in the context of tool use: agents will encounter tool failures, and the system needs to recover gracefully rather than halt. Design for failure from the start, not as an afterthought.

---

## Fallbacks — Degraded but Functional

Error handling prevents crashes. Fallbacks prevent empty results. The idea: when your primary data source fails, fall back to progressively less accurate but more reliable alternatives.

```python
def get_price(city, date):
    """
    Try multiple data sources in order of accuracy.
    Live API → cached data → historical data → None.
    Each level is less accurate but more reliable.
    """
    try:
        return api.fetch_live(city, date)        # Best: real-time price
    except APIError:
        pass

    cached = cache.get(f"{city}:{date}")
    if cached:
        return cached                             # Good: recent cache hit

    historical = db.get_similar(city, date)
    if historical:
        return historical                         # OK: historical average

    return None                                   # Flag as unavailable
```

This creates a reliability hierarchy. In the best case, you get real-time data. In the worst case, you get `None` — but even that is useful because you can tell the user "we couldn't get pricing for this destination" instead of crashing silently.

The same pattern applies to LLM calls. If your primary model (GPT-4, Claude) fails or times out, fall back to a faster, cheaper model. The response quality might be lower, but a slightly-less-good response beats no response.

The depth of your fallback chain depends on how critical the data is. For a travel recommendation, missing flight prices is annoying but manageable. For a medical diagnosis agent, missing data might mean you shouldn't return a result at all. Match your fallback strategy to your risk tolerance.

---

## Retries with Backoff

Not all failures are permanent. APIs recover. Rate limits reset. Network blips resolve themselves. Retrying a failed call after a short delay often succeeds.

But naive retries are dangerous. If you retry immediately and the server is overloaded, you're making the problem worse. If 1,000 clients all retry at the same time, you create a "thundering herd" that can take down the server entirely.

The solution is exponential backoff with jitter:

```python
import time
import random

def retry_with_backoff(fn, max_retries=3):
    """
    Retry a function with exponential backoff and jitter.

    Attempt 0: immediate
    Attempt 1: wait ~1-2 seconds
    Attempt 2: wait ~2-3 seconds
    Attempt 3: give up and raise

    The jitter (random noise) prevents thundering herd —
    multiple clients won't all retry at the exact same moment.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt failed — give up and let the caller handle it
                raise
            # Exponential wait: 1s, 2s, 4s, 8s...
            # Plus random jitter: 0-1 seconds of noise
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
```

The exponential part (`2 ** attempt`) means each retry waits longer than the last, giving the server more time to recover. The jitter (`random.uniform(0, 1)`) spreads out retry timing across clients so they don't all hit the server simultaneously.

A practical tip: set `max_retries` based on the operation. For an LLM call that costs money, 2-3 retries is reasonable. For a cheap health-check ping, you might retry 5-10 times. For an idempotent write operation, be extra careful — retrying a non-idempotent operation can cause duplicate actions.

---

## Caching — Avoid Redundant Work

Every external call your agent makes — API requests, LLM prompts, database queries — takes time and often costs money. Caching stores results so identical requests are served from memory instead of being recomputed.

**In-memory TTL cache:** Fast, simple, lost on restart. Perfect for caching API responses during a single user session.

```python
import time

class TTLCache:
    """
    Simple in-memory cache with time-to-live expiration.
    Entries are automatically invalid after `ttl` seconds.
    """
    def __init__(self, ttl=300):  # 5-minute default
        self.ttl = ttl
        self.store = {}

    def get(self, key):
        if key in self.store:
            value, timestamp = self.store[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.store[key]  # Expired — clean up
        return None

    def set(self, key, value):
        self.store[key] = (value, time.time())
```

**Persistent cache (SQLite, Redis):** Survives restarts. Good for historical data and expensive LLM results that don't change often.

**Cache key design** matters enormously. The key `f"{origin}:{destination}:{date}"` is specific enough to avoid stale data (different dates get different prices) and general enough to get cache hits (same route on the same date always returns the same cached result).

**When to cache:** API responses, LLM calls with identical prompts, expensive computations, data that changes slowly (weather forecasts, hotel availability).

**When NOT to cache:** Real-time data where staleness is dangerous (stock prices for trading), security-sensitive responses (authentication tokens), or anything where the freshness of data is more important than speed.

A particularly valuable pattern for agents: **cache LLM responses for identical prompts.** If your classification agent receives the same query twice, there's no reason to pay for a second LLM call. The response will be the same (or close enough). This alone can cut LLM costs by 20-40% in production systems with repetitive queries.

---

## Guardrails — Input and Output Validation

Guardrails are checks that run before and after your agent to catch problems early. OpenAI's "Practical Guide to Building Agents" emphasizes guardrails as a core component of any production agent system, not an afterthought.

**Input guardrails** validate before the agent runs. They catch bad data before it wastes LLM tokens and API calls:

```python
def validate_input(state):
    """
    Check input data BEFORE the agent pipeline runs.
    Fail fast with a clear error message.
    """
    if not state.get("destination"):
        raise ValueError("Destination is required")
    if state.get("duration_days", 0) < 1:
        raise ValueError("Duration must be at least 1 day")
    if state.get("budget", 0) <= 0:
        raise ValueError("Budget must be a positive number")
    # Prevent prompt injection: reject suspiciously long inputs
    if len(state.get("destination", "")) > 200:
        raise ValueError("Destination input is too long")
```

**Output guardrails** validate what the agent produces. They catch hallucinations and formatting errors before results reach the user:

```python
def validate_output(result):
    """
    Check agent output BEFORE returning to the user.
    Catch impossible values, missing fields, and format errors.
    """
    if not result.get("ranked_windows"):
        raise ValueError("Agent produced no recommendations")
    for window in result["ranked_windows"]:
        assert window["total_score"] >= 0, "Score can't be negative"
        assert window["total_score"] <= 1, "Score can't exceed 1"
        assert "start_date" in window, "Missing start date"
        assert "end_date" in window, "Missing end date"
```

Input guardrails prevent wasted work. Output guardrails prevent bad results from reaching users. Together, they form a safety envelope around your agent. Anthropic's guide notes that agents operating in loops can compound errors — a small hallucination in step 2 becomes a catastrophic failure by step 5. Guardrails at each stage prevent this compounding.

---

## When to Use LLM vs. Deterministic Code

This is perhaps the most impactful cost and reliability decision you'll make. Every task your agent performs falls on a spectrum from "requires understanding and judgment" to "can be expressed as a formula."

**Use an LLM for:** understanding natural language queries, generating human-readable text, making subjective judgments ("is this a good travel time?"), handling ambiguity ("cheap" means different things to different people), and summarizing complex information.

**Use deterministic code for:** math (scoring, averaging, ranking), sorting and filtering, API calls, data validation, format conversion, and anything with a clear algorithmic solution.

The rule of thumb: **if you can write a function for it, don't use an LLM.** Deterministic code is faster (microseconds vs. seconds), cheaper (free vs. per-token pricing), more predictable (same input always gives same output), and easier to test.

In our travel optimizer project, this split is clear. The Scorer is entirely deterministic — it takes weather data, flight prices, and event information, applies weighted formulas, and produces a ranked list. There's no ambiguity in "which week has the best combined score." The Synthesizer, on the other hand, uses an LLM because it needs to generate a natural-language explanation of why a particular travel window is recommended. That's a judgment call that requires language understanding.

Every LLM call you replace with deterministic code makes your system faster, cheaper, and more reliable. Audit your agent pipeline and ask: "Does this step truly need language understanding, or am I using an LLM because it was convenient?"

---

## Cost Management

Every LLM call costs money. In a multi-agent system with 5 agents, each making 2-3 LLM calls per request, costs add up fast. A few principles:

**Cache identical prompts.** If your classification agent sees the same query twice, serve the cached result. This is the single easiest cost reduction.

**Use tiered models.** Not every task needs GPT-4 or Claude Opus. Use a cheap, fast model (GPT-4o-mini, Claude Haiku) for classification, extraction, and simple routing. Reserve expensive models for complex reasoning and generation.

**Minimize prompt size.** Every token in your prompt costs money. Strip unnecessary context. Use concise system prompts. Don't send the entire conversation history when the agent only needs the last message.

**Set budgets and alerts.** Most LLM providers offer usage dashboards. Set a monthly budget. Alert when you hit 80%. Review which agents consume the most tokens and optimize them first.

---

## Observability — Debugging Multi-Agent Systems

When a single-agent system produces a bad result, debugging is straightforward: look at the prompt and the response. When a 5-agent pipeline produces a bad result, the bug could be anywhere — bad input parsing, a failed API call, a hallucination in agent 3 that corrupted state for agent 4, or a scoring error in the final step.

You need visibility into every step:

- **Log every node's input and output.** When something goes wrong, you need to see exactly what state each node received and what it returned.
- **Log tool calls with arguments and results.** If an agent called a tool with wrong arguments, you need to see that.
- **Trace the execution path.** In a graph with conditional edges, which path did execution actually take? Did it loop? Did it take the fallback branch?
- **Timestamp everything.** Performance problems are invisible without timing data. If your pipeline takes 30 seconds, you need to know which node took 25 of them.

LangSmith (LangGraph's companion tracing tool) provides all of this out of the box — every node execution, every LLM call, every state mutation, visualized as a timeline. If you're using LangGraph in production, LangSmith is worth the setup.

Even without a tracing tool, disciplined logging goes a long way. A simple pattern:

```python
import logging
logger = logging.getLogger("agent_pipeline")

def my_node(state):
    logger.info(f"[my_node] Input state keys: {list(state.keys())}")
    logger.info(f"[my_node] destination={state.get('destination')}")
    # ... do work ...
    logger.info(f"[my_node] Output: {result}")
    return result
```

Anthropic's guide is emphatic on this point: *"Test in realistic scenarios. Edge cases in agents often emerge from the LLM's interpretation of ambiguous inputs."* You can't test what you can't see. Observability isn't a nice-to-have — it's how you find the bugs that only appear in production.

---

## Runnable Example: A Resilient Agent Pipeline

Let's build a complete example that demonstrates every reliability pattern from this chapter. The scenario: an agent that fetches data from an unreliable source (fails 50% of the time), retries with backoff, falls back to cached data, accumulates errors, and returns partial results gracefully.

```python
"""
Resilient Agent Pipeline
=========================

Demonstrates production reliability patterns:
  - Retry with exponential backoff
  - Fallback to cached data
  - Error accumulation in state
  - Graceful degradation (partial results)

No external dependencies — runs with just the Python standard library.
"""

import random
import time
from typing import Optional


# ──────────────────────────────────────────────
# Simulated infrastructure
# ──────────────────────────────────────────────

# A simple in-memory cache with TTL
class TTLCache:
    """Cache that expires entries after a set number of seconds."""

    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.store: dict = {}

    def get(self, key: str) -> Optional[dict]:
        if key in self.store:
            value, timestamp = self.store[key]
            if time.time() - timestamp < self.ttl:
                print(f"    [cache] HIT for '{key}'")
                return value
            print(f"    [cache] EXPIRED for '{key}'")
            del self.store[key]
        else:
            print(f"    [cache] MISS for '{key}'")
        return None

    def set(self, key: str, value: dict):
        self.store[key] = (value, time.time())
        print(f"    [cache] STORED '{key}'")


# Global cache instance — shared across all agents
cache = TTLCache(ttl=600)

# Pre-populate the cache with some data so fallbacks work
cache.set("weather:tokyo", {"temp": 18, "condition": "partly cloudy", "source": "cache"})
cache.set("flights:tokyo", {"price": 850, "airline": "ANA", "source": "cache"})
cache.set("hotels:tokyo", {"price": 120, "name": "Shinjuku Inn", "source": "cache"})


# ──────────────────────────────────────────────
# Unreliable API simulator
# ──────────────────────────────────────────────

class UnreliableAPI:
    """
    Simulates an API that fails ~50% of the time.
    This is deliberately flaky to demonstrate retry and fallback patterns.
    """

    @staticmethod
    def fetch(endpoint: str) -> dict:
        # 50% chance of failure
        if random.random() < 0.5:
            raise ConnectionError(f"API timeout on {endpoint}")

        # Simulate some latency
        time.sleep(0.1)

        # Return "live" data (better than cached data)
        responses = {
            "weather": {"temp": 22, "condition": "sunny", "source": "live"},
            "flights": {"price": 780, "airline": "JAL", "source": "live"},
            "hotels": {"price": 95, "name": "Shibuya Grand", "source": "live"},
        }
        return responses.get(endpoint, {"source": "live", "data": "unknown"})


api = UnreliableAPI()


# ──────────────────────────────────────────────
# Retry with exponential backoff
# ──────────────────────────────────────────────

def retry_with_backoff(fn, max_retries: int = 3, label: str = ""):
    """
    Retry a function with exponential backoff and jitter.

    Args:
        fn: A callable that takes no arguments and returns a result.
        max_retries: Maximum number of attempts before giving up.
        label: A label for logging (e.g., "weather API").

    Returns:
        The result of fn() if successful.

    Raises:
        The last exception if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            result = fn()
            print(f"    [retry] {label} succeeded on attempt {attempt + 1}")
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    [retry] {label} FAILED after {max_retries} attempts: {e}")
                raise
            # Exponential backoff: 0.5s, 1s, 2s... (shortened for demo)
            # Plus jitter to prevent thundering herd
            wait = (2 ** attempt) * 0.3 + random.uniform(0, 0.2)
            print(f"    [retry] {label} attempt {attempt + 1} failed: {e}. "
                  f"Waiting {wait:.2f}s...")
            time.sleep(wait)


# ──────────────────────────────────────────────
# Data fetching with full resilience stack:
#   1. Try live API with retries
#   2. Fall back to cache
#   3. Return None if everything fails
# ──────────────────────────────────────────────

def fetch_with_fallback(endpoint: str, destination: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Fetch data with the full resilience stack.

    Returns:
        (data, error) — data is the result (or None), error is a message (or None).
        At most one of these will be None.
    """
    cache_key = f"{endpoint}:{destination}"

    # Level 1: Try the live API with retries
    try:
        data = retry_with_backoff(
            lambda: api.fetch(endpoint),
            max_retries=2,
            label=endpoint,
        )
        # Success! Cache the result for future fallback
        cache.set(cache_key, data)
        return data, None
    except Exception:
        pass  # All retries exhausted — try cache

    # Level 2: Fall back to cached data
    cached = cache.get(cache_key)
    if cached:
        return cached, f"{endpoint} API failed, using cached data"

    # Level 3: Nothing available
    return None, f"{endpoint} data completely unavailable"


# ──────────────────────────────────────────────
# Agent nodes (simulating a graph pipeline)
# ──────────────────────────────────────────────

def input_validator(state: dict) -> dict:
    """
    INPUT GUARDRAIL: Validate before any expensive work happens.
    """
    print("\n[input_validator] Checking inputs...")
    destination = state.get("destination", "").strip()

    if not destination:
        raise ValueError("Destination is required")
    if len(destination) > 100:
        raise ValueError("Destination name suspiciously long — possible injection")

    print(f"[input_validator] Destination '{destination}' is valid")
    return {"destination": destination, "errors": []}


def weather_agent(state: dict) -> dict:
    """Fetch weather data with full resilience."""
    print("\n[weather_agent] Fetching weather data...")
    errors = list(state.get("errors", []))

    data, error = fetch_with_fallback("weather", state["destination"])
    if error:
        errors.append(error)

    return {"weather_data": data, "errors": errors}


def flights_agent(state: dict) -> dict:
    """Fetch flight data with full resilience."""
    print("\n[flights_agent] Fetching flight data...")
    errors = list(state.get("errors", []))

    data, error = fetch_with_fallback("flights", state["destination"])
    if error:
        errors.append(error)

    return {"flight_data": data, "errors": errors}


def hotels_agent(state: dict) -> dict:
    """Fetch hotel data with full resilience."""
    print("\n[hotels_agent] Fetching hotel data...")
    errors = list(state.get("errors", []))

    data, error = fetch_with_fallback("hotels", state["destination"])
    if error:
        errors.append(error)

    return {"hotel_data": data, "errors": errors}


def scorer(state: dict) -> dict:
    """
    Score and combine results. Works with partial data.
    This is DETERMINISTIC — no LLM needed for math.
    """
    print("\n[scorer] Scoring results...")
    errors = list(state.get("errors", []))
    available_sources = []
    score = 0.0

    # Weather scoring (0-1)
    weather = state.get("weather_data")
    if weather:
        available_sources.append("weather")
        # Simple scoring: sunny=1.0, partly cloudy=0.7, else=0.4
        conditions = {"sunny": 1.0, "partly cloudy": 0.7, "cloudy": 0.4, "rain": 0.2}
        weather_score = conditions.get(weather.get("condition", ""), 0.5)
        score += weather_score * 0.4  # 40% weight
        print(f"    Weather score: {weather_score} (condition: {weather.get('condition')})")

    # Flight scoring (0-1, cheaper = better)
    flights = state.get("flight_data")
    if flights:
        available_sources.append("flights")
        price = flights.get("price", 1000)
        flight_score = max(0, 1 - (price / 2000))  # $0=1.0, $2000=0.0
        score += flight_score * 0.35  # 35% weight
        print(f"    Flight score: {flight_score:.2f} (price: ${price})")

    # Hotel scoring (0-1, cheaper = better)
    hotels = state.get("hotel_data")
    if hotels:
        available_sources.append("hotels")
        price = hotels.get("price", 200)
        hotel_score = max(0, 1 - (price / 500))  # $0=1.0, $500=0.0
        score += hotel_score * 0.25  # 25% weight
        print(f"    Hotel score: {hotel_score:.2f} (price: ${price}/night)")

    if not available_sources:
        errors.append("No data sources available — cannot produce recommendation")

    return {
        "final_score": round(score, 3),
        "sources_used": available_sources,
        "errors": errors,
    }


def output_validator(state: dict) -> dict:
    """
    OUTPUT GUARDRAIL: Check the final result before returning.
    """
    print("\n[output_validator] Checking outputs...")

    score = state.get("final_score", -1)
    if score < 0 or score > 1:
        state.setdefault("errors", []).append(
            f"Score {score} out of valid range [0, 1] — clamping"
        )
        state["final_score"] = max(0, min(1, score))

    return state


# ──────────────────────────────────────────────
# Run the pipeline
# ──────────────────────────────────────────────

def run_pipeline(destination: str):
    """
    Execute the full agent pipeline with all reliability patterns.

    In a real system, this would be a LangGraph with parallel edges.
    Here we run sequentially for clarity, but the patterns are identical.
    """
    print("=" * 60)
    print(f"Running travel analysis for: {destination}")
    print("=" * 60)

    # Set random seed for reproducibility (remove in production)
    # random.seed(42)  # Uncomment for deterministic demo

    state = {"destination": destination}

    # Step 1: Input validation (guardrail)
    try:
        state.update(input_validator(state))
    except ValueError as e:
        print(f"\nFATAL: Input validation failed: {e}")
        return

    # Step 2: Fetch data from all sources
    # In LangGraph, these three would run in PARALLEL.
    # The error handling and fallback patterns work the same either way.
    for agent in [weather_agent, flights_agent, hotels_agent]:
        updates = agent(state)
        # Merge errors (append, don't overwrite)
        existing_errors = state.get("errors", [])
        new_errors = updates.pop("errors", [])
        state.update(updates)
        state["errors"] = existing_errors + [e for e in new_errors
                                              if e not in existing_errors]

    # Step 3: Score results (deterministic — no LLM)
    updates = scorer(state)
    existing_errors = state.get("errors", [])
    new_errors = updates.pop("errors", [])
    state.update(updates)
    state["errors"] = existing_errors + [e for e in new_errors
                                          if e not in existing_errors]

    # Step 4: Output validation (guardrail)
    state = output_validator(state)

    # ── Final Report ──
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Destination:  {state['destination']}")
    print(f"Final Score:  {state['final_score']}")
    print(f"Sources Used: {', '.join(state.get('sources_used', ['none']))}")

    if state.get("weather_data"):
        w = state["weather_data"]
        print(f"Weather:      {w.get('condition', 'N/A')} "
              f"({w.get('temp', '?')}°C) [{w.get('source', '?')}]")

    if state.get("flight_data"):
        f = state["flight_data"]
        print(f"Flights:      ${f.get('price', '?')} on "
              f"{f.get('airline', '?')} [{f.get('source', '?')}]")

    if state.get("hotel_data"):
        h = state["hotel_data"]
        print(f"Hotels:       ${h.get('price', '?')}/night at "
              f"{h.get('name', '?')} [{h.get('source', '?')}]")

    if state.get("errors"):
        print(f"\nWarnings ({len(state['errors'])}):")
        for err in state["errors"]:
            print(f"  - {err}")
    else:
        print("\nNo errors — all data sources responded successfully.")


# Run it!
if __name__ == "__main__":
    run_pipeline("tokyo")
```

Run this multiple times. Because the API fails randomly, you'll see different outcomes each time:

- **Best case:** All three APIs respond on the first try. All sources show `[live]`. No warnings.
- **Partial failure:** One or two APIs fail, retries are exhausted, cached data is used. Sources show `[cache]`. Warnings list which APIs failed.
- **Worst case (rare):** All APIs fail and cache is empty. The scorer reports "no data sources available." The user gets a clear error message instead of a crash.

Every run produces *something* useful. That's the goal.

---

## Key Takeaways: 5 Rules for Reliable Agents

1. **Every component can fail — handle it.** Wrap external calls in try/except. Return empty data with error flags, not stack traces.

2. **Partial results beat no results.** If 2 out of 3 data sources respond, give the user a recommendation based on what you have, with a note about what's missing.

3. **Cache everything expensive.** API calls, LLM responses, computations. Same input should never trigger the same expensive work twice in the same session.

4. **Use LLM only where deterministic code can't do it.** Math, sorting, filtering, scoring, validation — these are all cheaper, faster, and more reliable as regular functions.

5. **Log everything — you'll need it.** When a multi-agent pipeline produces a bad result, you need to trace exactly what happened at every step. Add logging now, not after the first production incident.

---

## What's Next

You now have the full foundations: what agents are, how they reason, how they use tools, how to orchestrate them, and how to make them reliable. Head to the **Project Walkthrough** series to see all of these concepts applied together in a real multi-agent system — the travel optimizer you've been building toward throughout this series.
