# The Graph — Nodes, Edges, and State

The heart of our system is a graph. Not a fancy neural network graph — a simple directed graph where nodes are Python functions and edges define execution order. Let's read it line by line.

If you have not read [Foundations Chapter 06 — Graphs and Orchestration](../foundations/06-graphs-and-orchestration.md), do that first. This chapter applies those concepts to real code.

---

## TravelState — The Shared Communication Channel

Every node in our graph reads from and writes to a single shared data structure: `TravelState`. This is defined in `models.py`:

```python
class TravelState(TypedDict, total=False):
    # User input
    destination: str              # "Tokyo, Japan"
    origin: str                   # "Bangalore, India" (default)
    date_range: tuple[str, str]   # ("2026-06-01", "2026-09-30")
    duration_days: int            # 7
    num_travelers: int            # 1
    budget_max: float | None      # optional
    priorities: dict[str, float]  # {"weather": 0.4, "flights": 0.3, "hotels": 0.3}

    # Supervisor output
    candidate_windows: list[dict] # [{"start": "2026-06-01", "end": "2026-06-07"}, ...]

    # Data agent outputs
    weather_data: list[dict]      # [{window, avg_temp, rain_days, avg_humidity, score, is_historical}]
    flight_data: list[dict]       # [{window, min_price, avg_price, currency, score, is_historical}]
    hotel_data: list[dict]        # [{window, avg_nightly, currency, score, is_historical}]

    # Scorer output
    ranked_windows: list[dict]    # [{window, weather_score, flight_score, hotel_score, total_score, ...}]

    # Synthesizer output
    recommendation: str           # natural language summary

    # Errors from any agent
    errors: list[str]
```

This is the **only** way agents communicate. The weather agent writes `weather_data`. The scorer reads it. They never call each other directly. They do not import each other's modules. They share data exclusively through the state dictionary that LangGraph passes between them.

This is the state-as-communication pattern from [Foundations Chapter 03](../foundations/03-state-and-memory.md). The benefit is loose coupling: you can replace the weather agent's implementation entirely, and as long as it writes the same shape of data to `weather_data`, nothing else in the system changes.

Notice `total=False` on the `TypedDict`. This means every field is optional — the state starts nearly empty (just user inputs) and accumulates data as each node runs. A node can safely call `state.get("weather_data", [])` without worrying about whether the weather agent has run yet.

### Why a TypedDict and not a Pydantic model?

LangGraph expects a `TypedDict` for its `StateGraph` because it needs to merge partial dictionaries returned by each node. Each node returns only the fields it wants to update — not the entire state. LangGraph merges those partial returns into the accumulated state. A Pydantic model with required fields would reject these partial updates.

The individual data models (`CandidateWindow`, `WeatherResult`, `FlightResult`, etc.) are Pydantic models for validation within each agent. But the top-level state is a `TypedDict` for LangGraph compatibility.

---

## graph.py — Annotated

Here is the complete `graph.py` with inline annotations explaining every line:

```python
from langgraph.graph import StateGraph, END
from models import TravelState
from agents.supervisor import supervisor_node
from agents.weather import weather_node
from agents.flights import flights_node
from agents.hotels import hotels_node
from agents.scorer import scorer_node
from agents.synthesizer import synthesizer_node

def build_graph():
    # Create a graph that uses TravelState as its shared data structure.
    # Every node function will receive the current TravelState and return
    # a partial dict to merge back into it.
    graph = StateGraph(TravelState)

    # Add nodes — each is a plain Python function with signature:
    #   def node_fn(state: TravelState) -> dict
    graph.add_node("supervisor", supervisor_node)    # generates candidate windows
    graph.add_node("weather", weather_node)          # fetches weather data
    graph.add_node("flights", flights_node)          # fetches flight prices
    graph.add_node("hotels", hotels_node)            # fetches hotel prices
    graph.add_node("scorer", scorer_node)            # ranks windows
    graph.add_node("synthesizer", synthesizer_node)  # LLM recommendation

    # Entry point — execution starts here when you call graph.invoke()
    graph.set_entry_point("supervisor")

    # Fan-out: supervisor → 3 data agents IN PARALLEL
    # When LangGraph sees multiple edges leaving the same node,
    # it runs all target nodes concurrently.
    graph.add_edge("supervisor", "weather")
    graph.add_edge("supervisor", "flights")
    graph.add_edge("supervisor", "hotels")

    # Fan-in: all 3 → scorer
    # The scorer node has 3 incoming edges. LangGraph will NOT run it
    # until ALL THREE source nodes have completed and their outputs
    # have been merged into state.
    graph.add_edge("weather", "scorer")
    graph.add_edge("flights", "scorer")
    graph.add_edge("hotels", "scorer")

    # Sequential: scorer → synthesizer → done
    graph.add_edge("scorer", "synthesizer")
    graph.add_edge("synthesizer", END)

    # compile() validates the graph (checks for unreachable nodes,
    # missing edges, etc.) and returns a runnable object.
    return graph.compile()
```

That is the entire file. Twenty-one lines of actual logic (excluding imports). The graph is purely declarative — it says "what connects to what" and LangGraph figures out the execution strategy.

---

## What Happens When You Call `graph.invoke()`

Let's trace a complete execution. The caller (our Streamlit app) does:

```python
result = build_graph().invoke({
    "destination": "Tokyo",
    "origin": "Bangalore",
    "date_range": ("2026-07-01", "2026-09-30"),
    "duration_days": 7,
    "num_travelers": 1,
    "priorities": {"weather": 0.4, "flights": 0.3, "hotels": 0.3},
    "errors": []
})
```

Here is what LangGraph does internally, step by step:

**Step 1: Entry point — run `supervisor_node`.** LangGraph looks at the entry point, finds the supervisor, and calls `supervisor_node(state)`. The supervisor generates candidate windows using simple date arithmetic:

```python
def generate_candidate_windows(start_date, end_date, duration_days=7):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows = []
    current = start
    while current + timedelta(days=duration_days) <= end:
        windows.append({
            "start": current.isoformat(),
            "end": (current + timedelta(days=duration_days)).isoformat(),
        })
        current += timedelta(days=7)
    return windows
```

The supervisor returns `{"candidate_windows": [...], "errors": []}`. LangGraph merges this into the state.

**Step 2: Fan-out — spawn 3 parallel executions.** LangGraph sees three outgoing edges from "supervisor" (to weather, flights, hotels). It runs all three node functions concurrently. Each receives the full current state (which now includes `candidate_windows`). Each writes to a different field:

- Weather agent writes `weather_data`
- Flights agent writes `flight_data`
- Hotels agent writes `hotel_data`

This takes 10-30 seconds depending on API response times, but all three run simultaneously.

**Step 3: Fan-in — merge results.** When all three complete, LangGraph merges their returned dicts into the state. Since each agent writes to a different field, there are no conflicts. The state now contains all user inputs, candidate windows, and all three data sets.

**Step 4: Run `scorer_node`.** All three incoming edges to "scorer" are satisfied. LangGraph calls the scorer, which reads `weather_data`, `flight_data`, and `hotel_data`, normalizes scores, and returns `{"ranked_windows": [...]}`.

**Step 5: Run `synthesizer_node`.** The scorer's single outgoing edge leads to the synthesizer. It reads `ranked_windows`, calls an LLM to generate a natural language recommendation, and returns `{"recommendation": "..."}`.

**Step 6: END.** LangGraph returns the final accumulated state to the caller.

---

## The Merge Problem

When multiple nodes run in parallel and return results, LangGraph needs to merge their outputs into a single state. What if two agents tried to update the same field?

LangGraph uses **reducers** to handle this. The default reducer is "last write wins" — if two nodes both return `{"errors": [...]}`, only one value survives. This is generally fine for simple fields but dangerous for lists you want to accumulate.

In our system, we sidestep the problem entirely through deliberate state design:

| Node | Writes to |
|------|-----------|
| Weather | `weather_data`, `errors` |
| Flights | `flight_data`, `errors` |
| Hotels | `hotel_data`, `errors` |

Each data agent writes to its own unique field. The only shared field is `errors`, and each agent copies the existing errors list before appending:

```python
errors = list(state.get("errors", []))  # copy, don't mutate
# ... do work ...
errors.append("something went wrong")
return {"weather_data": results, "errors": errors}
```

Because each agent starts from the same base `errors` list (the supervisor's empty list) and appends independently, the last-write-wins merge means we might lose errors from agents that finish first. In practice, for this project, the trade-off is acceptable — the primary data fields are never in conflict, and the error list is for warnings, not critical control flow.

If you needed guaranteed error accumulation, you would define a custom reducer using LangGraph's `Annotated` type with an `operator.add` reducer. That is an advanced topic covered in the LangGraph documentation.

The broader lesson: **good state design avoids merge issues entirely.** Give each parallel agent its own output field, and merging becomes trivial.

---

## State at Each Step — A Trace

Here is what the state looks like after each node completes, for a search of "Tokyo" from "Bangalore" across July 2026:

```
Initial state:
  {destination: "Tokyo", origin: "Bangalore",
   date_range: ("2026-07-01", "2026-09-30"), duration_days: 7,
   priorities: {weather: 0.4, flights: 0.3, hotels: 0.3}, errors: []}

After supervisor:
  + {candidate_windows: [
      {start: "2026-07-01", end: "2026-07-08"},
      {start: "2026-07-08", end: "2026-07-15"},
      {start: "2026-07-15", end: "2026-07-22"},
      ... (12-13 windows total)
    ]}

After weather (parallel):
  + {weather_data: [
      {window: {start: "2026-07-01", ...}, avg_temp: 27.3, rain_days: 3,
       avg_humidity: 72.0, score: 0.612, is_historical: false},
      ...
    ]}

After flights (parallel):
  + {flight_data: [
      {window: {start: "2026-07-01", ...}, min_price: 32000,
       avg_price: 38500, currency: "INR", score: 0.0, is_historical: false},
      ...
    ]}

After hotels (parallel):
  + {hotel_data: [
      {window: {start: "2026-07-01", ...}, avg_nightly: 8500,
       currency: "INR", score: 0.0, is_historical: false},
      ...
    ]}

After scorer:
  + {ranked_windows: [
      {window: {start: "2026-09-16", ...}, weather_score: 0.85,
       flight_score: 0.92, hotel_score: 0.78, total_score: 0.856,
       estimated_flight_cost: 28000, estimated_hotel_cost: 7200,
       has_historical_data: false},
      ... (sorted by total_score descending)
    ]}

After synthesizer:
  + {recommendation: "The best time to visit Tokyo from Bangalore is
     September 16-23. Weather is pleasant at 24C with minimal rain,
     flights are near their lowest at ~28,000 INR, and hotels average
     ~7,200 INR/night. July is significantly more humid and rainy,
     making September the clear winner across all dimensions."}
```

Notice how data accumulates. Each node adds to the state without modifying what previous nodes wrote. By the end, the state is a complete record of the entire search — inputs, intermediate data, rankings, and the final recommendation.

---

## The Graph as Documentation

One underappreciated benefit of graph-based architectures: **the graph IS the documentation.** When a new developer opens `graph.py`, they can see the entire system architecture in 21 lines. They know:

- What nodes exist (the six `add_node` calls)
- What runs in parallel (multiple edges from one source)
- What waits for what (multiple edges into one target)
- Where execution starts and ends

Compare this to a procedural script where you have to read through hundreds of lines to understand the control flow. Or a single-agent LLM system where the execution path is determined at runtime by the LLM and is different every time.

The graph makes the architecture explicit, static, and readable. That is worth the small overhead of learning LangGraph's API.

---

## Key Takeaways

1. **`TravelState` is the shared communication channel.** Agents never call each other — they read from and write to state fields. This is loose coupling by design.

2. **The graph is declarative.** You say "supervisor connects to weather, flights, hotels" and LangGraph handles parallelism, synchronization, and state passing.

3. **Fan-out happens automatically** when multiple edges leave one node. Fan-in happens automatically when multiple edges enter one node (it waits for all).

4. **Good state design prevents merge conflicts.** Each parallel agent writes to its own field. The only shared field (`errors`) uses a copy-and-append pattern.

5. **`graph.invoke()` returns the full accumulated state** — a complete record of the entire search from inputs through to the final recommendation.

6. **The graph is the architecture diagram.** Reading `graph.py` tells you everything about the system's structure.

Next: [Chapter 03 — Data Agents](./03-data-agents.md), where we look inside the weather, flights, and hotels nodes to see how they call APIs, handle failures, and fall back to cached data.
