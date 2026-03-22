# Chapter 5: Multi-Agent Architectures

One agent can do a lot. But some problems naturally decompose into specialists. You wouldn't ask your accountant to fix your plumbing — same idea.

In the last chapter, we explored how a single agent decides what to do: ReAct, tool-use, plan-execute, reflection, routing. These patterns are powerful. A well-designed single agent with good tools can handle a surprising range of tasks. But at some point, you hit a wall.

Maybe your agent's system prompt is 4,000 words long because it needs to handle billing *and* technical support *and* onboarding. Maybe your tool list has 30 entries and the model keeps picking the wrong one. Maybe parts of the task could run in parallel but your single-threaded agent loop processes them sequentially. These are signals that you might need to split one agent into several.

This chapter covers the major multi-agent patterns, when each makes sense, and — critically — when to resist the temptation and stick with one agent.

---

## 1. Why Multiple Agents?

Four forces push you toward multi-agent architectures:

**Specialization.** A flight-booking agent and a hotel-booking agent can each have focused system prompts, curated tool sets, and domain-specific few-shot examples. One mega-agent trying to do both will have a diluted prompt and a bloated tool list. Models perform better with focused instructions — this is well-documented.

**Parallelism.** If you need weather data, flight options, and hotel availability, why fetch them sequentially? Three specialist agents can work simultaneously, and you combine results when they're all done. Wall-clock time drops from the sum of all tasks to the duration of the slowest one.

**Separation of concerns.** When your flight agent breaks, you fix the flight agent. You don't wade through 2,000 lines of a monolithic agent wondering which part handles flights vs. hotels vs. car rentals. Testing is easier too — you can unit test each agent in isolation.

**Context management.** Every agent has a context window. A single agent handling a complex task accumulates a long conversation history — tool calls, results, reasoning traces — that eventually pushes against the window limit or degrades performance (models get worse as context grows). Multiple agents each start with a clean, focused context.

### But Also: When NOT to Use Multi-Agent

Here's the contrarian take that experienced builders converge on: **most of the time, you don't need multiple agents.** The reasons:

- **Communication overhead.** Agents need to pass information to each other. That means serializing state, formatting messages, and sometimes losing nuance in translation. A single agent with access to all the context doesn't have this problem.
- **Debugging complexity.** When something goes wrong in a multi-agent system, you need to trace which agent did what, what messages were passed, and where the breakdown happened. This is harder than debugging one agent's conversation history.
- **Added latency.** Coordination takes time. An orchestrator deciding which worker to call, waiting for results, deciding what to do next — each step adds an LLM call.
- **Unpredictable interactions.** Agent A gives Agent B slightly ambiguous output. Agent B misinterprets it. Now you have a bug that neither agent would produce in isolation.

OpenAI's "Practical Guide to Building Agents" is blunt about this: **"Start with a single agent. Graduate to multi-agent when the single agent's tools or instructions become too complex."** Don't reach for multi-agent because it sounds impressive. Reach for it because a single agent is demonstrably failing.

---

## 2. Pattern 1: Orchestrator (Boss + Workers)

This is the most common multi-agent pattern in production systems, and it's what our travel-optimizer project uses. One supervisor agent (the "orchestrator") receives the user's request, decides which specialist workers to invoke, collects their results, and synthesizes a final response.

```
┌──────────────┐
│  Orchestrator │
│  (Supervisor) │
└───┬──┬──┬────┘
    │  │  │
    ▼  ▼  ▼
┌──┐ ┌──┐ ┌──┐
│W1│ │W2│ │W3│   ← Workers (can run in parallel)
└──┘ └──┘ └──┘
```

The orchestrator is itself an LLM-powered agent. It reads the user's request, reasons about which workers to call (and in what order), dispatches tasks, and combines results. Workers are simpler — often just a single LLM call with a focused prompt and a small set of tools.

```python
class Orchestrator:
    def __init__(self, workers: dict):
        self.workers = workers  # {"flights": FlightAgent, "hotels": HotelAgent, ...}

    def handle(self, user_request: str) -> str:
        # Step 1: Decide which workers to invoke
        plan = llm.invoke(
            f"Given this request, which specialists should handle it?\n"
            f"Available: {list(self.workers.keys())}\n"
            f"Request: {user_request}"
        )

        # Step 2: Dispatch to workers (potentially in parallel)
        results = {}
        for worker_name in plan.selected_workers:
            worker = self.workers[worker_name]
            results[worker_name] = worker.run(user_request)

        # Step 3: Synthesize results into a final response
        return llm.invoke(
            f"Combine these specialist results into a coherent response:\n"
            f"{json.dumps(results, indent=2)}"
        )
```

Anthropic describes this as the **"orchestrator-workers"** pattern: *"a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results."* OpenAI frames it similarly: *"a manager agent that delegates tasks and synthesizes results."*

**Why it works:** The control flow is clear. The orchestrator is always in charge. Workers don't talk to each other — they report back to the boss. This makes the system predictable and debuggable. You can trace any issue to either the orchestrator's decision-making or a specific worker's output.

**The main weakness:** The orchestrator is a bottleneck. Every piece of information flows through it. If Worker 1's output would be useful context for Worker 2, it has to go up to the orchestrator and back down — the workers can't communicate directly. For tasks that need tight collaboration between specialists, this round-tripping through the orchestrator adds latency and loses context.

**Design tips for orchestrators:**

- Keep the orchestrator's job narrow: decide who to call, combine results. Don't let it also do domain-specific work.
- Give workers clear, non-overlapping responsibilities. If two workers might handle the same request, the orchestrator will waste time deciding between them (or call both).
- Consider making workers stateless — they take a task, return a result, done. This makes them easy to retry, test, and parallelize.

---

## 3. Pattern 2: Peer-to-Peer (Agents Talk to Each Other)

In this pattern, there's no central coordinator. Agents communicate directly with each other, passing messages and building on each other's work.

```
┌──────┐    ┌──────┐
│Agent A│◄──►│Agent B│
└───┬───┘    └───┬───┘
    │            │
    └─────┬──────┘
          ▼
     ┌─────────┐
     │ Agent C  │
     └─────────┘
```

A classic example: a Researcher agent and a Writer agent collaborating on a blog post. The Researcher gathers information and passes findings to the Writer. The Writer drafts content, realizes it needs more detail on a topic, and asks the Researcher for clarification. They go back and forth until the post is done.

```python
def peer_to_peer_collaboration(task: str) -> str:
    researcher_context = []
    writer_context = []

    # Researcher starts by gathering information
    research = researcher_agent.invoke(
        f"Research the following topic thoroughly: {task}"
    )
    writer_context.append({"from": "researcher", "content": research})

    # Writer creates a draft based on research
    draft = writer_agent.invoke(
        f"Write a blog post based on this research:\n{research}"
    )
    researcher_context.append({"from": "writer", "content": draft})

    # Writer might request more information
    feedback = writer_agent.invoke(
        f"What additional information do you need to improve this draft?\n{draft}"
    )

    if feedback.needs_more_research:
        # Back to the researcher
        additional = researcher_agent.invoke(
            f"The writer needs more info on: {feedback.topics}"
        )
        # And back to the writer for a final draft
        final = writer_agent.invoke(
            f"Here's additional research. Revise your draft:\n{additional}"
        )
        return final

    return draft
```

**The appeal:** Flexibility. Agents self-organize around the task. The conversation can go wherever it needs to.

**The danger:** Without a coordinator, conversations can go in circles. Agent A asks Agent B for clarification, Agent B asks Agent A for clarification, and they loop forever. You need hard limits on the number of exchanges, or a termination condition that's reliably detectable.

Debugging is also harder. In the orchestrator pattern, you can follow one thread: orchestrator → worker → orchestrator. In peer-to-peer, messages flow in multiple directions, and understanding what happened requires reconstructing a multi-party conversation.

**When to use:** Tasks that genuinely require back-and-forth collaboration between specialists. Creative tasks (writing, brainstorming) where iterative refinement is the natural workflow. Small systems with 2-3 agents — the complexity is manageable.

**When to avoid:** Systems with many agents (the communication graph explodes), tasks with clear authority structure (someone needs to make final decisions), or production systems where predictability matters.

---

## 4. Pattern 3: Hierarchical (Tree Structure)

When systems get large, you need hierarchy. A top-level manager delegates to sub-managers, who delegate to workers. It's the corporate org chart applied to agents.

```
     ┌──────────┐
     │  Manager  │
     └───┬──┬───┘
         │  │
   ┌─────▼┐ ┌▼─────┐
   │Sub-M1│ │Sub-M2│
   └─┬──┬┘ └┬───┬─┘
     │  │   │   │
     W  W   W   W
```

Consider a travel planning system for complex multi-city trips. The top manager handles the overall itinerary. Sub-manager 1 handles transportation (flights, trains, car rentals), with workers for each mode. Sub-manager 2 handles accommodation (hotels, Airbnb, hostels), with workers for each platform. Sub-manager 3 handles activities (tours, restaurants, events).

```python
class HierarchicalSystem:
    """
    Each level only talks to the level directly below it.
    The top manager never directly calls a worker — it goes through sub-managers.
    """
    def __init__(self):
        # Workers — do the actual tool-calling work
        self.flight_worker = FlightSearchAgent()
        self.train_worker = TrainSearchAgent()
        self.hotel_worker = HotelSearchAgent()
        self.airbnb_worker = AirbnbSearchAgent()

        # Sub-managers — coordinate workers within their domain
        self.transport_manager = TransportManager(
            workers=[self.flight_worker, self.train_worker]
        )
        self.accommodation_manager = AccommodationManager(
            workers=[self.hotel_worker, self.airbnb_worker]
        )

        # Top manager — coordinates sub-managers
        self.manager = TopManager(
            sub_managers=[self.transport_manager, self.accommodation_manager]
        )

    def plan_trip(self, request: str) -> str:
        return self.manager.handle(request)
```

**The benefit:** Each level only needs to understand the level below it. The top manager doesn't know or care that there's a separate flight worker and train worker — it just tells the transport sub-manager "find the best way to get from A to B." This encapsulation makes large systems manageable.

**The cost:** Latency. A request that starts at the top and reaches a leaf worker has gone through multiple LLM calls just for delegation. The result has to travel all the way back up. Deep hierarchies (more than 2-3 levels) get slow and expensive fast.

**When to use:** Large systems with natural domain decomposition. Enterprise systems where different teams own different sub-managers. When you need clear authority boundaries.

**When to avoid:** Small systems (the overhead isn't worth it). Tasks that don't decompose cleanly into a tree. When latency matters — every level adds round trips.

---

## 5. Pattern 4: Swarm

The swarm pattern takes a fundamentally different approach: instead of designing the communication structure upfront, you let agents self-organize. Each agent follows simple rules about when to hand off to another agent, and complex behavior emerges from these simple interactions.

OpenAI released their "Swarm" framework to explore this idea. The core concept: every agent can "hand off" to any other agent. There's no fixed hierarchy — the conversation flows to whoever is best equipped to handle the current state.

```python
# Conceptual swarm agent definition (inspired by OpenAI Swarm)
flight_agent = Agent(
    name="Flight Specialist",
    instructions="You handle flight searches and bookings.",
    tools=[search_flights, book_flight],
    handoffs=[hotel_agent, general_agent]  # Can hand off to these agents
)

hotel_agent = Agent(
    name="Hotel Specialist",
    instructions="You handle hotel searches and bookings.",
    tools=[search_hotels, book_hotel],
    handoffs=[flight_agent, general_agent]  # Can hand off to these agents
)

# The conversation starts with one agent and flows naturally
# based on what the user asks
result = swarm.run(initial_agent=general_agent, messages=user_messages)
```

Think of it like an ant colony. Individual ants follow simple rules ("if you find food, leave a pheromone trail"). No ant has a master plan. But collectively, they find the shortest path to food sources, build complex structures, and allocate workers efficiently. Simple local rules, complex global behavior.

**The appeal:** Adaptability. The system handles unexpected inputs gracefully because there's always some agent that can take over. Adding a new capability is as simple as adding a new agent with handoff rules.

**The reality check:** Swarms are unpredictable. You can't easily guarantee the system will handle a specific request in a specific way. The conversation might bounce between agents in ways you didn't anticipate. For production systems where reliability matters, this unpredictability is a serious drawback.

**When to use:** Exploratory systems, research prototypes, conversational systems where the user drives the flow and you don't know in advance what they'll ask.

**When to avoid:** Mission-critical systems, production APIs, anywhere you need predictable behavior and SLAs.

---

## 6. Fan-Out / Fan-In: The Parallelization Execution Pattern

Fan-out/fan-in isn't an architecture per se — it's an execution pattern used *within* any of the architectures above. But it's so common and so important that it deserves dedicated attention.

**Fan-out:** One node dispatches work to multiple agents in parallel.
**Fan-in:** Collect all results, merge them, continue.

Anthropic calls this the **"parallelization"** pattern in their "Building Effective Agents" blog, and identifies it as one of the most practical patterns for real-world systems.

```python
import asyncio

async def parallel_search(user_request: str) -> dict:
    """
    Fan-out: dispatch to all specialist agents simultaneously.
    Fan-in: collect results and merge.
    """
    # Fan-out — all three searches start at the same time
    weather_task = asyncio.create_task(agent_weather.run(user_request))
    flights_task = asyncio.create_task(agent_flights.run(user_request))
    hotels_task = asyncio.create_task(agent_hotels.run(user_request))

    # Fan-in — wait for all results
    results = await asyncio.gather(weather_task, flights_task, hotels_task)

    # Merge results into a unified state
    combined = {
        "weather": results[0],
        "flights": results[1],
        "hotels": results[2],
    }

    # A scorer or synthesizer processes the combined results
    final = scorer_agent.run(combined)
    return final
```

The pattern is simple but the implementation details matter:

- **Error handling:** What if one agent fails? Do you wait for it, use a timeout, or proceed with partial results? In production, you usually want a timeout with a fallback: "if the hotel search takes more than 5 seconds, proceed without it and note that hotel data is unavailable."
- **Result merging:** How do you combine results from different agents? A shared state object (dictionary, dataclass) works well for simple cases. For complex cases, you might need a dedicated merging step (another LLM call to synthesize).
- **Partial dependencies:** Sometimes agents aren't fully independent. Maybe the hotel search needs the destination from the flight search. You handle this with dependency graphs: run independent agents first, then run dependent agents with the results.

```python
# Handling partial dependencies
async def search_with_dependencies(request: str) -> dict:
    # Phase 1: Independent agents (fan-out)
    flights, weather = await asyncio.gather(
        agent_flights.run(request),
        agent_weather.run(request),
    )

    # Phase 2: Dependent agents (need flight results to know the destination)
    destination = flights.destination
    hotels, activities = await asyncio.gather(
        agent_hotels.run(destination),
        agent_activities.run(destination, weather),
    )

    return {"flights": flights, "weather": weather,
            "hotels": hotels, "activities": activities}
```

---

## 7. Communication: Shared State vs. Message Passing

However you structure your agents, they need to communicate. There are two fundamental approaches:

### Shared State

All agents read from and write to a single state object. This is the approach LangGraph uses — a state dictionary (or typed state class) that flows through the graph, and each node reads what it needs and writes its results.

```python
# Shared state approach
state = {
    "user_request": "Plan a trip to Tokyo",
    "flights": None,
    "hotels": None,
    "weather": None,
    "recommendation": None,
}

# Each agent reads from and writes to the same state
def flight_agent(state: dict) -> dict:
    results = search_flights(state["user_request"])
    state["flights"] = results
    return state

def hotel_agent(state: dict) -> dict:
    results = search_hotels(state["user_request"])
    state["hotels"] = results
    return state
```

**Pros:** Simple mental model. No serialization issues. Every agent can see everything (if you want it to). Easy to debug — just print the state at each step.

**Cons:** Tight coupling. If you change the state schema, every agent that reads or writes to those fields needs to be updated. Race conditions are possible if agents run in parallel and both write to the same field (though most frameworks handle this with merge strategies). Every agent has access to the entire state, which means a misbehaving agent can corrupt data it shouldn't touch.

### Message Passing

Agents send discrete messages to each other, like microservices communicating over HTTP or a message queue. Each agent has defined inputs and outputs, and you explicitly route messages between them.

```python
# Message passing approach
class AgentMessage:
    sender: str
    recipient: str
    content: dict

# Agents receive messages and produce messages
def flight_agent(message: AgentMessage) -> AgentMessage:
    results = search_flights(message.content["request"])
    return AgentMessage(
        sender="flight_agent",
        recipient="orchestrator",
        content={"flights": results}
    )

# A message router handles delivery
def route_message(message: AgentMessage):
    agents[message.recipient].receive(message)
```

**Pros:** Loose coupling. Each agent defines its own interface. You can swap out agents without affecting others. Natural fit for distributed systems where agents run on different machines. Clear boundaries.

**Cons:** More complex infrastructure. You need message routing, potentially a message queue, serialization/deserialization. Harder to debug — you need to trace messages across agents.

**In practice:** Most multi-agent systems use shared state for simplicity, especially when all agents run in the same process. Message passing becomes worthwhile when you need loose coupling (different teams building different agents) or distribution (agents on different servers).

---

## 8. Decision Framework: Which Pattern to Use

Here's the practical decision guide:

| Situation | Pattern |
|-----------|---------|
| Clear task decomposition, independent subtasks | **Orchestrator + Fan-out** |
| Tasks need back-and-forth collaboration | **Peer-to-peer** |
| Large system with sub-systems, multiple teams | **Hierarchical** |
| Unpredictable, evolving tasks, conversational | **Swarm** |
| Just need to classify and route input | **Single agent with routing** (Ch. 4) |

And the meta-advice from both major labs:

OpenAI: **"Start with a single agent. Graduate to multi-agent when the single agent's tools or instructions become too complex."** They suggest looking for specific signals: tool list growing beyond what the model handles well, system prompt trying to cover too many domains, tasks that naturally parallelize but your agent processes sequentially.

Anthropic: **Parallelization and orchestrator-workers are the most common patterns in practice.** "These are the workhorses of multi-agent systems." They're predictable, debuggable, and handle the vast majority of real-world use cases. Save the exotic patterns (swarm, full peer-to-peer) for when simpler ones demonstrably fail.

Both converge on the same philosophy: **start simple, add agents only when you have evidence that a single agent isn't enough.** The progression is usually: single tool-use agent → single agent with routing → orchestrator with workers → more complex patterns. Most production systems never get past orchestrator-with-workers because it's sufficient.

---

## Runnable Example: A Simple Orchestrator with Two Workers

Here's a complete, runnable example of the orchestrator pattern. An orchestrator receives a task ("write a blog post about AI agents"), delegates research to a Researcher worker and writing to a Writer worker, and combines the results.

```python
"""
Multi-Agent Orchestrator Example
=================================
Demonstrates the Orchestrator (Boss + Workers) pattern.
One orchestrator delegates to two specialist workers:
  - Researcher: gathers key points about a topic
  - Writer: turns research into a blog post draft

No API keys needed — uses mock LLM responses.

Run: python 05_orchestrator_agents.py
"""

import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Shared State — the data structure that flows through the system.
# Each agent reads what it needs and writes its results.
# ---------------------------------------------------------------------------
@dataclass
class PipelineState:
    """
    Shared state that all agents can read from and write to.

    This is the 'shared state' communication pattern from the chapter.
    In LangGraph, this would be a TypedDict or Pydantic model that serves
    as the graph's state schema.
    """
    task: str = ""                    # The original user request
    research_notes: str = ""          # Output from the Researcher worker
    blog_draft: str = ""              # Output from the Writer worker
    final_output: str = ""            # Orchestrator's combined final output
    steps_log: list[str] = field(default_factory=list)  # Audit trail

    def log(self, message: str):
        """Add a timestamped entry to the audit trail."""
        step_num = len(self.steps_log) + 1
        self.steps_log.append(f"[Step {step_num}] {message}")
        print(f"  [Step {step_num}] {message}")


# ---------------------------------------------------------------------------
# Worker 1: Researcher Agent
# ---------------------------------------------------------------------------
def researcher_agent(state: PipelineState) -> PipelineState:
    """
    The Researcher worker.

    Responsibility: Given a topic, gather key points and facts.
    In a real system, this agent would:
      - Have tools like web_search, arxiv_search, wikipedia_lookup
      - Use an LLM to decide which tools to call and synthesize findings
      - Possibly run a ReAct loop to iteratively search and refine

    Here, we simulate the research with a mock response.
    """
    state.log(f"Researcher received task: '{state.task}'")

    # Simulate research — in production, this would be an LLM + tools
    # The researcher might make multiple search calls, read papers, etc.
    state.log("Researcher searching for information on AI agents...")
    state.log("Researcher found 3 relevant sources, synthesizing...")

    # Mock research output — pretend the agent gathered and summarized these
    state.research_notes = (
        "KEY FINDINGS ON AI AGENTS:\n"
        "\n"
        "1. DEFINITION: AI agents are systems where LLMs dynamically direct "
        "their own processes and tool usage to accomplish tasks. Unlike simple "
        "chatbots, agents take actions in the world — searching, computing, "
        "writing files, calling APIs.\n"
        "\n"
        "2. CORE LOOP: All agents share a common pattern — perceive (read input "
        "or tool results), reason (decide what to do), act (call a tool or "
        "respond). This loop continues until the task is complete.\n"
        "\n"
        "3. ARCHITECTURES: Major patterns include single-agent (ReAct, tool-use, "
        "plan-execute) and multi-agent (orchestrator-workers, peer-to-peer, "
        "hierarchical, swarm). Most production systems use orchestrator-workers.\n"
        "\n"
        "4. INDUSTRY TREND: Both OpenAI and Anthropic recommend starting simple. "
        "A single agent with good tools beats a complex multi-agent system for "
        "most use cases. Add complexity only when you have evidence it is needed.\n"
        "\n"
        "5. KEY CHALLENGES: Reliability (agents make mistakes), cost (each "
        "reasoning step costs tokens), latency (multiple LLM calls add up), "
        "and evaluation (how do you measure if an agent is good?).\n"
    )

    state.log("Researcher completed — research notes ready")
    return state


# ---------------------------------------------------------------------------
# Worker 2: Writer Agent
# ---------------------------------------------------------------------------
def writer_agent(state: PipelineState) -> PipelineState:
    """
    The Writer worker.

    Responsibility: Given research notes, produce a well-written blog post.
    In a real system, this agent would:
      - Receive the research notes in its context
      - Use an LLM to generate a draft
      - Possibly use reflection to self-edit (evaluator-optimizer pattern)

    Here, we simulate the writing with a mock response.
    """
    state.log(f"Writer received research notes ({len(state.research_notes)} chars)")

    # Check that we actually have research to work with
    if not state.research_notes:
        state.log("Writer ERROR: No research notes available!")
        state.blog_draft = "ERROR: Cannot write without research notes."
        return state

    state.log("Writer drafting blog post from research notes...")

    # Mock writing output — in production, the LLM generates this
    state.blog_draft = (
        "# AI Agents: What They Are and Why They Matter\n"
        "\n"
        "If you have used ChatGPT to answer a question, you have used an LLM. "
        "But you have not used an agent. The difference matters.\n"
        "\n"
        "An AI agent is an LLM that does not just talk — it acts. It searches "
        "the web, runs code, calls APIs, and makes decisions about what to do "
        "next. It is the difference between a librarian who answers your "
        "question and one who also goes and finds the book for you.\n"
        "\n"
        "## The Core Loop\n"
        "\n"
        "Every agent, from the simplest to the most sophisticated, runs the "
        "same fundamental loop: perceive, reason, act. The agent reads the "
        "current state (what the user asked, what tools returned), reasons "
        "about what to do next, and takes an action. Then it loops.\n"
        "\n"
        "## Simple Beats Complex\n"
        "\n"
        "The most surprising thing about building agents? Simpler is almost "
        "always better. Both OpenAI and Anthropic — the two labs pushing "
        "agents hardest — say the same thing: start with one agent and a "
        "few tools. Only add complexity when you have proof it is needed.\n"
        "\n"
        "Most production agent systems use a pattern called "
        "'orchestrator-workers' — one boss agent delegates to specialists. "
        "It is not glamorous, but it works.\n"
        "\n"
        "## The Challenges Ahead\n"
        "\n"
        "Agents are not magic. They make mistakes, they cost money (each "
        "thinking step uses tokens), and measuring whether they are "
        "actually good at their job is an open research problem. But they "
        "are the most practical path from 'AI that talks' to 'AI that does.'\n"
    )

    state.log("Writer completed — blog draft ready")
    return state


# ---------------------------------------------------------------------------
# The Orchestrator — the boss that coordinates workers
# ---------------------------------------------------------------------------
def orchestrator(task: str) -> PipelineState:
    """
    The orchestrator agent.

    This is the 'boss' in the boss-workers pattern. Its job:
      1. Initialize shared state with the user's task
      2. Decide which workers to call and in what order
      3. Dispatch work to workers
      4. Combine worker outputs into a final result

    In a real system, steps 2 and 4 would involve LLM calls —
    the orchestrator would reason about the task structure and
    synthesize results. Here we hard-code the workflow since
    the pattern (research then write) is always the same for
    "write a blog post" tasks.

    For more dynamic orchestration, the orchestrator would use
    an LLM to decide which workers to invoke based on the task.
    """
    # --- Initialize shared state ---
    state = PipelineState(task=task)

    print(f"\n{'='*60}")
    print(f"ORCHESTRATOR received task: {task}")
    print(f"{'='*60}\n")

    # --- Step 1: Orchestrator decides on a plan ---
    # In a real system: plan = llm.invoke("Which workers should handle this?")
    # For a "write a blog post" task, the plan is always:
    #   1. Researcher gathers information
    #   2. Writer turns research into a draft
    state.log("Orchestrator planning: need Research, then Writing")
    state.log("Orchestrator dispatching to Researcher worker...")

    print()

    # --- Step 2: Dispatch to Researcher ---
    # The researcher writes its output to state.research_notes
    state = researcher_agent(state)

    print()

    # --- Step 3: Dispatch to Writer ---
    # The writer reads state.research_notes and writes state.blog_draft
    state.log("Orchestrator dispatching to Writer worker...")

    print()

    state = writer_agent(state)

    print()

    # --- Step 4: Orchestrator synthesizes final output ---
    # In a real system, the orchestrator might:
    #   - Check quality of the draft
    #   - Add an introduction or conclusion
    #   - Request revisions from the Writer if needed
    state.log("Orchestrator reviewing and combining results...")

    # Combine the worker outputs into the final deliverable
    state.final_output = (
        f"{state.blog_draft}\n"
        f"---\n"
        f"*Research notes used for this post are available on request.*\n"
    )

    state.log("Orchestrator completed — final output ready")

    return state


# ---------------------------------------------------------------------------
# Run it!
# ---------------------------------------------------------------------------
def main():
    """
    Entry point. Sends a task to the orchestrator and displays the results.
    """
    task = "Write a blog post about AI agents"

    # Run the orchestrator
    final_state = orchestrator(task)

    # --- Display results ---
    print(f"\n{'='*60}")
    print("FINAL OUTPUT:")
    print(f"{'='*60}")
    print(final_state.final_output)

    print(f"\n{'='*60}")
    print("EXECUTION LOG (audit trail):")
    print(f"{'='*60}")
    for entry in final_state.steps_log:
        print(f"  {entry}")

    print(f"\n{'='*60}")
    print("STATE SUMMARY:")
    print(f"{'='*60}")
    print(f"  Task:           {final_state.task}")
    print(f"  Research notes:  {len(final_state.research_notes)} chars")
    print(f"  Blog draft:      {len(final_state.blog_draft)} chars")
    print(f"  Final output:    {len(final_state.final_output)} chars")
    print(f"  Total steps:     {len(final_state.steps_log)}")


if __name__ == "__main__":
    main()
```

Run this and you'll see the full orchestration flow: the orchestrator receives the task, dispatches to the researcher, then the writer, then combines results. The shared state (`PipelineState`) flows through the system, and the execution log gives you a complete audit trail of what happened and when.

In a production version, the mock responses would be replaced with real LLM calls, the researcher would use search tools, and the orchestrator would use an LLM to decide which workers to invoke and how to synthesize their outputs. But the structure — shared state, worker dispatch, result combination — stays exactly the same.

---

## Key Takeaways

1. **Multi-agent is a tool, not a goal.** Don't use multiple agents because it sounds impressive. Use them because a single agent is demonstrably failing — its tool list is too long, its prompt is too complex, or it's too slow because it can't parallelize.

2. **Orchestrator-workers is the workhorse.** It's predictable, debuggable, and handles most real-world scenarios. Start here when you do need multiple agents.

3. **Peer-to-peer is powerful but dangerous.** The flexibility comes at the cost of predictability. Use it for creative collaboration tasks with 2-3 agents, not for production systems with SLAs.

4. **Hierarchical is for large systems.** If you have sub-teams building different parts of the agent system, hierarchy gives you natural ownership boundaries. But deep hierarchies are slow.

5. **Swarm is experimental.** Exciting for research, risky for production. The self-organizing behavior is hard to predict and harder to debug.

6. **Fan-out/fan-in is your friend.** Wherever you can parallelize, do it. It's the single biggest performance win in multi-agent systems.

7. **Shared state is simpler; message passing scales better.** Use shared state until you need loose coupling or distribution, then invest in message passing.

8. **The progression:** single agent → single agent with routing → orchestrator with workers → more complex patterns. Most systems never need to go past step three.

---

## What's Next

You know the patterns. Now let's look at the frameworks that implement them. Chapter 6 dives into LangGraph and how it turns these architectural patterns into executable code — nodes, edges, state, and the graph abstraction that ties it all together.
