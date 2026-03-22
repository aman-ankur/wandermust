# AI Agents Learning Guide — Design Spec

## Overview

A chapter-based educational documentation series teaching how to build AI agents. Two separate series: **Foundations** (general agent knowledge) and **Project Walkthrough** (travel-optimizer as applied example). Target audience: technical developer comfortable with LLM APIs but new to agent architectures.

**Location:** `/Users/aankur/workspace/travel-optimizer/docs/learning/`

**Style:** Interleaved concept → snippet → explanation. Heavy on theory with diverse examples. Industry references from OpenAI and Anthropic agent guides. Runnable commented example at end of each chapter.

---

## Series 1: Foundations (`docs/learning/foundations/`)

7 chapters covering AI agents broadly. No project-specific assumptions.

### Chapter 01: What Are AI Agents?
- **Opening:** "You've called an LLM API. You sent a prompt, got a response. But what if the LLM could decide on its own to search the web, check a database, and retry if something failed — all without you writing the control flow?"
- **Concepts:**
  - Chatbot vs agent: the autonomy spectrum
  - The agent loop: perceive → reason → act → observe (with diagram in ASCII)
  - What makes something an "agent" vs a "workflow" (Anthropic's distinction)
  - Real-world examples: coding agents, research agents, customer support agents
- **Industry refs:**
  - OpenAI: "An agent is a system where an LLM dynamically directs its own workflows and tool usage"
  - Anthropic: "Agents are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks"
  - Anthropic's agents vs workflows: "Workflows are deterministic; agents are autonomous"
- **Inline snippets:** Chatbot (3 lines) vs agent loop (15 lines pseudocode)
- **Runnable example:** Simple agent loop in Python using raw Claude/OpenAI API — asks a question, decides if it needs a tool, calls it, returns answer
- **~2000 words**

### Chapter 02: Tools — Giving LLMs Hands
- **Opening:** "An LLM can write poetry, but it can't check the weather. Unless you give it a tool."
- **Concepts:**
  - What is a tool? (function with a schema that an LLM can invoke)
  - Tool schemas: name, description, parameters (JSON Schema)
  - How LLMs decide to call tools (not magic — pattern matching on descriptions)
  - The tool call cycle: LLM outputs tool call → you execute it → feed result back
  - Tool design principles: clear names, good descriptions, minimal parameters
  - Multiple tools: LLM choosing between them
- **Industry refs:**
  - OpenAI: tools as one of three core agent components (model + tools + instructions)
  - Anthropic: "the augmented LLM" — base unit is LLM + retrieval + tools + memory
- **Inline snippets:**
  - Tool schema definition (JSON)
  - Tool execution handler (10 lines)
  - LLM choosing between 2 tools (API call example)
- **Runnable example:** Build a mini-agent with 3 tools (calculator, weather lookup, web search mock) — show the full loop of LLM deciding which tool to call
- **~2500 words**

### Chapter 03: State & Memory
- **Opening:** "A function call is stateless. An agent is not. How does an agent remember what it's doing?"
- **Concepts:**
  - Three types of memory:
    - **Short-term:** conversation history (messages array)
    - **Working memory:** current task state (variables, intermediate results)
    - **Long-term:** persistent storage (databases, files, vector stores)
  - State management in agent systems — why it matters
  - TypedDict / Pydantic as state containers
  - State as the communication channel between agents
  - Checkpointing: saving state so you can resume or debug
- **Industry refs:**
  - OpenAI: instructions + context as memory
  - Anthropic: state management in workflows vs agents
- **Inline snippets:**
  - Conversation history as memory (5 lines)
  - TypedDict state container (10 lines)
  - State passing between two functions (10 lines)
- **Runnable example:** Build a stateful research agent that accumulates findings across multiple tool calls, showing state growing over the loop
- **~2000 words**

### Chapter 04: Agent Architectures — Single Agent
- **Opening:** "You have an LLM with tools. But how does it decide what to do? The architecture defines the decision loop."
- **Concepts:**
  - **ReAct pattern:** Thought → Action → Observation loop. Most common. LLM reasons about what to do, takes action, observes result, repeats.
  - **Tool-use agent:** simpler — LLM just picks tools, no explicit reasoning step
  - **Plan-then-Execute:** LLM makes a full plan first, then executes steps. Better for complex tasks.
  - **Reflection / self-correction:** agent checks its own output and retries if wrong
  - When to use which: ReAct for exploration, Plan-Execute for multi-step, Reflection for quality
- **Industry refs:**
  - OpenAI: start with single agent, add complexity only when needed
  - Anthropic: prompt chaining as simplest pattern, routing for classification
- **Inline snippets:**
  - ReAct loop (15 lines)
  - Plan-then-Execute pseudocode (10 lines)
  - Self-correction check (10 lines)
- **Runnable example:** ReAct agent that researches a topic — thinks about what to search, searches, reads results, decides if it needs more info, synthesizes answer
- **~2500 words**

### Chapter 05: Agent Architectures — Multi-Agent
- **Opening:** "One agent can do a lot. But some problems naturally decompose into specialists. That's where multi-agent systems come in."
- **Concepts:**
  - Why multiple agents? Specialization, parallelism, separation of concerns
  - **Orchestrator pattern:** one boss agent delegates to workers (what our project uses)
  - **Peer-to-peer:** agents talk to each other directly, no central coordinator
  - **Hierarchical:** tree structure — manager → sub-managers → workers
  - **Swarm:** autonomous agents that self-organize (OpenAI Swarm concept)
  - **Fan-out / Fan-in:** dispatch multiple agents in parallel, collect results (our project's pattern)
  - Communication: shared state vs message passing
  - When NOT to use multi-agent: added complexity, harder to debug
- **Industry refs:**
  - OpenAI: "manager agent delegates to specialist agents"
  - Anthropic: orchestrator-workers pattern, parallelization pattern
  - Comparison: how each company frames the same ideas differently
- **Inline snippets:**
  - Orchestrator dispatching to 2 workers (pseudocode)
  - Fan-out / fan-in pattern (15 lines)
  - Shared state vs message passing comparison
- **Runnable example:** Simple orchestrator with 2 worker agents — one researches, one writes. Orchestrator decides who to call and when.
- **~2500 words**

### Chapter 06: Graphs & Orchestration Frameworks
- **Opening:** "You could wire agents together with if/else and function calls. But as complexity grows, you need a framework. Enter graphs."
- **Concepts:**
  - Why graphs? Nodes = agents/functions, Edges = flow, State = data traveling through
  - **LangGraph deep dive:**
    - StateGraph: define state type, add nodes, add edges
    - Nodes: functions that take state and return updates
    - Edges: normal (always follow), conditional (route based on state)
    - Parallel execution: multiple edges from one node = fan-out
    - Fan-in: multiple edges into one node = wait for all, then proceed
    - Compilation: graph.compile() turns definition into runnable
  - **CrewAI comparison:** role-based (agents have roles, goals, backstories). More opinionated, less transparent.
  - **Raw orchestration:** just Python functions and loops. Maximum control, maximum boilerplate.
  - Framework comparison table: LangGraph vs CrewAI vs raw code
  - When to use a framework vs not
- **Industry refs:**
  - Anthropic: "frameworks can help with standard low-level tasks... but also create extra layers of abstraction that can obscure the underlying prompts"
  - OpenAI: orchestration as the mechanism connecting model, tools, and instructions
- **Inline snippets:**
  - Minimal LangGraph (8 lines: define state, add 2 nodes, edge, compile, invoke)
  - Conditional edge (10 lines)
  - Parallel fan-out (10 lines)
- **Runnable example:** Build a 3-node LangGraph: input parser → processor → formatter. Show state flowing through, show how to add a conditional branch.
- **~2500 words**

### Chapter 07: Reliability & Production Patterns
- **Opening:** "Your agent works in a demo. Now make it work when the API is down, the LLM hallucinates, and you've burned through your rate limit."
- **Concepts:**
  - **Error handling:** graceful degradation, partial results, agent-level try/catch
  - **Fallbacks:** historical data, cached responses, simpler models
  - **Retries with backoff:** exponential backoff, jitter, max attempts
  - **Caching:** in-memory TTL, persistent (SQLite/Redis), when to cache what
  - **Guardrails:** input validation, output validation, content filtering
  - **Human-in-the-loop:** approval gates, escalation, confidence thresholds
  - **Cost management:** token budgets, caching to avoid redundant calls, choosing when LLM is needed vs deterministic code
  - **Observability:** logging agent decisions, tracing tool calls, debugging multi-agent flows
- **Industry refs:**
  - OpenAI: guardrails as essential component, evaluation-driven development
  - Anthropic: "keep your agent's scope focused, test in realistic scenarios"
- **Inline snippets:**
  - Retry with backoff (10 lines)
  - Fallback pattern (10 lines)
  - Input guardrail (10 lines)
- **Runnable example:** Agent with a tool that randomly fails 50% of the time — show retry logic, fallback to cache, and graceful error message
- **~2000 words**

---

## Series 2: Project Walkthrough (`docs/learning/project-walkthrough/`)

5 chapters mapping concepts to the travel-optimizer codebase.

### Chapter 01: Project Overview & Why Multi-Agent
- Map the travel problem to agent architecture decisions
- Why not a single agent? (parallel data fetching, separation of concerns, deterministic scoring)
- The orchestrator fan-out/fan-in pattern — why it's the right fit
- Reference: Foundations Ch 05 (multi-agent) and Ch 06 (graphs)
- Code: show the high-level graph definition from `graph.py`
- **~1500 words**

### Chapter 02: The Graph — Nodes, Edges, State
- `graph.py` annotated line-by-line
- `models.py` — TravelState as the shared communication channel
- How LangGraph compiles and executes: what happens when you call `graph.invoke()`
- The parallel execution: how 3 data agents run concurrently
- Fan-in: how scorer waits for all 3 to complete
- Reference: Foundations Ch 03 (state) and Ch 06 (graphs)
- **~2000 words**

### Chapter 03: Data Agents — Tools in Action
- Weather agent: geocoding → API call → scoring. Pure deterministic.
- Flights agent: IATA lookup → Amadeus API → parse → SQLite save → fallback logic
- Hotels agent: same pattern as flights
- How these agents are "tools with logic" — not LLM-powered, just functions
- The fallback pattern: live API → historical SQLite → graceful error
- Reference: Foundations Ch 02 (tools) and Ch 07 (reliability)
- **~2500 words**

### Chapter 04: Scoring & Synthesis — Deterministic + LLM
- Scorer: normalization (0-1), weighted ranking, handling missing dimensions
- Why scorer is NOT an LLM — deterministic is better here (cheaper, faster, predictable)
- Synthesizer: where the LLM adds value — natural language explanation
- The fallback: when LLM fails, return raw data (still useful)
- Decision framework: "when to use LLM vs deterministic code" checklist
- Reference: Foundations Ch 04 (single agent patterns) and Ch 07 (reliability)
- **~2000 words**

### Chapter 05: Putting It Together — End-to-End Flow
- Full lifecycle: user enters "Tokyo, Jun-Sep" → Streamlit → graph → results
- Trace through a real request: supervisor generates 16 windows, 3 agents fetch in parallel, scorer ranks, synthesizer explains
- Error scenarios: what if Amadeus is down? What if LLM fails? What if geocoding can't find the city?
- State at each step (show the TravelState growing as it flows through nodes)
- Performance: ~65 API calls, caching strategy, SQLite building up over time
- Reference: all Foundations chapters
- **~2000 words**

---

## Cross-References

Each walkthrough chapter links back to the relevant foundations chapter. Each foundations chapter has a "See this in practice" callout pointing to the relevant walkthrough chapter (added after both series are written).

## File Naming

```
docs/learning/
├── foundations/
│   ├── 01-what-are-ai-agents.md
│   ├── 02-tools-giving-llms-hands.md
│   ├── 03-state-and-memory.md
│   ├── 04-single-agent-architectures.md
│   ├── 05-multi-agent-architectures.md
│   ├── 06-graphs-and-orchestration.md
│   └── 07-reliability-and-production.md
├── project-walkthrough/
│   ├── 01-project-overview.md
│   ├── 02-the-graph.md
│   ├── 03-data-agents.md
│   ├── 04-scoring-and-synthesis.md
│   └── 05-end-to-end-flow.md
└── README.md  (reading order + chapter summaries)
```

## Total Scope

- **Foundations:** ~16,000 words across 7 chapters
- **Walkthrough:** ~10,000 words across 5 chapters
- **Total:** ~26,000 words, 12 markdown files + 1 README
