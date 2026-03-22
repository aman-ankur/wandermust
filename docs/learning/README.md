# Learning to Build AI Agents

A chapter-based guide to understanding and building AI agent systems, from first principles to a working multi-agent project.

## How to Read This

**Two series, one learning path:**

1. **Start with Foundations** — general AI agent concepts, applicable to any project
2. **Then read the Project Walkthrough** — see every concept applied in the travel-optimizer

Each foundations chapter builds on the previous. The walkthrough assumes you've read the foundations.

## Series 1: Foundations

General AI agent knowledge. No project-specific assumptions.

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 01 | [What Are AI Agents?](foundations/01-what-are-ai-agents.md) | Agent vs chatbot, the agent loop, autonomy spectrum, agents vs workflows |
| 02 | [Tools — Giving LLMs Hands](foundations/02-tools-giving-llms-hands.md) | Tool schemas, how LLMs choose tools, the tool call cycle, design principles |
| 03 | [State & Memory](foundations/03-state-and-memory.md) | Short-term/working/long-term memory, state as communication, checkpointing |
| 04 | [Single Agent Architectures](foundations/04-single-agent-architectures.md) | ReAct, tool-use, plan-then-execute, reflection, routing patterns |
| 05 | [Multi-Agent Architectures](foundations/05-multi-agent-architectures.md) | Orchestrator, peer-to-peer, hierarchical, swarm, fan-out/fan-in |
| 06 | [Graphs & Orchestration](foundations/06-graphs-and-orchestration.md) | LangGraph deep dive, nodes/edges/state, CrewAI comparison, framework choice |
| 07 | [Reliability & Production](foundations/07-reliability-and-production.md) | Error handling, fallbacks, retries, caching, guardrails, cost management |

## Series 2: Project Walkthrough

How the travel-optimizer implements these concepts.

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 01 | [Project Overview](project-walkthrough/01-project-overview.md) | Why multi-agent, why orchestrator pattern, the architecture |
| 02 | [The Graph](project-walkthrough/02-the-graph.md) | graph.py line-by-line, state flow, parallel execution, fan-in mechanics |
| 03 | [Data Agents](project-walkthrough/03-data-agents.md) | Weather/flights/hotels agents, API integration, SQLite fallback pattern |
| 04 | [Scoring & Synthesis](project-walkthrough/04-scoring-and-synthesis.md) | Normalization, weighted ranking, LLM vs deterministic decision framework |
| 05 | [End-to-End Flow](project-walkthrough/05-end-to-end-flow.md) | Full request trace, error scenarios, performance budget |

## Industry References

These chapters draw from two authoritative guides:

- **OpenAI** — [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- **Anthropic** — [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

## After Reading

Ready to build? See [BUILD_GUIDE.md](../../BUILD_GUIDE.md) for the complete implementation plan — 14 tasks, test-first, with all the code.
