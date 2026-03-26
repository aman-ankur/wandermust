# Discovery v2 — Conversational Redesign

## Problem

The current "Discover Where" flow is a 10-question form dressed as a chat. Five fixed onboarding questions + five fixed discovery questions, all with hardcoded options, always in the same order. No reactions, no insights, no progressive reveal. It feels like filling out a travel agency intake form, not brainstorming with a well-traveled friend.

## Goal

Redesign discovery as an adaptive, insight-rich conversation that:
- Reacts to what you say (acknowledges, builds on it)
- Teaches you something with every option ("July? That's when Georgia has perfect weather AND wine harvest festivals")
- Progressively reveals destinations mid-conversation, not just dumps them at the end
- Respects real-world constraints (visa for your passport, INR exchange rates, flight connectivity)
- Responds in <1 second — feels instant, not like waiting for an API

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Architecture | State machine with LLM-generated content | Predictable phases + adaptive content within each |
| LLM | Model-agnostic with fallback chain | Primary: GPT-4o-mini. Fallbacks: Claude Haiku, Gemini Flash, local LLMs. Provider abstraction layer. |
| Backend | FastAPI (API-first) | Frontend-agnostic, enables React migration later |
| Frontend (now) | Streamlit (thin render layer) | Ship fast, iterate on conversation logic |
| Frontend (later) | React | Richer UX when ready |
| Knowledge | Pre-baked data modules + LLM reasoning | Accurate visa/budget/exchange data, not LLM guesses |
| Streaming | Yes, via SSE | Reaction text appears instantly while options generate |

---

## Architecture

### Conversation API

```
POST /api/discovery/start       → first ConversationTurn
POST /api/discovery/respond     → accepts answer, returns next ConversationTurn
GET  /api/discovery/state       → current session state
POST /api/discovery/select      → user picks destination → bridge to optimizer
```

### ConversationTurn (the core response type)

Every API response returns this structure. The LLM generates it. The UI renders it.

```python
class Option(BaseModel):
    id: str                    # "beaches", "georgia", etc.
    label: str                 # "Beaches & coast"
    insight: str               # "July is peak for Mediterranean + SE Asia coasts"
    emoji: str | None = None   # optional visual

class DestinationHint(BaseModel):
    name: str                  # "Tbilisi, Georgia"
    hook: str                  # "Visa-free, ₹3k/day, wine country with mountain views"
    match_reason: str          # "Matches your budget + food + culture interests"

class ConversationTurn(BaseModel):
    phase: str                 # "profile" | "discovery" | "narrowing" | "reveal"
    reaction: str | None       # Brief insight reacting to previous answer
    question: str              # The next question
    options: list[Option]      # 3-6 smart options with embedded insights
    multi_select: bool = False
    can_free_text: bool = True
    destination_hints: list[DestinationHint] | None = None  # Phase 3+
    thinking: str | None = None  # "I keep coming back to the Caucasus for you..."
    phase_complete: bool = False  # LLM signals ready to transition
```

### State Machine (4 phases)

```
Profile → Discovery → Narrowing → Reveal
```

| Phase | LLM's job | Transitions when | Min turns |
|-------|-----------|-----------------|-----------|
| **Profile** (first-time only) | Fill critical gaps: passport, budget level, travel style | LLM says it knows enough to reason about destinations | 2 |
| **Discovery** | Understand this trip: timing, companions, interests, constraints | LLM says it has a clear picture | 2 |
| **Narrowing** | Think out loud, show early ideas, refine based on reactions | User signals readiness or LLM detects convergence | 1 |
| **Reveal** | Present 3-5 polished final suggestions | User picks one | 1 |

Phase transitions are LLM-decided via the `phase_complete` flag, not hardcoded question counts. The state machine enforces minimum turns as a safety net.

Returning users skip Profile entirely (loaded from SQLite).

---

## Pre-Baked Knowledge Modules

Instead of relying on LLM general knowledge for factual travel data, we maintain small curated data files that get injected into the prompt based on user context. This keeps prompts compact AND accurate.

### Visa Tiers (by passport)

```python
# knowledge/visa_tiers.py
VISA_TIERS = {
    "IN": {
        "visa_free": ["Thailand", "Georgia", "Serbia", "Mauritius", "Nepal",
                       "Bhutan", "Fiji", "Maldives", "Indonesia", "Qatar"],
        "e_visa_easy": ["Vietnam", "Turkey", "Sri Lanka", "Cambodia", "Kenya",
                         "Ethiopia", "Myanmar", "Laos", "Azerbaijan"],
        "visa_required": ["Japan", "South Korea", "UAE", "Malaysia"],
        "hard_visa": ["US", "UK", "EU/Schengen", "Canada", "Australia", "NZ"]
    }
    # Expandable for other passport countries
}
```

### Budget Tiers (INR)

```python
# knowledge/budget_tiers.py
BUDGET_TIERS = {
    "IN": {
        "budget": {
            "daily_range": "₹2,000-4,000",
            "examples": ["Vietnam", "Nepal", "Georgia", "Cambodia", "Sri Lanka"],
            "note": "Hostels, street food, local transport"
        },
        "moderate": {
            "daily_range": "₹5,000-8,000",
            "examples": ["Thailand", "Turkey", "Bali", "Malaysia", "Azerbaijan"],
            "note": "Mid-range hotels, restaurants, some activities"
        },
        "comfortable": {
            "daily_range": "₹10,000-15,000",
            "examples": ["Japan", "South Korea", "Dubai", "Eastern Europe"],
            "note": "Good hotels, full experiences, domestic flights"
        },
        "luxury": {
            "daily_range": "₹15,000+",
            "examples": ["Western Europe", "Australia", "Scandinavia", "Switzerland"],
            "note": "INR exchange hurts here — budget 3-4x more than SE Asia"
        }
    }
}
```

### Exchange Rate Favorability

```python
# knowledge/exchange_rates.py
EXCHANGE_FAVORABILITY = {
    "IN": {
        "great_value": ["THB", "GEL", "VND", "LKR", "NPR", "KHR", "IDR"],
        "decent_value": ["TRY", "MYR", "AZN", "RSD", "KES"],
        "poor_value": ["EUR", "GBP", "USD", "AUD", "JPY", "CHF", "SGD", "NZD"],
        "note": "INR to EUR/GBP/USD is unfavorable — European/US trips cost 3-4x more than SE Asia equivalent"
    }
}
```

### Seasonality

```python
# knowledge/seasonality.py
SEASONALITY = {
    "Thailand": {"best": [11,12,1,2,3], "avoid": [4,5], "note": "Nov-Mar dry season"},
    "Georgia": {"best": [5,6,7,8,9], "avoid": [12,1,2], "note": "Summer wine harvest"},
    "Vietnam": {"best": [2,3,4,10,11], "avoid": [7,8,9], "note": "Varies by region"},
    # ... curated for top 40-50 destinations relevant to Indian travelers
}
```

### Context Injection

At each turn, the engine builds a compact knowledge block (~300 tokens) injected into the system prompt:

```python
def build_context(passport: str, budget: str, month: int | None) -> str:
    visa = VISA_TIERS.get(passport, {})
    budget_tier = BUDGET_TIERS.get(passport, {}).get(budget, {})
    exchange = EXCHANGE_FAVORABILITY.get(passport, {})

    context = f"""
TRAVEL INTELLIGENCE (for {passport} passport, {budget} budget):
- Visa-free: {', '.join(visa.get('visa_free', [])[:8])}
- E-visa easy: {', '.join(visa.get('e_visa_easy', [])[:6])}
- Hard visa (avoid unless asked): {', '.join(visa.get('hard_visa', [])[:5])}
- Budget range: {budget_tier.get('daily_range', 'unknown')}
- Great value currencies: {', '.join(exchange.get('great_value', [])[:5])}
- Poor value currencies: {', '.join(exchange.get('poor_value', [])[:5])}
- {exchange.get('note', '')}
"""
    if month:
        good_now = [d for d, s in SEASONALITY.items() if month in s["best"]]
        context += f"- Good in month {month}: {', '.join(good_now[:8])}\n"

    return context
```

This replaces pages of LLM prompt with ~300 tokens of curated, accurate data.

---

## System Prompt Design

One system prompt (~800 tokens) that stays constant, plus the injected knowledge context (~300 tokens).

```
You are a well-traveled friend helping plan a trip. You've been to 60+ countries
and have strong opinions. You're not a search engine — you're someone who says
"oh you HAVE to go to Georgia, the food scene there blew my mind."

PERSONALITY:
- Opinionated: you have favorites and you share them with reasons
- Insightful: every option teaches something the user didn't know
- Reactive: you acknowledge what they said before moving forward
- Honest: if a destination is expensive or has visa issues, say so upfront
- Concise: reactions are 1-2 sentences, not paragraphs

RULES:
- Never suggest hard-visa destinations unless the user specifically asks
- Always frame budgets in the user's currency
- Flag exchange rate issues honestly ("Europe is incredible but INR to EUR hurts")
- Each option MUST include a brief insight, not just a label
- When reacting, be specific to what they said, not generic ("Great choice!")

CURRENT PHASE: {phase}
PHASE RULES: {phase_specific_rules}

{injected_knowledge_context}

Respond with a valid ConversationTurn JSON object.
```

Phase-specific prompt additions:

- **Profile phase:** "Ask about the most important gaps. You need passport, budget comfort, and travel style to make good suggestions. Skip what you already know."
- **Discovery phase:** "Understand this specific trip. Timing, companions, interests, constraints. React to each answer with a brief insight. When you have enough to start thinking about destinations, set phase_complete: true."
- **Narrowing phase:** "Think out loud. Share 4-6 destination ideas in destination_hints. Include a 'thinking' field explaining your reasoning. The user can react to each destination. Refine based on their reactions."
- **Reveal phase:** "Present your final 3-5 curated suggestions. Each gets a rich reason (2-3 sentences), budget estimate, best timing, and one 'what you might not expect' insight."

---

## Streaming Strategy

For sub-1-second perceived response time:

1. **SSE (Server-Sent Events)** from FastAPI endpoint
2. LLM response streams as it generates
3. Parse incrementally:
   - `reaction` field appears first → render immediately as chat bubble
   - `question` field next → render below reaction
   - `options` array last → render as smart pills
4. User sees reaction text within 200-300ms of submitting their answer

```python
@app.post("/api/discovery/respond")
async def respond(request: DiscoveryRequest):
    return StreamingResponse(
        stream_conversation_turn(request),
        media_type="text/event-stream"
    )

async def stream_conversation_turn(request):
    async for chunk in llm.stream(messages, response_format=ConversationTurn):
        yield f"data: {chunk.json()}\n\n"
```

For Streamlit (which doesn't natively support SSE), we call the API non-streaming but use `st.empty()` containers to progressively render reaction → question → options as they arrive from parsing the complete response. Still fast since GPT-4o-mini returns ~300 tokens in ~500ms.

---

## Narrowing Phase Detail

This is the new phase that creates the "brainstorming session" feeling.

### Turn 1: Initial reveal
LLM presents 4-6 rough destination ideas based on everything learned so far:

```json
{
  "phase": "narrowing",
  "reaction": "Alright, I've got a pretty clear picture — solo food trip in July, moderate budget, Indian passport. Let me think out loud...",
  "thinking": "I keep coming back to three regions: the Caucasus (visa-free, insanely cheap, incredible food), coastal Vietnam (e-visa, street food paradise), and Turkey (e-visa, where East meets West on a plate).",
  "question": "Here's what I'm thinking. React to any of these — what catches your eye, what doesn't feel right?",
  "options": [
    {"id": "react_positive", "label": "Tell me more about...", "insight": "Pick a destination below to explore deeper"},
    {"id": "react_negative", "label": "Not feeling...", "insight": "I'll drop it and find alternatives"},
    {"id": "ready", "label": "I've seen enough, show me the final picks", "insight": "Skip to polished recommendations"}
  ],
  "destination_hints": [
    {"name": "Tbilisi, Georgia", "hook": "Visa-free, ₹3k/day, wine country + mountain views + the best khachapuri of your life", "match_reason": "Budget + food + culture trifecta"},
    {"name": "Istanbul, Turkey", "hook": "E-visa, ₹6k/day, where spice markets meet Bosphorus sunsets", "match_reason": "Food capital at the crossroads of continents"},
    {"name": "Da Nang, Vietnam", "hook": "E-visa, ₹2.5k/day, beach + bánh mì + the best central Vietnam food trail", "match_reason": "Beach + food + extremely budget-friendly"},
    {"name": "Penang, Malaysia", "hook": "Visa required but easy, ₹4k/day, arguably Asia's street food capital", "match_reason": "Food-first destination, July is fine weather"},
    {"name": "Oaxaca, Mexico", "hook": "E-visa, ₹5k/day, mole capital of the world + mezcal + Guelaguetza festival in JULY", "match_reason": "Wildcard — food + festival timing is perfect"}
  ]
}
```

### Turn 2+: Refinement
User says "Tell me more about Georgia" or "Not feeling Mexico — too far"

LLM reacts: "Fair — Mexico is a long haul from India. Let me swap it out. And Georgia? Let me tell you about Sioni district in Tbilisi..."

Refined list drops Mexico, adds a new option, goes deeper on Georgia.

### Transition
User selects "I've seen enough" or LLM detects convergence (user keeps liking the same 2-3) → move to Reveal phase.

---

## File Structure (changes)

```
knowledge/                    # NEW — pre-baked travel data
  __init__.py
  visa_tiers.py
  budget_tiers.py
  exchange_rates.py
  seasonality.py
  context_builder.py          # build_context() function

agents/
  conversation_engine.py      # NEW — replaces onboarding.py + discovery_chat.py
  suggestion_generator.py     # MODIFIED — uses knowledge modules
  discovery_bridge.py         # UNCHANGED

api/                          # NEW — FastAPI backend
  __init__.py
  main.py                     # FastAPI app
  routes.py                   # /start, /respond, /state, /select
  models.py                   # ConversationTurn, Option, DestinationHint

discovery_graph_v2.py         # NEW — 4-phase state machine
models.py                     # MODIFIED — add new types
app.py                        # MODIFIED — Streamlit renders API responses
config.py                     # MODIFIED — add gpt-4o-mini config
```

### What gets deleted/replaced
- `agents/onboarding.py` — merged into `conversation_engine.py`
- `agents/discovery_chat.py` — merged into `conversation_engine.py`
- `ONBOARDING_QUESTIONS` and `BASE_QUESTIONS` constants — gone entirely
- `discovery_graph.py` — replaced by `discovery_graph_v2.py`

### What stays unchanged
- `agents/discovery_bridge.py` — bridge logic is fine
- `db.py` — profile + session persistence
- `agents/mock_agents.py` — updated for new structure
- All optimizer code — untouched

---

## Streamlit UI Changes

The UI becomes a thin render layer for `ConversationTurn`:

```python
def render_turn(turn: ConversationTurn):
    # 1. Show reaction as assistant message
    if turn.reaction:
        with st.chat_message("assistant"):
            st.write(turn.reaction)

    # 2. Show thinking (narrowing phase)
    if turn.thinking:
        with st.chat_message("assistant"):
            st.write(f"*{turn.thinking}*")

    # 3. Show destination hints (narrowing/reveal phase)
    if turn.destination_hints:
        for hint in turn.destination_hints:
            with st.container(border=True):
                st.markdown(f"**{hint.name}**")
                st.write(hint.hook)
                st.caption(hint.match_reason)

    # 4. Show question
    with st.chat_message("assistant"):
        st.write(turn.question)

    # 5. Show smart options as pills
    cols = st.columns(min(len(turn.options), 3))
    for i, opt in enumerate(turn.options):
        with cols[i % 3]:
            if st.button(f"{opt.emoji or ''} {opt.label}", key=opt.id):
                handle_selection(opt.id)
            st.caption(opt.insight)
```

---

## Testing Strategy

- **Unit tests:** Conversation engine with mocked LLM — verify phase transitions, JSON structure
- **Knowledge module tests:** Verify visa/budget/seasonality data accuracy
- **Integration tests:** Full 4-phase flow with mocked LLM responses
- **Prompt tests:** Golden-file tests with canned inputs → verify output quality
- **Speed tests:** Assert API response time <1s with GPT-4o-mini
- **Existing tests:** All optimizer tests must continue to pass (untouched code)

---

## Open Details (to resolve during planning)

- **Session management:** API endpoints need a session ID (path param or header). Simplest: UUID generated on `/start`, passed back on subsequent calls.
- **LLM error handling:** If GPT-4o-mini returns invalid JSON or times out, retry once, then fall back to a canned "sorry, let me try that again" turn. Don't block the conversation.
- **Conversation history storage:** Extend `discovery_sessions` table with a `conversation_history` JSON column, or store in LangGraph checkpointer during session and persist only on completion.
- **Config update:** Change existing `discovery_model` from `gpt-4o` to `gpt-4o-mini` (not a new setting).
- **Python version:** Use `Optional[str]` style if runtime is <3.10, or upgrade Python.

## Migration Path

1. Build knowledge modules + conversation engine (backend only)
2. Build FastAPI endpoints
3. Test full flow via API (curl/httpie)
4. Update Streamlit to render ConversationTurn
5. Remove old onboarding + discovery chat agents
6. Later: React frontend consuming same API
