# Discovery v2: Hybrid Deterministic + LLM Conversation Engine

## Context

The current conversation engine delegates ALL logic to one LLM call per turn — topic selection, option generation, phase transitions, state tracking, AND natural language. This causes: stuck loops (budget question asked 5x), wrong options (destinations in profile phase), `phase_complete` never set, `multi_select` never used, and free-text answers ignored. Additionally, the Streamlit UI has a race condition where rapid clicks fire duplicate API calls causing phase jumps.

**Goal:** Split into deterministic logic (Python) + LLM personality (one focused call). Make stuck loops structurally impossible.

## Architecture

```
User clicks option
    ↓
[Deterministic Controller]  ← NEW: api/conversation_controller.py
  - Parse answer → update known_facts dict
  - Check phase transition (all required facts filled? max turns?)
  - Pick next topic (first unfilled fact in current phase)
  - Return: topic, option templates, multi_select, phase
    ↓
[LLM Personality Layer]  ← SIMPLIFIED: api/conversation_engine.py
  - Input: "Ask about {topic}. Options: {labels}. React to: {last_answer}"
  - Output: question phrasing, reaction, option insights
  - Cannot break flow — doesn't control logic
    ↓
[Assemble ConversationTurn] → render in UI
```

## Files to Change

| File | Action | Est. Lines |
|---|---|---|
| `api/conversation_controller.py` | CREATE | ~250 |
| `api/conversation_engine.py` | REWRITE | ~150 |
| `api/routes.py` | REWRITE | ~120 |
| `api/models.py` | MODIFY | ~5 |
| `api/session.py` | MODIFY | ~5 |
| `app.py` | MODIFY | ~25 |
| `tests/test_conversation_controller.py` | CREATE | ~200 |

## Implementation Steps

### Step 1: Create `api/conversation_controller.py` (~250 lines)

Pure Python, zero LLM calls. The brain of the system.

#### 1a. TopicConfig dataclass

```python
@dataclass
class TopicConfig:
    key: str                     # "passport", "budget_level", etc.
    phase: str                   # "profile" or "discovery"
    question_hint: str           # What the LLM should ask about
    option_templates: list       # [{id, label}] — LLM adds insights
    multi_select: bool = False
```

#### 1b. TOPIC_REGISTRY — all topics defined statically

```python
TOPIC_REGISTRY = [
    # Profile phase
    TopicConfig("passport", "profile", "what passport they hold",
                [{"id": "in", "label": "Indian"}, {"id": "us", "label": "US"},
                 {"id": "uk", "label": "UK"}, {"id": "eu", "label": "EU/Schengen"},
                 {"id": "other", "label": "Other"}]),
    TopicConfig("budget_level", "profile", "their budget comfort level",
                [{"id": "budget", "label": "Budget-friendly"},
                 {"id": "moderate", "label": "Mid-range"},
                 {"id": "comfortable", "label": "Comfortable"},
                 {"id": "luxury", "label": "Luxury"}]),
    TopicConfig("travel_style", "profile", "their preferred travel style",
                [{"id": "adventure", "label": "Adventure & outdoors"},
                 {"id": "culture", "label": "Culture & history"},
                 {"id": "relaxation", "label": "Relaxation & beaches"},
                 {"id": "foodie", "label": "Food & culinary"},
                 {"id": "mix", "label": "Mix of everything"}]),
    # Discovery phase
    TopicConfig("timing", "discovery", "when they plan to travel",
                [{"id": "soon", "label": "Next 1-2 months"},
                 {"id": "quarter", "label": "3-6 months out"},
                 {"id": "later", "label": "6+ months out"},
                 {"id": "flexible", "label": "Flexible"}]),
    TopicConfig("companions", "discovery", "who they're traveling with",
                [{"id": "solo", "label": "Solo"}, {"id": "couple", "label": "With partner"},
                 {"id": "friends", "label": "With friends"},
                 {"id": "family", "label": "Family with kids"}]),
    TopicConfig("interests", "discovery", "activities and experiences that excite them",
                [{"id": "food", "label": "Street food & local cuisine"},
                 {"id": "nature", "label": "Nature & hiking"},
                 {"id": "history", "label": "History & architecture"},
                 {"id": "nightlife", "label": "Nightlife & bars"},
                 {"id": "beaches", "label": "Beaches & water sports"},
                 {"id": "shopping", "label": "Markets & shopping"}],
                multi_select=True),
    TopicConfig("deal_breakers", "discovery", "any deal-breakers to avoid",
                [{"id": "long_flight", "label": "Long flights (10+ hours)"},
                 {"id": "extreme_heat", "label": "Extreme heat"},
                 {"id": "crowds", "label": "Overcrowded tourist spots"},
                 {"id": "visa_hassle", "label": "Complex visa process"},
                 {"id": "none", "label": "No deal-breakers"}],
                multi_select=True),
]
```

#### 1c. Core functions

- **`pick_next_topic(known_facts, phase)`**: Iterate TOPIC_REGISTRY, return first topic where `topic.phase == phase` and `topic.key not in known_facts`. Returns None when all filled.
- **`parse_answer(topic, option_ids, free_text)`**: If option_ids provided, map to fact value directly. For multi_select, store list of labels. For free text, keyword-match against option labels. Store raw text if no match.
- **`should_transition(phase, known_facts, turn_count)`**: Profile→Discovery when passport+budget_level+travel_style all in known_facts. Discovery→Narrowing when timing+companions+interests all filled. Safety net: force after MAX_TURNS (profile=5, discovery=6).
- **`build_controller_turn(session, option_ids, free_text)`**: Main orchestrator. Parse previous answer → check transition → pick next topic → return control dict.
- **`build_trip_intent_from_facts(known_facts)`**: Maps known_facts to trip_intent format (zero LLM).

### Step 2: Modify `api/models.py` (~5 lines)

```python
# Add to DiscoveryRespondRequest:
option_ids: Optional[List[str]] = None

# Add to ConversationTurn:
topic: Optional[str] = None
```

### Step 3: Modify `api/session.py` (~5 lines)

Add to session creation dict:
```python
"known_facts": {},
"last_topic_key": None,
```

### Step 4: Simplify `api/conversation_engine.py` (~150 lines)

Replace monolithic `generate_turn()` with two focused functions:

#### 4a. `generate_personality()` — for profile/discovery

Prompt (~40% smaller than current):
```
You are a well-traveled friend (60+ countries). Opinionated, insightful, honest.

User context: {known_facts_summary}
{knowledge_context}

Topic to ask about: {question_hint}
Options (keep these exact labels): {option_labels}
User's last answer: {last_answer}

Return JSON: {"reaction": "1-2 sentences", "question": "natural phrasing", "option_insights": ["insight per option"]}
```

#### 4b. `generate_destinations()` — for narrowing/reveal

LLM generates destination suggestions with hooks, reasoning, reaction options. This is where LLM reasoning is genuinely needed.

#### 4c. Delete `decide_next_phase()` (replaced by controller)
#### 4d. Keep `FALLBACK_TURNS` for resilience when LLM fails

### Step 5: Modify `api/routes.py` (~120 lines)

Wire controller + personality:

- `/start`: Init known_facts from saved profile → controller picks first topic → personality phrases it
- `/respond`: Controller parses answer + picks next topic → personality or destinations → assemble ConversationTurn
- Replace `_extract_and_save_profile` with `_save_profile_from_facts` (zero LLM — known_facts already structured)
- `_assemble_turn(ctrl, personality)`: Combine controller options with LLM insights into ConversationTurn
- `_assemble_destination_turn(ctrl, dest_data)`: Build narrowing/reveal turns from LLM destination data

### Step 6: Fix `app.py` race condition (~25 lines)

```python
# Session state init:
if "v2_submitting" not in st.session_state:
    st.session_state.v2_submitting = False

# Guard at top of render:
if st.session_state.v2_submitting:
    with st.spinner("Thinking..."):
        st.stop()

# In _submit_answer():
def _submit_answer(answer, option_ids=None):
    if st.session_state.v2_submitting:
        return
    st.session_state.v2_submitting = True
    # ... API call ...
    st.session_state.v2_submitting = False
    st.rerun()

# Button clicks pass option_ids:
if st.button(label, key=f"v2_opt_{opt.id}"):
    _submit_answer(opt.label, option_ids=[opt.id])
```

### Step 7: Tests — `tests/test_conversation_controller.py` (~200 lines)

Pure Python, zero LLM mocking:

1. **Topic selection**: empty facts → passport first; passport filled → budget_level; all profile filled → None
2. **Answer parsing**: option click maps to fact; multi-select stores list; free text keyword matches; no match stores raw
3. **Phase transitions**: all profile facts → "discovery"; missing facts → stays; max turns → forces; all discovery facts → "narrowing"
4. **Full controller flow**: empty session → passport topic; after answers → correct progression through phases
5. **Trip intent from facts**: known_facts → structured trip_intent dict

## Key Decisions

1. **Free-text parsing**: Keyword matching, not LLM. Raw text stored as fallback.
2. **Narrowing/reveal**: LLM controls destinations entirely (needs reasoning). Controller only manages transitions.
3. **Profile extraction**: Zero LLM — built from known_facts dict directly.
4. **Backward compatible**: New fields are all Optional with defaults.

## What This Fixes

| Bug | Root Cause | Fix |
|---|---|---|
| Budget question loops 5x | LLM can't track state | Controller tracks known_facts, picks unfilled topics |
| Destinations in profile options | LLM ignores rules | Controller generates options, not LLM |
| phase_complete never set | LLM forgets to set flag | Controller decides transitions deterministically |
| multi_select never true | LLM ignores instruction | Controller sets it per TopicConfig |
| Only 2-3 options | LLM generates too few | Controller provides fixed option templates (4-6) |
| Duplicate API calls (race) | No submission lock in Streamlit | v2_submitting flag + st.stop() |
| Profile extraction LLM call | Unnecessary — we have the facts | Direct dict lookup, zero LLM |

## Verification

1. `pytest tests/test_conversation_controller.py -v` — all deterministic tests pass
2. Full suite: `pytest tests/ -v --ignore=tests/test_flights_agent.py --ignore=tests/test_graph.py --ignore=tests/test_hotels_agent.py --ignore=tests/test_mock_agents.py --ignore=tests/test_serpapi_client.py`
3. Server: `uvicorn api.main:create_app --factory --port 8000 --reload`
4. Curl test: POST /start → 3 /respond calls (passport, budget, style) → verify phase transitions to discovery
5. Streamlit: `streamlit run app.py` — verify no duplicates on rapid clicks, multi-select works, options match questions
