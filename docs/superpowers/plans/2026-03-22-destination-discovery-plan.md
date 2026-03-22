# Destination Discovery Agent — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-03-22-destination-discovery-design.md`

---

## Phase 1: Data Layer (no UI, no LLM — pure foundation)

### Task 1.1: Extend `db.py` with profile and history tables

**Files:** `db.py`

Add to `HistoryDB._create_tables()`:
- `user_profile` table (single-row, `id=1` constraint) with fields: origin, nationality, preferred_climate (JSON), budget_tier, trip_styles (JSON), interests (JSON), dealbreakers (JSON), max_flight_hours, updated_at
- `travel_history` table with fields: destination, country, trip_type, month_visited, year_visited, duration_days, status, notes, created_at

Add methods:
- `save_profile(origin, nationality, ...)` — INSERT OR REPLACE with id=1
- `get_profile() -> dict | None`
- `save_history_entry(destination, country, ...)`
- `get_history() -> List[dict]`
- `get_history_by_status(status) -> List[dict]`

**Tests:** `tests/test_db_discovery.py` — CRUD operations, single-row profile constraint, history queries

**Verification:** `pytest tests/test_db_discovery.py -v`

---

### Task 1.2: Create discovery data models

**Files:** `discovery_models.py` (new)

Pydantic models:
- `UserProfile` — origin, nationality, preferred_climate, budget_tier, trip_styles, interests, dealbreakers, max_flight_hours
- `TravelHistoryEntry` — destination, country, trip_type, month_visited, year_visited, duration_days, status
- `DestinationSuggestion` — destination, country_code, why_you, visa_situation, flight_estimate, timing_verdict, budget_fit, novelty_note, tags, confidence

TypedDict:
- `DiscoveryState` — profile, history, profile_exists, conversation_messages, trip_intent, region_preference, date_range (Optional), duration_days, budget_max, questions_asked, suggestions, chosen_destination, errors

**Tests:** `tests/test_discovery_models.py` — Pydantic validation, optional fields, JSON serialization

**Verification:** `pytest tests/test_discovery_models.py -v`

---

### Task 1.3: Add `discovery_context` to TravelState

**Files:** `models.py`

Add `discovery_context: Optional[Dict[str, Any]]` field to `TravelState` TypedDict. Keys when present: `trip_intent`, `why_chosen`, `tags`.

**Tests:** Existing tests should still pass — field is optional with total=False.

**Verification:** `pytest tests/ -v`

---

### Task 1.4: Create `services/profile_manager.py`

**Files:** `services/profile_manager.py` (new)

High-level interface over db.py:
- `load_profile(db: HistoryDB) -> UserProfile | None`
- `save_profile(db: HistoryDB, profile: UserProfile)`
- `load_history(db: HistoryDB) -> List[TravelHistoryEntry]`
- `add_history_entry(db: HistoryDB, entry: TravelHistoryEntry)`
- `infer_preferences(history: List[TravelHistoryEntry]) -> dict` — analyzes history patterns to extract preferred climate, budget tier, trip styles, regions. Returns dict of inferred fields.
- `format_profile_for_llm(profile: UserProfile, history: List[TravelHistoryEntry]) -> str` — formats as structured text for LLM prompt injection.

**Tests:** `tests/test_profile_manager.py` — load/save round-trip, inference from sample histories, format output

**Verification:** `pytest tests/test_profile_manager.py -v`

---

## Phase 2: Discovery Agents (LLM-powered, no UI yet)

### Task 2.1: Create `agents/onboarding.py`

**Files:** `agents/onboarding.py` (new)

Re-entrant node using LangGraph `interrupt()`:
- First invocation: asks "Where have you travelled in the last few years?"
- Second invocation: receives free-text history → sends to LLM to extract structured entries → asks "What's your home city and nationality?"
- Third invocation: receives origin/nationality → asks about dealbreakers
- Fourth invocation: receives dealbreakers → returns (done)

LLM calls: one call to parse free-text travel history into structured `TravelHistoryEntry` list.

Uses `Command` to route back to self or forward to `build_profile`.

**Tests:** `tests/test_onboarding.py` — mock LLM, test each re-entry cycle, verify state accumulation, test with empty history

**Verification:** `pytest tests/test_onboarding.py -v`

---

### Task 2.2: Create `agents/discovery.py`

**Files:** `agents/discovery.py` (new)

Two functions in one file:

**A) `discovery_chat_node`** — re-entrant conversational node:
- Loads profile + history from state
- Generates adaptive questions (skips what's already known from profile)
- Uses `interrupt()` for each question
- Tracks `questions_asked`, caps at 5
- Uses `has_enough_info()` check: needs at minimum trip_intent + timing + duration
- Routes to `generate_suggestions` when ready

**B) `generate_suggestions_node`** — single LLM call:
- Constructs the rich prompt from spec (context, constraints, output format)
- Sends to LLM (same OpenRouter config as synthesizer)
- Parses JSON response into `List[DestinationSuggestion]`
- Saves to state as `suggestions`
- Handles LLM failure: retry once, then return generic fallback suggestions

**Tests:** `tests/test_discovery.py` — mock LLM for both nodes, verify question adaptation based on profile, verify suggestion format, test "surprise me" path, test fallback on LLM failure

**Verification:** `pytest tests/test_discovery.py -v`

---

### Task 2.3: Create helper nodes

**Files:** `agents/discovery.py` (append) or separate small functions

- `check_profile_node(state) -> state` — calls profile_manager.load_profile, sets `profile_exists` flag
- `load_profile_node(state) -> state` — loads full profile + history into state
- `build_profile_node(state) -> state` — takes onboarding answers from state, calls LLM to infer preferences, saves via profile_manager

**Tests:** Included in `tests/test_discovery.py`

**Verification:** `pytest tests/test_discovery.py -v`

---

## Phase 3: Discovery Graph + Bridge

### Task 3.1: Create `discovery_graph.py`

**Files:** `discovery_graph.py` (new)

```python
def build_discovery_graph():
    graph = StateGraph(DiscoveryState)
    # Nodes: check_profile, onboarding, build_profile, load_profile, discovery_chat, generate_suggestions
    # Conditional edge from check_profile: exists → load_profile, missing → onboarding
    # onboarding → build_profile → discovery_chat
    # load_profile → discovery_chat
    # Conditional edge from discovery_chat: need_more → discovery_chat, ready → generate_suggestions
    # generate_suggestions → END
    # Compile with interrupt support
    return graph.compile(checkpointer=MemorySaver())
```

Uses `MemorySaver` checkpointer for interrupt/resume support.

**Tests:** `tests/test_discovery_graph.py` — integration test with mocked LLM, full flow: check_profile (missing) → onboarding → build_profile → discovery_chat → generate_suggestions. Test both new-user and returning-user paths.

**Verification:** `pytest tests/test_discovery_graph.py -v`

---

### Task 3.2: Bridge function

**Files:** `discovery_graph.py` (add to same file)

```python
def bridge_discovery_to_optimizer(discovery_state: DiscoveryState, chosen: DestinationSuggestion) -> dict:
```

- Maps DiscoveryState → TravelState dict
- `infer_priorities(trip_intent)` — uses the priority table from spec
- Handles `date_range=None` (flexible) → defaults to next 6 months
- Sets `discovery_context` with trip_intent, why_chosen, tags

**Tests:** `tests/test_bridge.py` — test each trip intent maps to correct priorities, test flexible date handling, test discovery_context population

**Verification:** `pytest tests/test_bridge.py -v`

---

### Task 3.3: Update synthesizer for discovery context

**Files:** `agents/synthesizer.py`

Modify the synthesizer prompt to check for `state.get("discovery_context")`. If present, append a section:

```
PERSONALIZATION CONTEXT:
The user discovered this destination through our advisor. They wanted: {trip_intent}.
This destination was suggested because: {why_chosen}.
Key interests: {tags}.
Weave this context into your recommendation naturally.
```

If `discovery_context` is None, synthesizer behaves exactly as before.

**Tests:** `tests/test_synthesizer.py` — add test with discovery_context present, verify prompt includes personalization. Existing tests must still pass (no discovery_context case).

**Verification:** `pytest tests/ -v` (run all to ensure no regressions)

---

## Phase 4: Streamlit UI

### Task 4.1: Mode toggle and app restructuring

**Files:** `app.py`

Add tab-based mode toggle at the top:
```python
tab1, tab2 = st.tabs(["🔍 Discover Where", "📅 Optimize When"])
```

Move existing optimizer UI into `tab2`. Create `tab1` skeleton for discovery.

**Tests:** Manual — verify existing optimizer still works in tab2, tab switching works.

**Verification:** `streamlit run app.py` — both tabs render, optimizer tab unchanged.

---

### Task 4.2: Profile sidebar

**Files:** `app.py`

In the sidebar, add "👤 My Travel Profile" expander:
- If profile exists: show summary (origin, preferences, trip count)
- "Update preferences" button → sets `st.session_state.force_onboarding = True`
- "Add past trip" → small form (destination, when, type) that adds to travel_history

**Tests:** Manual — profile displays correctly, add-trip form works.

**Verification:** `streamlit run app.py`

---

### Task 4.3: Discovery chat UI

**Files:** `app.py`

In the Discover tab:
- Use `st.chat_message` for conversation display
- Display agent questions with option buttons (rendered as `st.button` in columns)
- On user response: resume discovery graph with `Command(resume=answer)`
- Track graph thread_id in `st.session_state` for interrupt/resume
- Show typing indicator while LLM processes

Flow:
1. User clicks "Discover Where" tab
2. If no profile → onboarding chat starts
3. After onboarding (or if profile exists) → discovery chat starts
4. After suggestions generated → show destination cards

**Tests:** Manual — full conversation flow, verify interrupt/resume works with Streamlit's rerun model.

**Verification:** `streamlit run app.py` — complete a discovery conversation.

---

### Task 4.4: Destination cards + bridge trigger

**Files:** `app.py`

After suggestions are generated, display 5 cards using `st.container`:
- Each card shows: destination, why_you, visa, flight estimate, timing, budget fit
- "Explore this →" button on each card
- On click: calls `bridge_discovery_to_optimizer()`, stores result in session_state, switches to Optimize tab, auto-runs optimizer

Also: after optimizer completes, auto-log to travel_history with status='searched'. Add "Mark as booked ✓" button that updates status.

**Tests:** Manual — click card, verify optimizer runs with correct pre-filled values, verify history logging.

**Verification:** `streamlit run app.py` — end-to-end: discover → pick → optimize.

---

## Phase 5: Polish + Integration Testing

### Task 5.1: Mock discovery agents

**Files:** `agents/mock_agents.py`

Add `mock_onboarding_node`, `mock_discovery_chat_node`, `mock_generate_suggestions_node` with preset responses for demo mode. Follow existing mock patterns.

Update `discovery_graph.py` `build_discovery_graph(demo=True)` to use mocks.

**Tests:** `tests/test_mock_discovery.py` — verify mock graph runs without LLM.

**Verification:** `pytest tests/test_mock_discovery.py -v`

---

### Task 5.2: End-to-end integration test

**Files:** `tests/test_e2e_discovery.py` (new)

Full pipeline test with mocked LLM:
1. New user → onboarding → profile saved
2. Discovery chat → suggestions generated
3. Bridge → TravelState correct
4. Optimizer graph (with mocks) → results include discovery context in recommendation

Test returning user path: profile already exists → skip onboarding.

**Verification:** `pytest tests/test_e2e_discovery.py -v`

---

### Task 5.3: Run full test suite

**Verification:** `pytest tests/ -v` — all tests pass, including existing 41 tests + new discovery tests. Zero regressions.

---

## Dependency Order

```
Phase 1 (no dependencies between tasks, can parallelize):
  1.1 (db) ──┐
  1.2 (models)├── Phase 2 (depends on 1.1-1.4)
  1.3 (state) │     2.1 (onboarding) ─┐
  1.4 (profile)┘    2.2 (discovery) ──├── Phase 3
                     2.3 (helpers) ───┘     3.1 (graph)
                                            3.2 (bridge) ── Phase 4
                                            3.3 (synthesizer)  (4.1-4.4 sequential)
                                                            ── Phase 5 (after all)
```

## Summary

- **New files:** 5 (`discovery_models.py`, `discovery_graph.py`, `agents/onboarding.py`, `agents/discovery.py`, `services/profile_manager.py`)
- **Modified files:** 4 (`db.py`, `models.py`, `agents/synthesizer.py`, `app.py`, `agents/mock_agents.py`)
- **New test files:** 7
- **Estimated new tests:** ~30-40
- **No new dependencies** — reuses existing LangGraph, Streamlit, OpenRouter, SQLite
