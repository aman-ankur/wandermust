# Destination Discovery — Implementation Plan

**Goal:** Add a conversational "Discover Where" mode to the Travel Optimizer that helps users find destinations, then feeds the chosen destination into the existing optimizer pipeline.

**Spec:** `docs/superpowers/specs/2026-03-22-destination-discovery-design.md`

---

## Phase 1: Data Layer (Independent Tasks — can be done in parallel)

### Task 1.1: Discovery Data Models
**Files:** Modify `models.py`
- Add `UserProfile` Pydantic model
- Add `DiscoveryState` TypedDict
- Add `DestinationSuggestion` model
- **Tests:** `tests/test_discovery_models.py`
- **Verify:** `pytest tests/test_discovery_models.py -v`

### Task 1.2: Discovery DB Tables
**Files:** Modify `db.py`
- Add `user_profiles` table
- Add `discovery_sessions` table
- Add CRUD methods: `save_profile`, `get_profile`, `save_session`, `get_session`
- **Tests:** `tests/test_discovery_db.py`
- **Verify:** `pytest tests/test_discovery_db.py -v`

### Task 1.3: Discovery Config
**Files:** Modify `config.py`
- Add `discovery_model` setting (can use same LLM or cheaper one)
- Add `max_onboarding_questions` and `max_discovery_questions` settings
- **Tests:** Inline validation
- **Verify:** `pytest tests/test_discovery_models.py tests/test_discovery_db.py -v`

### Task 1.4: Discovery Mock Data
**Files:** Modify `mock_data.py`
- Add `get_mock_onboarding_response`, `get_mock_discovery_response`, `get_mock_suggestions`
- **Tests:** `tests/test_discovery_mocks.py`
- **Verify:** `pytest tests/test_discovery_mocks.py -v`

---

## Phase 2: Conversational Agents (Sequential)

### Task 2.1: Onboarding Agent
**Files:** Create `agents/onboarding.py`
- Uses LangGraph `interrupt()` for human-in-the-loop
- Collects travel history, preferences, budget level, passport country
- Saves `UserProfile` to DB
- **Tests:** `tests/test_onboarding.py`
- **Verify:** `pytest tests/test_onboarding.py -v`

### Task 2.2: Discovery Chat Agent
**Files:** Create `agents/discovery_chat.py`
- Uses LangGraph `interrupt()` for human-in-the-loop
- Asks 3-5 adaptive questions about trip intent
- Extracts structured `trip_intent` from conversation
- **Tests:** `tests/test_discovery_chat.py`
- **Verify:** `pytest tests/test_discovery_chat.py -v`

---

## Phase 3: Suggestion & Bridge (Sequential)

### Task 3.1: Suggestion Generator
**Files:** Create `agents/suggestion_generator.py`
- Takes user_profile + trip_intent
- LLM reasons about visa, budget, flights, seasonality
- Returns 3-5 destination suggestions
- **Tests:** `tests/test_suggestion_generator.py`
- **Verify:** `pytest tests/test_suggestion_generator.py -v`

### Task 3.2: Bridge Function
**Files:** Create `agents/discovery_bridge.py`
- Converts chosen destination + trip_intent into TravelState
- **Tests:** `tests/test_discovery_bridge.py`
- **Verify:** `pytest tests/test_discovery_bridge.py -v`

---

## Phase 4: Graph & Integration

### Task 4.1: Discovery Graph
**Files:** Create `discovery_graph.py`
- LangGraph with: onboarding → discovery_chat → suggestion_generator
- Uses interrupt/resume pattern
- **Tests:** `tests/test_discovery_graph.py`
- **Verify:** `pytest tests/test_discovery_graph.py -v`

### Task 4.2: Synthesizer Modification
**Files:** Modify `agents/synthesizer.py`
- Add discovery context to prompt when destination came from discovery
- Small addition, not a rewrite
- **Verify:** `pytest tests/test_synthesizer.py -v`

---

## Phase 5: UI & Final

### Task 5.1: Streamlit UI
**Files:** Modify `app.py`
- Add mode toggle: "Discover Where" | "Optimize When"
- Chat interface for discovery mode
- Suggestion cards with "Use this destination" button
- **Verify:** Manual testing + `pytest tests/ -v`

### Task 5.2: Discovery Mock Agents
**Files:** Modify `agents/mock_agents.py`
- Add mock versions of onboarding, discovery_chat, suggestion_generator
- **Verify:** `pytest tests/ -v`

### Task 5.3: Final Regression
- **Verify:** `pytest tests/ -v` — all 75+ existing tests pass, plus new discovery tests
