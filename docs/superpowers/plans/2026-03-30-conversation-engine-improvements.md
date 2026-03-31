# Discovery v2 Conversation Engine Improvements

**Date:** 2026-03-30
**Branch:** `feature/discovery-v2-conversational-redesign`
**Goal:** Make the conversational travel recommendation flow feel like talking to a knowledgeable friend, not filling out a form.

---

## What Changed (9 commits)

### 1. Model Split — Two LLM Singletons (`aa0efda`)

**Problem:** Single `gpt-4o` model used for both personality phrasing (lightweight) and destination generation (reasoning-heavy). Personality responses were slow (~800ms+).

**Change:**
- `config.py`: Added `discovery_v2_personality_model` (gpt-4o-mini) and `discovery_v2_destination_model` (gpt-4o)
- `api/conversation_engine.py`: Replaced single `_llm`/`_get_llm()` with `_personality_llm`/`_get_personality_llm()` (max_tokens=512) and `_destination_llm`/`_get_destination_llm()` (max_tokens=1024)
- `generate_personality()` uses fast model, `generate_destinations()` uses smart model

**Files:** `config.py`, `api/conversation_engine.py`, `tests/test_conversation_engine.py`

---

### 2. Seasonality Grounding (`7c6e8e4`)

**Problem:** `knowledge/context_builder.py` already supported a `month` param for seasonality data ("Good in month X: Thailand, Bali...") but it was never passed. Destination recs ignored travel timing.

**Change:**
- `api/routes.py`: Added `_resolve_month(known_facts)` that maps timing preferences to calendar months:
  - "Next 1-2 months" → current month
  - "3-6 months out" → current + 3
  - "Flexible" → current month
  - "6+ months out" → None
- Updated `_build_knowledge_context(profile, known_facts)` to pass month to `build_context()`
- All 3 call sites updated

**Files:** `api/routes.py`, `tests/test_app_discovery_v2.py`

---

### 3. Conversation Memory Threading (`3dea51b`)

**Problem:** LLM received flat `known_facts_summary` ("passport: Indian, budget_level: Mid-range") but never the actual conversation. Reactions felt generic and disconnected.

**Change:**
- `api/conversation_engine.py`: Added `build_conversation_summary(messages, max_turns=6)` — formats last N messages as "You: .../User: ..." compact string
- Updated `PERSONALITY_PROMPT` and `DESTINATION_PROMPT` to include `{conversation_history}` placeholder
- Updated `generate_personality()` signature to accept `messages` param
- `api/routes.py`: Both `/start` and `/respond` now pass `session["messages"]` to the engine

**Files:** `api/conversation_engine.py`, `api/routes.py`, `tests/test_conversation_engine.py`

---

### 4. Few-Shot Examples (`2aa68c6`)

**Problem:** Prompts said "Return ONLY valid JSON" but gave no examples of quality output. LLM guessed tone/detail every time, producing inconsistent results.

**Change:**
- `api/conversation_engine.py`: Added `PERSONALITY_FEW_SHOT` (2 examples) and `DESTINATION_FEW_SHOT` (1 example) constants
- Examples are INR-aware (rupee budget hints, Indian passport visa-free destinations)
- Injected via `{few_shot}` placeholder in both prompt templates
- Personality examples show: null reaction on first turn, specific passport-aware reactions, personalized option insights with INR amounts

**Files:** `api/conversation_engine.py`, `tests/test_conversation_engine.py`

---

### 5. Intent Extractor (`9d4e253`)

**Problem:** `parse_answer()` did basic keyword matching. "I want to explore temples in SE Asia on a shoestring budget, maybe next month, traveling solo" only captured what matched the current topic's option labels. Rich context was lost.

**Change:**
- **New file** `api/intent_extractor.py`: LLM-based multi-fact extraction using gpt-4o-mini
  - `extract_facts_from_text(free_text, known_facts)` → `{topic_key: value}`
  - Validates against 7 allowed topic keys
  - Never overwrites already-known facts
  - Falls back to `{}` on failure
  - Skips texts < 5 chars
- `api/routes.py`: Integrated into `/respond` — runs BEFORE controller when user sends free text (no option_ids). Extracted facts merge into `known_facts` so the controller sees them.

**Files:** `api/intent_extractor.py` (new), `api/routes.py`, `tests/test_intent_extractor.py` (new)

---

### 6. Min-Turn Guards + Bonus Topics (`ba24bce`)

**Problem:** With intent extraction filling 3+ facts at once, users could jump from turn 1 to the narrowing phase — feels jarring and rushed.

**Change:**
- `api/conversation_controller.py`:
  - `should_transition()` now requires BOTH required facts filled AND minimum turns reached (`settings.discovery_v2_min_profile_turns=2`, `settings.discovery_v2_min_discovery_turns=2`). Safety net (max turns) still forces transition.
  - Added `pick_bonus_topic(known_facts, phase)` — returns non-required unfilled topics (e.g., `deal_breakers` in discovery) as bonus questions when min turns aren't met
  - `build_controller_turn()` falls back to bonus topic when `pick_next_topic` returns None

**Files:** `api/conversation_controller.py`, `tests/test_conversation_controller.py`

---

### 7. Narrowing → Reveal Phase Transition (`25fec15`)

**Problem:** Clicking "Tell me more about X" in the narrowing phase just regenerated new narrowing suggestions instead of advancing to reveal. The `should_transition()` function had no `narrowing → reveal` case.

**Change:**
- `api/conversation_controller.py`: Added narrowing → reveal transition after `discovery_v2_min_narrowing_turns` (default 1). So the first "Tell me more" click advances to reveal.
- `api/conversation_engine.py`: Updated reveal phase instructions to tell the LLM to focus on destinations the user showed interest in (from conversation history), with detailed budget breakdowns, best time to visit, and insider tips.
- Updated test `test_narrowing_never_auto_transitions` → split into `test_narrowing_transitions_to_reveal` and `test_narrowing_stays_below_min_turns`

**Files:** `api/conversation_controller.py`, `api/conversation_engine.py`, `tests/test_conversation_controller.py`

---

### 8. Deterministic Reveal Options (`aa71878`)

**Problem:** The LLM kept generating "Tell me more about X" options in the reveal phase, ignoring prompt instructions to use "I'm sold!", "Compare top 2", "Start over". This created infinite loops where clicking any option just regenerated more suggestions.

**Change:**
- `api/routes.py`: In `_assemble_destination_turn()`, reveal phase options are now **hardcoded in Python** instead of trusting LLM output:
  - "I'm sold!" (id: `sold`)
  - "Compare top 2" (id: `compare`)
  - "Start over" (id: `start_over`)
- LLM-generated options are still used for narrowing phase (where "Tell me more about X" is correct behavior)

**Lesson learned:** Never trust LLMs for UI control flow. Deterministic code should control buttons/actions; LLMs should only generate content (text, descriptions, insights).

**Files:** `api/routes.py`

---

### 9. Session Recovery on Mode Switch (`25fec15`)

**Problem:** Switching between "Discover Where" and "Optimize When" modes in Streamlit and switching back showed a blank screen. The Streamlit session still had a `v2_session_id` but the in-memory `SessionStore` had been wiped (e.g., by hot reload), so the UI tried to render a non-existent session.

**Change:**
- `app.py`: Before rendering, verify the backend session still exists via `_get_session_store().get(session_id)`. If gone, auto-start a fresh session instead of showing blank.

**Files:** `app.py`

---

## Architecture (unchanged philosophy)

```
User → /respond → Intent Extractor (pre-controller, free text only)
                → Deterministic Controller (topic selection, fact parsing, phase transitions)
                → LLM Personality Layer (phrasing only, can't break the flow)
                → ConversationTurn response
```

**4-phase flow:** `profile → discovery → narrowing → reveal`

- Controller drives ALL logic (topic order, transitions, fact storage)
- LLM is a personality wrapper — it phrases questions, it doesn't decide them
- Intent extractor enriches facts BEFORE controller runs
- Two-model strategy: fast (gpt-4o-mini) for personality + extraction, smart (gpt-4o) for destinations
- Reveal phase options are deterministic Python (not LLM-generated)

## Config Reference

```python
# config.py — Discovery v2 settings
discovery_v2_personality_model: str = "gpt-4o-mini"   # personality + extraction
discovery_v2_destination_model: str = "gpt-4o"         # destination generation
discovery_v2_min_profile_turns: int = 2                # min turns before profile→discovery
discovery_v2_min_discovery_turns: int = 2              # min turns before discovery→narrowing
discovery_v2_min_narrowing_turns: int = 1              # min turns before narrowing→reveal
```

## Phase Transition Rules

| From | To | Condition |
|------|----|-----------|
| profile | discovery | All required facts (passport, budget_level, travel_style) filled AND turn_count >= 2 |
| profile | discovery | Safety net: turn_count >= 5 (forced, even if facts missing) |
| discovery | narrowing | All required facts (timing, companions, interests) filled AND turn_count >= 2 |
| discovery | narrowing | Safety net: turn_count >= 6 (forced) |
| narrowing | reveal | turn_count >= 1 (after first user interaction with suggestions) |
| reveal | (terminal) | User clicks "Select" on a destination or "Start over" |

## Key Design Decisions

1. **Deterministic controller, not LLM-driven flow.** The controller decides which topic to ask, when to transition phases, and what options to show. The LLM only adds natural language personality. This prevents the LLM from breaking the conversation flow.

2. **Pre-controller intent extraction.** Free text is extracted into structured facts BEFORE the controller runs, so topic skipping and phase transitions work correctly with enriched data.

3. **Hardcoded reveal options.** LLMs reliably ignore instructions about specific button labels. Reveal phase options are Python constants, not LLM output. Narrowing phase still uses LLM-generated options (where dynamic "Tell me more about X" is desired).

4. **Min-turn guards with bonus topics.** Even when intent extraction fills all required facts at once, the conversation doesn't feel rushed — bonus topics (like deal_breakers) fill the remaining turns before transition.

5. **Two-model strategy.** gpt-4o-mini for personality phrasing + intent extraction (~4s), gpt-4o for destination generation (~8-10s). Personality doesn't need strong reasoning; destinations do.

## Test Coverage

82 tests across 4 test files:
- `tests/test_conversation_engine.py` — 14 tests (model split, conversation summary, few-shot, fallbacks)
- `tests/test_intent_extractor.py` — 7 tests (single/multi-fact extraction, known-fact protection, failure handling)
- `tests/test_conversation_controller.py` — 53 tests (topic selection, parsing, transitions, min-turn guards, bonus topics, narrowing→reveal)
- `tests/test_app_discovery_v2.py` — 8 tests (model rendering, seasonality month resolution)

## How to Test

### Streamlit UI (recommended)
```bash
python3 -m streamlit run app.py
```
Open http://localhost:8501, click "Discover Where" in sidebar.

### FastAPI API
```bash
python3 -m uvicorn "api.main:create_app" --factory --reload --port 8000

# Start session
curl -s -X POST http://localhost:8000/api/discovery/start \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"test"}' | python3 -m json.tool

# Rich free text (fills multiple facts at once)
curl -s -X POST http://localhost:8000/api/discovery/respond \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SESSION_ID","answer":"I want a budget-friendly solo adventure next month, love hiking and street food"}' | python3 -m json.tool
```

### What to verify
- Reactions reference what user actually said (not generic)
- Option insights mention INR for Indian passport
- Topics filled by extraction are skipped
- Phase transitions don't happen before min turns
- Destination recs include seasonality data
- Personality responses use gpt-4o-mini (check `[PERF]` log lines, ~4s)
- Destination responses use gpt-4o (check logs, ~8-10s)
- Narrowing → reveal transition works on first "Tell me more" click
- Reveal phase shows "I'm sold!", "Compare top 2", "Start over" (not "Tell me more")
- Switching Streamlit modes and back doesn't show blank screen

## Observed Performance (from live testing)

| Phase | Model | Avg Latency |
|-------|-------|-------------|
| Personality (profile/discovery) | gpt-4o-mini | 4-6s |
| Intent extraction (free text) | gpt-4o-mini | ~3s |
| Destination narrowing | gpt-4o | 8-10s |
| Destination reveal | gpt-4o | 6-8s |
