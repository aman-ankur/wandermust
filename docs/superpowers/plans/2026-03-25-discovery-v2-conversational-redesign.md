# Discovery v2 — Conversational Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed 10-question discovery flow with an adaptive, insight-rich 4-phase conversation (Profile -> Discovery -> Narrowing -> Reveal) powered by pre-baked knowledge modules and a FastAPI backend.

**Architecture:** State machine with 4 LLM-decided phases. Pre-baked knowledge modules (~300 tokens) injected into compact system prompts replace LLM general knowledge for factual data (visa, budget, exchange rates, seasonality). FastAPI backend serves `ConversationTurn` responses; Streamlit becomes a thin render layer. Conversation engine replaces separate onboarding + discovery_chat agents.

**Tech Stack:** Python 3.9, FastAPI, Pydantic v2, LangChain-OpenAI (ChatOpenAI), LangGraph (state machine), SQLite (HistoryDB), Streamlit (UI), pytest (testing). Primary LLM: gpt-4o-mini via existing `get_llm()` from `agents/llm_helper.py`.

**Spec:** `docs/superpowers/specs/2026-03-25-discovery-v2-conversational-redesign.md`

**Important codebase constraints (Python 3.9):**
- Use `Optional[str]` not `str | None`
- Use `List[str]` not `list[str]` in type annotations
- Use `from __future__ import annotations` where needed

**Test runner:** `pytest tests/ -v`

**Existing patterns to follow:**
- Module-level `_llm` cache with `_get_llm()` lazy init (see `agents/discovery_chat.py`)
- `parse_json_response()` from `agents/llm_helper.py` for LLM JSON parsing
- `HistoryDB` class in `db.py` for SQLite access
- `Settings` in `config.py` via pydantic-settings
- Tests use `unittest.mock.patch` + `MagicMock` for LLM mocking

**Spec deviations (intentional):**
- Spec shows `discovery_graph_v2.py` as separate file; plan puts state machine logic in `api/conversation_engine.py` and route orchestration in `api/routes.py` — cleaner separation without an extra file.
- Spec says change existing `discovery_model` to `gpt-4o-mini`; plan adds `discovery_v2_model` to allow v1/v2 side-by-side during migration.
- SSE streaming: Spec describes full SSE implementation. Plan implements sync endpoints first (sufficient for Streamlit). Streaming is a follow-up task (see TODO in Task 10).
- Session persistence: Plan uses in-memory `SessionStore`. SQLite-backed session persistence is a follow-up task. Sessions are persisted to DB on completion via `/select`.

**Known Python 3.9 fix needed:** Existing code in `agents/onboarding.py` and `agents/discovery_chat.py` uses `list[dict]` type hints (Python 3.10+). These work at runtime on 3.9 because they're only used as annotations (not evaluated), but should be fixed if static type checking is enabled. Task 8 includes `from __future__ import annotations` in all new files.

---

## File Structure

```
knowledge/                          # NEW — pre-baked travel intelligence
  __init__.py                       # Package init, re-exports build_context
  visa_tiers.py                     # VISA_TIERS dict by passport country
  budget_tiers.py                   # BUDGET_TIERS dict by passport + level
  exchange_rates.py                 # EXCHANGE_FAVORABILITY dict by passport
  seasonality.py                    # SEASONALITY dict by destination
  context_builder.py                # build_context() — assembles ~300 token block

api/                                # NEW — FastAPI backend
  __init__.py                       # Package init
  models.py                         # Option, DestinationHint, ConversationTurn, request/response models
  session.py                        # In-memory session store (dict[str, SessionData])
  conversation_engine.py            # Core LLM conversation logic — generates ConversationTurn
  routes.py                         # /start, /respond, /state, /select endpoints
  main.py                           # FastAPI app factory, CORS, lifespan

models.py                           # MODIFY — add DiscoveryV2State TypedDict
config.py                           # MODIFY — add discovery_v2_model setting (gpt-4o-mini)
db.py                               # MODIFY — add conversation_history column to discovery_sessions
app.py                              # MODIFY — Streamlit renders ConversationTurn from API

tests/test_knowledge.py             # NEW — knowledge module tests
tests/test_api_models.py            # NEW — API model validation tests
tests/test_conversation_engine.py   # NEW — conversation engine tests
tests/test_api_routes.py            # NEW — FastAPI route integration tests
tests/test_session.py               # NEW — session store tests
tests/test_app_discovery_v2.py      # NEW — Streamlit v2 rendering tests
```

### What gets replaced (after v2 is complete)
- `agents/onboarding.py` — merged into `api/conversation_engine.py`
- `agents/discovery_chat.py` — merged into `api/conversation_engine.py`
- `discovery_graph.py` — replaced by FastAPI state machine in `api/routes.py`

### What stays unchanged
- `agents/discovery_bridge.py` — bridge logic is reused
- `agents/suggestion_generator.py` — reused in reveal phase
- `agents/llm_helper.py` — shared LLM factory, no changes
- `db.py` — extended, not replaced
- All optimizer code — untouched

---

## Task 1: Knowledge Module — Visa Tiers

**Files:**
- Create: `knowledge/__init__.py`
- Create: `knowledge/visa_tiers.py`
- Create: `tests/test_knowledge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_knowledge.py
import pytest
from knowledge.visa_tiers import VISA_TIERS


def test_visa_tiers_has_indian_passport():
    assert "IN" in VISA_TIERS


def test_indian_visa_free_contains_expected():
    tiers = VISA_TIERS["IN"]
    assert "visa_free" in tiers
    assert "Thailand" in tiers["visa_free"]
    assert "Georgia" in tiers["visa_free"]


def test_indian_hard_visa_contains_expected():
    tiers = VISA_TIERS["IN"]
    assert "hard_visa" in tiers
    assert "US" in tiers["hard_visa"]
    assert "UK" in tiers["hard_visa"]


def test_all_tier_keys_present():
    tiers = VISA_TIERS["IN"]
    assert set(tiers.keys()) == {"visa_free", "e_visa_easy", "visa_required", "hard_visa"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'knowledge'`

- [ ] **Step 3: Create package and implement visa_tiers.py**

```python
# knowledge/__init__.py
"""Pre-baked travel intelligence modules."""
```

```python
# knowledge/visa_tiers.py
"""Visa tier data by passport country.

Curated visa accessibility tiers for destination suggestions.
Data should be periodically reviewed for accuracy.
"""

VISA_TIERS = {
    "IN": {
        "visa_free": [
            "Thailand", "Georgia", "Serbia", "Mauritius", "Nepal",
            "Bhutan", "Fiji", "Maldives", "Indonesia", "Qatar",
        ],
        "e_visa_easy": [
            "Vietnam", "Turkey", "Sri Lanka", "Cambodia", "Kenya",
            "Ethiopia", "Myanmar", "Laos", "Azerbaijan",
        ],
        "visa_required": ["Japan", "South Korea", "UAE", "Malaysia"],
        "hard_visa": ["US", "UK", "EU/Schengen", "Canada", "Australia", "NZ"],
    },
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add knowledge/__init__.py knowledge/visa_tiers.py tests/test_knowledge.py
git commit -m "feat(knowledge): add visa tiers module for Indian passport"
```

---

## Task 2: Knowledge Module — Budget Tiers

**Files:**
- Create: `knowledge/budget_tiers.py`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_knowledge.py`:

```python
from knowledge.budget_tiers import BUDGET_TIERS


def test_budget_tiers_has_indian_context():
    assert "IN" in BUDGET_TIERS


def test_budget_tier_keys():
    tiers = BUDGET_TIERS["IN"]
    assert set(tiers.keys()) == {"budget", "moderate", "comfortable", "luxury"}


def test_budget_tier_has_daily_range():
    for tier_name, tier_data in BUDGET_TIERS["IN"].items():
        assert "daily_range" in tier_data, f"{tier_name} missing daily_range"
        assert "examples" in tier_data, f"{tier_name} missing examples"
        assert "note" in tier_data, f"{tier_name} missing note"


def test_budget_tier_examples_are_lists():
    for tier_name, tier_data in BUDGET_TIERS["IN"].items():
        assert isinstance(tier_data["examples"], list)
        assert len(tier_data["examples"]) >= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge.py::test_budget_tiers_has_indian_context -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement budget_tiers.py**

```python
# knowledge/budget_tiers.py
"""Budget tier data by passport country.

Daily budget ranges in local currency with example destinations.
"""

BUDGET_TIERS = {
    "IN": {
        "budget": {
            "daily_range": "₹2,000-4,000",
            "examples": ["Vietnam", "Nepal", "Georgia", "Cambodia", "Sri Lanka"],
            "note": "Hostels, street food, local transport",
        },
        "moderate": {
            "daily_range": "₹5,000-8,000",
            "examples": ["Thailand", "Turkey", "Bali", "Malaysia", "Azerbaijan"],
            "note": "Mid-range hotels, restaurants, some activities",
        },
        "comfortable": {
            "daily_range": "₹10,000-15,000",
            "examples": ["Japan", "South Korea", "Dubai", "Eastern Europe"],
            "note": "Good hotels, full experiences, domestic flights",
        },
        "luxury": {
            "daily_range": "₹15,000+",
            "examples": ["Western Europe", "Australia", "Scandinavia", "Switzerland"],
            "note": "INR exchange hurts here — budget 3-4x more than SE Asia",
        },
    },
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add knowledge/budget_tiers.py tests/test_knowledge.py
git commit -m "feat(knowledge): add budget tiers module for Indian passport"
```

---

## Task 3: Knowledge Module — Exchange Rates + Seasonality

**Files:**
- Create: `knowledge/exchange_rates.py`
- Create: `knowledge/seasonality.py`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_knowledge.py`:

```python
from knowledge.exchange_rates import EXCHANGE_FAVORABILITY
from knowledge.seasonality import SEASONALITY


def test_exchange_favorability_has_indian_context():
    assert "IN" in EXCHANGE_FAVORABILITY
    fx = EXCHANGE_FAVORABILITY["IN"]
    assert "great_value" in fx
    assert "poor_value" in fx
    assert "note" in fx


def test_exchange_great_value_currencies():
    fx = EXCHANGE_FAVORABILITY["IN"]
    assert "THB" in fx["great_value"]
    assert "VND" in fx["great_value"]


def test_exchange_poor_value_currencies():
    fx = EXCHANGE_FAVORABILITY["IN"]
    assert "EUR" in fx["poor_value"]
    assert "GBP" in fx["poor_value"]


def test_seasonality_has_destinations():
    assert len(SEASONALITY) >= 15


def test_seasonality_structure():
    for dest, data in SEASONALITY.items():
        assert "best" in data, f"{dest} missing 'best'"
        assert "avoid" in data, f"{dest} missing 'avoid'"
        assert "note" in data, f"{dest} missing 'note'"
        assert all(1 <= m <= 12 for m in data["best"]), f"{dest} has invalid best months"
        assert all(1 <= m <= 12 for m in data["avoid"]), f"{dest} has invalid avoid months"


def test_seasonality_thailand():
    th = SEASONALITY["Thailand"]
    assert 12 in th["best"]  # Dec is peak season
    assert "dry" in th["note"].lower() or "nov" in th["note"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge.py::test_exchange_favorability_has_indian_context -v`
Expected: FAIL

- [ ] **Step 3: Implement exchange_rates.py**

```python
# knowledge/exchange_rates.py
"""Exchange rate favorability by passport country.

Helps frame budget expectations relative to home currency.
"""

EXCHANGE_FAVORABILITY = {
    "IN": {
        "great_value": ["THB", "GEL", "VND", "LKR", "NPR", "KHR", "IDR"],
        "decent_value": ["TRY", "MYR", "AZN", "RSD", "KES"],
        "poor_value": ["EUR", "GBP", "USD", "AUD", "JPY", "CHF", "SGD", "NZD"],
        "note": "INR to EUR/GBP/USD is unfavorable — European/US trips cost 3-4x more than SE Asia equivalent",
    },
}
```

- [ ] **Step 4: Implement seasonality.py**

```python
# knowledge/seasonality.py
"""Destination seasonality data.

Best and avoid months for top destinations relevant to Indian travelers.
"""

SEASONALITY = {
    "Thailand": {"best": [11, 12, 1, 2, 3], "avoid": [4, 5], "note": "Nov-Mar dry season"},
    "Georgia": {"best": [5, 6, 7, 8, 9], "avoid": [12, 1, 2], "note": "Summer wine harvest, warm weather"},
    "Vietnam": {"best": [2, 3, 4, 10, 11], "avoid": [7, 8, 9], "note": "Varies by region, central coast best in spring"},
    "Turkey": {"best": [4, 5, 6, 9, 10], "avoid": [12, 1, 2], "note": "Spring and autumn ideal, summer hot"},
    "Sri Lanka": {"best": [1, 2, 3, 4, 12], "avoid": [5, 6], "note": "West coast best Dec-Apr, east coast May-Sep"},
    "Nepal": {"best": [10, 11, 3, 4], "avoid": [6, 7, 8], "note": "Oct-Nov best for trekking, monsoon Jun-Aug"},
    "Maldives": {"best": [1, 2, 3, 4, 12], "avoid": [6, 7, 8], "note": "Dry season Dec-Apr, monsoon May-Oct"},
    "Indonesia": {"best": [5, 6, 7, 8, 9], "avoid": [12, 1, 2], "note": "Dry season May-Sep, Bali best Jun-Aug"},
    "Cambodia": {"best": [11, 12, 1, 2, 3], "avoid": [5, 6, 7], "note": "Cool dry season Nov-Feb"},
    "Japan": {"best": [3, 4, 10, 11], "avoid": [6, 7, 8], "note": "Cherry blossom Mar-Apr, autumn foliage Oct-Nov"},
    "South Korea": {"best": [4, 5, 9, 10], "avoid": [7, 8], "note": "Spring cherry blossom, autumn foliage"},
    "Malaysia": {"best": [3, 4, 5, 6, 9], "avoid": [11, 12], "note": "West coast best Dec-Apr, Borneo Mar-Oct"},
    "Azerbaijan": {"best": [4, 5, 6, 9, 10], "avoid": [12, 1, 2], "note": "Spring and autumn mild, winter cold"},
    "Serbia": {"best": [5, 6, 9, 10], "avoid": [12, 1, 2], "note": "Summer music festivals, mild spring/autumn"},
    "Mauritius": {"best": [5, 6, 7, 8, 9, 10], "avoid": [1, 2, 3], "note": "Winter (May-Oct) dry and mild"},
    "Kenya": {"best": [1, 2, 7, 8, 9, 10], "avoid": [4, 5], "note": "Great Migration Jul-Oct, dry Jan-Feb"},
    "Dubai": {"best": [11, 12, 1, 2, 3], "avoid": [6, 7, 8], "note": "Winter pleasant, summer extreme heat"},
    "Qatar": {"best": [11, 12, 1, 2, 3], "avoid": [6, 7, 8], "note": "Winter mild, summer extreme heat"},
    "Mexico": {"best": [11, 12, 1, 2, 3, 4], "avoid": [6, 7, 8, 9], "note": "Dry season Nov-Apr, hurricane season Jun-Nov"},
    "Bhutan": {"best": [3, 4, 5, 10, 11], "avoid": [6, 7, 8], "note": "Spring festivals, autumn clear skies"},
    "Fiji": {"best": [5, 6, 7, 8, 9, 10], "avoid": [1, 2, 3], "note": "Dry season May-Oct"},
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_knowledge.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add knowledge/exchange_rates.py knowledge/seasonality.py tests/test_knowledge.py
git commit -m "feat(knowledge): add exchange rates and seasonality modules"
```

---

## Task 4: Knowledge Module — Context Builder

**Files:**
- Create: `knowledge/context_builder.py`
- Modify: `knowledge/__init__.py`
- Modify: `tests/test_knowledge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_knowledge.py`:

```python
from knowledge.context_builder import build_context


def test_build_context_basic():
    ctx = build_context(passport="IN", budget="moderate", month=None)
    assert "Visa-free" in ctx
    assert "Thailand" in ctx
    assert "moderate" in ctx.lower() or "5,000" in ctx


def test_build_context_with_month():
    ctx = build_context(passport="IN", budget="budget", month=7)
    assert "Good in month 7" in ctx or "month 7" in ctx


def test_build_context_unknown_passport():
    ctx = build_context(passport="XX", budget="moderate", month=None)
    # Should not crash, just return minimal context
    assert isinstance(ctx, str)


def test_build_context_token_size():
    """Context should be compact — under 500 tokens (rough: ~4 chars/token)."""
    ctx = build_context(passport="IN", budget="moderate", month=7)
    estimated_tokens = len(ctx) / 4
    assert estimated_tokens < 500, f"Context too large: ~{estimated_tokens:.0f} tokens"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge.py::test_build_context_basic -v`
Expected: FAIL

- [ ] **Step 3: Implement context_builder.py**

```python
# knowledge/context_builder.py
"""Builds compact knowledge context (~300 tokens) for LLM system prompt injection.

Assembles visa, budget, exchange, and seasonality data based on user context.
"""
from typing import Optional

from knowledge.visa_tiers import VISA_TIERS
from knowledge.budget_tiers import BUDGET_TIERS
from knowledge.exchange_rates import EXCHANGE_FAVORABILITY
from knowledge.seasonality import SEASONALITY


def build_context(
    passport: str,
    budget: str,
    month: Optional[int] = None,
) -> str:
    visa = VISA_TIERS.get(passport, {})
    budget_tier = BUDGET_TIERS.get(passport, {}).get(budget, {})
    exchange = EXCHANGE_FAVORABILITY.get(passport, {})

    parts = [f"TRAVEL INTELLIGENCE (for {passport} passport, {budget} budget):"]

    if visa.get("visa_free"):
        parts.append(f"- Visa-free: {', '.join(visa['visa_free'][:8])}")
    if visa.get("e_visa_easy"):
        parts.append(f"- E-visa easy: {', '.join(visa['e_visa_easy'][:6])}")
    if visa.get("hard_visa"):
        parts.append(f"- Hard visa (avoid unless asked): {', '.join(visa['hard_visa'][:5])}")
    if budget_tier.get("daily_range"):
        parts.append(f"- Budget range: {budget_tier['daily_range']}")
    if exchange.get("great_value"):
        parts.append(f"- Great value currencies: {', '.join(exchange['great_value'][:5])}")
    if exchange.get("poor_value"):
        parts.append(f"- Poor value currencies: {', '.join(exchange['poor_value'][:5])}")
    if exchange.get("note"):
        parts.append(f"- {exchange['note']}")

    if month:
        good_now = [d for d, s in SEASONALITY.items() if month in s["best"]]
        if good_now:
            parts.append(f"- Good in month {month}: {', '.join(good_now[:8])}")

    return "\n".join(parts)
```

- [ ] **Step 4: Update knowledge/__init__.py**

```python
# knowledge/__init__.py
"""Pre-baked travel intelligence modules."""
from knowledge.context_builder import build_context

__all__ = ["build_context"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_knowledge.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add knowledge/context_builder.py knowledge/__init__.py tests/test_knowledge.py
git commit -m "feat(knowledge): add context builder — assembles ~300 token knowledge block"
```

---

## Task 5: API Models — ConversationTurn + Request/Response Types

**Files:**
- Create: `api/__init__.py`
- Create: `api/models.py`
- Create: `tests/test_api_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_models.py
import pytest
from api.models import Option, DestinationHint, ConversationTurn, DiscoveryStartRequest, DiscoveryRespondRequest


def test_option_creation():
    opt = Option(id="beaches", label="Beaches & coast", insight="Great in summer")
    assert opt.id == "beaches"
    assert opt.emoji is None


def test_option_with_emoji():
    opt = Option(id="food", label="Foodie", insight="Street food paradise", emoji="🍜")
    assert opt.emoji == "🍜"


def test_destination_hint_creation():
    hint = DestinationHint(
        name="Tbilisi, Georgia",
        hook="Visa-free, ₹3k/day, wine country",
        match_reason="Budget + food + culture trifecta",
    )
    assert hint.name == "Tbilisi, Georgia"


def test_conversation_turn_minimal():
    turn = ConversationTurn(
        phase="profile",
        question="What passport do you hold?",
        options=[Option(id="in", label="Indian", insight="Many visa-free options")],
    )
    assert turn.phase == "profile"
    assert turn.reaction is None
    assert turn.destination_hints is None
    assert turn.multi_select is False
    assert turn.can_free_text is True
    assert turn.phase_complete is False


def test_conversation_turn_full():
    turn = ConversationTurn(
        phase="narrowing",
        reaction="Great choices!",
        question="What catches your eye?",
        options=[Option(id="more", label="Tell me more", insight="Explore deeper")],
        destination_hints=[
            DestinationHint(name="Georgia", hook="Wine country", match_reason="Budget fit")
        ],
        thinking="I keep coming back to the Caucasus...",
        phase_complete=False,
        multi_select=True,
    )
    assert turn.thinking is not None
    assert len(turn.destination_hints) == 1


def test_conversation_turn_valid_phases():
    for phase in ("profile", "discovery", "narrowing", "reveal"):
        turn = ConversationTurn(
            phase=phase,
            question="Test?",
            options=[Option(id="a", label="A", insight="a")],
        )
        assert turn.phase == phase


def test_start_request():
    req = DiscoveryStartRequest(user_id="default")
    assert req.user_id == "default"


def test_respond_request():
    req = DiscoveryRespondRequest(
        session_id="abc-123",
        answer="July",
    )
    assert req.session_id == "abc-123"
    assert req.answer == "July"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create api package and models**

```python
# api/__init__.py
"""FastAPI backend for Discovery v2."""
```

```python
# api/models.py
"""Pydantic models for the Discovery v2 conversation API.

ConversationTurn is the core response type — every API response returns one.
The LLM generates it. The UI renders it.
"""
from typing import List, Optional

from pydantic import BaseModel


class Option(BaseModel):
    id: str
    label: str
    insight: str
    emoji: Optional[str] = None


class DestinationHint(BaseModel):
    name: str
    hook: str
    match_reason: str


class ConversationTurn(BaseModel):
    phase: str  # "profile" | "discovery" | "narrowing" | "reveal"
    reaction: Optional[str] = None
    question: str
    options: List[Option]
    multi_select: bool = False
    can_free_text: bool = True
    destination_hints: Optional[List[DestinationHint]] = None
    thinking: Optional[str] = None
    phase_complete: bool = False


class DiscoveryStartRequest(BaseModel):
    user_id: str = "default"


class DiscoveryRespondRequest(BaseModel):
    session_id: str
    answer: str


class DiscoverySelectRequest(BaseModel):
    session_id: str
    destination: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/__init__.py api/models.py tests/test_api_models.py
git commit -m "feat(api): add ConversationTurn and request/response Pydantic models"
```

---

## Task 6: Session Store

**Files:**
- Create: `api/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session.py
import pytest
from api.session import SessionStore


def test_create_session():
    store = SessionStore()
    session_id = store.create(user_id="default")
    assert isinstance(session_id, str)
    assert len(session_id) > 0


def test_get_session():
    store = SessionStore()
    sid = store.create(user_id="default")
    session = store.get(sid)
    assert session is not None
    assert session["user_id"] == "default"
    assert session["phase"] == "profile"
    assert session["messages"] == []
    assert session["turn_count"] == 0


def test_get_nonexistent_session():
    store = SessionStore()
    assert store.get("nonexistent") is None


def test_update_session():
    store = SessionStore()
    sid = store.create(user_id="default")
    store.update(sid, phase="discovery", turn_count=3)
    session = store.get(sid)
    assert session["phase"] == "discovery"
    assert session["turn_count"] == 3


def test_add_message():
    store = SessionStore()
    sid = store.create(user_id="default")
    store.add_message(sid, role="user", content="July")
    session = store.get(sid)
    assert len(session["messages"]) == 1
    assert session["messages"][0]["role"] == "user"


def test_create_session_with_existing_profile():
    store = SessionStore()
    profile = {"passport_country": "IN", "budget_level": "moderate"}
    sid = store.create(user_id="default", profile=profile)
    session = store.get(sid)
    assert session["profile"] == profile
    assert session["phase"] == "discovery"  # Skip profile phase


def test_delete_session():
    store = SessionStore()
    sid = store.create(user_id="default")
    store.delete(sid)
    assert store.get(sid) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session.py -v`
Expected: FAIL

- [ ] **Step 3: Implement session store**

```python
# api/session.py
"""In-memory session store for discovery conversations.

Each session tracks conversation state across the 4-phase flow.
Sessions are ephemeral — persisted to DB only on completion.
"""
import uuid
from typing import Any, Dict, List, Optional


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create(
        self,
        user_id: str = "default",
        profile: Optional[Dict] = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        # If profile exists, skip profile phase
        initial_phase = "discovery" if profile else "profile"
        self._sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "phase": initial_phase,
            "messages": [],
            "turn_count": 0,
            "profile": profile or {},
            "trip_intent": {},
            "destination_hints": [],
            "suggestions": [],
        }
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def update(self, session_id: str, **kwargs: Any) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.update(kwargs)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session["messages"].append({"role": role, "content": content})

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/session.py tests/test_session.py
git commit -m "feat(api): add in-memory session store for discovery conversations"
```

---

## Task 7: Config Update — Discovery v2 Model Setting

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Write the failing test**

Append to a new section in existing tests or add inline:

```python
# tests/test_config_v2.py
import pytest


def test_discovery_v2_model_setting():
    from config import Settings
    s = Settings(openai_api_key="test")
    assert hasattr(s, "discovery_v2_model")
    assert s.discovery_v2_model == "gpt-4o-mini"


def test_discovery_v2_min_turns():
    from config import Settings
    s = Settings(openai_api_key="test")
    assert hasattr(s, "discovery_v2_min_profile_turns")
    assert s.discovery_v2_min_profile_turns == 2
    assert hasattr(s, "discovery_v2_min_discovery_turns")
    assert s.discovery_v2_min_discovery_turns == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_v2.py -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add v2 settings to config.py**

Add these fields to the `Settings` class in `config.py` (after `max_discovery_questions`):

```python
    # Discovery v2
    discovery_v2_model: str = "gpt-4o-mini"
    discovery_v2_min_profile_turns: int = 2
    discovery_v2_min_discovery_turns: int = 2
    discovery_v2_min_narrowing_turns: int = 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_v2.py -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add config.py tests/test_config_v2.py
git commit -m "feat(config): add discovery v2 settings (model, min turns)"
```

---

## Task 8: Conversation Engine — Core LLM Logic

This is the heart of the redesign. The conversation engine takes session state + user answer, calls the LLM with the system prompt + knowledge context, and returns a `ConversationTurn`.

**Files:**
- Create: `api/conversation_engine.py`
- Create: `tests/test_conversation_engine.py`

- [ ] **Step 1: Write the failing test — basic turn generation**

```python
# tests/test_conversation_engine.py
import pytest
from unittest.mock import patch, MagicMock
from api.conversation_engine import generate_turn
from api.models import ConversationTurn


MOCK_PROFILE_TURN_JSON = '''{
    "phase": "profile",
    "reaction": null,
    "question": "What passport do you hold?",
    "options": [
        {"id": "in", "label": "Indian", "insight": "Many visa-free destinations in SE Asia"},
        {"id": "us", "label": "US", "insight": "Visa-free almost everywhere"},
        {"id": "other", "label": "Other", "insight": "Tell me and I'll check"}
    ],
    "multi_select": false,
    "can_free_text": true,
    "phase_complete": false
}'''


def test_generate_turn_returns_conversation_turn():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=MOCK_PROFILE_TURN_JSON)

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        turn = generate_turn(
            phase="profile",
            messages=[],
            profile={},
            knowledge_context="",
        )

    assert isinstance(turn, ConversationTurn)
    assert turn.phase == "profile"
    assert len(turn.options) == 3


def test_generate_turn_includes_reaction():
    mock_json = '''{
        "phase": "discovery",
        "reaction": "India has great visa-free options in SE Asia!",
        "question": "When are you thinking of traveling?",
        "options": [
            {"id": "summer", "label": "Summer", "insight": "Monsoon in India, dry in Europe"}
        ],
        "phase_complete": false
    }'''
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=mock_json)

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        turn = generate_turn(
            phase="discovery",
            messages=[
                {"role": "assistant", "content": "What passport?"},
                {"role": "user", "content": "Indian"},
            ],
            profile={"passport_country": "IN"},
            knowledge_context="TRAVEL INTELLIGENCE...",
        )

    assert turn.reaction is not None
    assert "India" in turn.reaction


def test_generate_turn_llm_failure_returns_fallback():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM timeout")

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        turn = generate_turn(
            phase="profile",
            messages=[],
            profile={},
            knowledge_context="",
        )

    assert isinstance(turn, ConversationTurn)
    assert turn.phase == "profile"
    assert len(turn.options) >= 1  # Fallback should still have options


def test_generate_turn_narrowing_has_destination_hints():
    mock_json = '''{
        "phase": "narrowing",
        "reaction": "Let me think out loud...",
        "thinking": "I keep coming back to the Caucasus",
        "question": "What catches your eye?",
        "options": [
            {"id": "more", "label": "Tell me more", "insight": "Explore deeper"}
        ],
        "destination_hints": [
            {"name": "Tbilisi, Georgia", "hook": "Visa-free, wine country", "match_reason": "Budget fit"}
        ],
        "phase_complete": false
    }'''
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=mock_json)

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        turn = generate_turn(
            phase="narrowing",
            messages=[],
            profile={"passport_country": "IN", "budget_level": "moderate"},
            knowledge_context="TRAVEL INTELLIGENCE...",
        )

    assert turn.destination_hints is not None
    assert len(turn.destination_hints) == 1
    assert turn.thinking is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conversation_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement conversation engine**

```python
# api/conversation_engine.py
"""Core conversation engine — generates ConversationTurn responses via LLM.

Replaces the separate onboarding + discovery_chat agents with a single
adaptive engine that uses pre-baked knowledge context.
"""
import json
import logging
from typing import Dict, List, Optional

from api.models import ConversationTurn, Option
from agents.llm_helper import get_llm, parse_json_response
from config import settings

logger = logging.getLogger("wandermust.conversation_engine")

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(settings.discovery_v2_model)
    return _llm


SYSTEM_PROMPT = """You are a well-traveled friend helping plan a trip. You've been to 60+ countries
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
{phase_rules}

{knowledge_context}

Respond with a valid JSON object matching this schema:
{{
    "phase": "{phase}",
    "reaction": "brief insight reacting to previous answer (null if first turn)",
    "question": "your next question",
    "options": [{{"id": "string", "label": "string", "insight": "string", "emoji": "optional"}}],
    "multi_select": false,
    "can_free_text": true,
    "destination_hints": null,
    "thinking": null,
    "phase_complete": false
}}"""

PHASE_RULES = {
    "profile": (
        "Ask about the most important gaps. You need passport, budget comfort, "
        "and travel style to make good suggestions. Skip what you already know."
    ),
    "discovery": (
        "Understand this specific trip. Timing, companions, interests, constraints. "
        "React to each answer with a brief insight. When you have enough to start "
        "thinking about destinations, set phase_complete: true."
    ),
    "narrowing": (
        "Think out loud. Share 4-6 destination ideas in destination_hints. Include a "
        "'thinking' field explaining your reasoning. The user can react to each "
        "destination. Refine based on their reactions."
    ),
    "reveal": (
        "Present your final 3-5 curated suggestions in destination_hints. Each gets "
        "a rich hook (2-3 sentences), budget estimate, and one 'what you might not "
        "expect' insight. Set phase_complete: true."
    ),
}

FALLBACK_TURNS = {
    "profile": ConversationTurn(
        phase="profile",
        question="Let's start with the basics — what passport do you hold? This helps me suggest visa-friendly destinations.",
        options=[
            Option(id="in", label="Indian", insight="Many visa-free options in SE Asia and Caucasus"),
            Option(id="us", label="US", insight="Visa-free almost everywhere"),
            Option(id="other", label="Other", insight="Tell me and I'll check"),
        ],
    ),
    "discovery": ConversationTurn(
        phase="discovery",
        question="When are you thinking of traveling?",
        options=[
            Option(id="soon", label="Next 1-2 months", insight="Limited planning time, focus on easy destinations"),
            Option(id="quarter", label="3-6 months out", insight="Good lead time for most destinations"),
            Option(id="flexible", label="Flexible", insight="We can optimize for best season"),
        ],
    ),
    "narrowing": ConversationTurn(
        phase="narrowing",
        question="I'm having trouble generating ideas right now. Could you tell me more about what you're looking for?",
        options=[
            Option(id="retry", label="Try again", insight="I'll give it another shot"),
        ],
    ),
    "reveal": ConversationTurn(
        phase="reveal",
        question="I'm having trouble finalizing recommendations. Let me try again.",
        options=[
            Option(id="retry", label="Try again", insight="I'll give it another shot"),
        ],
    ),
}


def _build_prompt(
    phase: str,
    messages: List[Dict],
    profile: Dict,
    knowledge_context: str,
) -> List[Dict[str, str]]:
    phase_rules = PHASE_RULES.get(phase, "")
    system = SYSTEM_PROMPT.format(
        phase=phase,
        phase_rules=phase_rules,
        knowledge_context=knowledge_context,
    )

    prompt_messages = [{"role": "system", "content": system}]

    # Add profile context if available
    if profile:
        profile_summary = json.dumps(profile, indent=2)
        prompt_messages.append({
            "role": "system",
            "content": f"User profile:\n{profile_summary}",
        })

    # Add conversation history
    for msg in messages:
        prompt_messages.append({"role": msg["role"], "content": msg["content"]})

    return prompt_messages


def generate_turn(
    phase: str,
    messages: List[Dict],
    profile: Dict,
    knowledge_context: str,
) -> ConversationTurn:
    """Generate the next ConversationTurn via LLM.

    Args:
        phase: Current conversation phase.
        messages: Conversation history (role/content dicts).
        profile: User profile dict (may be empty for first-time users).
        knowledge_context: Pre-built knowledge context string from context_builder.

    Returns:
        ConversationTurn with the next question, options, and optional insights.
    """
    prompt_messages = _build_prompt(phase, messages, profile, knowledge_context)

    try:
        llm = _get_llm()
        logger.info(f"ConversationEngine: generating turn (phase={phase}, msgs={len(messages)})")
        response = llm.invoke(prompt_messages)
        logger.info(f"ConversationEngine: got response ({len(response.content)} chars)")
        data = parse_json_response(response.content)
        turn = ConversationTurn(**data)
        return turn
    except Exception as e:
        logger.error(f"ConversationEngine: LLM failed — {e}")
        return FALLBACK_TURNS.get(phase, FALLBACK_TURNS["profile"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_conversation_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/conversation_engine.py tests/test_conversation_engine.py
git commit -m "feat(api): add conversation engine — core LLM logic for adaptive turns"
```

---

## Task 9: Conversation Engine — Phase Transition Logic + Trip Intent Extraction

**Files:**
- Modify: `api/conversation_engine.py`
- Modify: `tests/test_conversation_engine.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_conversation_engine.py`:

```python
from api.conversation_engine import decide_next_phase


def test_profile_stays_if_under_min_turns():
    phase = decide_next_phase(
        current_phase="profile",
        turn_count=1,
        phase_complete=True,
        profile={},
    )
    assert phase == "profile"  # min 2 turns enforced


def test_profile_transitions_to_discovery():
    phase = decide_next_phase(
        current_phase="profile",
        turn_count=2,
        phase_complete=True,
        profile={"passport_country": "IN"},
    )
    assert phase == "discovery"


def test_discovery_transitions_to_narrowing():
    phase = decide_next_phase(
        current_phase="discovery",
        turn_count=3,
        phase_complete=True,
        profile={"passport_country": "IN"},
    )
    assert phase == "narrowing"


def test_narrowing_transitions_to_reveal():
    phase = decide_next_phase(
        current_phase="narrowing",
        turn_count=1,
        phase_complete=True,
        profile={},
    )
    assert phase == "reveal"


def test_discovery_stays_if_llm_says_not_complete():
    phase = decide_next_phase(
        current_phase="discovery",
        turn_count=5,
        phase_complete=False,
        profile={},
    )
    assert phase == "discovery"


def test_reveal_stays_reveal():
    phase = decide_next_phase(
        current_phase="reveal",
        turn_count=1,
        phase_complete=True,
        profile={},
    )
    assert phase == "reveal"  # No phase after reveal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conversation_engine.py::test_profile_stays_if_under_min_turns -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement decide_next_phase**

Add to `api/conversation_engine.py`:

```python
PHASE_ORDER = ["profile", "discovery", "narrowing", "reveal"]

MIN_TURNS = {
    "profile": settings.discovery_v2_min_profile_turns,
    "discovery": settings.discovery_v2_min_discovery_turns,
    "narrowing": settings.discovery_v2_min_narrowing_turns,
    "reveal": 1,
}


def decide_next_phase(
    current_phase: str,
    turn_count: int,
    phase_complete: bool,
    profile: Dict,
) -> str:
    """Decide whether to transition to the next phase.

    Phase transitions are LLM-decided via phase_complete, with minimum
    turn counts as a safety net.
    """
    min_turns = MIN_TURNS.get(current_phase, 1)

    # Enforce minimum turns
    if turn_count < min_turns:
        return current_phase

    # LLM says not complete — stay in current phase
    if not phase_complete:
        return current_phase

    # Move to next phase
    try:
        idx = PHASE_ORDER.index(current_phase)
        if idx + 1 < len(PHASE_ORDER):
            return PHASE_ORDER[idx + 1]
    except ValueError:
        pass

    return current_phase
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_conversation_engine.py -v`
Expected: PASS

- [ ] **Step 5: Write test for trip intent extraction**

This function extracts structured trip intent from the conversation when transitioning from discovery to narrowing. It is called by the `/respond` route to populate `session["trip_intent"]` so the `/select` endpoint can bridge to the optimizer.

Append to `tests/test_conversation_engine.py`:

```python
from api.conversation_engine import extract_trip_intent_from_messages


def test_extract_trip_intent_returns_structured_data():
    messages = [
        {"role": "assistant", "content": "When are you traveling?"},
        {"role": "user", "content": "July"},
        {"role": "assistant", "content": "How long?"},
        {"role": "user", "content": "7 days"},
        {"role": "assistant", "content": "What excites you?"},
        {"role": "user", "content": "Food and beaches"},
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='{"travel_month": "July", "duration_days": 7, '
                '"interests": ["food", "beaches"], "constraints": [], '
                '"travel_companions": "solo", "region_preference": "", "budget_total": 50000}'
    )

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        intent = extract_trip_intent_from_messages(messages)

    assert intent["travel_month"] == "July"
    assert intent["duration_days"] == 7
    assert "food" in intent["interests"]


def test_extract_trip_intent_fallback_on_failure():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM timeout")

    with patch("api.conversation_engine._get_llm", return_value=mock_llm):
        intent = extract_trip_intent_from_messages([])

    assert intent["duration_days"] == 7
    assert intent["travel_companions"] == "solo"
```

- [ ] **Step 6: Implement extract_trip_intent_from_messages**

Add to `api/conversation_engine.py`:

```python
INTENT_EXTRACTION_PROMPT = """Based on this conversation, extract the trip intent.

Return ONLY valid JSON:
{{
    "travel_month": "month name or season",
    "duration_days": <integer>,
    "interests": ["list of interests"],
    "constraints": ["list of constraints"],
    "travel_companions": "solo|couple|family|group",
    "region_preference": "specific region or empty string",
    "budget_total": <estimated total budget number or 0>
}}

Conversation:
{conversation}"""


def extract_trip_intent_from_messages(messages: List[Dict]) -> Dict:
    """Extract structured trip intent from conversation history via LLM.

    Called when transitioning from discovery to narrowing phase.
    Populates session trip_intent for the optimizer bridge.
    """
    conversation = "\n".join(
        f"{'Assistant' if m['role'] == 'assistant' else 'User'}: {m['content']}"
        for m in messages
    )
    prompt = INTENT_EXTRACTION_PROMPT.format(conversation=conversation)

    try:
        llm = _get_llm()
        logger.info(f"ConversationEngine: extracting trip intent from {len(messages)} messages")
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        logger.error(f"ConversationEngine: trip intent extraction failed — {e}")
        return {
            "travel_month": "",
            "duration_days": 7,
            "interests": [],
            "constraints": [],
            "travel_companions": "solo",
            "region_preference": "",
            "budget_total": 0,
        }
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_conversation_engine.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add api/conversation_engine.py tests/test_conversation_engine.py
git commit -m "feat(api): add phase transition logic + trip intent extraction"
```

---

## Task 10: FastAPI Routes — /start and /respond

**Files:**
- Create: `api/routes.py`
- Create: `api/main.py`
- Create: `tests/test_api_routes.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add FastAPI dependency**

Add to `requirements.txt`:

```
fastapi>=0.115.0
uvicorn>=0.30.0
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_api_routes.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import create_app
from api.models import ConversationTurn, Option


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


MOCK_TURN = ConversationTurn(
    phase="profile",
    question="What passport do you hold?",
    options=[Option(id="in", label="Indian", insight="Great SE Asia access")],
)

MOCK_TURN_JSON = MOCK_TURN.model_dump()


def test_start_creates_session(client):
    with patch("api.routes.generate_turn", return_value=MOCK_TURN):
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "turn" in data
    assert data["turn"]["phase"] == "profile"


def test_start_with_existing_profile_skips_profile_phase(client):
    mock_profile = {
        "user_id": "default",
        "passport_country": "IN",
        "budget_level": "moderate",
        "travel_history": ["Japan"],
        "preferences": {"style": "culture"},
    }
    discovery_turn = ConversationTurn(
        phase="discovery",
        question="When are you traveling?",
        options=[Option(id="summer", label="Summer", insight="Hot but cheap")],
    )

    with patch("api.routes.HistoryDB") as MockDB, \
         patch("api.routes.generate_turn", return_value=discovery_turn):
        MockDB.return_value.get_profile.return_value = mock_profile
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    data = resp.json()
    assert data["turn"]["phase"] == "discovery"


def test_respond_returns_next_turn(client):
    # First start a session
    with patch("api.routes.generate_turn", return_value=MOCK_TURN):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    # Then respond
    next_turn = ConversationTurn(
        phase="profile",
        reaction="Indian passport — great options in SE Asia!",
        question="What's your budget comfort level?",
        options=[Option(id="mod", label="Moderate", insight="₹5-8k/day")],
    )
    with patch("api.routes.generate_turn", return_value=next_turn):
        resp = client.post("/api/discovery/respond", json={
            "session_id": session_id,
            "answer": "Indian passport",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["turn"]["reaction"] is not None


def test_respond_invalid_session(client):
    resp = client.post("/api/discovery/respond", json={
        "session_id": "nonexistent",
        "answer": "hello",
    })
    assert resp.status_code == 404


def test_get_state(client):
    with patch("api.routes.generate_turn", return_value=MOCK_TURN):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    resp = client.get(f"/api/discovery/state?session_id={session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["phase"] in ("profile", "discovery")


def test_get_state_invalid_session(client):
    resp = client.get("/api/discovery/state?session_id=nonexistent")
    assert resp.status_code == 404


def test_select_bridges_to_optimizer(client):
    """POST /select should bridge to optimizer and clean up session."""
    with patch("api.routes.generate_turn", return_value=MOCK_TURN):
        start_resp = client.post("/api/discovery/start", json={"user_id": "default"})
    session_id = start_resp.json()["session_id"]

    # Manually set trip_intent in session so bridge works
    from api.routes import _get_session_store
    store = _get_session_store()
    store.update(session_id, trip_intent={
        "travel_month": "July", "duration_days": 7,
        "interests": ["beaches"], "constraints": [],
        "travel_companions": "solo", "budget_total": 50000,
    })

    with patch("api.routes.HistoryDB"):
        resp = client.post("/api/discovery/select", json={
            "session_id": session_id,
            "destination": "Tbilisi, Georgia",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["destination"] == "Tbilisi, Georgia"
    assert "optimizer_state" in data
    assert data["optimizer_state"]["destination"] == "Tbilisi, Georgia"

    # Session should be cleaned up
    assert store.get(session_id) is None


def test_select_invalid_session(client):
    resp = client.post("/api/discovery/select", json={
        "session_id": "nonexistent",
        "destination": "Tokyo",
    })
    assert resp.status_code == 404
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_api_routes.py -v`
Expected: FAIL

- [ ] **Step 4: Implement routes.py**

```python
# api/routes.py
"""FastAPI routes for Discovery v2 conversation API.

Endpoints:
  POST /api/discovery/start   — start a new conversation, returns first turn
  POST /api/discovery/respond  — submit answer, returns next turn
  GET  /api/discovery/state    — get current session state
  POST /api/discovery/select   — pick a destination, bridge to optimizer

TODO: Add streaming /respond variant via SSE (StreamingResponse + text/event-stream)
for future React frontend. Current sync endpoints are sufficient for Streamlit.
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from api.models import (
    ConversationTurn,
    DiscoveryRespondRequest,
    DiscoverySelectRequest,
    DiscoveryStartRequest,
)
from api.session import SessionStore
from api.conversation_engine import (
    generate_turn,
    decide_next_phase,
    extract_trip_intent_from_messages,
)
from knowledge.context_builder import build_context
from agents.llm_helper import get_llm, parse_json_response
from config import settings
from db import HistoryDB

logger = logging.getLogger("wandermust.api.routes")

router = APIRouter(prefix="/api/discovery")

# Module-level session store (shared across requests)
_session_store = SessionStore()


def _get_session_store() -> SessionStore:
    return _session_store


def _get_db() -> HistoryDB:
    """Get a shared HistoryDB instance. Reuses connection."""
    if not hasattr(_get_db, "_instance"):
        _get_db._instance = HistoryDB(settings.db_path)
    return _get_db._instance


@router.post("/start")
def start(request: DiscoveryStartRequest) -> Dict[str, Any]:
    store = _get_session_store()

    # Check for existing profile
    profile = None
    try:
        db = _get_db()
        profile = db.get_profile(request.user_id)
    except Exception as e:
        logger.warning(f"Failed to load profile: {e}")

    session_id = store.create(user_id=request.user_id, profile=profile)
    session = store.get(session_id)

    # Build knowledge context
    knowledge_ctx = ""
    if profile:
        knowledge_ctx = build_context(
            passport=profile.get("passport_country", "IN"),
            budget=profile.get("budget_level", "moderate"),
        )

    # Generate first turn
    turn = generate_turn(
        phase=session["phase"],
        messages=[],
        profile=profile or {},
        knowledge_context=knowledge_ctx,
    )

    # Store the assistant message
    store.add_message(session_id, role="assistant", content=turn.question)
    store.update(session_id, turn_count=1)

    return {"session_id": session_id, "turn": turn.model_dump()}


@router.post("/respond")
def respond(request: DiscoveryRespondRequest) -> Dict[str, Any]:
    store = _get_session_store()
    session = store.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Record user answer
    store.add_message(request.session_id, role="user", content=request.answer)

    # Build knowledge context
    profile = session["profile"]
    knowledge_ctx = ""
    if profile:
        passport = profile.get("passport_country", "IN")
        budget = profile.get("budget_level", "moderate")
        knowledge_ctx = build_context(passport=passport, budget=budget)

    # Generate next turn
    turn = generate_turn(
        phase=session["phase"],
        messages=session["messages"],
        profile=profile,
        knowledge_context=knowledge_ctx,
    )

    # Update session
    new_turn_count = session["turn_count"] + 1
    next_phase = decide_next_phase(
        current_phase=session["phase"],
        turn_count=new_turn_count,
        phase_complete=turn.phase_complete,
        profile=profile,
    )

    # If phase changed, reset turn count for new phase
    if next_phase != session["phase"]:
        new_turn_count = 0
        turn.phase = next_phase  # Override phase in response

    store.add_message(request.session_id, role="assistant", content=turn.question)
    store.update(
        request.session_id,
        phase=next_phase,
        turn_count=new_turn_count,
    )

    # On profile -> discovery transition: extract and save profile
    if session["phase"] == "profile" and next_phase == "discovery":
        _extract_and_save_profile(request.session_id, session)

    # On discovery -> narrowing transition: extract trip intent
    if session["phase"] == "discovery" and next_phase == "narrowing":
        trip_intent = extract_trip_intent_from_messages(session["messages"])
        store.update(request.session_id, trip_intent=trip_intent)

    return {"session_id": request.session_id, "turn": turn.model_dump()}


@router.get("/state")
def get_state(session_id: str) -> Dict[str, Any]:
    store = _get_session_store()
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "phase": session["phase"],
        "turn_count": session["turn_count"],
        "message_count": len(session["messages"]),
    }


@router.post("/select")
def select(request: DiscoverySelectRequest) -> Dict[str, Any]:
    store = _get_session_store()
    session = store.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from agents.discovery_bridge import build_optimizer_state

    trip_intent = session.get("trip_intent", {})
    optimizer_state = build_optimizer_state(request.destination, trip_intent)

    # Save to DB
    try:
        db = _get_db()
        db.save_discovery_session(
            session["user_id"],
            trip_intent,
            session.get("suggestions", []),
            request.destination,
        )
    except Exception as e:
        logger.warning(f"Failed to save discovery session: {e}")

    # Clean up session
    store.delete(request.session_id)

    return {"destination": request.destination, "optimizer_state": optimizer_state}


# --- Profile extraction prompt (inline, avoids dependency on deprecated onboarding module) ---

_PROFILE_EXTRACTION_PROMPT = """Extract a user travel profile from this conversation.

Return ONLY valid JSON:
{{
    "travel_history": ["countries/cities"],
    "preferences": {{"climate": "warm|cold|moderate|tropical", "pace": "fast|relaxed|moderate", "style": "adventure|relaxation|culture|foodie|mix"}},
    "budget_level": "budget|moderate|luxury",
    "passport_country": "2-letter ISO code"
}}

Conversation:
{conversation}"""


def _extract_and_save_profile(session_id: str, session: Dict) -> None:
    """Extract profile from conversation via LLM and save to DB + session."""
    try:
        conversation = "\n".join(
            f"{'Assistant' if m['role'] == 'assistant' else 'User'}: {m['content']}"
            for m in session["messages"]
        )
        prompt = _PROFILE_EXTRACTION_PROMPT.format(conversation=conversation)
        llm = get_llm(settings.discovery_v2_model)
        response = llm.invoke(prompt)
        extracted = parse_json_response(response.content)

        profile = {
            "user_id": session["user_id"],
            "travel_history": extracted.get("travel_history", []),
            "preferences": extracted.get("preferences", {}),
            "budget_level": extracted.get("budget_level", "moderate"),
            "passport_country": extracted.get("passport_country", "IN"),
        }
        db = _get_db()
        db.save_profile(
            session["user_id"],
            profile["travel_history"],
            profile["preferences"],
            profile["budget_level"],
            profile["passport_country"],
        )
        store = _get_session_store()
        store.update(session_id, profile=profile)
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}")
```

- [ ] **Step 5: Implement main.py**

```python
# api/main.py
"""FastAPI application factory for Discovery v2."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Wandermust Discovery v2", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_api_routes.py -v`
Expected: PASS

- [ ] **Step 7: Run all tests to verify no regressions**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add api/routes.py api/main.py tests/test_api_routes.py requirements.txt
git commit -m "feat(api): add FastAPI routes — /start, /respond, /state, /select endpoints"
```

---

## Task 11: DB Schema Update — Conversation History

**Files:**
- Modify: `db.py`
- Modify: `tests/test_discovery_db.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_discovery_db.py or create tests/test_db_v2.py
import pytest
import json
from db import HistoryDB


def test_save_discovery_session_with_conversation_history():
    db = HistoryDB(":memory:")
    history = [
        {"role": "assistant", "content": "What passport?"},
        {"role": "user", "content": "Indian"},
    ]
    db.save_discovery_session(
        user_id="default",
        trip_intent={"travel_month": "July"},
        suggestions=[{"destination": "Georgia"}],
        chosen_destination="Tbilisi, Georgia",
        conversation_history=history,
    )
    sessions = db.get_discovery_sessions("default")
    assert len(sessions) == 1
    assert sessions[0]["conversation_history"] == history


def test_save_discovery_session_without_conversation_history():
    """Backward compatibility — conversation_history is optional."""
    db = HistoryDB(":memory:")
    db.save_discovery_session(
        user_id="default",
        trip_intent={},
        suggestions=[],
        chosen_destination="Tokyo",
    )
    sessions = db.get_discovery_sessions("default")
    assert len(sessions) == 1
    assert sessions[0]["conversation_history"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_v2.py -v`
Expected: FAIL (signature doesn't accept `conversation_history`)

- [ ] **Step 3: Modify db.py**

Update `save_discovery_session` in `db.py` to accept optional `conversation_history` parameter, and update `_create_tables` to add the column, and update `get_discovery_sessions` to return it.

In `_create_tables`, modify the `discovery_sessions` CREATE TABLE to add:
```sql
conversation_history TEXT NOT NULL DEFAULT '[]',
```

Also add a migration for existing databases (after `_create_tables` call):
```python
def _migrate_tables(self):
    """Add columns that may be missing in older databases."""
    try:
        self._conn.execute("SELECT conversation_history FROM discovery_sessions LIMIT 1")
    except sqlite3.OperationalError:
        self._conn.execute(
            "ALTER TABLE discovery_sessions ADD COLUMN conversation_history TEXT NOT NULL DEFAULT '[]'"
        )
        self._conn.commit()
```

Call `self._migrate_tables()` at the end of `__init__`, after `self._create_tables()`.

Update `save_discovery_session`:
```python
def save_discovery_session(self, user_id, trip_intent, suggestions, chosen_destination,
                           conversation_history=None):
    import json
    self._conn.execute(
        "INSERT INTO discovery_sessions "
        "(user_id, trip_intent, suggestions, chosen_destination, conversation_history) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, json.dumps(trip_intent), json.dumps(suggestions),
         chosen_destination, json.dumps(conversation_history or [])))
    self._conn.commit()
```

Update `get_discovery_sessions` to include `conversation_history` in the SELECT and return:
```python
def get_discovery_sessions(self, user_id="default", limit=5):
    import json
    rows = self._conn.execute(
        "SELECT trip_intent, suggestions, chosen_destination, conversation_history, created_at "
        "FROM discovery_sessions WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit)).fetchall()
    return [{
        "trip_intent": json.loads(r["trip_intent"]),
        "suggestions": json.loads(r["suggestions"]),
        "chosen_destination": r["chosen_destination"],
        "conversation_history": json.loads(r["conversation_history"]) if r["conversation_history"] else [],
        "created_at": r["created_at"],
    } for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_v2.py -v`
Expected: PASS

- [ ] **Step 5: Run all existing DB tests**

Run: `pytest tests/test_db.py tests/test_discovery_db.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add db.py tests/test_db_v2.py
git commit -m "feat(db): add conversation_history column to discovery_sessions"
```

---

## Task 12: Streamlit UI — Render ConversationTurn from API

**Files:**
- Modify: `app.py`

This task replaces the hardcoded ONBOARDING_STEPS/DISCOVERY_STEPS UI with a thin render layer that calls the FastAPI routes and renders `ConversationTurn` responses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app_discovery_v2.py
import pytest
from api.models import ConversationTurn, Option, DestinationHint


def test_conversation_turn_renders_options():
    """Verify ConversationTurn has the fields needed for rendering."""
    turn = ConversationTurn(
        phase="profile",
        question="What passport?",
        options=[
            Option(id="in", label="Indian", insight="SE Asia visa-free", emoji="🇮🇳"),
            Option(id="us", label="US", insight="Global access", emoji="🇺🇸"),
        ],
    )
    assert len(turn.options) == 2
    assert turn.options[0].emoji == "🇮🇳"


def test_conversation_turn_renders_destination_hints():
    turn = ConversationTurn(
        phase="narrowing",
        reaction="Let me think...",
        thinking="I keep coming back to Georgia",
        question="What catches your eye?",
        options=[Option(id="more", label="Tell me more", insight="Dig deeper")],
        destination_hints=[
            DestinationHint(
                name="Tbilisi, Georgia",
                hook="Visa-free, ₹3k/day",
                match_reason="Budget fit",
            ),
        ],
    )
    assert turn.destination_hints is not None
    assert turn.destination_hints[0].name == "Tbilisi, Georgia"


def test_conversation_turn_reveal_phase():
    turn = ConversationTurn(
        phase="reveal",
        reaction="Here are my top picks!",
        question="Which destination excites you?",
        options=[Option(id="pick", label="Pick one below", insight="Click to select")],
        destination_hints=[
            DestinationHint(name="Georgia", hook="Wine + mountains", match_reason="Perfect match"),
            DestinationHint(name="Vietnam", hook="Street food + coast", match_reason="Budget winner"),
        ],
        phase_complete=True,
    )
    assert len(turn.destination_hints) == 2
    assert turn.phase_complete is True
```

- [ ] **Step 2: Run test to verify it passes (these are model validation tests)**

Run: `pytest tests/test_app_discovery_v2.py -v`
Expected: PASS (model tests)

- [ ] **Step 3: Update app.py — Replace the Discover Where section**

Replace the `if mode == "🔍 Discover Where":` block in `app.py` with new code that:

1. Uses `httpx` to call FastAPI endpoints (or imports route functions directly for Streamlit-in-process use)
2. Renders `ConversationTurn` using a `render_turn()` function
3. Stores `session_id` in `st.session_state`
4. Shows reaction as chat bubble, question below, options as pills, destination hints as cards

The key changes:
- Remove `ONBOARDING_STEPS` and `DISCOVERY_STEPS` constants
- Remove `_handle_answer`, `_do_onboarding_extraction`, `_do_discovery_extraction` functions
- Add `render_turn(turn)` function that renders any ConversationTurn
- Call API routes directly (in-process, not HTTP) for Streamlit compatibility

```python
# In the "Discover Where" section of app.py, replace with:
if mode == "🔍 Discover Where":
    st.title("🔍 Discover Where to Travel")
    st.caption("Tell me about yourself — I'll suggest destinations that match")

    from api.models import ConversationTurn, Option, DestinationHint
    from api.routes import start as api_start, respond as api_respond, select as api_select
    from api.routes import get_state as api_get_state, _get_session_store
    from api.models import DiscoveryStartRequest, DiscoveryRespondRequest, DiscoverySelectRequest

    # Session state
    if "v2_session_id" not in st.session_state:
        st.session_state.v2_session_id = None
    if "v2_turns" not in st.session_state:
        st.session_state.v2_turns = []  # List of (role, content_or_turn) tuples
    if "v2_current_turn" not in st.session_state:
        st.session_state.v2_current_turn = None

    def _start_session():
        resp = api_start(DiscoveryStartRequest(user_id="default"))
        st.session_state.v2_session_id = resp["session_id"]
        turn = ConversationTurn(**resp["turn"])
        st.session_state.v2_current_turn = turn
        st.session_state.v2_turns = []

    def _submit_answer(answer):
        # Record user message
        st.session_state.v2_turns.append(("user", answer))
        # Record current turn's question as assistant message
        cur = st.session_state.v2_current_turn
        if cur:
            st.session_state.v2_turns.append(("assistant_turn", cur))

        resp = api_respond(DiscoveryRespondRequest(
            session_id=st.session_state.v2_session_id,
            answer=answer,
        ))
        turn = ConversationTurn(**resp["turn"])
        st.session_state.v2_current_turn = turn
        st.rerun()

    def render_turn_history():
        """Render all previous turns as chat history."""
        for role, content in st.session_state.v2_turns:
            if role == "user":
                with st.chat_message("user"):
                    st.write(content)
            elif role == "assistant_turn":
                turn = content
                if turn.reaction:
                    with st.chat_message("assistant"):
                        st.write(turn.reaction)
                if turn.thinking:
                    with st.chat_message("assistant"):
                        st.markdown(f"*{turn.thinking}*")
                if turn.destination_hints:
                    for hint in turn.destination_hints:
                        with st.container(border=True):
                            st.markdown(f"**{hint.name}**")
                            st.write(hint.hook)
                            st.caption(hint.match_reason)
                with st.chat_message("assistant"):
                    st.write(turn.question)

    def render_current_turn(turn):
        """Render the current turn's interactive elements."""
        if turn.reaction:
            with st.chat_message("assistant"):
                st.write(turn.reaction)
        if turn.thinking:
            with st.chat_message("assistant"):
                st.markdown(f"*{turn.thinking}*")
        if turn.destination_hints:
            for hint in turn.destination_hints:
                with st.container(border=True):
                    col_main, col_btn = st.columns([5, 1])
                    with col_main:
                        st.markdown(f"**{hint.name}**")
                        st.write(hint.hook)
                        st.caption(hint.match_reason)
                    with col_btn:
                        if turn.phase == "reveal":
                            if st.button("Select", key=f"sel_{hint.name}", use_container_width=True):
                                resp = api_select(DiscoverySelectRequest(
                                    session_id=st.session_state.v2_session_id,
                                    destination=hint.name,
                                ))
                                st.session_state.optimizer_prefill = resp["optimizer_state"]
                                st.session_state.chosen_destination = hint.name
                                st.session_state.v2_session_id = None
                                st.session_state.v2_current_turn = None
                                st.session_state.v2_turns = []
                                st.balloons()
                                st.rerun()

        with st.chat_message("assistant"):
            st.write(turn.question)

        # Render options as buttons
        if turn.multi_select:
            if "v2_multi_selected" not in st.session_state:
                st.session_state.v2_multi_selected = set()
            cols = st.columns(min(len(turn.options), 4))
            for j, opt in enumerate(turn.options):
                with cols[j % len(cols)]:
                    is_sel = opt.id in st.session_state.v2_multi_selected
                    label = f"✅ {opt.emoji or ''} {opt.label}" if is_sel else f"{opt.emoji or ''} {opt.label}"
                    if st.button(label, key=f"v2_opt_{opt.id}", use_container_width=True):
                        if opt.id in st.session_state.v2_multi_selected:
                            st.session_state.v2_multi_selected.discard(opt.id)
                        else:
                            st.session_state.v2_multi_selected.add(opt.id)
                        st.rerun()
                    st.caption(opt.insight)
            if st.session_state.v2_multi_selected:
                if st.button("✅ Confirm", type="primary"):
                    selected_labels = [o.label for o in turn.options if o.id in st.session_state.v2_multi_selected]
                    answer = ", ".join(selected_labels)
                    st.session_state.v2_multi_selected = set()
                    _submit_answer(answer)
        else:
            cols = st.columns(min(len(turn.options), 4))
            for j, opt in enumerate(turn.options):
                with cols[j % len(cols)]:
                    label = f"{opt.emoji or ''} {opt.label}".strip()
                    if st.button(label, key=f"v2_opt_{opt.id}", use_container_width=True):
                        _submit_answer(opt.label)
                    st.caption(opt.insight)

        if turn.can_free_text:
            custom = st.chat_input("Type your own answer...")
            if custom:
                if hasattr(st.session_state, "v2_multi_selected"):
                    st.session_state.v2_multi_selected = set()
                _submit_answer(custom)

    # --- Main flow ---
    if st.session_state.v2_session_id is None:
        _start_session()
        st.rerun()
    else:
        render_turn_history()
        turn = st.session_state.v2_current_turn
        if turn:
            render_current_turn(turn)

    if st.button("🔄 Start over", use_container_width=True):
        st.session_state.v2_session_id = None
        st.session_state.v2_turns = []
        st.session_state.v2_current_turn = None
        st.rerun()
```

**Note:** This is a structural guide. The implementing agent should read the current `app.py` completely and make surgical changes — replacing only the "Discover Where" section while keeping the "Optimize When" section untouched. The old `ONBOARDING_STEPS`, `DISCOVERY_STEPS`, `_handle_answer`, `_do_onboarding_extraction`, `_do_discovery_extraction`, and all the old phase handlers should be removed.

- [ ] **Step 4: Manual verification**

Run: `streamlit run app.py`
- Verify "Discover Where" mode shows the new conversational UI
- Verify "Optimize When" mode is unchanged
- Verify conversation flows through profile -> discovery -> narrowing -> reveal

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_discovery_v2.py
git commit -m "feat(ui): replace discovery UI with ConversationTurn renderer + API integration"
```

---

## Task 13: Integration Test — Full 4-Phase Flow

**Files:**
- Create: `tests/test_discovery_v2_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_discovery_v2_integration.py
"""End-to-end integration test for the Discovery v2 4-phase flow."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import create_app
from api.models import ConversationTurn, Option, DestinationHint


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def _mock_turn(phase, question, options, phase_complete=False, reaction=None,
               thinking=None, destination_hints=None):
    return ConversationTurn(
        phase=phase,
        question=question,
        options=[Option(id=o[0], label=o[1], insight=o[2]) for o in options],
        phase_complete=phase_complete,
        reaction=reaction,
        thinking=thinking,
        destination_hints=destination_hints,
    )


def test_full_4_phase_flow(client):
    """Walk through all 4 phases: profile -> discovery -> narrowing -> reveal."""
    turns = [
        # Profile phase - turn 1
        _mock_turn("profile", "What passport?", [("in", "Indian", "SE Asia access")]),
        # Profile phase - turn 2 (completes)
        _mock_turn("profile", "Budget level?", [("mod", "Moderate", "₹5-8k/day")],
                   phase_complete=True, reaction="Indian passport — great options!"),
        # Discovery phase - turn 1
        _mock_turn("discovery", "When traveling?", [("jul", "July", "Monsoon in India")],
                   reaction="Moderate budget opens up most of Asia"),
        # Discovery phase - turn 2 (completes)
        _mock_turn("discovery", "Interests?", [("food", "Food", "Street food tours")],
                   phase_complete=True, reaction="July is great for the Caucasus"),
        # Narrowing phase - turn 1 (completes)
        _mock_turn("narrowing", "What catches your eye?",
                   [("more", "Tell me more", "Dig deeper")],
                   phase_complete=True,
                   reaction="Here's what I'm thinking...",
                   thinking="I keep coming back to Georgia",
                   destination_hints=[
                       DestinationHint(name="Tbilisi", hook="Wine country", match_reason="Budget fit"),
                   ]),
        # Reveal phase
        _mock_turn("reveal", "Pick your destination!",
                   [("pick", "Pick below", "Click to select")],
                   phase_complete=True,
                   reaction="My top picks for you!",
                   destination_hints=[
                       DestinationHint(name="Tbilisi, Georgia", hook="₹3k/day", match_reason="Perfect match"),
                   ]),
    ]
    turn_idx = [0]

    def mock_generate(*args, **kwargs):
        t = turns[turn_idx[0]]
        turn_idx[0] += 1
        return t

    with patch("api.routes.generate_turn", side_effect=mock_generate):
        # Start
        resp = client.post("/api/discovery/start", json={"user_id": "default"})
        assert resp.status_code == 200
        data = resp.json()
        session_id = data["session_id"]
        assert data["turn"]["phase"] == "profile"

        # Profile turn 1 -> answer
        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "Indian passport"})
        assert resp.status_code == 200

        # Profile turn 2 (completes) -> answer
        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "Moderate"})
        assert resp.status_code == 200

        # Discovery turn 1 -> answer
        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "July"})
        assert resp.status_code == 200

        # Discovery turn 2 (completes) -> answer
        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "Food and culture"})
        assert resp.status_code == 200

        # Narrowing -> answer
        resp = client.post("/api/discovery/respond",
                           json={"session_id": session_id, "answer": "Tell me more about Georgia"})
        assert resp.status_code == 200

        # Verify we reached reveal
        state_resp = client.get(f"/api/discovery/state?session_id={session_id}")
        assert state_resp.status_code == 200
        assert state_resp.json()["phase"] == "reveal"


def test_returning_user_skips_profile(client):
    """Returning user with profile in DB should start in discovery phase."""
    mock_profile = {
        "user_id": "default",
        "passport_country": "IN",
        "budget_level": "moderate",
        "travel_history": ["Japan"],
        "preferences": {},
    }
    discovery_turn = _mock_turn("discovery", "When traveling?",
                                [("jul", "July", "Good timing")])

    with patch("api.routes.HistoryDB") as MockDB, \
         patch("api.routes.generate_turn", return_value=discovery_turn):
        MockDB.return_value.get_profile.return_value = mock_profile
        resp = client.post("/api/discovery/start", json={"user_id": "default"})

    assert resp.json()["turn"]["phase"] == "discovery"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_discovery_v2_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_discovery_v2_integration.py
git commit -m "test: add full 4-phase integration test for Discovery v2"
```

---

## Task 14: Clean Up — Remove Old Discovery Code

**Files:**
- Delete content from: `agents/onboarding.py` (keep file but deprecate)
- Delete content from: `agents/discovery_chat.py` (keep file but deprecate)
- Modify: `discovery_graph.py` (add deprecation notice, keep for backward compat)

- [ ] **Step 1: Add deprecation notices**

At the top of `agents/onboarding.py`:
```python
"""DEPRECATED: Use api/conversation_engine.py for Discovery v2.

This module is kept for backward compatibility with existing tests.
New code should use the ConversationTurn-based API.
"""
```

At the top of `agents/discovery_chat.py`:
```python
"""DEPRECATED: Use api/conversation_engine.py for Discovery v2.

This module is kept for backward compatibility with existing tests.
New code should use the ConversationTurn-based API.
"""
```

At the top of `discovery_graph.py`:
```python
"""DEPRECATED: Discovery v2 uses FastAPI routes instead of LangGraph.

This module is kept for reference. New discovery flow is in api/routes.py.
"""
```

- [ ] **Step 2: Verify all tests still pass**

Run: `pytest tests/ -v`
Expected: All PASS (old tests still work with deprecated modules)

- [ ] **Step 3: Commit**

```bash
git add agents/onboarding.py agents/discovery_chat.py discovery_graph.py
git commit -m "chore: add deprecation notices to old discovery modules"
```

---

## Task 15: Final Verification + Run Script

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Run the FastAPI server standalone**

Run: `uvicorn api.main:create_app --factory --port 8000 --reload`
Test with curl:
```bash
# Start session
curl -s -X POST http://localhost:8000/api/discovery/start -H 'Content-Type: application/json' -d '{"user_id":"default"}' | python -m json.tool

# Respond (use session_id from above)
curl -s -X POST http://localhost:8000/api/discovery/respond -H 'Content-Type: application/json' -d '{"session_id":"SESSION_ID","answer":"Indian passport"}' | python -m json.tool
```

- [ ] **Step 3: Run Streamlit app**

Run: `streamlit run app.py`
- Test "Discover Where" flows through all 4 phases
- Test "Optimize When" is unchanged
- Test "Start over" button works

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: Discovery v2 — adaptive 4-phase conversational redesign complete"
```

---

## Summary

| Task | Component | Dependencies |
|------|-----------|-------------|
| 1 | Knowledge: Visa Tiers | None |
| 2 | Knowledge: Budget Tiers | None |
| 3 | Knowledge: Exchange + Seasonality | None |
| 4 | Knowledge: Context Builder | Tasks 1-3 |
| 5 | API Models | None |
| 6 | Session Store | None |
| 7 | Config Update | None |
| 8 | Conversation Engine: Core | Tasks 4, 5 |
| 9 | Conversation Engine: Phase Transitions + Intent Extraction | Task 8 |
| 10 | FastAPI Routes | Tasks 6, 8, 9 |
| 11 | DB Schema Update | None |
| 12 | Streamlit UI | Task 10 |
| 13 | Integration Test | Task 10 |
| 14 | Deprecation Notices | Task 12 |
| 15 | Final Verification | All |

**Parallelizable:** Tasks 1, 2, 3, 5, 6, 7, 11 can all run in parallel (zero dependencies).

## Follow-up Tasks (not in this plan)

These are out of scope for the initial implementation but should be tracked:

1. **SSE Streaming /respond endpoint** — Add `StreamingResponse` variant for React frontend. Streamlit uses sync endpoints.
2. **Session persistence** — SQLite-backed session store for crash recovery. Current in-memory store is sufficient for single-server Streamlit.
3. **Suggestion generator integration in reveal phase** — Optionally call `suggestion_generator.py` to enrich `destination_hints` with budget/seasonality data from knowledge modules.
4. **React frontend** — Consume the same FastAPI endpoints with proper SSE streaming support.
