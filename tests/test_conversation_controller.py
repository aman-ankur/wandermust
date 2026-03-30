"""Tests for the deterministic conversation controller.

Pure Python, zero LLM mocking — validates topic selection, answer parsing,
phase transitions, full controller flow, and trip intent extraction.
"""
import pytest

from api.conversation_controller import (
    REQUIRED_DISCOVERY_FACTS,
    REQUIRED_PROFILE_FACTS,
    TOPIC_REGISTRY,
    ControllerResult,
    TopicConfig,
    build_controller_turn,
    build_first_turn,
    build_profile_from_facts,
    build_trip_intent_from_facts,
    get_topic,
    parse_answer,
    pick_bonus_topic,
    pick_next_topic,
    should_transition,
)


# ── Topic Selection ──────────────────────────────────────────────────────

class TestPickNextTopic:
    def test_empty_facts_returns_passport(self):
        topic = pick_next_topic({}, "profile")
        assert topic is not None
        assert topic.key == "passport"

    def test_passport_filled_returns_budget(self):
        topic = pick_next_topic({"passport": "Indian"}, "profile")
        assert topic is not None
        assert topic.key == "budget_level"

    def test_all_profile_filled_returns_none(self):
        facts = {"passport": "US", "budget_level": "Mid-range", "travel_style": "Adventure & outdoors"}
        topic = pick_next_topic(facts, "profile")
        assert topic is None

    def test_discovery_phase_starts_with_timing(self):
        topic = pick_next_topic({}, "discovery")
        assert topic is not None
        assert topic.key == "timing"

    def test_discovery_skips_filled(self):
        topic = pick_next_topic({"timing": "Flexible"}, "discovery")
        assert topic is not None
        assert topic.key == "companions"

    def test_all_discovery_filled_returns_none(self):
        facts = {
            "timing": "Flexible",
            "companions": "Solo",
            "interests": ["Nature & hiking"],
            "deal_breakers": ["No deal-breakers"],
        }
        topic = pick_next_topic(facts, "discovery")
        assert topic is None

    def test_unknown_phase_returns_none(self):
        topic = pick_next_topic({}, "narrowing")
        assert topic is None


# ── Answer Parsing ───────────────────────────────────────────────────────

class TestParseAnswer:
    def test_option_click_maps_to_label(self):
        topic = get_topic("passport")
        result = parse_answer(topic, option_ids=["us"])
        assert result == "US"

    def test_multi_select_stores_list(self):
        topic = get_topic("interests")
        result = parse_answer(topic, option_ids=["food", "nature"])
        assert result == ["Street food & local cuisine", "Nature & hiking"]

    def test_free_text_keyword_match(self):
        topic = get_topic("budget_level")
        result = parse_answer(topic, free_text="I'm on a budget")
        assert result == "Budget-friendly"

    def test_free_text_no_match_stores_raw(self):
        topic = get_topic("passport")
        result = parse_answer(topic, free_text="Canadian")
        assert result == "Canadian"

    def test_free_text_multi_select_no_match(self):
        topic = get_topic("interests")
        result = parse_answer(topic, free_text="I love scuba diving")
        assert result == ["I love scuba diving"]

    def test_option_ids_take_priority_over_free_text(self):
        topic = get_topic("passport")
        result = parse_answer(topic, option_ids=["in"], free_text="US citizen")
        assert result == "Indian"

    def test_no_input_returns_none(self):
        topic = get_topic("passport")
        result = parse_answer(topic)
        assert result is None

    def test_unknown_option_id_returns_id(self):
        topic = get_topic("passport")
        result = parse_answer(topic, option_ids=["xx"])
        assert result == "xx"

    def test_multi_select_filters_unknown_ids(self):
        topic = get_topic("interests")
        result = parse_answer(topic, option_ids=["food", "unknown_id"])
        assert result == ["Street food & local cuisine"]

    def test_free_text_id_match(self):
        topic = get_topic("budget_level")
        result = parse_answer(topic, free_text="moderate")
        assert result == "Mid-range"


# ── Phase Transitions ────────────────────────────────────────────────────

class TestShouldTransition:
    def test_all_profile_facts_transitions_to_discovery(self):
        facts = {"passport": "US", "budget_level": "Mid-range", "travel_style": "Mix of everything"}
        result = should_transition("profile", facts, 3)
        assert result == "discovery"

    def test_missing_profile_facts_stays(self):
        facts = {"passport": "US"}
        result = should_transition("profile", facts, 1)
        assert result is None

    def test_profile_max_turns_forces_transition(self):
        facts = {"passport": "US"}
        result = should_transition("profile", facts, 5)
        assert result == "discovery"

    def test_all_discovery_facts_transitions_to_narrowing(self):
        facts = {"timing": "Flexible", "companions": "Solo", "interests": ["Nature"]}
        result = should_transition("discovery", facts, 3)
        assert result == "narrowing"

    def test_missing_discovery_facts_stays(self):
        facts = {"timing": "Flexible"}
        result = should_transition("discovery", facts, 1)
        assert result is None

    def test_discovery_max_turns_forces_transition(self):
        facts = {"timing": "Flexible"}
        result = should_transition("discovery", facts, 6)
        assert result == "narrowing"

    def test_narrowing_transitions_to_reveal(self):
        result = should_transition("narrowing", {}, 1)
        assert result == "reveal"

    def test_narrowing_stays_below_min_turns(self):
        result = should_transition("narrowing", {}, 0)
        assert result is None

    def test_reveal_never_auto_transitions(self):
        result = should_transition("reveal", {}, 10)
        assert result is None


# ── Full Controller Flow ─────────────────────────────────────────────────

class TestBuildFirstTurn:
    def test_new_session_returns_passport_topic(self):
        session = {
            "phase": "profile",
            "known_facts": {},
            "turn_count": 0,
            "last_topic_key": None,
        }
        ctrl = build_first_turn(session)
        assert ctrl.topic is not None
        assert ctrl.topic.key == "passport"
        assert ctrl.phase == "profile"
        assert ctrl.phase_changed is False
        assert len(ctrl.option_templates) == 5

    def test_returning_user_skips_to_discovery(self):
        session = {
            "phase": "discovery",
            "known_facts": {"passport": "US", "budget_level": "Mid-range", "travel_style": "Mix"},
            "turn_count": 0,
            "last_topic_key": None,
        }
        ctrl = build_first_turn(session)
        assert ctrl.topic is not None
        assert ctrl.topic.key == "timing"
        assert ctrl.phase == "discovery"


class TestBuildControllerTurn:
    def test_passport_answer_advances_to_budget(self):
        session = {
            "phase": "profile",
            "known_facts": {},
            "turn_count": 1,
            "last_topic_key": "passport",
        }
        ctrl = build_controller_turn(session, option_ids=["us"])
        assert ctrl.known_facts["passport"] == "US"
        assert ctrl.topic is not None
        assert ctrl.topic.key == "budget_level"
        assert ctrl.phase == "profile"
        assert ctrl.phase_changed is False

    def test_final_profile_answer_transitions_to_discovery(self):
        session = {
            "phase": "profile",
            "known_facts": {"passport": "US", "budget_level": "Mid-range"},
            "turn_count": 3,
            "last_topic_key": "travel_style",
        }
        ctrl = build_controller_turn(session, option_ids=["adventure"])
        assert "travel_style" in ctrl.known_facts
        assert ctrl.phase == "discovery"
        assert ctrl.phase_changed is True
        assert ctrl.topic is not None
        assert ctrl.topic.key == "timing"

    def test_full_profile_to_discovery_progression(self):
        session = {
            "phase": "profile",
            "known_facts": {},
            "turn_count": 0,
            "last_topic_key": None,
        }

        ctrl = build_first_turn(session)
        assert ctrl.topic.key == "passport"

        session["last_topic_key"] = "passport"
        session["turn_count"] = 1
        ctrl = build_controller_turn(session, option_ids=["in"])
        assert ctrl.known_facts["passport"] == "Indian"
        assert ctrl.topic.key == "budget_level"

        session["known_facts"] = ctrl.known_facts
        session["last_topic_key"] = "budget_level"
        session["turn_count"] = 2
        ctrl = build_controller_turn(session, option_ids=["moderate"])
        assert ctrl.known_facts["budget_level"] == "Mid-range"
        assert ctrl.topic.key == "travel_style"

        session["known_facts"] = ctrl.known_facts
        session["last_topic_key"] = "travel_style"
        session["turn_count"] = 3
        ctrl = build_controller_turn(session, option_ids=["mix"])
        assert ctrl.known_facts["travel_style"] == "Mix of everything"
        assert ctrl.phase == "discovery"
        assert ctrl.phase_changed is True
        assert ctrl.topic.key == "timing"

    def test_multi_select_answer(self):
        session = {
            "phase": "discovery",
            "known_facts": {"timing": "Flexible", "companions": "Solo"},
            "turn_count": 3,
            "last_topic_key": "interests",
        }
        ctrl = build_controller_turn(session, option_ids=["food", "nature", "history"])
        assert ctrl.known_facts["interests"] == [
            "Street food & local cuisine",
            "Nature & hiking",
            "History & architecture",
        ]
        assert ctrl.last_answer_text == "Street food & local cuisine, Nature & hiking, History & architecture"

    def test_free_text_answer(self):
        session = {
            "phase": "profile",
            "known_facts": {},
            "turn_count": 1,
            "last_topic_key": "passport",
        }
        ctrl = build_controller_turn(session, free_text="I have an Indian passport")
        assert ctrl.known_facts["passport"] == "Indian"


# ── Trip Intent from Facts ───────────────────────────────────────────────

class TestBuildTripIntentFromFacts:
    def test_basic_facts(self):
        facts = {
            "timing": "Next 1-2 months",
            "companions": "With partner",
            "interests": ["Street food & local cuisine", "Nature & hiking"],
            "deal_breakers": ["Long flights (10+ hours)"],
        }
        intent = build_trip_intent_from_facts(facts)
        assert intent["travel_month"] == "next month"
        assert intent["travel_companions"] == "couple"
        assert intent["interests"] == ["Street food & local cuisine", "Nature & hiking"]
        assert intent["constraints"] == ["Long flights (10+ hours)"]
        assert intent["duration_days"] == 7

    def test_no_deal_breakers(self):
        facts = {
            "timing": "Flexible",
            "companions": "Solo",
            "interests": ["Nature & hiking"],
            "deal_breakers": ["No deal-breakers"],
        }
        intent = build_trip_intent_from_facts(facts)
        assert intent["constraints"] == []

    def test_empty_facts(self):
        intent = build_trip_intent_from_facts({})
        assert intent["travel_month"] == "flexible"
        assert intent["travel_companions"] == "solo"
        assert intent["interests"] == []
        assert intent["constraints"] == []


# ── Profile from Facts ───────────────────────────────────────────────────

class TestBuildProfileFromFacts:
    def test_basic_profile(self):
        facts = {
            "passport": "Indian",
            "budget_level": "Mid-range",
            "travel_style": "Adventure & outdoors",
        }
        profile = build_profile_from_facts(facts, user_id="test-user")
        assert profile["passport_country"] == "IN"
        assert profile["budget_level"] == "moderate"
        assert profile["preferences"]["style"] == "adventure"
        assert profile["user_id"] == "test-user"

    def test_unknown_passport_defaults(self):
        facts = {"passport": "Martian"}
        profile = build_profile_from_facts(facts)
        assert profile["passport_country"] == "XX"

    def test_empty_facts_defaults(self):
        profile = build_profile_from_facts({})
        assert profile["passport_country"] == "XX"
        assert profile["budget_level"] == "moderate"
        assert profile["preferences"]["style"] == "mix"


# ── Topic Registry Integrity ─────────────────────────────────────────────

class TestTopicRegistry:
    def test_all_profile_topics_exist(self):
        profile_keys = {t.key for t in TOPIC_REGISTRY if t.phase == "profile"}
        assert REQUIRED_PROFILE_FACTS.issubset(profile_keys)

    def test_all_discovery_topics_exist(self):
        discovery_keys = {t.key for t in TOPIC_REGISTRY if t.phase == "discovery"}
        assert REQUIRED_DISCOVERY_FACTS.issubset(discovery_keys)

    def test_multi_select_topics(self):
        multi = [t for t in TOPIC_REGISTRY if t.multi_select]
        multi_keys = {t.key for t in multi}
        assert "interests" in multi_keys
        assert "deal_breakers" in multi_keys
        assert len(multi) == 2

    def test_all_topics_have_options(self):
        for topic in TOPIC_REGISTRY:
            assert len(topic.option_templates) >= 3, f"{topic.key} has too few options"

    def test_option_ids_unique_per_topic(self):
        for topic in TOPIC_REGISTRY:
            ids = [o["id"] for o in topic.option_templates]
            assert len(ids) == len(set(ids)), f"{topic.key} has duplicate option ids"

    def test_get_topic_returns_correct(self):
        assert get_topic("passport").key == "passport"
        assert get_topic("interests").multi_select is True
        assert get_topic("nonexistent") is None


# ── Min-Turn Guards ─────────────────────────────────────────────────────

class TestMinTurnGuards:
    def test_transition_blocked_by_min_turns(self):
        """Even with all required facts, transition should wait for min turns."""
        facts = {"passport": "Indian", "budget_level": "Mid-range", "travel_style": "Mix of everything"}
        result = should_transition("profile", facts, turn_count=1)
        assert result is None

    def test_transition_allowed_after_min_turns(self):
        facts = {"passport": "Indian", "budget_level": "Mid-range", "travel_style": "Mix of everything"}
        result = should_transition("profile", facts, turn_count=2)
        assert result == "discovery"

    def test_discovery_transition_blocked_by_min_turns(self):
        facts = {"timing": "Flexible", "companions": "Solo", "interests": ["Food"]}
        result = should_transition("discovery", facts, turn_count=1)
        assert result is None

    def test_discovery_transition_allowed(self):
        facts = {"timing": "Flexible", "companions": "Solo", "interests": ["Food"]}
        result = should_transition("discovery", facts, turn_count=2)
        assert result == "narrowing"

    def test_safety_net_still_forces_transition(self):
        """Max turns safety net should override min-turn requirement."""
        facts = {"passport": "Indian"}  # Not all required, but max turns reached
        result = should_transition("profile", facts, turn_count=5)
        assert result == "discovery"


# ── Bonus Topics ────────────────────────────────────────────────────────

class TestPickBonusTopic:
    def test_pick_bonus_topic_profile(self):
        """No non-required profile topics exist."""
        facts = {"passport": "Indian", "budget_level": "Mid-range", "travel_style": "Mix"}
        bonus = pick_bonus_topic(facts, "profile")
        assert bonus is None

    def test_pick_bonus_topic_discovery(self):
        """deal_breakers is not required, should be offered as bonus."""
        facts = {"timing": "Flexible", "companions": "Solo", "interests": ["Food"]}
        bonus = pick_bonus_topic(facts, "discovery")
        assert bonus is not None
        assert bonus.key == "deal_breakers"

    def test_pick_bonus_topic_all_filled(self):
        facts = {"timing": "Flexible", "companions": "Solo", "interests": ["Food"], "deal_breakers": ["None"]}
        bonus = pick_bonus_topic(facts, "discovery")
        assert bonus is None
