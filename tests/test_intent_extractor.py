"""Tests for LLM-based intent extraction from free text."""
import pytest
from unittest.mock import patch, MagicMock

from api.intent_extractor import extract_facts_from_text


def _mock_extraction(json_str: str):
    """Helper to mock the extraction LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json_str)
    return patch("api.intent_extractor._get_extractor_llm", return_value=mock_llm)


def test_single_fact_extraction():
    with _mock_extraction('{"budget_level": "Budget-friendly"}'):
        result = extract_facts_from_text("I'm on a tight budget", known_facts={})
    assert result == {"budget_level": "Budget-friendly"}


def test_multi_fact_extraction():
    with _mock_extraction('{"companions": "Solo", "timing": "Next 1-2 months", "budget_level": "Budget-friendly"}'):
        result = extract_facts_from_text(
            "Solo trip next month on a budget",
            known_facts={},
        )
    assert result == {
        "companions": "Solo",
        "timing": "Next 1-2 months",
        "budget_level": "Budget-friendly",
    }


def test_does_not_overwrite_known_facts():
    with _mock_extraction('{"budget_level": "Luxury", "timing": "Next 1-2 months"}'):
        result = extract_facts_from_text(
            "Luxury trip soon",
            known_facts={"budget_level": "Budget-friendly"},
        )
    assert "budget_level" not in result
    assert result == {"timing": "Next 1-2 months"}


def test_empty_extraction():
    with _mock_extraction('{}'):
        result = extract_facts_from_text("hello", known_facts={})
    assert result == {}


def test_llm_failure_returns_empty():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM down")
    with patch("api.intent_extractor._get_extractor_llm", return_value=mock_llm):
        result = extract_facts_from_text("solo budget trip", known_facts={})
    assert result == {}


def test_invalid_topic_keys_filtered():
    with _mock_extraction('{"budget_level": "Budget-friendly", "favorite_color": "blue"}'):
        result = extract_facts_from_text("budget trip", known_facts={})
    assert "favorite_color" not in result
    assert result == {"budget_level": "Budget-friendly"}


def test_short_text_skipped():
    """Text shorter than 5 chars should return empty without calling LLM."""
    result = extract_facts_from_text("hi", known_facts={})
    assert result == {}
