import pytest
from knowledge.context_builder import build_context
from knowledge.visa_tiers import VISA_TIERS
from knowledge.budget_tiers import BUDGET_TIERS
from knowledge.exchange_rates import EXCHANGE_FAVORABILITY
from knowledge.seasonality import SEASONALITY


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
    assert 12 in th["best"]
    assert "dry" in th["note"].lower() or "nov" in th["note"].lower()


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
    assert isinstance(ctx, str)


def test_build_context_token_size():
    """Context should be compact — under 500 tokens (rough: ~4 chars/token)."""
    ctx = build_context(passport="IN", budget="moderate", month=7)
    estimated_tokens = len(ctx) / 4
    assert estimated_tokens < 500, f"Context too large: ~{estimated_tokens:.0f} tokens"
