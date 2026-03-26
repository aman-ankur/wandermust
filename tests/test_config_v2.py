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
