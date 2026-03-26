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
    assert session["phase"] == "discovery"


def test_delete_session():
    store = SessionStore()
    sid = store.create(user_id="default")
    store.delete(sid)
    assert store.get(sid) is None
