import time
from cache import TTLCache

def test_cache_set_and_get():
    c = TTLCache(ttl_seconds=60)
    c.set("key1", {"data": 42})
    assert c.get("key1") == {"data": 42}

def test_cache_miss():
    c = TTLCache(ttl_seconds=60)
    assert c.get("missing") is None

def test_cache_expiry():
    c = TTLCache(ttl_seconds=1)
    c.set("key1", "value")
    time.sleep(1.1)
    assert c.get("key1") is None
