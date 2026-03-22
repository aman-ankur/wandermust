import time
from typing import Any

class TTLCache:
    def __init__(self, ttl_seconds: int = 86400):
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any:
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        self._store.clear()
