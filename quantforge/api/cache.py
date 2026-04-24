"""Cache layer: Redis if available, in-memory TTL dict otherwise."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from typing import Any


class _InMemoryCache:
    """Thread-safe-enough TTL cache. Fallback when Redis is not configured."""

    def __init__(self, max_entries: int = 10_000):
        self._store: dict[str, tuple[float, bytes]] = {}
        self._max = max_entries

    def get(self, key: str) -> bytes | None:
        item = self._store.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.time():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: bytes, ttl_seconds: int = 300) -> None:
        if len(self._store) >= self._max:
            # drop oldest — cheap eviction
            oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
            self._store.pop(oldest, None)
        self._store[key] = (time.time() + ttl_seconds, value)

    def ping(self) -> bool:
        return True

    def clear(self) -> None:
        self._store.clear()


class _RedisCache:
    def __init__(self, url: str):
        import redis  # type: ignore

        self.client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2,
                                             decode_responses=False)

    def get(self, key: str) -> bytes | None:
        try:
            return self.client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: bytes, ttl_seconds: int = 300) -> None:
        try:
            self.client.set(key, value, ex=ttl_seconds)
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False

    def clear(self) -> None:
        try:
            self.client.flushdb()
        except Exception:
            pass


def _build_cache():
    url = os.environ.get("REDIS_URL", "").strip()
    if url:
        try:
            c = _RedisCache(url)
            if c.ping():
                return c
        except Exception:
            pass
    return _InMemoryCache()


_cache = _build_cache()


def cache_backend() -> Any:
    return _cache


def reset_cache() -> None:
    global _cache
    _cache = _build_cache()


def make_key(namespace: str, **parts) -> str:
    """Deterministic key with a namespace prefix."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:24]
    return f"qf:{namespace}:{h}"


def cached(namespace: str, ttl_seconds: int = 300):
    """Decorator that caches JSON-serializable return values by kwargs."""
    def decorator(fn: Callable[..., Any]):
        def wrapper(**kwargs):
            key = make_key(namespace, **kwargs)
            hit = _cache.get(key)
            if hit is not None:
                try:
                    return json.loads(hit)
                except Exception:
                    pass
            val = fn(**kwargs)
            try:
                _cache.set(key, json.dumps(val, default=str).encode(), ttl_seconds)
            except Exception:
                pass
            return val
        wrapper.__wrapped__ = fn
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator
