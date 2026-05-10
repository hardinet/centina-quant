from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

TTL_MAP = {
    "1m":  60,
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
    "1d":  86400,
    "1w":  604800,
}

_LOCKS: dict[str, asyncio.Lock] = {}


class RedisManager:
    """
    Async Redis cache with adaptive TTL.
    Falls back to in-process dict if Redis is unavailable.
    """

    def __init__(self, url: str | None = None):
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None
        self._fallback: dict[str, str] = {}
        self._available = False

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(self._url, decode_responses=True)
            await self._client.ping()
            self._available = True
            logger.info("Redis connected: %s", self._url)
        except Exception as e:
            logger.warning("Redis unavailable (%s) – using in-process fallback", e)
            self._available = False

    async def get(self, key: str) -> Any | None:
        if self._available and self._client:
            raw = await self._client.get(key)
            return json.loads(raw) if raw else None
        return json.loads(self._fallback[key]) if key in self._fallback else None

    async def set(self, key: str, value: Any, ttl: int = 60) -> None:
        serialised = json.dumps(value)
        if self._available and self._client:
            await self._client.set(key, serialised, ex=ttl)
        else:
            self._fallback[key] = serialised

    async def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Coroutine],
        interval: str = "1h",
    ) -> Any:
        lock = _LOCKS.setdefault(key, asyncio.Lock())
        async with lock:
            cached = await self.get(key)
            if cached is not None:
                return cached
            result = await fetch_fn()
            ttl = TTL_MAP.get(interval, 3600)
            await self.set(key, result, ttl)
            return result

    async def delete(self, key: str) -> None:
        if self._available and self._client:
            await self._client.delete(key)
        self._fallback.pop(key, None)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
