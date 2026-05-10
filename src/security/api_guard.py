from __future__ import annotations

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from binance import BinanceAPIException, BinanceRequestException

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Binance REST weight limits
WEIGHT_LIMIT    = 1200   # per minute
WEIGHT_WINDOW   = 60     # seconds

# Backoff settings
MAX_RETRIES     = 5
BASE_DELAY      = 1.0    # seconds
MAX_DELAY       = 60.0

# Error codes that warrant a retry
RETRYABLE_CODES = {-1003, -1015, -1021, 429, 418}


class WeightBucket:
    """Sliding-window weight tracker (shared across all BinanceREST instances)."""

    def __init__(self, limit: int = WEIGHT_LIMIT, window: int = WEIGHT_WINDOW):
        self._limit  = limit
        self._window = window
        self._slots: list[tuple[float, int]] = []  # (timestamp, weight)

    def _prune(self) -> None:
        now    = time.monotonic()
        cutoff = now - self._window
        self._slots = [(t, w) for t, w in self._slots if t > cutoff]

    def used(self) -> int:
        self._prune()
        return sum(w for _, w in self._slots)

    async def acquire(self, weight: int) -> None:
        while True:
            self._prune()
            if self.used() + weight <= self._limit:
                self._slots.append((time.monotonic(), weight))
                return
            oldest = min(t for t, _ in self._slots) if self._slots else time.monotonic()
            sleep_for = max(0.1, oldest + self._window - time.monotonic())
            logger.debug("API weight budget exhausted – sleeping %.1fs", sleep_for)
            await asyncio.sleep(sleep_for)


# Module-level shared bucket
_global_bucket = WeightBucket()


class APIGuard:
    """
    Decorator/context helper that wraps Binance API calls with:
    - Weight-aware rate limiting (shared global bucket)
    - Exponential backoff with jitter on retryable errors
    - Endpoint rotation for fallback (e.g. testnet)
    """

    def __init__(self, weight: int = 1):
        self.weight = weight

    def __call__(self, func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await _global_bucket.acquire(self.weight)
            delay = BASE_DELAY
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    return await func(*args, **kwargs)
                except BinanceAPIException as e:
                    if e.code in RETRYABLE_CODES:
                        jitter = random.uniform(0, delay * 0.3)
                        wait = min(delay + jitter, MAX_DELAY)
                        logger.warning(
                            "Binance rate/error %d (attempt %d/%d) – retry in %.1fs",
                            e.code, attempt, MAX_RETRIES, wait,
                        )
                        await asyncio.sleep(wait)
                        delay = min(delay * 2, MAX_DELAY)
                        await _global_bucket.acquire(self.weight)
                    else:
                        raise
                except BinanceRequestException as e:
                    if attempt < MAX_RETRIES:
                        jitter = random.uniform(0, delay * 0.3)
                        wait = min(delay + jitter, MAX_DELAY)
                        logger.warning("Network error (attempt %d/%d) – retry in %.1fs", attempt, MAX_RETRIES, wait)
                        await asyncio.sleep(wait)
                        delay = min(delay * 2, MAX_DELAY)
                    else:
                        raise
            raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded for {func.__name__}")
        return wrapper  # type: ignore


def rate_limited(weight: int = 1) -> APIGuard:
    """Shorthand decorator factory: @rate_limited(weight=10)."""
    return APIGuard(weight=weight)


async def with_backoff(coro, weight: int = 1, max_retries: int = MAX_RETRIES):
    """Wrap an arbitrary coroutine with rate-limit + backoff (imperative form)."""
    await _global_bucket.acquire(weight)
    delay = BASE_DELAY
    for attempt in range(1, max_retries + 1):
        try:
            return await coro
        except BinanceAPIException as e:
            if e.code not in RETRYABLE_CODES or attempt == max_retries:
                raise
            await asyncio.sleep(min(delay * (1 + random.random() * 0.3), MAX_DELAY))
            delay = min(delay * 2, MAX_DELAY)
            await _global_bucket.acquire(weight)
