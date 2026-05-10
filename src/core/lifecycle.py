from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Callable

logger = logging.getLogger(__name__)

_shutdown_event = asyncio.Event()
_shutdown_callbacks: list[Callable] = []


def register_shutdown(callback: Callable) -> None:
    _shutdown_callbacks.append(callback)


def request_shutdown() -> None:
    logger.info("Lifecycle: shutdown requested")
    _shutdown_event.set()


async def wait_for_shutdown() -> None:
    await _shutdown_event.wait()


def setup_signal_handlers() -> None:
    loop = asyncio.get_event_loop()

    def _handle(sig: signal.Signals) -> None:
        logger.info("Lifecycle: received signal %s", sig.name)
        loop.call_soon_threadsafe(request_shutdown)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle, sig)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda s, f: request_shutdown())


async def graceful_shutdown(timeout: float = 30.0) -> None:
    logger.info("Lifecycle: beginning graceful shutdown (timeout=%ss)", timeout)
    tasks = []
    for cb in _shutdown_callbacks:
        try:
            result = cb()
            if asyncio.iscoroutine(result):
                tasks.append(asyncio.create_task(result))
        except Exception as e:
            logger.error("Shutdown callback error: %s", e)

    if tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
        except asyncio.TimeoutError:
            logger.critical("Lifecycle: shutdown timeout — force exit")

    logger.info("Lifecycle: shutdown complete")
