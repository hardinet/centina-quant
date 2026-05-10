from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable

from binance import AsyncClient, BinanceSocketManager

logger = logging.getLogger(__name__)

Callback = Callable[[dict[str, Any]], None]


RECONNECT_SEQUENCE = [1, 2, 4, 8, 16, 60]   # exponential back-off seconds
HEARTBEAT_INTERVAL = 30                       # seconds
CIRCULAR_BUFFER_SIZE = 100


class BinanceWebSocket:
    """
    v5.0 Persistent WebSocket manager.
    - Heartbeat every 30s, silent-disconnect detection
    - Exponential reconnect: 1→2→4→8→16→60s
    - Circular message buffer (100 msgs) per stream
    """

    def __init__(self, api_key: str, api_secret: str):
        self._key = api_key
        self._secret = api_secret
        self._client: AsyncClient | None = None
        self._bsm: BinanceSocketManager | None = None
        self._callbacks: dict[str, list[Callback]] = defaultdict(list)
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._last_msg: dict[str, float] = {}   # key → timestamp
        self._buffers: dict[str, list[dict]] = defaultdict(list)

    async def connect(self) -> None:
        self._client = await AsyncClient.create(self._key, self._secret)
        self._bsm = BinanceSocketManager(self._client)
        self._running = True
        logger.info("WebSocket manager ready")

    async def disconnect(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._client:
            await self._client.close_connection()
        logger.info("WebSocket disconnected")

    # ------------------------------------------------------------------
    # Stream subscriptions
    # ------------------------------------------------------------------

    def subscribe_kline(self, symbol: str, interval: str, callback: Callback) -> None:
        key = f"kline_{symbol}_{interval}"
        self._callbacks[key].append(callback)

    def subscribe_ticker(self, symbol: str, callback: Callback) -> None:
        key = f"ticker_{symbol}"
        self._callbacks[key].append(callback)

    def subscribe_user_data(self, callback: Callback) -> None:
        self._callbacks["user_data"].append(callback)

    async def start_streams(self) -> None:
        for key in list(self._callbacks.keys()):
            task = asyncio.create_task(self._stream_loop(key))
            self._tasks.append(task)
        hb_task = asyncio.create_task(self._heartbeat_loop())
        self._tasks.append(hb_task)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _stream_loop(self, key: str) -> None:
        attempt = 0
        import time
        while self._running:
            try:
                socket = self._get_socket(key)
                if socket is None:
                    logger.error("No socket factory for key: %s", key)
                    return
                attempt = 0
                async with socket as stream:
                    logger.info("Stream started: %s", key)
                    async for msg in stream:
                        self._last_msg[key] = time.monotonic()
                        # Circular buffer
                        buf = self._buffers[key]
                        buf.append(msg)
                        if len(buf) > CIRCULAR_BUFFER_SIZE:
                            buf.pop(0)
                        for cb in self._callbacks[key]:
                            try:
                                cb(msg)
                            except Exception as e:
                                logger.error("Callback error on %s: %s", key, e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                delay = RECONNECT_SEQUENCE[min(attempt, len(RECONNECT_SEQUENCE) - 1)]
                logger.warning("Stream %s dropped (%s), reconnecting in %ds", key, e, delay)
                attempt += 1
                await asyncio.sleep(delay)

    async def _heartbeat_loop(self) -> None:
        import time
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            now = time.monotonic()
            for key, last_ts in list(self._last_msg.items()):
                if now - last_ts > HEARTBEAT_INTERVAL * 3:
                    logger.warning("Silent disconnect detected on stream: %s", key)

    def _get_socket(self, key: str):
        if key.startswith("kline_"):
            _, symbol, interval = key.split("_", 2)
            return self._bsm.kline_socket(symbol, interval)
        if key.startswith("ticker_"):
            symbol = key[len("ticker_"):]
            return self._bsm.symbol_ticker_socket(symbol)
        if key == "user_data":
            return self._bsm.user_socket()
        return None

    # ------------------------------------------------------------------
    # Convenience: subscribe and start in one call
    # ------------------------------------------------------------------

    async def watch_klines(
        self, symbols: list[str], interval: str, callback: Callback
    ) -> None:
        for sym in symbols:
            self.subscribe_kline(sym, interval, callback)
        await self.start_streams()
