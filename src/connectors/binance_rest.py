from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import pandas as pd
from binance import AsyncClient, BinanceAPIException
from binance.enums import (
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_STOP_LOSS_LIMIT,
    SIDE_BUY,
    SIDE_SELL,
    TIME_IN_FORCE_GTC,
)

logger = logging.getLogger(__name__)

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "trades", "taker_buy_base",
              "taker_buy_quote", "ignore"]


class RateLimiter:
    """Token bucket – Binance allows 1200 weight/min on REST."""

    def __init__(self, max_weight: int = 1200, window_sec: int = 60):
        self._max = max_weight
        self._window = window_sec
        self._used = 0
        self._reset_at = time.monotonic() + window_sec

    async def acquire(self, weight: int = 1) -> None:
        now = time.monotonic()
        if now >= self._reset_at:
            self._used = 0
            self._reset_at = now + self._window

        if self._used + weight > self._max:
            sleep_for = self._reset_at - now
            logger.debug("Rate limit hit, sleeping %.1fs", sleep_for)
            await asyncio.sleep(sleep_for)
            self._used = 0
            self._reset_at = time.monotonic() + self._window

        self._used += weight


class BinanceREST:
    """
    Async REST wrapper around python-binance.
    All public methods are coroutines.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self._key = api_key
        self._secret = api_secret
        self._testnet = testnet
        self._client: AsyncClient | None = None
        self._rate = RateLimiter()

    async def connect(self) -> None:
        self._client = await AsyncClient.create(self._key, self._secret, testnet=self._testnet)
        logger.info("Binance REST connected (testnet=%s)", self._testnet)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close_connection()
            logger.info("Binance REST disconnected")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_top_usdt_pairs(self, n: int = 50) -> list[str]:
        await self._rate.acquire(40)
        tickers = await self._client.get_ticker()
        usdt = [t for t in tickers if t["symbol"].endswith("USDT")
                and not any(s in t["symbol"] for s in ("UP", "DOWN", "BULL", "BEAR"))]
        usdt.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
        return [t["symbol"] for t in usdt[:n]]

    async def get_klines(
        self, symbol: str, interval: str, limit: int = 300
    ) -> pd.DataFrame:
        await self._rate.acquire(2)
        raw = await self._client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(raw, columns=KLINE_COLS)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df

    async def get_exchange_info(self, symbol: str) -> dict[str, Any]:
        await self._rate.acquire(10)
        info = await self._client.get_symbol_info(symbol)
        return info or {}

    async def get_account_balance(self) -> dict[str, float]:
        await self._rate.acquire(10)
        account = await self._client.get_account()
        return {
            b["asset"]: float(b["free"])
            for b in account["balances"]
            if float(b["free"]) > 0 or float(b["locked"]) > 0
        }

    async def get_usdt_balance(self) -> float:
        balances = await self.get_account_balance()
        return balances.get("USDT", 0.0)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_limit_buy(
        self, symbol: str, qty: float, price: float
    ) -> dict[str, Any]:
        await self._rate.acquire(1)
        logger.info("LIMIT BUY %s qty=%.6f @ %.8f", symbol, qty, price)
        return await self._client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=f"{qty:.6f}",
            price=f"{price:.8f}",
        )

    async def place_oco_sell(
        self,
        symbol: str,
        qty: float,
        tp_price: float,
        sl_price: float,
        sl_limit_price: float,
    ) -> dict[str, Any]:
        await self._rate.acquire(2)
        logger.info(
            "OCO SELL %s qty=%.6f TP=%.8f SL=%.8f",
            symbol, qty, tp_price, sl_price,
        )
        return await self._client.create_oco_order(
            symbol=symbol,
            side=SIDE_SELL,
            quantity=f"{qty:.6f}",
            price=f"{tp_price:.8f}",
            stopPrice=f"{sl_price:.8f}",
            stopLimitPrice=f"{sl_limit_price:.8f}",
            stopLimitTimeInForce=TIME_IN_FORCE_GTC,
        )

    async def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        await self._rate.acquire(1)
        return await self._client.cancel_order(symbol=symbol, orderId=order_id)

    async def cancel_oco(self, symbol: str, order_list_id: int) -> dict[str, Any]:
        await self._rate.acquire(1)
        return await self._client.delete_oco_order(symbol=symbol, orderListId=order_list_id)

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        await self._rate.acquire(3)
        if symbol:
            return await self._client.get_open_orders(symbol=symbol)
        return await self._client.get_open_orders()

    async def get_symbol_price(self, symbol: str) -> float:
        await self._rate.acquire(1)
        ticker = await self._client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    async def place_market_sell(self, symbol: str, qty: float) -> dict[str, Any]:
        from binance.enums import ORDER_TYPE_MARKET
        await self._rate.acquire(1)
        logger.info("MARKET SELL %s qty=%.6f", symbol, qty)
        return await self._client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=f"{qty:.6f}",
        )

    async def place_market_buy(self, symbol: str, qty: float) -> dict[str, Any]:
        from binance.enums import ORDER_TYPE_MARKET
        await self._rate.acquire(1)
        logger.info("MARKET BUY %s qty=%.6f", symbol, qty)
        return await self._client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=f"{qty:.6f}",
        )
