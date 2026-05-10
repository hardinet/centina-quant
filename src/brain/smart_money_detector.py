"""
SmartMoneyDetector — détection des flux institutionnels et whale activity.
Sources : Binance large trades, WhaleAlert API (optionnel), on-chain heuristics.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

WHALE_ALERT_BASE = "https://api.whale-alert.io/v1/transactions"
LARGE_TRADE_USD  = 100_000   # seuil whale en USD
BLOCK_MINUTES    = 15


@dataclass
class SmartMoneyResult:
    whale_buys:        int   = 0     # nb gros achats détectés
    whale_sells:       int   = 0
    exchange_inflow:   float = 0.0   # flux vers exchanges ($M)
    exchange_outflow:  float = 0.0
    net_flow:          float = 0.0   # positif = bullish (sortie nette)
    large_buy_ratio:   float = 0.0   # ratio grands achats / total
    smart_signal:      str   = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL
    bonus:             int   = 0
    veto:              bool  = False
    details:           dict  = field(default_factory=dict)


class SmartMoneyDetector:
    """
    Détecte l'activité institutionnelle :
    - Gros trades Binance (>100k$) → whale_buys / whale_sells
    - WhaleAlert API (optionnel, WHALE_ALERT_API_KEY)
    - Heuristique on-chain : outflow exchange > inflow → accumulation → +20 pts
    - large_buy_ratio > 0.6 → +15 pts
    - Dump soudain whale → VETO
    """

    def __init__(self):
        self._wa_key = os.getenv("WHALE_ALERT_API_KEY", "")
        self._cache: dict[str, tuple[float, SmartMoneyResult]] = {}

    async def analyze(self, symbol: str, current_price: float = 0) -> SmartMoneyResult:
        coin = symbol.replace("USDT", "").replace("BUSD", "")
        now  = datetime.now(tz=timezone.utc).timestamp()

        if symbol in self._cache:
            ts, cached = self._cache[symbol]
            if now - ts < 180:   # TTL 3 min
                return cached

        try:
            tasks = [
                self._fetch_large_trades(symbol, current_price),
                self._fetch_whale_alert(coin) if self._wa_key else asyncio.coroutine(lambda: [])(),
            ]

            large_trades, whale_txs = await asyncio.gather(*tasks, return_exceptions=True)

            result = SmartMoneyResult()
            bonus  = 0

            # ── Large trades Binance ──────────────────────────────────
            if isinstance(large_trades, dict):
                result.whale_buys      = large_trades.get("buys", 0)
                result.whale_sells     = large_trades.get("sells", 0)
                result.large_buy_ratio = large_trades.get("buy_ratio", 0.5)

                total_whales = result.whale_buys + result.whale_sells
                if total_whales >= 3:
                    if result.large_buy_ratio >= 0.65:
                        bonus += 15
                        result.smart_signal = "BULLISH"
                        result.details["large_trades"] = (
                            f"{result.whale_buys}B/{result.whale_sells}S"
                        )
                    elif result.large_buy_ratio <= 0.35:
                        bonus -= 15
                        result.smart_signal = "BEARISH"
                        if result.whale_sells >= 5:
                            result.veto = True
                            result.details["whale_dump"] = f"{result.whale_sells} sells"

            # ── WhaleAlert on-chain ───────────────────────────────────
            if isinstance(whale_txs, list) and whale_txs:
                inflow  = sum(t.get("usd", 0) for t in whale_txs if t.get("dir") == "exchange")
                outflow = sum(t.get("usd", 0) for t in whale_txs if t.get("dir") == "wallet")
                result.exchange_inflow  = round(inflow / 1e6, 2)
                result.exchange_outflow = round(outflow / 1e6, 2)
                result.net_flow         = round((outflow - inflow) / 1e6, 2)

                if result.net_flow > 5:    # >5M sortie nette → accumulation
                    bonus += 20
                    result.smart_signal = "BULLISH"
                    result.details["net_outflow"] = f"+${result.net_flow:.1f}M accumulation"
                elif result.net_flow < -5:  # >5M entrée nette → pression vente
                    bonus -= 10
                    result.details["net_inflow"] = f"-${abs(result.net_flow):.1f}M selling"

            result.bonus = max(-20, min(20, bonus))
            self._cache[symbol] = (now, result)
            return result

        except Exception as e:
            logger.debug("SmartMoneyDetector error: %s", e)
            return SmartMoneyResult()

    # ── Fetchers ──────────────────────────────────────────────────────

    async def _fetch_large_trades(self, symbol: str, price: float) -> dict:
        """Agrège les gros trades récents depuis Binance aggTrades."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    "https://api.binance.com/api/v3/aggTrades",
                    params={"symbol": symbol, "limit": 500},
                    timeout=aiohttp.ClientTimeout(total=6),
                )
                if r.status != 200:
                    return {}
                trades = await r.json()

            ref_price = price or 1.0
            buys  = 0
            sells = 0
            for t in trades:
                qty = float(t.get("q", 0))
                usd = qty * ref_price
                if usd < LARGE_TRADE_USD:
                    continue
                if t.get("m"):    # maker = sell
                    sells += 1
                else:
                    buys  += 1

            total = buys + sells
            buy_ratio = buys / total if total > 0 else 0.5
            return {"buys": buys, "sells": sells, "buy_ratio": round(buy_ratio, 3)}

        except Exception as e:
            logger.debug("Large trades fetch error: %s", e)
            return {}

    async def _fetch_whale_alert(self, coin: str) -> list[dict]:
        """
        WhaleAlert API v1 — requiert WHALE_ALERT_API_KEY.
        Retourne transactions récentes pour la crypto.
        """
        try:
            import aiohttp
            import time

            since = int(time.time()) - BLOCK_MINUTES * 60
            params = {
                "api_key":    self._wa_key,
                "min_value":  500000,
                "limit":      20,
                "currency":   coin.lower(),
                "start":      since,
            }
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    WHALE_ALERT_BASE, params=params,
                    timeout=aiohttp.ClientTimeout(total=8),
                )
                if r.status != 200:
                    return []
                data = await r.json()

            txs    = data.get("transactions", [])
            result = []
            for tx in txs:
                to_   = tx.get("to",   {}).get("owner_type", "")
                from_ = tx.get("from", {}).get("owner_type", "")
                usd   = tx.get("amount_usd", 0)
                direction = "exchange" if to_   == "exchange" else (
                            "wallet"   if from_ == "exchange" else "unknown")
                result.append({"usd": usd, "dir": direction})
            return result

        except Exception as e:
            logger.debug("WhaleAlert fetch error: %s", e)
            return []
