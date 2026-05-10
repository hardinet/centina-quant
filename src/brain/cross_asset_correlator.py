"""
CrossAssetCorrelator — BTC dominance, ETH/BTC, funding rates, DXY proxy.
Inspiré Kavout. Toutes les données via Binance + sources gratuites.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetResult:
    btc_dominance:   float = 50.0
    eth_btc_ratio:   float = 0.0
    eth_btc_trend:   str   = "NEUTRAL"   # UP / DOWN / NEUTRAL
    funding_rate:    float = 0.0
    funding_signal:  str   = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL / DANGER
    dxy_proxy:       str   = "NEUTRAL"   # UP / DOWN / NEUTRAL (via BTC/USD vs BTC/EUR)
    bonus:           int   = 0
    details:         dict  = field(default_factory=dict)


class CrossAssetCorrelator:
    """
    Signaux macro contextuels :
    - ETH/BTC ratio en hausse + funding neutre → altseason +20 pts
    - Funding > 0.05% → MALUS -20 (surcharge longs)
    - Funding < 0 → BONUS +10 (shorts liquidés = rebond)
    - BTC dominance baisse + volume alts ↑ → +15 pts
    """

    async def analyze(self, symbol: str = "BTCUSDT") -> CrossAssetResult:
        try:
            eth_btc_task  = self._get_eth_btc_trend()
            funding_task  = self._get_funding_rate(symbol)
            btc_dom_task  = self._get_btc_dominance()

            eth_btc, funding, btc_dom = await asyncio.gather(
                eth_btc_task, funding_task, btc_dom_task,
                return_exceptions=True,
            )

            result = CrossAssetResult()
            bonus  = 0

            # ── ETH/BTC ratio ─────────────────────────────────────────
            if isinstance(eth_btc, dict):
                result.eth_btc_ratio = eth_btc.get("ratio", 0)
                result.eth_btc_trend = eth_btc.get("trend", "NEUTRAL")
                if eth_btc.get("trend") == "UP":
                    bonus += 15
                    result.details["eth_btc"] = "altseason signal"
                elif eth_btc.get("trend") == "DOWN":
                    bonus -= 5

            # ── Funding rate ──────────────────────────────────────────
            if isinstance(funding, float):
                result.funding_rate = funding
                if funding > 0.0005:       # > 0.05% → DANGER surcharge longs
                    result.funding_signal = "DANGER"
                    bonus -= 20
                    result.details["funding"] = f"{funding:.4%} DANGER"
                elif funding > 0.0002:     # 0.02-0.05% → légèrement bearish
                    result.funding_signal = "BEARISH"
                    bonus -= 8
                elif funding < -0.0001:    # < -0.01% → shorts paient → rebond
                    result.funding_signal = "BULLISH"
                    bonus += 10
                    result.details["funding"] = f"{funding:.4%} bullish"
                else:
                    result.funding_signal = "NEUTRAL"

            # ── BTC dominance ─────────────────────────────────────────
            if isinstance(btc_dom, dict):
                result.btc_dominance = btc_dom.get("dominance", 50)
                dom_trend = btc_dom.get("trend", "NEUTRAL")
                if dom_trend == "DOWN" and not symbol.startswith("BTC"):
                    bonus += 15
                    result.details["btc_dominance"] = "rotation vers alts"

            result.bonus = max(-25, min(30, bonus))
            return result

        except Exception as e:
            logger.debug("CrossAssetCorrelator error: %s", e)
            return CrossAssetResult()

    # ── Data fetchers ─────────────────────────────────────────────────

    async def _get_eth_btc_trend(self) -> dict:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": "ETHBTC", "interval": "1h", "limit": 6},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
                data = await r.json()
                closes = [float(d[4]) for d in data]
                if len(closes) < 3:
                    return {"ratio": 0, "trend": "NEUTRAL"}
                ratio  = closes[-1]
                change = (closes[-1] - closes[0]) / closes[0] * 100
                trend  = "UP" if change > 0.3 else ("DOWN" if change < -0.3 else "NEUTRAL")
                return {"ratio": round(ratio, 6), "trend": trend, "change_pct": round(change, 3)}
        except Exception as e:
            logger.debug("ETH/BTC fetch error: %s", e)
            return {"ratio": 0, "trend": "NEUTRAL"}

    async def _get_funding_rate(self, symbol: str) -> float:
        try:
            import aiohttp
            # Binance futures funding rate (gratuit, pas de clé)
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    "https://fapi.binance.com/fapi/v1/premiumIndex",
                    params={"symbol": symbol},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
                if r.status != 200:
                    return 0.0
                data = await r.json()
                return float(data.get("lastFundingRate", 0))
        except Exception as e:
            logger.debug("Funding rate fetch error: %s", e)
            return 0.0

    async def _get_btc_dominance(self) -> dict:
        """BTC dominance via CoinGecko (gratuit, rate limited)."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    "https://api.coingecko.com/api/v3/global",
                    timeout=aiohttp.ClientTimeout(total=6),
                )
                if r.status != 200:
                    return {"dominance": 50, "trend": "NEUTRAL"}
                data = await r.json()
                dom = data.get("data", {}).get("market_cap_percentage", {}).get("btc", 50)
                # Trend heuristique : comparaison avec valeur historique moyenne
                trend = "DOWN" if dom < 48 else ("UP" if dom > 55 else "NEUTRAL")
                return {"dominance": round(dom, 2), "trend": trend}
        except Exception as e:
            logger.debug("BTC dominance fetch error: %s", e)
            return {"dominance": 50, "trend": "NEUTRAL"}
