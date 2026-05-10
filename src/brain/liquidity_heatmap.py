"""
LiquidityHeatmap — zones de liquidation Binance Futures.
Identifie les murs de liquidation au-dessus/en-dessous du prix courant.
Utilise Coinglass API (clé optionnelle) ou Binance open interest comme proxy.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"


@dataclass
class LiquidityResult:
    nearest_liq_above: float = 0.0   # prix du premier mur de liq au-dessus
    nearest_liq_below: float = 0.0   # prix du premier mur de liq en-dessous
    liq_above_size:    float = 0.0   # taille estimée ($M)
    liq_below_size:    float = 0.0
    above_distance_pct: float = 0.0  # % entre prix courant et mur au-dessus
    below_distance_pct: float = 0.0
    bonus:             int   = 0
    veto:              bool  = False
    details:           dict  = field(default_factory=dict)


class LiquidityHeatmap:
    """
    Si un mur de liquidations short est juste EN-DESSOUS → +15 pts (protection, prix poussé ↑)
    Si un mur de liquidations long est juste AU-DESSUS → -10 pts (résistance)
    Données : Coinglass API (COINGLASS_API_KEY) ou proxy Binance funding/OI.
    """

    def __init__(self):
        self._api_key = os.getenv("COINGLASS_API_KEY", "")

    async def analyze(self, symbol: str, current_price: float) -> LiquidityResult:
        if not current_price:
            return LiquidityResult()
        try:
            if self._api_key:
                result = await self._fetch_coinglass(symbol, current_price)
            else:
                result = await self._estimate_from_binance(symbol, current_price)
            return result
        except Exception as e:
            logger.debug("LiquidityHeatmap error %s: %s", symbol, e)
            return LiquidityResult()

    async def _fetch_coinglass(self, symbol: str, price: float) -> LiquidityResult:
        coin = symbol.replace("USDT", "")
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    f"{COINGLASS_BASE}/liquidation_map",
                    params={"symbol": coin, "exchangeName": "Binance"},
                    headers={"coinglassSecret": self._api_key},
                    timeout=aiohttp.ClientTimeout(total=6),
                )
                data = await r.json()
                zones = data.get("data", {}).get("liqLevels", [])
                if not zones:
                    return LiquidityResult()

                above = [(z["price"], z["size"]) for z in zones if z["price"] > price * 1.001]
                below = [(z["price"], z["size"]) for z in zones if z["price"] < price * 0.999]

                return self._build_result(price, above, below)
        except Exception as e:
            logger.debug("Coinglass fetch error: %s", e)
            return await self._estimate_from_binance(symbol, price)

    async def _estimate_from_binance(self, symbol: str, price: float) -> LiquidityResult:
        """
        Proxy gratuit : utilise les données de funding rate et open interest Binance Futures.
        Les niveaux ronds (±2%, ±5%, ±10%) sont souvent des zones de liquidation.
        """
        try:
            import aiohttp
            # Binance Futures OI history (gratuit)
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    "https://fapi.binance.com/futures/data/openInterestHist",
                    params={"symbol": symbol, "period": "1h", "limit": 10},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
                if r.status != 200:
                    return self._synthetic_estimate(price)
                oi_data = await r.json()

            if not oi_data:
                return self._synthetic_estimate(price)

            # Tendance OI : si OI monte + prix monte → longs s'accumulent
            oi_vals = [float(d.get("sumOpenInterest", 0)) for d in oi_data]
            oi_trend = (oi_vals[-1] - oi_vals[0]) / (oi_vals[0] + 1e-10)

            # Estimation heuristique des murs
            # Typiquement ±2.5% pour les liquidations tier-1
            above_pct = 0.025
            below_pct = 0.025

            if oi_trend > 0.05:   # accumulation longs → risque squeeze shorts en-dessous
                below_pct = 0.015   # mur plus proche

            liq_above = price * (1 + above_pct)
            liq_below = price * (1 - below_pct)

            above_size = abs(oi_trend) * 50   # estimation fictive en $M
            below_size = 100.0

            above = [(liq_above, above_size)]
            below = [(liq_below, below_size)]
            return self._build_result(price, above, below)

        except Exception as e:
            logger.debug("Binance OI estimate error: %s", e)
            return self._synthetic_estimate(price)

    def _synthetic_estimate(self, price: float) -> LiquidityResult:
        """Estimation synthétique basée sur les niveaux psychologiques."""
        # Niveaux ronds = zones de liquidation classiques
        round_levels = self._round_levels(price)
        above = [(l, 50.0) for l in round_levels if l > price * 1.005][:3]
        below = [(l, 50.0) for l in round_levels if l < price * 0.995][:3]
        return self._build_result(price, above, below)

    @staticmethod
    def _round_levels(price: float) -> list[float]:
        """Retourne les niveaux ronds autour du prix."""
        magnitude = 10 ** (len(str(int(price))) - 2)
        base = round(price / magnitude) * magnitude
        return [base + i * magnitude for i in range(-5, 6)]

    @staticmethod
    def _build_result(price: float, above: list, below: list) -> LiquidityResult:
        result = LiquidityResult()
        bonus  = 0

        if above:
            above.sort(key=lambda x: x[0])
            liq_a, size_a = above[0]
            dist_above = (liq_a - price) / price * 100
            result.nearest_liq_above  = round(liq_a, 6)
            result.liq_above_size     = round(size_a, 2)
            result.above_distance_pct = round(dist_above, 2)
            # Mur de liq long proche au-dessus → résistance
            if 1.0 < dist_above < 3.0 and size_a > 30:
                bonus -= 10
                result.details["liq_resistance"] = f"{dist_above:.1f}% above"

        if below:
            below.sort(key=lambda x: -x[0])
            liq_b, size_b = below[0]
            dist_below = (price - liq_b) / price * 100
            result.nearest_liq_below  = round(liq_b, 6)
            result.liq_below_size     = round(size_b, 2)
            result.below_distance_pct = round(dist_below, 2)
            # Mur de liq short proche en-dessous → prix poussé vers le haut
            if 0.5 < dist_below < 2.5 and size_b > 30:
                bonus += 15
                result.details["liq_support"] = f"{dist_below:.1f}% below"

        result.bonus = max(-15, min(20, bonus))
        return result
