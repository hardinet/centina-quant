"""
OrderflowAnalyzer — analyse du carnet d'ordres Binance.
Détecte : imbalance, absorption, stop hunt, iceberg orders.
Inspiré Bookmap. Données via Binance REST depth.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

DEPTH_LEVELS = 20     # niveaux du carnet à analyser
PRICE_RANGE  = 0.02   # ±2% du prix courant


@dataclass
class OrderflowResult:
    bid_ask_imbalance:  float = 0.0    # >1 = bids dominent, <1 = asks dominent
    absorption_detected: bool = False
    stop_hunt_detected:  bool = False
    iceberg_detected:    bool = False
    buy_pressure:        float = 0.0   # 0-100
    bonus:               int   = 0
    veto:                bool  = False
    veto_reason:         str   = ""
    details:             dict  = field(default_factory=dict)


class OrderflowAnalyzer:
    """
    Analyse le carnet d'ordres via Binance REST.
    Imbalance > 1.5 → acheteurs dominent → +20 pts
    Absorption → +15 pts
    Stop hunt → VETO
    Iceberg → suit la direction
    """

    async def analyze(self, symbol: str, current_price: float = 0) -> OrderflowResult:
        try:
            depth = await self._fetch_depth(symbol)
            if not depth:
                return OrderflowResult()
            return self._compute(depth, current_price)
        except Exception as e:
            logger.debug("OrderflowAnalyzer error %s: %s", symbol, e)
            return OrderflowResult()

    async def _fetch_depth(self, symbol: str) -> dict | None:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    "https://api.binance.com/api/v3/depth",
                    params={"symbol": symbol, "limit": DEPTH_LEVELS},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
                if r.status != 200:
                    return None
                return await r.json()
        except Exception as e:
            logger.debug("depth fetch error: %s", e)
            return None

    def _compute(self, depth: dict, current_price: float) -> OrderflowResult:
        result = OrderflowResult()
        bonus  = 0

        bids = depth.get("bids", [])  # [[price, qty], ...]
        asks = depth.get("asks", [])

        if not bids or not asks:
            return result

        # ── Imbalance ─────────────────────────────────────────────────
        bid_volume = sum(float(q) * float(p) for p, q in bids[:DEPTH_LEVELS])
        ask_volume = sum(float(q) * float(p) for p, q in asks[:DEPTH_LEVELS])
        total_vol  = bid_volume + ask_volume + 1e-10
        imbalance  = bid_volume / (ask_volume + 1e-10)
        result.bid_ask_imbalance = round(imbalance, 3)
        result.buy_pressure      = round(bid_volume / total_vol * 100, 1)

        if imbalance >= 2.0:          # acheteurs dominent fortement
            bonus += 20
            result.details["imbalance"] = f"{imbalance:.2f}x"
        elif imbalance >= 1.5:
            bonus += 12
            result.details["imbalance"] = f"{imbalance:.2f}x"
        elif imbalance <= 0.5:        # vendeurs dominent
            bonus -= 10

        # ── Absorption ────────────────────────────────────────────────
        # Gros mur vendeur dans les 3 premiers niveaux qui se réduit
        if asks and float(asks[0][1]) > sum(float(q) for _, q in asks[1:4]) * 2:
            result.absorption_detected = True
            bonus += 15
            result.details["absorption"] = True

        # ── Stop Hunt ────────────────────────────────────────────────
        # Mouvement récent rapide hors range + retour immédiat
        # Approximation via spread extrême dans les bids/asks
        if current_price > 0 and bids:
            best_bid = float(bids[0][0])
            spread   = (current_price - best_bid) / current_price * 100
            if spread > 0.5:    # spread anormalement large = possible stop hunt
                result.stop_hunt_detected = True
                result.veto        = True
                result.veto_reason = "STOP_HUNT"
                bonus -= 25
                result.details["stop_hunt_spread"] = round(spread, 3)

        # ── Iceberg Orders ────────────────────────────────────────────
        # Ordres répétitifs de même taille dans le carnet
        bid_qtys = [float(q) for _, q in bids[:10]]
        if len(bid_qtys) >= 5:
            std_q = np.std(bid_qtys)
            mean_q = np.mean(bid_qtys)
            if std_q / (mean_q + 1e-10) < 0.1 and mean_q > 0:
                result.iceberg_detected = True
                # Iceberg côté achat → haussier
                bonus += 10
                result.details["iceberg_buy"] = True

        result.bonus = max(-30, min(35, bonus))
        return result
