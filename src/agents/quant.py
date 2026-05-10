from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from binance.enums import KLINE_INTERVAL_4HOUR

from src.agents.scout import Opportunity
from src.brain.opportunity_scorer import OpportunityScorer, ScoreBreakdown
from src.brain.trend_analyzer import TrendAnalyzer
from src.connectors.binance_rest import BinanceREST
from src.security.manipulation_detector import ManipulationDetector

logger = logging.getLogger(__name__)

BTC_SYMBOL = "BTCUSDT"


@dataclass
class QuantifiedOpportunity:
    opportunity: Opportunity
    breakdown: ScoreBreakdown
    final_score: float
    verdict: str
    metadata: dict = field(default_factory=dict)


class QuantAgent:
    """
    Receives raw Scout opportunities and enriches each with a full
    OpportunityScorer breakdown including regime, manipulation check,
    and BTC context.

    Returns a list sorted by final_score descending.
    """

    MIN_FINAL_SCORE = 55.0

    def __init__(self, rest: BinanceREST, cb_level_value: int = 0):
        self._rest       = rest
        self._scorer     = OpportunityScorer()
        self._analyst    = TrendAnalyzer()
        self._manip      = ManipulationDetector()
        self._cb_val     = cb_level_value
        self._btc_crash  = False

    def set_btc_crash(self, crashed: bool) -> None:
        self._btc_crash = crashed

    def set_cb_level(self, level: int) -> None:
        self._cb_val = level

    async def evaluate(self, opportunities: list[Opportunity]) -> list[QuantifiedOpportunity]:
        logger.info("Quant: evaluating %d opportunities", len(opportunities))

        # Refresh BTC context once
        btc_score = await self._get_btc_score()
        self._scorer.set_btc_score(btc_score)

        tasks = [self._eval_one(opp) for opp in opportunities]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        qualified: list[QuantifiedOpportunity] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Quant eval error: %s", r)
                continue
            if r and r.final_score >= self.MIN_FINAL_SCORE:
                qualified.append(r)

        qualified.sort(key=lambda q: q.final_score, reverse=True)
        logger.info("Quant: %d qualified opportunities", len(qualified))
        return qualified

    async def _eval_one(self, opp: Opportunity) -> QuantifiedOpportunity | None:
        try:
            df_4h = await self._rest.get_klines(opp.symbol, KLINE_INTERVAL_4HOUR, limit=300)
        except Exception as e:
            logger.debug("Quant: failed to fetch 4h for %s: %s", opp.symbol, e)
            return None

        signal_4h = opp.signals.get(KLINE_INTERVAL_4HOUR)
        if signal_4h is None:
            try:
                signal_4h = self._analyst.analyze(df_4h, opp.symbol, KLINE_INTERVAL_4HOUR)
            except Exception as e:
                logger.debug("Quant: analysis failed for %s: %s", opp.symbol, e)
                return None

        manip_flags      = self._manip.check(df_4h, opp.symbol)
        market_cap_rank: int | None = getattr(opp, "market_cap_rank", None)

        breakdown = self._scorer.score(
            signal_4h=signal_4h,
            df_4h=df_4h,
            manipulation_flags=manip_flags,
            cb_level_value=self._cb_val,
            market_cap_rank=market_cap_rank,
            btc_crash=self._btc_crash,
        )

        if breakdown.vetoed:
            return None

        return QuantifiedOpportunity(
            opportunity=opp,
            breakdown=breakdown,
            final_score=breakdown.final_score,
            verdict=breakdown.verdict,
            metadata={
                "manip_flags":    manip_flags,
                "btc_score":      self._scorer._btc_score,
                "btc_crash":      self._btc_crash,
                "is_prime":       breakdown.is_prime,
                "veto_reason":    breakdown.veto_reason,
            },
        )

    async def _get_btc_score(self) -> float:
        try:
            df = await self._rest.get_klines(BTC_SYMBOL, KLINE_INTERVAL_4HOUR, limit=300)
            sig = self._analyst.analyze(df, BTC_SYMBOL, KLINE_INTERVAL_4HOUR)
            return sig.score
        except Exception as e:
            logger.warning("Quant: BTC score fetch failed: %s", e)
            return 50.0
