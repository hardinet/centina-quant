from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from binance.enums import (
    KLINE_INTERVAL_15MINUTE,
    KLINE_INTERVAL_1HOUR,
    KLINE_INTERVAL_4HOUR,
    KLINE_INTERVAL_1DAY,
)

from src.brain.trend_analyzer import TrendAnalyzer, TrendSignal
from src.brain.regime_detector import Regime, RegimeDetector
from src.connectors.binance_rest import BinanceREST
from src.core.event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)

TIMEFRAMES = [
    KLINE_INTERVAL_15MINUTE,
    KLINE_INTERVAL_1HOUR,
    KLINE_INTERVAL_4HOUR,
    KLINE_INTERVAL_1DAY,
]

TF_WEIGHT = {
    KLINE_INTERVAL_15MINUTE: 0.10,
    KLINE_INTERVAL_1HOUR:    0.20,
    KLINE_INTERVAL_4HOUR:    0.35,
    KLINE_INTERVAL_1DAY:     0.35,
}

MIN_COMPOSITE_SCORE = 60.0
TOP_N               = 10
SCAN_INTERVAL       = 300   # seconds between full scans
TOP_SYMBOLS         = 50


@dataclass
class Opportunity:
    symbol:          str
    composite_score: float
    signals:         dict[str, TrendSignal] = field(default_factory=dict)
    best_atr:        float = 0.0
    best_price:      float = 0.0
    suggested_strategy: str = "B"    # A | B | C
    market_cap_rank:    int | None = None
    scanned_at:      datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def verdict(self) -> str:
        if self.composite_score >= 80:
            return "STRONG_BUY"
        if self.composite_score >= 65:
            return "BUY"
        return "NEUTRAL"


class Scout:
    """
    Sub-agent 1 — continuous market scanner.

    Every SCAN_INTERVAL seconds:
      • Fetches top-50 USDT pairs by volume
      • Analyzes 4 timeframes per pair
      • Emits OPPORTUNITY_DETECTED for each qualifying signal
      • Hints at best strategy (A: breakout, B: golden cross, C: news/momentum)
    """

    def __init__(self, rest: BinanceREST, bus: EventBus, top_n_symbols: int = TOP_SYMBOLS):
        self._rest     = rest
        self._bus      = bus
        self._top_n    = top_n_symbols
        self._analyzer = TrendAnalyzer()
        self._regime   = RegimeDetector()
        self._last_scan: list[Opportunity] = []

    # ------------------------------------------------------------------
    # Main loop (called by orchestrator)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("Scout: started (scan every %ds)", SCAN_INTERVAL)
        while True:
            try:
                opps = await self.scan()
                self._last_scan = opps
                for opp in opps:
                    await self._bus.emit(Event(
                        EventType.OPPORTUNITY_DETECTED,
                        self._opp_to_payload(opp),
                        source="scout",
                    ))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Scout scan error: %s", e)
            await asyncio.sleep(SCAN_INTERVAL)

    def get_last_scan(self) -> list[Opportunity]:
        return self._last_scan

    # ------------------------------------------------------------------
    # Core scan (also called directly by main.py interactive loop)
    # ------------------------------------------------------------------

    async def scan(self) -> list[Opportunity]:
        logger.info("Scout: fetching top %d USDT pairs", self._top_n)
        symbols = await self._rest.get_top_usdt_pairs(self._top_n)
        logger.info("Scout: scanning %d symbols across %d timeframes", len(symbols), len(TIMEFRAMES))

        tasks   = [self._analyze_symbol(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        opportunities: list[Opportunity] = []
        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug("Scout skip %s: %s", sym, result)
                continue
            if result and result.composite_score >= MIN_COMPOSITE_SCORE:
                opportunities.append(result)

        opportunities.sort(key=lambda o: o.composite_score, reverse=True)
        top = opportunities[:TOP_N]
        logger.info("Scout: %d opportunities found (top %d returned)", len(opportunities), len(top))
        return top

    # ------------------------------------------------------------------
    # Per-symbol analysis
    # ------------------------------------------------------------------

    async def _analyze_symbol(self, symbol: str) -> Opportunity | None:
        signals: dict[str, TrendSignal] = {}

        for tf in TIMEFRAMES:
            try:
                df     = await self._rest.get_klines(symbol, tf, limit=300)
                signal = self._analyzer.analyze(df, symbol, tf)
                signals[tf] = signal
            except Exception as e:
                logger.debug("%s/%s analysis failed: %s", symbol, tf, e)

        if not signals:
            return None

        composite = sum(
            signals[tf].score * TF_WEIGHT[tf]
            for tf in signals
        ) / sum(TF_WEIGHT[tf] for tf in signals)

        ref_tf = KLINE_INTERVAL_4HOUR if KLINE_INTERVAL_4HOUR in signals else next(iter(signals))
        ref    = signals[ref_tf]

        # Detect best strategy hint
        try:
            df4h = await self._rest.get_klines(symbol, KLINE_INTERVAL_4HOUR, limit=200)
            strategy = self._hint_strategy(df4h, ref)
        except Exception:
            strategy = "B"

        return Opportunity(
            symbol=symbol,
            composite_score=round(composite, 2),
            signals=signals,
            best_atr=ref.atr,
            best_price=ref.current_price,
            suggested_strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Strategy hint (A / B / C)
    # ------------------------------------------------------------------

    def _hint_strategy(self, df, sig: TrendSignal) -> str:
        """
        A — Breakout Momentum : new 48h high + volume spike ×2.5
        B — Golden Cross       : EMA7 just crossed above EMA25
        C — News/Momentum      : RSI 55-70 + volume expansion 3 candles
        Default B if none match clearly.
        """
        import pandas_ta as ta
        import pandas as pd

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        volume = df["volume"].astype(float)

        try:
            ema7  = ta.ema(close, length=7)
            ema25 = ta.ema(close, length=25)

            # A: Breakout
            if len(high) > 50:
                is_breakout = close.iloc[-1] >= high.iloc[-48:-1].max()
                vol_spike   = volume.iloc[-1] > volume.iloc[-20:-1].mean() * 2.5
                if is_breakout and vol_spike:
                    return "A"

            # B: Golden Cross (EMA7 crossed above EMA25 in last 3 bars)
            if len(ema7) > 5:
                crossed = (
                    ema7.iloc[-1] > ema25.iloc[-1]
                    and (ema7.iloc[-3] < ema25.iloc[-3] or ema7.iloc[-2] < ema25.iloc[-2])
                )
                if crossed:
                    return "B"

            # C: RSI momentum + volume expansion
            if 55 <= sig.rsi <= 70:
                if (len(volume) >= 4
                        and volume.iloc[-1] > volume.iloc[-2] > volume.iloc[-3]):
                    return "C"

        except Exception:
            pass

        return "B"

    # ------------------------------------------------------------------
    # Payload helper
    # ------------------------------------------------------------------

    def _opp_to_payload(self, opp: Opportunity) -> dict:
        sig4h = opp.signals.get(KLINE_INTERVAL_4HOUR)
        return {
            "symbol":             opp.symbol,
            "composite_score":    opp.composite_score,
            "verdict":            opp.verdict,
            "best_price":         opp.best_price,
            "best_atr":           opp.best_atr,
            "suggested_strategy": opp.suggested_strategy,
            "rsi":                sig4h.rsi if sig4h else 50.0,
            "ema7":               sig4h.ema7 if sig4h else 0.0,
            "ema25":              sig4h.ema25 if sig4h else 0.0,
            "ema99":              sig4h.ema99 if sig4h else 0.0,
            "ema200":             sig4h.ema200 if sig4h else 0.0,
            "scanned_at":         opp.scanned_at.isoformat(),
        }
