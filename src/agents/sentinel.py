from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from src.brain.news_momentum_correlator import NewsMomentumCorrelator
from src.connectors.binance_rest import BinanceREST
from src.connectors.n8n_webhook import get_next_signal
from src.core.event_bus import Event, EventBus, EventType
from src.data.web_scraper import CryptoWebScraper
from src.notifications.apprise_notifier import AppriseNotifier, NotifLevel

logger = logging.getLogger(__name__)

BTC_DANGER_PCT     = 2.0
BTC_CHECK_INTERVAL = 60   # seconds
SCRAPE_INTERVAL    = 300  # scraping every 5 min


class Sentinel:
    """
    Sub-agent 5: External intelligence & BTC monitoring.

    - Receives n8n webhook signals (news, whale alerts)
    - Scores each news event via LLM (if OpenAI key available)
    - Monitors BTC for systemic danger
    - Emits: NewsAlert, WhaleAlert, BTCDanger, AltSeason
    """

    def __init__(self, rest: BinanceREST, bus: EventBus, notif: AppriseNotifier):
        self._rest       = rest
        self._bus        = bus
        self._notif      = notif
        self._correlator = NewsMomentumCorrelator()
        self._scraper    = CryptoWebScraper()
        self._btc_prices: list[float] = []
        self._btc_last_check = datetime.now(timezone.utc) - timedelta(seconds=100)
        self._web_cache: dict[str, dict] = {}  # symbol → last scrape result

    async def run(self) -> None:
        logger.info("Sentinel: started")
        await asyncio.gather(
            self._webhook_loop(),
            self._btc_monitor_loop(),
            self._scrape_loop(),
        )

    async def _scrape_loop(self) -> None:
        """Scrape les sources web toutes les 5 min pour enrichir les signaux."""
        while True:
            try:
                # Fear & Greed global — mis en cache
                fng = await self._scraper.get_fear_greed()
                self._web_cache["__fng__"] = {"fear_greed": fng}
                logger.debug("Sentinel: Fear&Greed=%d", fng)
            except Exception as e:
                logger.debug("Sentinel scrape_loop error: %s", e)
            await asyncio.sleep(SCRAPE_INTERVAL)

    async def get_web_bonus(self, symbol: str) -> dict:
        """
        Retourne les bonus/malus web pour un symbole (appelable par l'Analyst).
        trending_bonus, reddit_bonus, fng_malus.
        """
        try:
            data = await self._scraper.collect_all(symbol)
            self._web_cache[symbol] = data
            return data
        except Exception as e:
            logger.debug("Sentinel web_bonus error %s: %s", symbol, e)
            return {"total_bonus": 0, "trending_bonus": 0, "reddit_bonus": 0, "fng_malus": 0}

    async def _webhook_loop(self) -> None:
        while True:
            signal = await get_next_signal(timeout=2.0)
            if signal:
                await self._process_signal(signal)

    async def _btc_monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(BTC_CHECK_INTERVAL)
            await self._check_btc()

    async def _process_signal(self, signal: dict) -> None:
        sig_type = signal.get("type", "news")

        if sig_type == "news":
            score = await self._score_news_llm(signal)
            urgency = "critical" if score >= 80 else ("high" if score >= 60 else "normal")

            # Record for correlation learning
            self._correlator.record_news({**signal, "score": score})
            pump_prob = self._correlator.predict_pump_probability(signal)

            event_payload = {
                "symbol":        signal.get("symbol"),
                "score_bullish": score,
                "urgency":       urgency,
                "pump_prob":     pump_prob,
                "headline":      signal.get("headline", ""),
                "source":        signal.get("source", ""),
                "sentiment":     signal.get("sentiment", 0.0),
            }
            await self._bus.emit(Event(EventType.NEWS_RECEIVED, event_payload, source="sentinel"))

            if urgency == "critical":
                self._notif.notify(
                    f"NEWS CRITICAL: {signal.get('headline','')[:80]}",
                    level=NotifLevel.CRITICAL,
                )

        elif sig_type == "whale":
            await self._bus.emit(Event(EventType.WHALE_ALERT, signal, source="sentinel"))
            logger.info("Sentinel: whale alert — %s", signal)

    async def _check_btc(self) -> None:
        try:
            df = await self._rest.get_klines("BTCUSDT", "1h", limit=3)
            close = df["close"].astype(float)
            latest = close.iloc[-1]
            prev   = close.iloc[-2]
            change_pct = (latest - prev) / prev * 100

            self._btc_prices.append(latest)
            if len(self._btc_prices) > 60:
                self._btc_prices = self._btc_prices[-60:]

            if change_pct <= -BTC_DANGER_PCT:
                logger.warning("Sentinel: BTC DANGER — %.2f%% drop in 1h", change_pct)
                await self._bus.emit(Event(
                    EventType.BTC_DANGER,
                    {"change_pct": change_pct, "price": latest},
                    source="sentinel",
                ))
                self._notif.notify(
                    f"BTC chute {change_pct:.2f}% en 1h — PAUSE TRADING",
                    level=NotifLevel.CRITICAL,
                )
        except Exception as e:
            logger.debug("Sentinel: BTC check error — %s", e)

    async def _score_news_llm(self, signal: dict) -> float:
        """Score news 0-100. Uses OpenAI if key present, else heuristic."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        headline = signal.get("headline", "")
        sentiment = signal.get("sentiment", 0.0)

        if api_key and headline:
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=api_key)
                resp = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Rate the bullish impact of this crypto news on a scale 0-100. "
                            f"Respond with only a number.\n\nNews: {headline}"
                        ),
                    }],
                    max_tokens=5,
                )
                score_str = resp.choices[0].message.content.strip()
                return float(score_str)
            except Exception as e:
                logger.debug("LLM scoring failed: %s", e)

        # Heuristic fallback
        return max(0.0, min(100.0, 50.0 + sentiment * 50))
