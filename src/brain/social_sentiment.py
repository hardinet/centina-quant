"""
SocialSentiment — Reddit + CryptoPanic NLP sentiment analysis.
Sources gratuites : Reddit (PRAW ou pushshift), CryptoPanic (gratuit).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1/posts/"
REDDIT_SEARCH   = "https://www.reddit.com/r/CryptoCurrency/search.json"

BULLISH_WORDS = {
    "moon", "pump", "breakout", "bullish", "buy", "long", "rally", "ath",
    "accumulate", "green", "surge", "bullrun", "hodl", "uptrend", "gains",
    "outperform", "recovery", "support", "bounce", "strong",
}
BEARISH_WORDS = {
    "dump", "crash", "sell", "short", "bearish", "drop", "fall", "fear",
    "liquidation", "scam", "rug", "correction", "oversold", "panic",
    "bearrun", "breakdown", "resistance", "weak", "bubble",
}


@dataclass
class SentimentResult:
    reddit_score:     float = 50.0   # 0-100
    reddit_trend:     str   = "NEUTRAL"
    cryptopanic_score: float = 50.0
    fear_greed:       float = 50.0
    combined_score:   float = 50.0
    bonus:            int   = 0
    veto:             bool  = False
    details:          dict  = field(default_factory=dict)


def _simple_nlp_score(texts: list[str]) -> float:
    """Score NLP simple : ratio mots haussiers / (haussiers + baissiers)."""
    bull = 0
    bear = 0
    for text in texts:
        words = re.findall(r"\w+", text.lower())
        bull += sum(1 for w in words if w in BULLISH_WORDS)
        bear += sum(1 for w in words if w in BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return 50.0
    return round(bull / total * 100, 1)


class SocialSentiment:
    """
    Agrège le sentiment social depuis Reddit et CryptoPanic.
    Score > 65 → +10 pts, < 35 → -10 pts, < 20 → VETO (panic extreme)
    CryptoPanic API key optionnelle (CRYPTOPANIC_API_KEY).
    """

    CACHE_TTL = 300   # 5 min

    def __init__(self):
        self._cp_key  = os.getenv("CRYPTOPANIC_API_KEY", "")
        self._cache:  dict[str, tuple[float, SentimentResult]] = {}

    async def analyze(self, symbol: str = "BTC") -> SentimentResult:
        coin = symbol.replace("USDT", "").replace("BUSD", "")
        now  = datetime.now(tz=timezone.utc).timestamp()

        if coin in self._cache:
            ts, cached = self._cache[coin]
            if now - ts < self.CACHE_TTL:
                return cached

        try:
            reddit_task = self._fetch_reddit(coin)
            cp_task     = self._fetch_cryptopanic(coin)
            fg_task     = self._fetch_fear_greed()

            reddit_texts, cp_texts, fg = await asyncio.gather(
                reddit_task, cp_task, fg_task, return_exceptions=True,
            )

            result = SentimentResult()
            bonus  = 0

            # ── Reddit ────────────────────────────────────────────────
            if isinstance(reddit_texts, list) and reddit_texts:
                score = _simple_nlp_score(reddit_texts)
                result.reddit_score = score
                if score >= 65:
                    result.reddit_trend = "BULLISH"
                    bonus += 8
                    result.details["reddit"] = f"{score:.0f}% bull"
                elif score <= 35:
                    result.reddit_trend = "BEARISH"
                    bonus -= 8
                    result.details["reddit"] = f"{score:.0f}% bear"
                else:
                    result.reddit_trend = "NEUTRAL"

            # ── CryptoPanic ───────────────────────────────────────────
            if isinstance(cp_texts, list) and cp_texts:
                cp_score = _simple_nlp_score(cp_texts)
                result.cryptopanic_score = cp_score
                if cp_score >= 70:
                    bonus += 5
                elif cp_score <= 30:
                    bonus -= 5

            # ── Fear & Greed ──────────────────────────────────────────
            if isinstance(fg, float):
                result.fear_greed = fg
                if fg <= 20:
                    result.veto = True
                    bonus -= 15
                    result.details["fear_greed"] = f"EXTREME FEAR {fg:.0f}"
                elif fg <= 30:
                    bonus -= 8
                elif fg >= 80:
                    bonus -= 5
                    result.details["fear_greed"] = f"GREED {fg:.0f}"

            combined = (result.reddit_score * 0.5 + result.cryptopanic_score * 0.3 +
                        result.fear_greed * 0.2)
            result.combined_score = round(combined, 1)
            result.bonus          = max(-15, min(15, bonus))

            self._cache[coin] = (now, result)
            return result

        except Exception as e:
            logger.debug("SocialSentiment error: %s", e)
            return SentimentResult()

    # ── Fetchers ──────────────────────────────────────────────────────

    async def _fetch_reddit(self, coin: str) -> list[str]:
        try:
            import aiohttp
            params = {
                "q":    coin,
                "sort": "new",
                "limit": 25,
                "restrict_sr": "true",
                "t":    "day",
            }
            headers = {"User-Agent": "centina-quant/1.0"}
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    REDDIT_SEARCH, params=params, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=8),
                )
                if r.status != 200:
                    return []
                data = await r.json()
                posts = data.get("data", {}).get("children", [])
                texts = []
                for p in posts:
                    d = p.get("data", {})
                    texts.append(d.get("title", "") + " " + d.get("selftext", ""))
                return texts
        except Exception as e:
            logger.debug("Reddit fetch error: %s", e)
            return []

    async def _fetch_cryptopanic(self, coin: str) -> list[str]:
        try:
            import aiohttp
            params: dict = {
                "currencies": coin,
                "public":     "true",
                "filter":     "hot",
                "limit":      20,
            }
            if self._cp_key:
                params["auth_token"] = self._cp_key

            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    CRYPTOPANIC_BASE, params=params,
                    timeout=aiohttp.ClientTimeout(total=8),
                )
                if r.status != 200:
                    return []
                data = await r.json()
                results = data.get("results", [])
                return [item.get("title", "") for item in results]
        except Exception as e:
            logger.debug("CryptoPanic fetch error: %s", e)
            return []

    async def _fetch_fear_greed(self) -> float:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                r = await s.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=aiohttp.ClientTimeout(total=5),
                )
                if r.status != 200:
                    return 50.0
                data = await r.json()
                val  = data.get("data", [{}])[0].get("value", "50")
                return float(val)
        except Exception as e:
            logger.debug("Fear&Greed fetch error: %s", e)
            return 50.0
