from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path("data/web_knowledge.db")

SEARCH_QUERIES = [
    "crypto trading strategy 2026 high win rate",
    "best altcoin momentum indicator parameters",
    "bitcoin scalping EMA RSI settings",
    "crypto breakout strategy backtest results",
    "binance trading bot open source strategy",
]


class WebKnowledgeSeeker:
    """
    Daily web search for trading strategies.
    Discovers, stores, and auto-tests promising strategies.
    Integrates those with WR > 60% AND avg gain > 4% as Strategy F/G/H.
    """

    MIN_WIN_RATE   = 0.60
    MIN_AVG_GAIN   = 4.0
    TEST_DAYS      = 30

    def __init__(self):
        self._init_db()

    async def daily_search(self) -> list[dict]:
        """Main daily search coroutine. Returns list of discovered strategies."""
        logger.info("WebKnowledgeSeeker: starting daily search")
        discovered = []
        for query in SEARCH_QUERIES:
            results = await self._search_web(query)
            for r in results:
                strategy = self._extract_strategy(r)
                if strategy:
                    self._store_strategy(strategy)
                    discovered.append(strategy)
        logger.info("WebKnowledgeSeeker: discovered %d strategies", len(discovered))
        return discovered

    async def weekly_test(self, rest) -> list[dict]:
        """Test 3 best untested strategies on 30d historical data."""
        untested = self._get_untested(n=3)
        results = []
        for strat in untested:
            result = await self._test_strategy(strat, rest)
            if result:
                results.append(result)
                if result.get("win_rate", 0) >= self.MIN_WIN_RATE and \
                   result.get("avg_gain", 0) >= self.MIN_AVG_GAIN:
                    self._mark_integrated(strat["id"])
                    logger.info(
                        "WebKnowledgeSeeker: strategy '%s' INTEGRATED (WR=%.1f%%, gain=%.1f%%)",
                        strat["name"], result["win_rate"] * 100, result["avg_gain"],
                    )
        return results

    # ── Internal ─────────────────────────────────────────────────────

    async def _search_web(self, query: str) -> list[dict]:
        """Fetch search results. Uses aiohttp + BeautifulSoup."""
        try:
            import aiohttp
            from bs4 import BeautifulSoup

            headers = {"User-Agent": "Mozilla/5.0 CentinaResearch/5.0"}
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num=5"

            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            results = []
            for item in soup.select(".tF2Cxc")[:5]:
                title = item.select_one("h3")
                snippet = item.select_one(".VwiC3b")
                link = item.select_one("a")
                if title and snippet:
                    results.append({
                        "title":   title.get_text(),
                        "snippet": snippet.get_text(),
                        "url":     link["href"] if link else "",
                    })
            return results
        except Exception as e:
            logger.debug("Web search failed for '%s': %s", query, e)
            return []

    def _extract_strategy(self, result: dict) -> dict | None:
        """Extract strategy parameters from search result text."""
        text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
        params = {}

        # EMA detection
        import re
        ema_matches = re.findall(r"ema[-\s]?(\d+)", text)
        if ema_matches:
            params["ema_periods"] = [int(m) for m in ema_matches[:3]]

        # RSI detection
        rsi_matches = re.findall(r"rsi[-\s]?(\d+)", text)
        if rsi_matches:
            params["rsi_period"] = int(rsi_matches[0])

        # Win rate detection
        wr_matches = re.findall(r"(\d{2,3})%?\s*win\s*rate", text)
        if wr_matches:
            params["claimed_win_rate"] = int(wr_matches[0]) / 100

        if not params:
            return None

        return {
            "name":             result.get("title", "unknown")[:100],
            "source":           result.get("url", ""),
            "params":           params,
            "credibility_score": self._credibility(result),
            "date_discovered":  datetime.utcnow().isoformat(),
            "tested_yet":       False,
        }

    def _credibility(self, result: dict) -> float:
        text = result.get("snippet", "").lower()
        score = 0.5
        if "backtest" in text:    score += 0.1
        if "live" in text:        score += 0.1
        if "years" in text:       score += 0.1
        if "claimed" in text:     score -= 0.1
        if "guaranteed" in text:  score -= 0.2
        return round(max(0.0, min(1.0, score)), 2)

    async def _test_strategy(self, strat: dict, rest) -> dict | None:
        try:
            from src.learning.backtest_engine import BacktestEngine
            engine = BacktestEngine()
            params = strat.get("params", {})

            ema_fast = params.get("ema_periods", [7])[0]
            ema_slow = params.get("ema_periods", [7, 25])[-1] if len(params.get("ema_periods", [])) > 1 else 25

            df = await rest.get_klines("BTCUSDT", "4h", limit=500)
            result = engine.run(df, "BTCUSDT", "4h", ema_fast=ema_fast, ema_slow=ema_slow)

            win_rate = result.win_rate
            avg_gain = result.total_return_pct / max(result.total_trades, 1)

            with self._conn() as conn:
                conn.execute(
                    "UPDATE web_knowledge SET tested_yet=1, win_rate_if_tested=? WHERE id=?",
                    (win_rate, strat["id"]),
                )
            return {"win_rate": win_rate, "avg_gain": avg_gain, "total_trades": result.total_trades}
        except Exception as e:
            logger.error("Strategy test failed: %s", e)
            return None

    def _store_strategy(self, strat: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO web_knowledge
                   (name, source, params_json, credibility_score, date_discovered)
                   VALUES (?,?,?,?,?)""",
                (strat["name"], strat.get("source", ""), json.dumps(strat.get("params", {})),
                 strat.get("credibility_score", 0.5), strat.get("date_discovered", "")),
            )

    def _get_untested(self, n: int = 3) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, params_json FROM web_knowledge WHERE tested_yet=0 ORDER BY credibility_score DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [{"id": r[0], "name": r[1], "params": json.loads(r[2] or "{}")} for r in rows]

    def _mark_integrated(self, strategy_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE web_knowledge SET integrated=1 WHERE id=?", (strategy_id,))

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS web_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE, source TEXT, params_json TEXT,
                    credibility_score REAL, date_discovered TEXT,
                    tested_yet INTEGER DEFAULT 0,
                    win_rate_if_tested REAL, integrated INTEGER DEFAULT 0
                )
            """)

    def _conn(self):
        import contextlib
        @contextlib.contextmanager
        def _ctx():
            c = sqlite3.connect(DB_PATH)
            try:
                yield c
                c.commit()
            finally:
                c.close()
        return _ctx()
