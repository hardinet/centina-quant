from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class NotifLevel(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "failure"   # Apprise uses "failure" for critical


class AppriseNotifier:
    """
    Multi-channel notifier via Apprise.
    Supports Telegram, Discord, SMS (Twilio), email, and any Apprise URL.

    Channels are loaded from env vars:
      NOTIFY_URL          – primary channel (required)
      NOTIFY_URL_2        – secondary (optional)
      NOTIFY_URL_3        – tertiary  (optional)
    """

    _PREFIXES = {
        NotifLevel.INFO:     "ℹ️ CENTINA",
        NotifLevel.WARNING:  "⚠️ CENTINA",
        NotifLevel.CRITICAL: "🚨 CENTINA ALERT",
    }

    def __init__(self, urls: list[str] | None = None):
        self._urls = urls or self._load_urls()
        self._apprise = self._build_apprise()

    # ── Public API ───────────────────────────────────────────────────

    def notify(self, message: str, level: NotifLevel = NotifLevel.INFO, title: str | None = None) -> bool:
        if not self._apprise:
            logger.warning("Apprise not configured – notification skipped")
            return False
        tag = title or self._PREFIXES[level]
        try:
            result = self._apprise.notify(body=message, title=tag, notify_type=level.value)
            if not result:
                logger.warning("Apprise notify returned False for level=%s", level)
            return bool(result)
        except Exception as e:
            logger.error("Notification failed: %s", e)
            return False

    async def async_notify(self, message: str, level: NotifLevel = NotifLevel.INFO, title: str | None = None) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.notify, message, level, title)

    # ── Convenience shortcuts ─────────────────────────────────────────

    def trade_opened(self, symbol: str, price: float, qty: float, score: float) -> None:
        self.notify(
            f"BUY {symbol} @ {price:.6f}  qty={qty:.4f}  score={score:.0f}/100",
            level=NotifLevel.INFO,
        )

    def trade_closed(self, symbol: str, pnl: float, reason: str) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        level = NotifLevel.INFO if pnl >= 0 else NotifLevel.WARNING
        self.notify(
            f"{emoji} CLOSE {symbol}  PnL={pnl:+.2f} USDT  reason={reason}",
            level=level,
        )

    def circuit_breaker(self, level_name: str, drawdown: float) -> None:
        self.notify(
            f"Circuit Breaker [{level_name}] triggered — drawdown {drawdown:.2%}",
            level=NotifLevel.CRITICAL,
        )

    def opportunity_found(self, symbol: str, score: float, regime: str) -> None:
        self.notify(
            f"Opportunity: {symbol}  score={score:.0f}/100  regime={regime}",
            level=NotifLevel.INFO,
        )

    def manipulation_alert(self, symbol: str, flags: dict) -> None:
        active = [k for k, v in flags.items() if v]
        self.notify(
            f"Manipulation detected on {symbol}: {', '.join(active)}",
            level=NotifLevel.WARNING,
        )

    # ── Internal ─────────────────────────────────────────────────────

    def _build_apprise(self):
        if not self._urls:
            return None
        try:
            import apprise
            ap = apprise.Apprise()
            for url in self._urls:
                if url:
                    ap.add(url)
            logger.info("AppriseNotifier: %d channel(s) loaded", len(self._urls))
            return ap
        except ImportError:
            logger.warning("apprise package not installed – notifications disabled")
            return None

    @staticmethod
    def _load_urls() -> list[str]:
        return [
            u for u in [
                os.getenv("NOTIFY_URL",   ""),
                os.getenv("NOTIFY_URL_2", ""),
                os.getenv("NOTIFY_URL_3", ""),
            ] if u
        ]
