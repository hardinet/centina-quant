from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    OPPORTUNITY_DETECTED            = "opportunity_detected"
    OPPORTUNITY_VALIDATED           = "opportunity_validated"
    OPPORTUNITY_PRIME               = "opportunity_prime"
    NEWS_RECEIVED                   = "news_received"
    WHALE_ALERT                     = "whale_alert"
    TRADE_EXECUTED                  = "trade_executed"
    TRADE_CLOSED                    = "trade_closed"
    TP1_HIT                         = "tp1_hit"
    TP2_HIT                         = "tp2_hit"
    CIRCUIT_BREAKER_TRIGGERED       = "circuit_breaker_triggered"
    BTC_DANGER                      = "btc_danger"
    ALT_SEASON                      = "alt_season"
    LEARNING_UPDATE                 = "learning_update"
    STRATEGY_DISCOVERED             = "strategy_discovered"
    HISTORICAL_SIMULATION_COMPLETED = "historical_simulation_completed"
    STRESS_TEST_ALERT               = "stress_test_alert"
    VOICE_COMMAND                   = "voice_command"
    PAPER_MODE_EXPIRY               = "paper_mode_expiry"


@dataclass
class Event:
    type:      EventType
    payload:   dict[str, Any] = field(default_factory=dict)
    timestamp: datetime       = field(default_factory=datetime.utcnow)
    source:    str            = ""


class EventBus:
    """
    Lightweight async pub/sub.
    Agents subscribe to event types and receive events via asyncio.Queue.
    """

    MAX_QUEUE = 256

    def __init__(self) -> None:
        self._subs: dict[EventType, list[asyncio.Queue[Event]]] = {}
        self._history: list[Event] = []

    def subscribe(self, *event_types: EventType) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self.MAX_QUEUE)
        for et in event_types:
            self._subs.setdefault(et, []).append(q)
        return q

    async def emit(self, event: Event) -> None:
        self._history.append(event)
        if len(self._history) > 1000:
            self._history = self._history[-500:]
        queues = self._subs.get(event.type, [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("EventBus queue full for %s — dropping event", event.type)

    def emit_nowait(self, event: Event) -> None:
        self._history.append(event)
        for q in self._subs.get(event.type, []):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def recent(self, event_type: EventType, n: int = 10) -> list[Event]:
        return [e for e in self._history if e.type == event_type][-n:]
