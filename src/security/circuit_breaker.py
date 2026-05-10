from __future__ import annotations

import json
import logging
from datetime import date, datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class CBLevel(Enum):
    NORMAL      = 0
    ALERT       = 1   # 2%   drawdown  – reduce size, notify
    RESTRICTED  = 2   # 3.5% drawdown  – no new positions
    STOPPED     = 3   # 5%   drawdown  – halt all trading today
    TERMINATED  = 4   # 15%  drawdown  – definitive stop, manual reset needed


CB_THRESHOLDS = {
    CBLevel.ALERT:      0.020,
    CBLevel.RESTRICTED: 0.035,
    CBLevel.STOPPED:    0.050,
    CBLevel.TERMINATED: 0.150,
}

STATE_FILE = Path("data/circuit_breaker_state.json")


class CircuitBreaker:
    """
    4-level drawdown protection.
    Tracks intraday P&L and cumulative P&L from a capital baseline.
    """

    def __init__(self, capital_baseline: float, notify_callback=None):
        self.capital_baseline = capital_baseline
        self.notify = notify_callback or (lambda msg: logger.warning(msg))
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def level(self) -> CBLevel:
        return CBLevel(self._state["level"])

    @property
    def can_open_new_position(self) -> bool:
        return self.level in (CBLevel.NORMAL, CBLevel.ALERT)

    @property
    def position_size_multiplier(self) -> float:
        """ALERT level reduces size to 50%; other allowed levels = 100%."""
        return 0.5 if self.level == CBLevel.ALERT else 1.0

    def record_pnl(self, pnl_usdt: float) -> CBLevel:
        """
        Called after every closed trade or mark-to-market update.
        Returns the current circuit-breaker level.
        """
        today = date.today().isoformat()
        if self._state["date"] != today:
            self._reset_daily(today)

        self._state["daily_pnl"] += pnl_usdt
        self._state["cumulative_pnl"] += pnl_usdt

        daily_dd   = -self._state["daily_pnl"] / self.capital_baseline
        cumul_dd   = -self._state["cumulative_pnl"] / self.capital_baseline

        worst_dd = max(daily_dd, cumul_dd)
        new_level = self._evaluate(worst_dd)

        if new_level.value > self._state["level"]:
            self._state["level"] = new_level.value
            self._on_level_change(new_level, worst_dd)

        self._save_state()
        return CBLevel(self._state["level"])

    def manual_reset(self) -> None:
        """Operator reset – only allowed after TERMINATED to resume."""
        logger.critical("CIRCUIT BREAKER manually reset by operator")
        self._state["level"] = CBLevel.NORMAL.value
        self._state["daily_pnl"] = 0.0
        self._save_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, drawdown: float) -> CBLevel:
        if drawdown >= CB_THRESHOLDS[CBLevel.TERMINATED]:
            return CBLevel.TERMINATED
        if drawdown >= CB_THRESHOLDS[CBLevel.STOPPED]:
            return CBLevel.STOPPED
        if drawdown >= CB_THRESHOLDS[CBLevel.RESTRICTED]:
            return CBLevel.RESTRICTED
        if drawdown >= CB_THRESHOLDS[CBLevel.ALERT]:
            return CBLevel.ALERT
        return CBLevel.NORMAL

    def _on_level_change(self, level: CBLevel, dd: float) -> None:
        msgs = {
            CBLevel.ALERT:      f"[ALERT] Drawdown {dd:.2%} – position size halved",
            CBLevel.RESTRICTED: f"[RESTRICTED] Drawdown {dd:.2%} – no new positions",
            CBLevel.STOPPED:    f"[STOPPED] Drawdown {dd:.2%} – trading halted for today",
            CBLevel.TERMINATED: f"[TERMINATED] Drawdown {dd:.2%} – DEFINITIVE STOP. Manual reset required.",
        }
        msg = msgs.get(level, "")
        logger.critical(msg)
        self.notify(msg)

    def _reset_daily(self, today: str) -> None:
        self._state["date"] = today
        self._state["daily_pnl"] = 0.0
        # Daily level resets unless TERMINATED
        if self._state["level"] != CBLevel.TERMINATED.value:
            self._state["level"] = CBLevel.NORMAL.value
        logger.info("Circuit breaker daily reset")

    def _load_state(self) -> dict:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except Exception:
                pass
        return {
            "date": date.today().isoformat(),
            "level": CBLevel.NORMAL.value,
            "daily_pnl": 0.0,
            "cumulative_pnl": 0.0,
        }

    def _save_state(self) -> None:
        STATE_FILE.write_text(json.dumps(self._state, indent=2))
