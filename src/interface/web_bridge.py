"""
WebBridge — pont temps réel entre l'agent CENTINA et le dashboard Streamlit.
Écrit un fichier JSON partagé toutes les 2s que le dashboard peut lire.
Pas de Redis requis — fonctionne avec un simple fichier partagé.
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.guardian import Guardian
    from src.agents.scout import Scout
    from src.security.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

BRIDGE_FILE = pathlib.Path("data/bridge_state.json")
REFRESH_INTERVAL = 2.0


class WebBridge:
    """
    Écrit l'état complet de l'agent dans data/bridge_state.json.
    Le dashboard Streamlit le lit en polling (st_autorefresh 5s).
    """

    def __init__(
        self,
        scout: "Scout | None" = None,
        guardian: "Guardian | None" = None,
        cb: "CircuitBreaker | None" = None,
        mode: str = "PAPER",
        capital: float = 200.0,
        daily_target: float = 30.0,
    ):
        self._scout   = scout
        self._guardian = guardian
        self._cb      = cb
        self._mode    = mode
        self._capital = capital
        self._target  = daily_target
        self._daily_pnl = 0.0
        self._total_pnl = 0.0
        self._n_trades  = 0
        self._status_log: list[str] = []
        self._events: list[dict] = []
        self._validated_opps: list[dict] = []   # full analyst payloads
        self._voice_active: bool = False

        BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────

    def record_pnl(self, pnl_usdt: float) -> None:
        self._daily_pnl += pnl_usdt
        self._total_pnl += pnl_usdt
        self._n_trades  += 1

    def log_status(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._status_log.append(f"{ts} {msg}")
        if len(self._status_log) > 50:
            self._status_log = self._status_log[-50:]

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0
        self._n_trades  = 0

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_voice_active(self, active: bool) -> None:
        self._voice_active = active

    def update_opportunity(self, payload: dict) -> None:
        """Store full analyst result (satellite data, LLM council, etc.)."""
        sym = payload.get("symbol", "")
        if not sym:
            return
        # Compute estimated_gain_eur from net_gain_pct and kelly_pct
        net_gain_pct = payload.get("net_gain_pct", 0.0)
        kelly_pct    = payload.get("kelly_pct", 0.03)
        position_eur = self._capital * kelly_pct
        estimated_gain_eur = round(position_eur * net_gain_pct / 100, 2) if net_gain_pct else 0.0

        enriched = {**payload, "estimated_gain_eur": estimated_gain_eur}
        # Replace existing entry for same symbol or append
        self._validated_opps = [o for o in self._validated_opps if o.get("symbol") != sym]
        self._validated_opps.insert(0, enriched)
        # Keep top 15 by score
        self._validated_opps = sorted(self._validated_opps, key=lambda x: x.get("score", 0), reverse=True)[:15]

    # ── Main coroutine ────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("WebBridge: started → %s", BRIDGE_FILE)
        while True:
            try:
                self._write()
            except Exception as e:
                logger.debug("WebBridge write error: %s", e)
            await asyncio.sleep(REFRESH_INTERVAL)

    # ── Serialization ─────────────────────────────────────────────────

    def add_event(self, text: str, ev_type: str = "INFO") -> None:
        """Appelé par l'orchestrateur pour pousser un événement vers le dashboard."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._events.append({"ts": ts, "text": text, "type": ev_type})
        if len(self._events) > 50:
            self._events = self._events[-50:]

    def _write(self) -> None:
        positions = self._positions_snapshot()
        cb_name   = self._cb.level.name  if self._cb else "NORMAL"
        paper_left = self._paper_days_left()
        state: dict[str, Any] = {
            # ── compat ancien dashboard ──────────────────────────────
            "ts":              datetime.now(timezone.utc).isoformat(),
            "mode":            self._mode,
            "capital":         self._capital,
            "daily_target":    self._target,
            "daily_pnl":       round(self._daily_pnl, 4),
            "total_pnl":       round(self._total_pnl, 4),
            "n_trades_today":  self._n_trades,
            "cb_level":        cb_name,
            "cb_name":         cb_name,
            # ── champs attendus par le dashboard Streamlit ───────────
            "capital_usdt":    self._capital,
            "pnl_day_eur":     round(self._daily_pnl, 4),
            "open_positions":  len(positions),
            "paper_days_left": paper_left,
            "positions":       positions,
            "opportunities":   self._opps_snapshot(),
            "events":          self._events[-30:],
            "status_log":      self._status_log[-20:],
            "voice_active":    self._voice_active,
        }
        BRIDGE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _paper_days_left(self) -> int:
        activation_file = pathlib.Path("data/.activation_date")
        try:
            if activation_file.exists():
                from datetime import date
                activated = date.fromisoformat(activation_file.read_text().strip())
                remaining = 7 - (date.today() - activated).days
                return max(0, remaining)
        except Exception:
            pass
        return 7

    def _positions_snapshot(self) -> list[dict]:
        if not self._guardian:
            return []
        try:
            out = []
            for sym, pos in self._guardian.get_positions().items():
                elapsed = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600
                out.append({
                    "symbol":     sym,
                    "entry":      pos.entry_price,
                    "qty_total":  pos.qty_total,
                    "qty_rem":    pos.qty_remaining,
                    "strategy":   pos.strategy,
                    "score":      pos.score,
                    "tp1":        pos.oco_levels.tp1,
                    "tp2":        pos.oco_levels.tp2,
                    "sl":         pos.oco_levels.sl,
                    "tp1_done":   pos.tp1_done,
                    "tp2_done":   pos.tp2_done,
                    "trailing":   pos.trailing_active,
                    "elapsed_h":  round(elapsed, 2),
                    "paper":      pos.paper,
                    "opened_at":  pos.opened_at.isoformat(),
                })
            return out
        except Exception:
            return []

    def _opps_snapshot(self) -> list[dict]:
        # Priority: full analyst-validated results
        if self._validated_opps:
            return self._validated_opps[:12]
        # Fallback: raw scout scan
        if not self._scout:
            return []
        try:
            opps = self._scout.get_last_scan() or []
            return [
                {
                    "symbol":            o.symbol,
                    "score":             o.score,
                    "strategy":          getattr(o, "suggested_strategy", "B"),
                    "regime":            getattr(o, "regime", "?"),
                    "price":             getattr(o, "best_price", 0),
                    "rsi":               getattr(o, "rsi", 0),
                    "estimated_gain_eur": 0,
                    "satellite_bonus":   0,
                    "council_verdict":   "",
                    "council_text":      "",
                }
                for o in sorted(opps, key=lambda x: x.score, reverse=True)[:12]
            ]
        except Exception:
            return []


# ── Dashboard reader helper ───────────────────────────────────────────────

def read_bridge() -> dict:
    """Called by Streamlit dashboard to get the current agent state."""
    try:
        if BRIDGE_FILE.exists():
            return json.loads(BRIDGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "ts": "", "mode": "—", "capital": 0, "daily_target": 30,
        "daily_pnl": 0, "total_pnl": 0, "n_trades_today": 0,
        "cb_level": 0, "cb_name": "—",
        "positions": [], "opportunities": [], "status_log": [],
    }
