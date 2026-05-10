from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.brain.oco_calculator import OCOCalculator, OCOLevels
from src.connectors.binance_rest import BinanceREST
from src.core.event_bus import Event, EventBus, EventType
from src.database.cortex_db import CortexDB, TradeRecord
from src.notifications.apprise_notifier import AppriseNotifier, NotifLevel
from src.security.audit_logger import AuditLogger
from src.security.circuit_breaker import CBLevel, CircuitBreaker

logger = logging.getLogger(__name__)

POLL_INTERVAL     = 10    # seconds
MAX_POSITIONS     = 3
MAX_DURATION_H    = 5.0   # force exit after 5h
BE_AFTER_H        = 3.0   # move SL to breakeven after 3h if TP1 not hit


@dataclass
class ActivePosition:
    symbol:         str
    entry_price:    float
    qty_total:      float
    qty_remaining:  float
    oco_levels:     OCOLevels
    buy_order_id:   int
    strategy:       str = "B"
    score:          float = 0.0
    oco_order_id:   int | None = None
    tp1_done:       bool = False
    tp2_done:       bool = False
    trailing_active: bool = False
    trail_high:     float = 0.0
    sl_moved_to_be: bool = False   # SL moved to breakeven after TP1
    opened_at:      datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    paper:          bool = False


class Guardian:
    """
    Sub-agent 4 — open-position lifecycle manager.

    • Listens for TRADE_EXECUTED events (from orchestrator)
    • Places OCO (TP1 + SL) immediately after fill
    • After TP1 hit: moves SL to breakeven, places TP2 OCO
    • After TP2 hit: activates ATR trailing stop on remaining 20%
    • Emits TP1_HIT, TP2_HIT, TRADE_CLOSED events
    • Enforces MAX_POSITIONS=3 cap
    • Notifies on every position close
    """

    def __init__(
        self,
        rest:  BinanceREST,
        db:    CortexDB,
        bus:   EventBus,
        audit: AuditLogger,
        cb:    CircuitBreaker,
        notif: AppriseNotifier,
        voice=None,
    ):
        self._rest   = rest
        self._db     = db
        self._bus    = bus
        self._audit  = audit
        self._cb     = cb
        self._notif  = notif
        self._voice  = voice   # VoiceOrchestrator | None
        self._oco    = OCOCalculator()
        self._positions: dict[str, ActivePosition] = {}
        self._queue  = bus.subscribe(EventType.TRADE_EXECUTED)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("Guardian: started (max %d positions)", MAX_POSITIONS)
        monitor_task = asyncio.create_task(self._monitor_loop())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    if event.type == EventType.TRADE_EXECUTED:
                        await self._on_trade_executed(event.payload)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            monitor_task.cancel()
            raise

    # ------------------------------------------------------------------
    # Event handler: new trade filled
    # ------------------------------------------------------------------

    async def _on_trade_executed(self, payload: dict) -> None:
        symbol  = payload.get("symbol", "")
        if not symbol:
            return
        if symbol in self._positions:
            logger.warning("Guardian: %s already tracked — skip", symbol)
            return
        if len(self._positions) >= MAX_POSITIONS:
            logger.warning("Guardian: MAX_POSITIONS=%d reached, cannot track %s", MAX_POSITIONS, symbol)
            return

        entry_price  = float(payload.get("entry_price", 0))
        qty          = float(payload.get("qty", 0))
        atr          = float(payload.get("atr", entry_price * 0.01))
        buy_order_id = int(payload.get("order_id", 0))
        strategy     = payload.get("strategy", "B")
        score        = float(payload.get("score", 0))
        paper        = bool(payload.get("paper", False))

        levels = self._oco.calculate(entry_price, atr)
        pos = ActivePosition(
            symbol=symbol,
            entry_price=entry_price,
            qty_total=qty,
            qty_remaining=qty,
            oco_levels=levels,
            buy_order_id=buy_order_id,
            strategy=strategy,
            score=score,
            paper=paper,
        )

        if not paper:
            await self._place_initial_oco(pos)

        self._positions[symbol] = pos
        self._audit.log("POSITION_OPENED", {
            "symbol": symbol, "entry": entry_price, "qty": qty,
            "strategy": strategy, "score": score, "paper": paper,
        })
        logger.info("Guardian: tracking %s entry=%.6f qty=%.6f strategy=%s",
                    symbol, entry_price, qty, strategy)

    # ------------------------------------------------------------------
    # OCO placement helpers
    # ------------------------------------------------------------------

    async def _place_initial_oco(self, pos: ActivePosition) -> None:
        """OCO for TP1 (50% qty) with SL."""
        tp1_qty = round(pos.qty_total * pos.oco_levels.qty_tp1, 6)
        sl_limit = round(pos.oco_levels.sl * 0.999, 8)
        try:
            resp = await self._rest.place_oco_sell(
                pos.symbol, tp1_qty, pos.oco_levels.tp1, pos.oco_levels.sl, sl_limit
            )
            pos.oco_order_id = resp.get("orderListId")
            logger.info("Guardian: TP1 OCO placed %s TP1=%.6f SL=%.6f",
                        pos.symbol, pos.oco_levels.tp1, pos.oco_levels.sl)
        except Exception as e:
            logger.error("Guardian: TP1 OCO failed for %s: %s", pos.symbol, e)

    async def _place_tp2_oco(self, pos: ActivePosition) -> None:
        """After TP1 hit: place TP2 OCO with SL at breakeven."""
        tp2_qty    = round(pos.qty_total * pos.oco_levels.qty_tp2, 6)
        sl_be      = round(pos.entry_price * 1.001, 8)   # breakeven + 0.1% fee
        sl_be_lim  = round(sl_be * 0.999, 8)
        try:
            resp = await self._rest.place_oco_sell(
                pos.symbol, tp2_qty, pos.oco_levels.tp2, sl_be, sl_be_lim
            )
            pos.oco_order_id = resp.get("orderListId")
            pos.sl_moved_to_be = True
            logger.info("Guardian: TP2 OCO placed %s TP2=%.6f SL-BE=%.6f",
                        pos.symbol, pos.oco_levels.tp2, sl_be)
        except Exception as e:
            logger.error("Guardian: TP2 OCO failed for %s: %s", pos.symbol, e)

    # ------------------------------------------------------------------
    # Position monitoring loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            for symbol in list(self._positions.keys()):
                try:
                    await self._update_position(self._positions[symbol])
                except Exception as e:
                    logger.error("Guardian monitor error %s: %s", symbol, e)

    async def _update_position(self, pos: ActivePosition) -> None:
        elapsed_h = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600

        # ── Forced exit after 5h ─────────────────────────────────────
        if elapsed_h >= MAX_DURATION_H and pos.qty_remaining > 0:
            logger.warning("Guardian: forced 5h exit %s (%.1fh elapsed)", pos.symbol, elapsed_h)
            try:
                price = float(await self._rest.get_symbol_price(pos.symbol))
            except Exception:
                price = pos.entry_price
            if not pos.paper:
                try:
                    await self._rest.place_market_sell(pos.symbol, pos.qty_remaining)
                except Exception as e:
                    logger.error("Guardian: forced sell failed %s: %s", pos.symbol, e)
            await self._finalize_position(pos, price, "FORCED_5H_EXIT")
            return

        if pos.paper:
            await self._paper_tick(pos)
            return

        open_orders = await self._rest.get_open_orders(pos.symbol)

        # ── SL → breakeven after 3h if TP1 not hit ──────────────────
        if not pos.tp1_done and not pos.sl_moved_to_be and elapsed_h >= BE_AFTER_H:
            logger.info("Guardian: 3h elapsed, moving SL to breakeven for %s", pos.symbol)
            pos.sl_moved_to_be = True
            if pos.oco_order_id is not None and not pos.paper:
                try:
                    await self._rest.cancel_oco(pos.symbol, pos.oco_order_id)
                    sl_be = round(pos.entry_price * 1.001, 8)
                    sl_be_lim = round(sl_be * 0.999, 8)
                    tp1_qty = round(pos.qty_total * pos.oco_levels.qty_tp1, 6)
                    resp = await self._rest.place_oco_sell(
                        pos.symbol, tp1_qty, pos.oco_levels.tp1, sl_be, sl_be_lim
                    )
                    pos.oco_order_id = resp.get("orderListId")
                    logger.info("Guardian: SL moved to BE %.6f for %s", sl_be, pos.symbol)
                except Exception as e:
                    logger.error("Guardian: SL→BE failed %s: %s", pos.symbol, e)

        # TP1 check: OCO gone → TP1 hit (or SL hit)
        if not pos.tp1_done and pos.oco_order_id is not None:
            oco_still_open = any(
                o.get("orderListId") == pos.oco_order_id for o in open_orders
            )
            if not oco_still_open:
                await self._handle_tp1_hit(pos)

        # TP2 check
        elif pos.tp1_done and not pos.tp2_done and pos.oco_order_id is not None:
            oco_still_open = any(
                o.get("orderListId") == pos.oco_order_id for o in open_orders
            )
            if not oco_still_open:
                await self._handle_tp2_hit(pos)

        # Trailing stop on last 20%
        elif pos.trailing_active:
            await self._manage_trailing(pos, open_orders)

    # ------------------------------------------------------------------
    # TP handlers
    # ------------------------------------------------------------------

    async def _handle_tp1_hit(self, pos: ActivePosition) -> None:
        pos.tp1_done = True
        tp1_qty = round(pos.qty_total * pos.oco_levels.qty_tp1, 6)
        pos.qty_remaining -= tp1_qty

        pnl_pct = (pos.oco_levels.tp1 - pos.entry_price) / pos.entry_price * 100
        logger.info("Guardian: TP1 HIT %s +%.2f%% — placing TP2 OCO", pos.symbol, pnl_pct)

        await self._bus.emit(Event(EventType.TP1_HIT, {
            "symbol": pos.symbol, "tp1_price": pos.oco_levels.tp1,
            "pnl_pct": round(pnl_pct, 2), "strategy": pos.strategy,
        }, source="guardian"))

        self._notif.notify(
            f"✅ TP1 {pos.symbol} +{pnl_pct:.1f}% — SL→BE activé",
            level=NotifLevel.INFO,
        )
        if self._voice:
            self._voice.announce_tp1_hit(pos.symbol, pnl_pct)
        await self._place_tp2_oco(pos)

    async def _handle_tp2_hit(self, pos: ActivePosition) -> None:
        pos.tp2_done = True
        tp2_qty = round(pos.qty_total * pos.oco_levels.qty_tp2, 6)
        pos.qty_remaining -= tp2_qty
        pos.trailing_active = True
        pos.trail_high = pos.oco_levels.tp2

        pnl_pct = (pos.oco_levels.tp2 - pos.entry_price) / pos.entry_price * 100
        logger.info("Guardian: TP2 HIT %s +%.2f%% — trailing ATR×1.5 active", pos.symbol, pnl_pct)

        await self._bus.emit(Event(EventType.TP2_HIT, {
            "symbol": pos.symbol, "tp2_price": pos.oco_levels.tp2,
            "pnl_pct": round(pnl_pct, 2), "strategy": pos.strategy,
        }, source="guardian"))

        self._notif.notify(
            f"🚀 TP2 {pos.symbol} +{pnl_pct:.1f}% — trailing ACTIF",
            level=NotifLevel.INFO,
        )
        if self._voice:
            self._voice.announce_tp2_hit(pos.symbol, pnl_pct)

    async def _manage_trailing(self, pos: ActivePosition, open_orders: list) -> None:
        """Trailing stop: if price drops ATR×1.5 from trail_high → close."""
        try:
            ticker = await self._rest.get_symbol_price(pos.symbol)
            price  = float(ticker)
        except Exception:
            return

        if price > pos.trail_high:
            pos.trail_high = price

        trail_distance = pos.oco_levels.tp3_trail_atr
        stop_price     = pos.trail_high - trail_distance

        if price <= stop_price and pos.qty_remaining > 0:
            logger.info("Guardian: trailing stop hit %s price=%.6f stop=%.6f",
                        pos.symbol, price, stop_price)
            await self._close_trailing(pos, price)

    async def _close_trailing(self, pos: ActivePosition, exit_price: float) -> None:
        if not pos.paper:
            try:
                await self._rest.place_market_sell(pos.symbol, pos.qty_remaining)
            except Exception as e:
                logger.error("Guardian: market sell failed %s: %s", pos.symbol, e)

        await self._finalize_position(pos, exit_price, "TRAILING_STOP")

    # ------------------------------------------------------------------
    # Paper mode tick
    # ------------------------------------------------------------------

    async def _paper_tick(self, pos: ActivePosition) -> None:
        """Simulate TP/SL in paper mode using live prices."""
        try:
            price = float(await self._rest.get_symbol_price(pos.symbol))
        except Exception:
            return

        if not pos.tp1_done and price >= pos.oco_levels.tp1:
            await self._handle_tp1_hit(pos)
        elif not pos.tp1_done and price <= pos.oco_levels.sl:
            await self._finalize_position(pos, price, "SL_HIT")
        elif pos.tp1_done and not pos.tp2_done and price >= pos.oco_levels.tp2:
            await self._handle_tp2_hit(pos)
        elif pos.tp1_done and not pos.tp2_done and price <= pos.entry_price * 1.001:
            await self._finalize_position(pos, price, "SL_BE_HIT")
        elif pos.trailing_active:
            await self._manage_trailing(pos, [])

    # ------------------------------------------------------------------
    # Finalize (record to DB, emit TRADE_CLOSED)
    # ------------------------------------------------------------------

    async def _finalize_position(self, pos: ActivePosition, exit_price: float, reason: str) -> None:
        pnl_usdt = (exit_price - pos.entry_price) / pos.entry_price * pos.qty_total * pos.entry_price
        pnl_pct  = (exit_price - pos.entry_price) / pos.entry_price * 100

        self._db.close_trade_by_symbol(
            pos.symbol,
            exit_price=exit_price,
            pnl_usdt=round(pnl_usdt, 4),
            pnl_pct=round(pnl_pct, 4),
            exit_reason=reason,
            tp1_hit=pos.tp1_done,
            tp2_hit=pos.tp2_done,
        )

        self._cb.record_pnl(pnl_usdt)

        await self._bus.emit(Event(EventType.TRADE_CLOSED, {
            "symbol":    pos.symbol,
            "exit_price": exit_price,
            "pnl_usdt":  round(pnl_usdt, 4),
            "pnl_pct":   round(pnl_pct, 4),
            "reason":    reason,
            "strategy":  pos.strategy,
            "tp1_hit":   pos.tp1_done,
            "tp2_hit":   pos.tp2_done,
        }, source="guardian"))

        colour = "✅" if pnl_usdt >= 0 else "❌"
        self._notif.notify(
            f"{colour} CLOSE {pos.symbol} {reason} {pnl_pct:+.2f}% ({pnl_usdt:+.2f} USDT)",
            level=NotifLevel.INFO if pnl_usdt >= 0 else NotifLevel.WARNING,
        )
        self._audit.log("TRADE_CLOSED", {
            "symbol": pos.symbol, "exit": exit_price,
            "pnl_usdt": pnl_usdt, "reason": reason,
        })
        if self._voice:
            self._voice.announce_trade_result(pos.symbol, pnl_pct, reason, pnl_usdt)

        del self._positions[pos.symbol]
        logger.info("Guardian: closed %s reason=%s pnl=%+.4f USDT", pos.symbol, reason, pnl_usdt)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def open_positions_count(self) -> int:
        return len(self._positions)

    def get_positions(self) -> dict[str, ActivePosition]:
        return dict(self._positions)

    async def open_position(
        self,
        symbol: str,
        entry_price: float,
        qty: float,
        atr: float,
        buy_order_id: int,
        strategy: str = "B",
        score: float = 0.0,
        paper: bool = False,
    ) -> ActivePosition:
        """Direct call from interactive loop (bypass event queue)."""
        payload = {
            "symbol": symbol, "entry_price": entry_price, "qty": qty,
            "atr": atr, "order_id": buy_order_id, "strategy": strategy,
            "score": score, "paper": paper,
        }
        await self._on_trade_executed(payload)
        return self._positions.get(symbol)
