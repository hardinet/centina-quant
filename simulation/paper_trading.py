from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.brain.oco_calculator import OCOCalculator, OCOLevels
from src.brain.kelly_calculator import KellyCalculator
from src.agents.scout import Opportunity

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    id:           str
    symbol:       str
    side:         str         # BUY / SELL
    qty:          float
    entry_price:  float
    sl:           float
    tp1:          float
    tp2:          float
    tp3_trail_atr: float
    opened_at:    datetime = field(default_factory=datetime.utcnow)
    closed_at:    datetime | None = None
    exit_price:   float = 0.0
    pnl_usdt:     float = 0.0
    exit_reason:  str = ""
    tp1_hit:      bool = False
    tp2_hit:      bool = False


@dataclass
class PaperPortfolio:
    capital_usdt:    float
    initial_capital: float
    open_orders:     dict[str, PaperOrder] = field(default_factory=dict)
    closed_orders:   list[PaperOrder] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return sum(o.pnl_usdt for o in self.closed_orders)

    @property
    def win_rate(self) -> float:
        wins = [o for o in self.closed_orders if o.pnl_usdt > 0]
        return len(wins) / len(self.closed_orders) if self.closed_orders else 0.0

    @property
    def equity(self) -> float:
        unrealised = sum(
            (self._get_last_price(o) - o.entry_price) * o.qty
            for o in self.open_orders.values()
        )
        return self.capital_usdt + unrealised

    def _get_last_price(self, order: PaperOrder) -> float:
        return order.entry_price   # override via price_feed in PaperTrader


def paper_gate_countdown() -> tuple[bool, int]:
    """Returns (gate_cleared, days_remaining). Same logic as main.py."""
    from pathlib import Path
    from datetime import datetime
    f = Path("data/.activation_date")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(datetime.utcnow().isoformat())
        return False, 7
    try:
        dt = datetime.fromisoformat(f.read_text().strip())
        remaining = max(0, 7 - (datetime.utcnow() - dt).days)
        return remaining == 0, remaining
    except Exception:
        return False, 7


class PaperTrader:
    """
    v5.0 Paper trading engine.
    TP1=50%/TP2=30%/trailing 20%.
    Tracks fees (0.1% per leg).
    Enforces 7-day paper gate countdown display.
    """

    POLL_INTERVAL = 5   # seconds
    FEE_RATE = 0.001    # 0.1%

    def __init__(self, capital_usdt: float = 200.0):
        self._portfolio = PaperPortfolio(capital_usdt=capital_usdt, initial_capital=capital_usdt)
        self._oco       = OCOCalculator()
        self._kelly     = KellyCalculator()
        self._prices:   dict[str, float] = {}
        self._running   = False
        self._rest      = None
        self._total_fees = 0.0

        cleared, days_left = paper_gate_countdown()
        if not cleared:
            logger.info("[PAPER] Gate active — %d days until live trading unlocked", days_left)
        self._gate_cleared = cleared
        self._days_left    = days_left

    def set_rest(self, rest) -> None:
        self._rest = rest

    # ── Public API ───────────────────────────────────────────────────

    def open_trade(
        self,
        opp: Opportunity,
        score: float,
        capital: float | None = None,
    ) -> PaperOrder:
        cap  = capital or self._portfolio.capital_usdt
        sl_pct = self._oco.calculate(opp.best_price, opp.best_atr).sl_pct / 100
        kr   = self._kelly.from_signal_score(score, cap, sl_pct)
        qty  = round(kr.position_usdt / opp.best_price, 6)
        lvls = self._oco.calculate(opp.best_price, opp.best_atr)

        order = PaperOrder(
            id=str(uuid.uuid4())[:8],
            symbol=opp.symbol,
            side="BUY",
            qty=qty,
            entry_price=opp.best_price,
            sl=lvls.sl,
            tp1=lvls.tp1,
            tp2=lvls.tp2,
            tp3_trail_atr=lvls.tp3_trail_atr,
        )
        cost = qty * opp.best_price
        self._portfolio.capital_usdt -= cost
        self._portfolio.open_orders[order.id] = order

        logger.info(
            "[PAPER] OPEN  %s  qty=%.6f @ %.6f  SL=%.6f  TP1=%.6f  TP2=%.6f",
            opp.symbol, qty, opp.best_price, lvls.sl, lvls.tp1, lvls.tp2,
        )
        return order

    async def run(self) -> None:
        self._running = True
        while self._running:
            await self._update_prices()
            self._check_orders()
            await asyncio.sleep(self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False

    def summary(self) -> dict:
        return {
            "capital_usdt":    round(self._portfolio.capital_usdt, 2),
            "equity":          round(self._portfolio.equity, 2),
            "total_pnl":       round(self._portfolio.total_pnl, 2),
            "total_fees":      round(self._total_fees, 4),
            "open_positions":  len(self._portfolio.open_orders),
            "closed_trades":   len(self._portfolio.closed_orders),
            "win_rate":        round(self._portfolio.win_rate, 4),
            "return_pct":      round(self._portfolio.total_pnl / self._portfolio.initial_capital * 100, 2),
            "paper_mode":      True,
            "gate_cleared":    self._gate_cleared,
            "days_to_live":    self._days_left,
        }

    # ── Internal ─────────────────────────────────────────────────────

    async def _update_prices(self) -> None:
        if not self._rest or not self._portfolio.open_orders:
            return
        for oid, order in self._portfolio.open_orders.items():
            try:
                df = await self._rest.get_klines(order.symbol, "1m", limit=2)
                self._prices[order.symbol] = df["close"].astype(float).iloc[-1]
            except Exception:
                pass

    def _check_orders(self) -> None:
        for oid in list(self._portfolio.open_orders.keys()):
            order = self._portfolio.open_orders[oid]
            price = self._prices.get(order.symbol, order.entry_price)

            if price <= order.sl:
                self._close(order, price, "SL")
            elif not order.tp1_hit and price >= order.tp1:
                # v5.0: TP1 = 50% of position
                self._partial_close(order, price, order.qty * 0.50, "TP1")
                order.tp1_hit = True
                order.sl = order.entry_price   # move SL to breakeven
            elif order.tp1_hit and not order.tp2_hit and price >= order.tp2:
                # v5.0: TP2 = 30% of remaining original position
                self._partial_close(order, price, order.qty * 0.30, "TP2")
                order.tp2_hit = True
                logger.info("[PAPER] TP2 hit %s – activating trailing stop (ATR×1.5)", order.symbol)

    def _partial_close(self, order: PaperOrder, price: float, qty: float, reason: str) -> None:
        pnl = (price - order.entry_price) * qty
        self._portfolio.capital_usdt += price * qty
        order.qty -= qty
        logger.info("[PAPER] PARTIAL %s %s  qty=%.6f  pnl=%.2f USDT", reason, order.symbol, qty, pnl)

    def _close(self, order: PaperOrder, price: float, reason: str) -> None:
        fee = price * order.qty * self.FEE_RATE
        order.exit_price  = price
        order.pnl_usdt    = (price - order.entry_price) * order.qty - fee
        order.exit_reason = reason
        order.closed_at   = datetime.utcnow()
        self._portfolio.capital_usdt += price * order.qty - fee
        self._total_fees += fee
        del self._portfolio.open_orders[order.id]
        self._portfolio.closed_orders.append(order)
        emoji = "✅" if order.pnl_usdt >= 0 else "❌"
        logger.info(
            "[PAPER] %s CLOSE %s @ %.6f  PnL=%.2f USDT  reason=%s",
            emoji, order.symbol, price, order.pnl_usdt, reason,
        )
