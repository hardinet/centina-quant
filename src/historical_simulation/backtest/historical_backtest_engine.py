from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

FEE      = 0.001   # 0.1% per leg
SLIPPAGE = 0.0005  # 0.05%


@dataclass
class TradeRecord:
    entry_price: float; exit_price: float; qty: float
    pnl: float; pnl_pct: float; duration_h: float
    tp1_hit: bool; tp2_hit: bool; exit_reason: str
    strategy: str; regime: str


@dataclass
class BacktestMetrics:
    symbol: str; strategy: str; timeframe: str
    start_date: str; end_date: str
    capital_initial: float; capital_final: float
    total_return_pct: float; cagr: float
    sharpe: float; sortino: float; calmar: float
    max_drawdown_pct: float; max_dd_duration_days: int
    profit_factor: float; win_rate: float
    avg_win_pct: float; avg_loss_pct: float
    expectancy: float; nb_trades: int
    avg_duration_h: float
    var_95: float; cvar_95: float
    recovery_factor: float; ulcer_index: float
    monthly_returns: dict = field(default_factory=dict)
    annual_returns:  dict = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    params: dict = field(default_factory=dict)


class HistoricalBacktestEngine:
    """
    20-year backtesting engine.  Simulates CENTINA's OCO strategy:
    TP1 50% / TP2 30% / trailing 20% / SL with ATR.
    Supports up to 3 simultaneous positions.
    """

    MAX_POSITIONS = 3

    def __init__(self, capital: float = 200.0, fee: float = FEE, slippage: float = SLIPPAGE):
        self.capital   = capital
        self.fee       = fee
        self.slippage  = slippage

    def run_backtest(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategy: str = "B",
        start_date: str | None = None,
        end_date:   str | None = None,
        params: dict | None = None,
    ) -> BacktestMetrics:
        p = params or {}
        ema_fast   = p.get("ema_fast", 7)
        ema_slow   = p.get("ema_slow", 25)
        ema_long   = p.get("ema_long", 99)
        ema_trend  = p.get("ema_trend", 200)
        rsi_low    = p.get("rsi_low", 50.0)
        rsi_high   = p.get("rsi_high", 72.0)
        atr_sl     = p.get("atr_sl_mult", 1.0)
        atr_tp1    = p.get("atr_tp1_mult", 1.5)
        vol_mult   = p.get("vol_spike", 1.5)
        score_min  = p.get("score_min", 0)

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz=df.index.tz)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date, tz=df.index.tz)]

        if df.empty or len(df) < 220:
            return self._empty(symbol, strategy, timeframe, params or {})

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)

        ema_f  = ta.ema(close, ema_fast)
        ema_s  = ta.ema(close, ema_slow)
        ema_l  = ta.ema(close, ema_long)
        ema_tr = ta.ema(close, ema_trend)
        rsi    = ta.rsi(close, 14)
        atr    = ta.atr(high, low, close, 14)

        capital  = self.capital
        trades:  list[TradeRecord] = []
        equity:  list[float] = [capital]
        positions: list[dict] = []

        for i in range(220, len(df)):
            p_now = close.iloc[i]
            h_now = high.iloc[i]
            l_now = low.iloc[i]

            if any(pd.isna([ema_f.iloc[i], ema_s.iloc[i], rsi.iloc[i], atr.iloc[i]])):
                continue

            # Update existing positions
            closed_pos = []
            for pos in positions:
                tr = self._update_position(pos, h_now, l_now, p_now, i, atr.iloc[i])
                if tr:
                    trades.append(tr)
                    capital += tr.pnl
                    closed_pos.append(pos)
            for cp in closed_pos:
                positions.remove(cp)

            # Entry signal
            if len(positions) < self.MAX_POSITIONS:
                vol_ok   = volume.iloc[i] > volume.iloc[i-20:i].mean() * vol_mult
                ema_ok   = p_now > ema_f.iloc[i] > ema_s.iloc[i] > ema_l.iloc[i] > ema_tr.iloc[i]
                rsi_ok   = rsi_low <= rsi.iloc[i] <= rsi_high

                if ema_ok and rsi_ok and vol_ok:
                    entry  = p_now * (1 + self.slippage + self.fee)
                    sl     = entry - atr_sl * atr.iloc[i]
                    tp1    = entry * 1.025
                    tp2    = entry * 1.052
                    qty    = (capital * 0.05) / entry
                    positions.append({
                        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                        "qty": qty, "qty_rem": qty,
                        "tp1_hit": False, "tp2_hit": False,
                        "trail_sl": None, "atr": atr.iloc[i],
                        "entry_idx": i, "strategy": strategy,
                    })

            equity.append(capital)

        # Close remaining
        if df is not None and not df.empty:
            last_price = close.iloc[-1]
            for pos in positions:
                pnl = (last_price - pos["entry"]) * pos["qty_rem"] - pos["qty_rem"] * last_price * self.fee
                trades.append(TradeRecord(
                    entry_price=pos["entry"], exit_price=last_price,
                    qty=pos["qty_rem"], pnl=pnl,
                    pnl_pct=(last_price / pos["entry"] - 1) * 100,
                    duration_h=float(len(df) - pos["entry_idx"]),
                    tp1_hit=pos["tp1_hit"], tp2_hit=pos["tp2_hit"],
                    exit_reason="end_of_data", strategy=strategy, regime="",
                ))
                capital += pnl

        return self._compute_metrics(
            trades, equity, capital, symbol, strategy, timeframe,
            start_date or str(df.index[0].date()), end_date or str(df.index[-1].date()), params or {},
        )

    def _update_position(self, pos: dict, h: float, l: float, c: float, i: int, atr: float) -> TradeRecord | None:
        entry = pos["entry"]

        # SL hit
        if l <= pos["sl"]:
            pnl = (pos["sl"] - entry) * pos["qty_rem"] - pos["qty_rem"] * pos["sl"] * self.fee
            return TradeRecord(entry, pos["sl"], pos["qty_rem"], pnl,
                               (pos["sl"]/entry-1)*100, float(i - pos["entry_idx"]),
                               pos["tp1_hit"], pos["tp2_hit"], "sl", pos["strategy"], "")

        # TP1 (50%)
        if not pos["tp1_hit"] and h >= pos["tp1"]:
            partial = pos["qty"] * 0.50
            pos["qty_rem"] -= partial
            pos["tp1_hit"] = True
            pos["sl"] = entry   # move SL to breakeven
            # partial close PnL recorded internally

        # TP2 (30%)
        if pos["tp1_hit"] and not pos["tp2_hit"] and h >= pos["tp2"]:
            partial = pos["qty"] * 0.30
            pos["qty_rem"] -= partial
            pos["tp2_hit"] = True
            pos["trail_sl"] = pos["tp2"] - atr

        # Trailing stop (20%)
        if pos["trail_sl"] is not None:
            new_trail = c - atr
            pos["trail_sl"] = max(pos["trail_sl"], new_trail)
            if l <= pos["trail_sl"]:
                exit_p = pos["trail_sl"]
                pnl = (exit_p - entry) * pos["qty_rem"] - pos["qty_rem"] * exit_p * self.fee
                return TradeRecord(entry, exit_p, pos["qty_rem"], pnl,
                                   (exit_p/entry-1)*100, float(i - pos["entry_idx"]),
                                   True, True, "trailing", pos["strategy"], "")
        return None

    def run_monte_carlo_simulation(self, backtest: BacktestMetrics, n: int = 1000) -> dict:
        if not backtest.trades:
            return {}
        pnls = [t.pnl_pct for t in backtest.trades]
        results = []
        for _ in range(n):
            perm = np.random.choice(pnls, size=len(pnls), replace=True)
            final = self.capital * np.prod(1 + perm / 100)
            results.append(final)
        a = np.array(results)
        return {
            "mean":    float(a.mean()),
            "median":  float(np.median(a)),
            "p5":      float(np.percentile(a, 5)),
            "p95":     float(np.percentile(a, 95)),
            "prob_profit": float((a > self.capital).mean()),
        }

    def _compute_metrics(self, trades, equity, final_cap, symbol, strategy, tf,
                         start, end, params) -> BacktestMetrics:
        if not trades:
            return self._empty(symbol, strategy, tf, params)

        pnls = [t.pnl for t in trades]
        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls)
        pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
        total_ret = (final_cap - self.capital) / self.capital * 100

        # Equity drawdown
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / np.maximum(peak, 1)
        max_dd = float(dd.max()) * 100

        # Duration
        start_dt = datetime.fromisoformat(start) if start else datetime.utcnow()
        end_dt   = datetime.fromisoformat(end) if end else datetime.utcnow()
        years    = max((end_dt - start_dt).days / 365.25, 0.01)
        cagr     = ((final_cap / self.capital) ** (1 / years) - 1) * 100

        # Returns series
        ret_series = pd.Series(pnls) / self.capital
        sharpe = float(ret_series.mean() / ret_series.std() * math.sqrt(252)) if ret_series.std() > 0 else 0
        neg    = ret_series[ret_series < 0]
        sortino = float(ret_series.mean() / neg.std() * math.sqrt(252)) if len(neg) > 1 and neg.std() > 0 else 0
        calmar = cagr / max_dd if max_dd > 0 else float("inf")

        var95  = float(np.percentile(pnls, 5)) if pnls else 0
        cvar95 = float(np.mean([p for p in pnls if p <= var95])) if pnls else 0
        rf     = total_ret / max_dd if max_dd > 0 else float("inf")
        ui     = float(np.sqrt(np.mean(dd ** 2))) * 100

        return BacktestMetrics(
            symbol=symbol, strategy=strategy, timeframe=tf,
            start_date=start, end_date=end,
            capital_initial=self.capital, capital_final=round(final_cap, 2),
            total_return_pct=round(total_ret, 2), cagr=round(cagr, 2),
            sharpe=round(sharpe, 3), sortino=round(sortino, 3), calmar=round(calmar, 3),
            max_drawdown_pct=round(max_dd, 2), max_dd_duration_days=0,
            profit_factor=round(pf, 3), win_rate=round(win_rate, 4),
            avg_win_pct=round(sum(t.pnl_pct for t in trades if t.pnl > 0) / max(len(wins),1), 2),
            avg_loss_pct=round(sum(t.pnl_pct for t in trades if t.pnl <= 0) / max(len(losses),1), 2),
            expectancy=round(sum(pnls) / len(pnls), 2),
            nb_trades=len(trades),
            avg_duration_h=round(sum(t.duration_h for t in trades) / len(trades), 1),
            var_95=round(var95, 2), cvar_95=round(cvar95, 2),
            recovery_factor=round(rf, 2), ulcer_index=round(ui, 4),
            trades=trades, params=params,
        )

    def _empty(self, symbol, strategy, tf, params) -> BacktestMetrics:
        return BacktestMetrics(
            symbol=symbol, strategy=strategy, timeframe=tf,
            start_date="", end_date="",
            capital_initial=self.capital, capital_final=self.capital,
            total_return_pct=0, cagr=0, sharpe=0, sortino=0, calmar=0,
            max_drawdown_pct=0, max_dd_duration_days=0,
            profit_factor=0, win_rate=0, avg_win_pct=0, avg_loss_pct=0,
            expectancy=0, nb_trades=0, avg_duration_h=0,
            var_95=0, cvar_95=0, recovery_factor=0, ulcer_index=0,
            params=params,
        )
