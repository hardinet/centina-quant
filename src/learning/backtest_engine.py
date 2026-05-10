from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    total_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    params: dict


class BacktestEngine:
    """
    VectorBT-based backtesting wrapper.
    Runs CENTINA's trend strategy on historical OHLCV data.

    Falls back to a pure-pandas implementation if vectorbt is unavailable,
    ensuring tests can run without the full dependency.
    """

    def __init__(self, initial_capital: float = 1000.0, fee_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self._vbt_available = self._check_vbt()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        ema_fast: int = 7,
        ema_slow: int = 25,
        rsi_low: float = 45.0,
        rsi_high: float = 65.0,
        atr_sl_mult: float = 1.0,
        atr_tp_mult: float = 1.5,
    ) -> BacktestResult:
        params = dict(
            ema_fast=ema_fast, ema_slow=ema_slow,
            rsi_low=rsi_low, rsi_high=rsi_high,
            atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
        )

        if self._vbt_available:
            return self._run_vbt(df, symbol, timeframe, params)
        return self._run_pandas(df, symbol, timeframe, params)

    # ── VectorBT path ─────────────────────────────────────────────────

    def _run_vbt(self, df: pd.DataFrame, symbol: str, tf: str, params: dict) -> BacktestResult:
        import vectorbt as vbt

        close  = df["close"].astype(float).values
        high   = df["high"].astype(float).values
        low    = df["low"].astype(float).values
        index  = pd.to_datetime(df["open_time"]) if "open_time" in df.columns else df.index

        import pandas_ta as ta
        close_s = pd.Series(close, index=index)
        high_s  = pd.Series(high,  index=index)
        low_s   = pd.Series(low,   index=index)

        ema_f = ta.ema(close_s, length=params["ema_fast"])
        ema_s = ta.ema(close_s, length=params["ema_slow"])
        rsi   = ta.rsi(close_s, length=14)
        atr   = ta.atr(high_s, low_s, close_s, length=14)

        entries = (
            (close_s > ema_f)
            & (ema_f > ema_s)
            & (rsi >= params["rsi_low"])
            & (rsi <= params["rsi_high"])
        )

        sl_stop = params["atr_sl_mult"] * atr / close_s
        tp_stop = params["atr_tp_mult"] * atr / close_s

        pf = vbt.Portfolio.from_signals(
            close_s,
            entries=entries,
            exits=~entries,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            fees=self.fee_pct,
            init_cash=self.initial_capital,
            freq="1d",
        )

        stats = pf.stats()
        return BacktestResult(
            symbol=symbol,
            timeframe=tf,
            total_trades=int(stats.get("Total Trades", 0)),
            win_rate=float(stats.get("Win Rate [%]", 0)) / 100,
            total_return_pct=float(stats.get("Total Return [%]", 0)),
            max_drawdown_pct=float(stats.get("Max Drawdown [%]", 0)),
            sharpe_ratio=float(stats.get("Sharpe Ratio", 0)),
            profit_factor=float(stats.get("Profit Factor", 0)) if stats.get("Profit Factor") else 0.0,
            params=params,
        )

    # ── Pandas fallback path ──────────────────────────────────────────

    def _run_pandas(self, df: pd.DataFrame, symbol: str, tf: str, params: dict) -> BacktestResult:
        import pandas_ta as ta

        close = df["close"].astype(float)
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)

        ema_f = ta.ema(close, length=params["ema_fast"])
        ema_s = ta.ema(close, length=params["ema_slow"])
        rsi   = ta.rsi(close, length=14)
        atr   = ta.atr(high, low, close, length=14)

        trades: list[dict] = []
        in_trade = False
        entry_price = sl = tp = 0.0

        for i in range(1, len(close)):
            if pd.isna(ema_f.iloc[i]) or pd.isna(rsi.iloc[i]):
                continue
            p = close.iloc[i]

            if not in_trade:
                if (
                    p > ema_f.iloc[i] > ema_s.iloc[i]
                    and params["rsi_low"] <= rsi.iloc[i] <= params["rsi_high"]
                ):
                    entry_price = p * (1 + self.fee_pct)
                    sl = p - params["atr_sl_mult"] * atr.iloc[i]
                    tp = p + params["atr_tp_mult"] * atr.iloc[i]
                    in_trade = True
            else:
                if p <= sl or p >= tp or (not (p > ema_f.iloc[i])):
                    exit_price = p * (1 - self.fee_pct)
                    trades.append({"pnl": exit_price - entry_price})
                    in_trade = False

        if not trades:
            return BacktestResult(symbol, tf, 0, 0, 0, 0, 0, 0, params)

        pnls  = [t["pnl"] for t in trades]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_ret = sum(pnls) / self.initial_capital * 100

        from src.learning.performance_tracker import PerformanceTracker
        pt  = PerformanceTracker()
        rep = pt.compute(pnls, self.initial_capital)

        return BacktestResult(
            symbol=symbol, timeframe=tf,
            total_trades=len(trades),
            win_rate=rep.win_rate,
            total_return_pct=round(total_ret, 2),
            max_drawdown_pct=round(rep.max_drawdown * 100, 2),
            sharpe_ratio=rep.sharpe_ratio,
            profit_factor=rep.profit_factor,
            params=params,
        )

    @staticmethod
    def _check_vbt() -> bool:
        try:
            import vectorbt  # noqa: F401
            return True
        except ImportError:
            logger.info("vectorbt not installed – using pandas fallback for backtests")
            return False
