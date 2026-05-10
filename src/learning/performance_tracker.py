from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class PerformanceReport:
    total_trades:  int
    win_rate:      float
    profit_factor: float
    total_pnl:     float
    avg_win:       float
    avg_loss:      float
    sharpe_ratio:  float
    calmar_ratio:  float
    max_drawdown:  float   # as fraction (0.15 = 15%)
    recovery_factor: float
    expectancy:    float   # avg expected $ per trade
    annualised_return: float

    def summary(self) -> str:
        return (
            f"Trades={self.total_trades}  WR={self.win_rate:.1%}  "
            f"PF={self.profit_factor:.2f}  Sharpe={self.sharpe_ratio:.2f}  "
            f"Calmar={self.calmar_ratio:.2f}  MDD={self.max_drawdown:.1%}  "
            f"PnL={self.total_pnl:+.2f}"
        )


class PerformanceTracker:
    """
    Computes institutional-grade metrics from a list of trade P&L values (USDT).

    Usage:
        tracker = PerformanceTracker()
        report  = tracker.compute(pnl_list, capital_baseline=1000.0)
    """

    RISK_FREE_RATE  = 0.05   # 5% annual
    PERIODS_PER_YEAR = 365   # daily compounding assumption

    def compute(
        self,
        pnl_series: Sequence[float],
        capital_baseline: float,
        periods_per_year: int | None = None,
    ) -> PerformanceReport:
        pnls = list(pnl_series)
        n = len(pnls)
        if n == 0:
            return self._empty()

        wins   = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p < 0]

        win_rate      = len(wins) / n
        avg_win       = sum(wins)  / len(wins)  if wins   else 0.0
        avg_loss      = sum(losses)/ len(losses) if losses else 0.0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
        expectancy    = win_rate * avg_win + (1 - win_rate) * avg_loss
        total_pnl     = sum(pnls)

        # Equity curve & drawdown
        equity_curve = self._equity_curve(capital_baseline, pnls)
        max_dd       = self._max_drawdown(equity_curve)

        # Returns as fractions for ratio calculations
        ppa = periods_per_year or self.PERIODS_PER_YEAR
        returns_frac = [p / capital_baseline for p in pnls]
        sharpe  = self._sharpe(returns_frac, ppa)
        calmar  = self._calmar(returns_frac, max_dd, ppa)
        ann_ret = self._annualised_return(returns_frac, ppa)
        rf      = total_pnl / (capital_baseline * max_dd) if max_dd > 0 else float("inf")

        return PerformanceReport(
            total_trades=n,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 3),
            total_pnl=round(total_pnl, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            sharpe_ratio=round(sharpe, 3),
            calmar_ratio=round(calmar, 3),
            max_drawdown=round(max_dd, 4),
            recovery_factor=round(rf, 3),
            expectancy=round(expectancy, 2),
            annualised_return=round(ann_ret, 4),
        )

    # ── Metrics ──────────────────────────────────────────────────────

    def _equity_curve(self, start: float, pnls: list[float]) -> list[float]:
        curve = [start]
        for p in pnls:
            curve.append(curve[-1] + p)
        return curve

    def _max_drawdown(self, equity: list[float]) -> float:
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _sharpe(self, returns: list[float], ppa: int) -> float:
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        if std_r == 0:
            return 0.0
        daily_rf = (1 + self.RISK_FREE_RATE) ** (1 / ppa) - 1
        return (mean_r - daily_rf) / std_r * math.sqrt(ppa)

    def _calmar(self, returns: list[float], max_dd: float, ppa: int) -> float:
        ann = self._annualised_return(returns, ppa)
        return ann / max_dd if max_dd > 0 else float("inf")

    def _annualised_return(self, returns: list[float], ppa: int) -> float:
        n = len(returns)
        if n == 0:
            return 0.0
        total_return = 1.0
        for r in returns:
            total_return *= (1 + r)
        return total_return ** (ppa / n) - 1

    @staticmethod
    def _empty() -> PerformanceReport:
        return PerformanceReport(
            total_trades=0, win_rate=0.0, profit_factor=0.0, total_pnl=0.0,
            avg_win=0.0, avg_loss=0.0, sharpe_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, recovery_factor=0.0, expectancy=0.0, annualised_return=0.0,
        )
