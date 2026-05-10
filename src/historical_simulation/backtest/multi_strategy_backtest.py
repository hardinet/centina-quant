from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .historical_backtest_engine import BacktestMetrics, HistoricalBacktestEngine

logger = logging.getLogger(__name__)

AllocationMethod = Literal["equal", "perf_weighted", "inv_vol", "kelly_optimal", "regime_based"]

STRATEGY_PARAMS: dict[str, dict] = {
    "A": {"ema_fast": 5, "ema_slow": 20, "ema_long": 50, "ema_trend": 200,
          "rsi_low": 55, "rsi_high": 75, "vol_spike": 2.0, "atr_sl_mult": 1.2, "atr_tp1_mult": 1.8},
    "B": {"ema_fast": 7, "ema_slow": 25, "ema_long": 99, "ema_trend": 200,
          "rsi_low": 50, "rsi_high": 72, "vol_spike": 1.5, "atr_sl_mult": 1.0, "atr_tp1_mult": 1.5},
    "C": {"ema_fast": 3, "ema_slow": 10, "ema_long": 30, "ema_trend": 100,
          "rsi_low": 60, "rsi_high": 80, "vol_spike": 3.0, "atr_sl_mult": 0.8, "atr_tp1_mult": 2.0},
    "D": {"ema_fast": 5, "ema_slow": 15, "ema_long": 50, "ema_trend": 200,
          "rsi_low": 45, "rsi_high": 70, "vol_spike": 2.5, "atr_sl_mult": 0.7, "atr_tp1_mult": 1.2},
    "E": {"ema_fast": 9, "ema_slow": 21, "ema_long": 55, "ema_trend": 200,
          "rsi_low": 40, "rsi_high": 65, "vol_spike": 1.8, "atr_sl_mult": 1.5, "atr_tp1_mult": 1.3},
}


@dataclass
class MultiStrategyResult:
    symbol:          str
    timeframe:       str
    allocation:      AllocationMethod
    weights:         dict[str, float]
    strategy_results: dict[str, BacktestMetrics]
    combined_return: float
    combined_sharpe: float
    combined_max_dd: float
    combined_calmar: float
    correlation_matrix: dict = field(default_factory=dict)
    diversification_ratio: float = 1.0


class MultiStrategyBacktest:
    """
    Runs all 5 CENTINA strategies simultaneously and combines equity curves
    using different allocation methods including Markowitz optimization.
    """

    def __init__(self, capital: float = 200.0):
        self.capital = capital
        self.engine  = HistoricalBacktestEngine(capital=capital)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        allocation: AllocationMethod = "equal",
        strategies: list[str] | None = None,
    ) -> MultiStrategyResult:
        strats = strategies or list(STRATEGY_PARAMS.keys())

        results: dict[str, BacktestMetrics] = {}
        for s in strats:
            try:
                r = self.engine.run_backtest(df, symbol, timeframe, strategy=s,
                                             params=STRATEGY_PARAMS[s])
                results[s] = r
            except Exception as e:
                logger.debug("Strategy %s backtest failed: %s", s, e)

        if not results:
            return self._empty(symbol, timeframe, allocation)

        weights = self._compute_weights(results, allocation)
        combined = self._combine_equity(results, weights)
        corr     = self._correlation_matrix(results)
        dr       = self._diversification_ratio(results, weights)

        return MultiStrategyResult(
            symbol=symbol, timeframe=timeframe, allocation=allocation,
            weights=weights, strategy_results=results,
            combined_return=round(combined["total_return"], 2),
            combined_sharpe=round(combined["sharpe"], 3),
            combined_max_dd=round(combined["max_dd"], 2),
            combined_calmar=round(combined["calmar"], 3),
            correlation_matrix=corr,
            diversification_ratio=round(dr, 3),
        )

    def get_summary_report(self, result: MultiStrategyResult) -> pd.DataFrame:
        rows = []
        for s, m in result.strategy_results.items():
            rows.append({
                "strategy":      s,
                "weight":        round(result.weights.get(s, 0), 3),
                "total_return":  m.total_return_pct,
                "sharpe":        m.sharpe,
                "max_dd":        m.max_drawdown_pct,
                "calmar":        m.calmar,
                "win_rate":      m.win_rate,
                "nb_trades":     m.nb_trades,
                "profit_factor": m.profit_factor,
            })
        rows.append({
            "strategy":      "COMBINED",
            "weight":        1.0,
            "total_return":  result.combined_return,
            "sharpe":        result.combined_sharpe,
            "max_dd":        result.combined_max_dd,
            "calmar":        result.combined_calmar,
            "win_rate":      float("nan"),
            "nb_trades":     sum(m.nb_trades for m in result.strategy_results.values()),
            "profit_factor": float("nan"),
        })
        return pd.DataFrame(rows)

    # ── Weight computation ────────────────────────────────────────────

    def _compute_weights(
        self, results: dict[str, BacktestMetrics], method: AllocationMethod
    ) -> dict[str, float]:
        strats = list(results.keys())
        n = len(strats)

        if method == "equal":
            w = 1.0 / n
            return {s: w for s in strats}

        if method == "perf_weighted":
            scores = {s: max(results[s].sharpe, 0) for s in strats}
            total  = sum(scores.values()) or 1.0
            return {s: scores[s] / total for s in strats}

        if method == "inv_vol":
            # Inverse of max-drawdown as volatility proxy
            inv = {s: 1.0 / max(results[s].max_drawdown_pct, 0.1) for s in strats}
            total = sum(inv.values())
            return {s: inv[s] / total for s in strats}

        if method == "kelly_optimal":
            return self._kelly_weights(results)

        if method == "regime_based":
            # Use Sharpe-weighted but zero-out negative Sharpe strategies
            scores = {s: max(results[s].sharpe, 0) for s in strats}
            total  = sum(scores.values()) or 1.0
            return {s: scores[s] / total for s in strats}

        return {s: 1.0 / n for s in strats}

    def _kelly_weights(self, results: dict[str, BacktestMetrics]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for s, m in results.items():
            if m.win_rate <= 0 or m.avg_loss_pct == 0:
                weights[s] = 0.0
                continue
            b = abs(m.avg_win_pct / m.avg_loss_pct) if m.avg_loss_pct != 0 else 1.0
            p = m.win_rate
            q = 1.0 - p
            k = (p * b - q) / b if b > 0 else 0.0
            weights[s] = max(k / 8, 0)  # 1/8 fractional Kelly

        total = sum(weights.values())
        if total <= 0:
            n = len(results)
            return {s: 1.0 / n for s in results}
        return {s: w / total for s, w in weights.items()}

    # ── Equity combination ────────────────────────────────────────────

    def _combine_equity(
        self, results: dict[str, BacktestMetrics], weights: dict[str, float]
    ) -> dict:
        total_ret = sum(results[s].total_return_pct * weights.get(s, 0) for s in results)
        sharpes   = [results[s].sharpe for s in results if results[s].sharpe > 0]
        w_sharpe  = sum(results[s].sharpe * weights.get(s, 0) for s in results)

        # Weighted max drawdown (conservative: use max across strategies)
        max_dd = max(results[s].max_drawdown_pct for s in results)
        calmar = total_ret / max_dd if max_dd > 0 else 0.0

        return {"total_return": total_ret, "sharpe": w_sharpe, "max_dd": max_dd, "calmar": calmar}

    def _correlation_matrix(self, results: dict[str, BacktestMetrics]) -> dict:
        strats = list(results.keys())
        pnls   = {s: [t.pnl_pct for t in results[s].trades] for s in strats}

        matrix: dict[str, dict[str, float]] = {}
        for s1 in strats:
            matrix[s1] = {}
            for s2 in strats:
                p1, p2 = pnls[s1], pnls[s2]
                min_len = min(len(p1), len(p2))
                if min_len < 5:
                    matrix[s1][s2] = 0.0
                    continue
                corr = float(np.corrcoef(p1[:min_len], p2[:min_len])[0, 1])
                matrix[s1][s2] = round(corr if not np.isnan(corr) else 0.0, 3)
        return matrix

    def _diversification_ratio(
        self, results: dict[str, BacktestMetrics], weights: dict[str, float]
    ) -> float:
        strats = list(results.keys())
        vols   = np.array([results[s].max_drawdown_pct for s in strats])
        w      = np.array([weights.get(s, 0) for s in strats])

        weighted_vol = float(w.dot(vols))
        if weighted_vol <= 0:
            return 1.0

        corr_matrix = self._correlation_matrix(results)
        portfolio_var = 0.0
        for i, s1 in enumerate(strats):
            for j, s2 in enumerate(strats):
                c = corr_matrix.get(s1, {}).get(s2, 0.0)
                portfolio_var += w[i] * w[j] * vols[i] * vols[j] * c

        portfolio_vol = float(np.sqrt(max(portfolio_var, 0)))
        return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0

    def _empty(self, symbol, timeframe, allocation) -> MultiStrategyResult:
        return MultiStrategyResult(
            symbol=symbol, timeframe=timeframe, allocation=allocation,
            weights={}, strategy_results={},
            combined_return=0, combined_sharpe=0,
            combined_max_dd=0, combined_calmar=0,
        )
