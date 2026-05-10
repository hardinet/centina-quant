from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..backtest.historical_backtest_engine import BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class OverfittingReport:
    strategy:      str
    symbol:        str
    pbo:           float      # Probability of Backtest Overfitting (0-1)
    is_overfit:    bool       # pbo > 0.5
    oos_degradation: float    # (IS_sharpe - OOS_sharpe) / IS_sharpe
    param_sensitivity: dict[str, float] = field(default_factory=dict)
    verdict:       str = ""


class OverfittingDetector:
    """
    Implements Bailey et al. (2016) Probability of Backtest Overfitting (PBO)
    via Combinatorially Symmetric Cross-Validation (CSCV).
    Also checks parameter sensitivity and IS/OOS Sharpe degradation.
    """

    def detect(
        self,
        is_metrics: BacktestMetrics,
        oos_metrics: BacktestMetrics | None = None,
        param_sweep: list[BacktestMetrics] | None = None,
    ) -> OverfittingReport:
        pbo         = self._compute_pbo(param_sweep or [is_metrics])
        is_sharpe   = is_metrics.sharpe
        oos_sharpe  = oos_metrics.sharpe if oos_metrics else is_sharpe * 0.6
        degradation = ((is_sharpe - oos_sharpe) / abs(is_sharpe)) if is_sharpe != 0 else 0.0
        sensitivity = self._param_sensitivity(param_sweep or [])

        is_overfit = pbo > 0.5 or degradation > 0.5

        verdict = self._make_verdict(pbo, degradation, is_sharpe, oos_sharpe)

        return OverfittingReport(
            strategy=is_metrics.strategy, symbol=is_metrics.symbol,
            pbo=round(pbo, 3), is_overfit=is_overfit,
            oos_degradation=round(degradation, 3),
            param_sensitivity=sensitivity,
            verdict=verdict,
        )

    def detect_batch(
        self,
        is_list: list[BacktestMetrics],
        oos_list: list[BacktestMetrics] | None = None,
    ) -> pd.DataFrame:
        rows = []
        oos_map = {m.strategy: m for m in (oos_list or [])}
        for m in is_list:
            r = self.detect(m, oos_map.get(m.strategy))
            rows.append({
                "strategy":   r.strategy,
                "symbol":     r.symbol,
                "pbo":        r.pbo,
                "is_overfit": r.is_overfit,
                "oos_degradation": r.oos_degradation,
                "verdict":    r.verdict,
            })
        return pd.DataFrame(rows)

    # ── PBO via CSCV ─────────────────────────────────────────────────

    def _compute_pbo(self, sweep: list[BacktestMetrics]) -> float:
        """
        Simplified CSCV: partition trade PnL series into S subsets,
        test every combination of IS vs OOS halves.
        """
        if len(sweep) < 4:
            return 0.5

        pnl_matrix = []
        for m in sweep:
            if m.trades:
                pnl_matrix.append([t.pnl_pct for t in m.trades])

        if len(pnl_matrix) < 4:
            return 0.5

        min_len = min(len(p) for p in pnl_matrix)
        if min_len < 4:
            return 0.5

        M = np.array([p[:min_len] for p in pnl_matrix])  # shape: (n_configs, n_trades)
        S = min(8, min_len)
        n_configs = M.shape[0]

        chunk = min_len // S
        if chunk < 1:
            return 0.5

        chunks = [M[:, i*chunk:(i+1)*chunk] for i in range(S)]

        n_overfit = 0
        n_total   = 0

        half = S // 2
        for is_idx in itertools.combinations(range(S), half):
            oos_idx = tuple(i for i in range(S) if i not in is_idx)

            is_data  = np.concatenate([chunks[i] for i in is_idx],  axis=1)
            oos_data = np.concatenate([chunks[i] for i in oos_idx], axis=1)

            is_sharpes  = [self._quick_sharpe(is_data[j])  for j in range(n_configs)]
            oos_sharpes = [self._quick_sharpe(oos_data[j]) for j in range(n_configs)]

            best_is  = int(np.argmax(is_sharpes))
            oos_rank = sorted(range(n_configs), key=lambda x: oos_sharpes[x], reverse=True)
            oos_pos  = oos_rank.index(best_is)

            # Relative rank 0 = best OOS, n-1 = worst
            r = oos_pos / max(n_configs - 1, 1)
            n_overfit += 1 if r > 0.5 else 0
            n_total   += 1

        return float(n_overfit / n_total) if n_total > 0 else 0.5

    def _quick_sharpe(self, pnl: np.ndarray) -> float:
        std = pnl.std()
        return float(pnl.mean() / std * math.sqrt(252)) if std > 0 else 0.0

    def _param_sensitivity(self, sweep: list[BacktestMetrics]) -> dict[str, float]:
        if len(sweep) < 3:
            return {}
        sharpes = np.array([m.sharpe for m in sweep])
        # Coefficient of variation: high CV → sensitive to parameters
        cv = float(sharpes.std() / abs(sharpes.mean())) if sharpes.mean() != 0 else 0.0
        return {"sharpe_cv": round(cv, 3)}

    def _make_verdict(self, pbo: float, degradation: float, is_sr: float, oos_sr: float) -> str:
        if pbo > 0.7:
            return "HIGH OVERFITTING — do not use these parameters live"
        if pbo > 0.5:
            return "MODERATE OVERFITTING — reduce complexity or expand dataset"
        if degradation > 0.5:
            return "IS/OOS degradation >50% — revisit feature engineering"
        if is_sr < 0.5:
            return "WEAK STRATEGY — Sharpe below 0.5 even in-sample"
        if oos_sr > is_sr * 0.7:
            return "ROBUST — OOS performance >70% of IS, low overfitting risk"
        return "ACCEPTABLE — monitor live closely"
