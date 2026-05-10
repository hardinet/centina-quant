from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..backtest.historical_backtest_engine import BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    strategy:           str
    symbol:             str
    sharpe_ratio:       float
    deflated_sharpe:    float     # DSR — corrects for multiple testing
    psr:                float     # Probabilistic Sharpe Ratio
    min_track_record:   float     # Minimum Track Record Length (years)
    actual_track_record: float
    is_sufficient:      bool      # actual_track_record >= min_track_record
    t_stat_return:      float
    p_value_return:     float
    is_significant:     bool      # p_value < 0.05
    jarque_bera_p:      float     # normality of returns
    skewness:           float
    kurtosis:           float
    autocorrelation:    float     # lag-1 autocorrelation of returns


class StatisticalValidator:
    """
    Bailey et al. DSR + PSR methodology.
    Validates that backtest results are statistically sound and not flukes.
    """

    BENCHMARK_SHARPE = 0.0     # null hypothesis: SR <= 0
    N_TRIALS_FACTOR  = 20      # assumed number of parameter trials before choosing

    def validate(self, metrics: BacktestMetrics) -> ValidationResult:
        trades = metrics.trades
        if not trades or len(trades) < 5:
            return self._empty(metrics)

        returns = np.array([t.pnl_pct / 100 for t in trades])
        n       = len(returns)
        mu      = float(returns.mean())
        sigma   = float(returns.std(ddof=1)) if n > 1 else 1e-10

        sr_annual = self._annualized_sharpe(returns, metrics.avg_duration_h)
        dsr       = self._deflated_sharpe(sr_annual, n)
        psr       = self._probabilistic_sharpe(sr_annual, n, self.BENCHMARK_SHARPE)
        mtrl      = self._min_track_record_length(sr_annual, n)

        years = (n * metrics.avg_duration_h) / (365 * 24)

        t_stat = float(mu / sigma * math.sqrt(n)) if sigma > 0 else 0.0
        p_val  = float(stats.t.sf(abs(t_stat), df=n-1) * 2)

        jb_stat, jb_p = stats.jarque_bera(returns) if n >= 8 else (0, 1)
        skew  = float(stats.skew(returns))
        kurt  = float(stats.kurtosis(returns))

        ac = float(pd.Series(returns).autocorr(lag=1)) if n > 10 else 0.0

        return ValidationResult(
            strategy=metrics.strategy, symbol=metrics.symbol,
            sharpe_ratio=round(sr_annual, 3),
            deflated_sharpe=round(dsr, 3),
            psr=round(psr, 4),
            min_track_record=round(mtrl, 2),
            actual_track_record=round(years, 2),
            is_sufficient=years >= mtrl,
            t_stat_return=round(t_stat, 3),
            p_value_return=round(p_val, 4),
            is_significant=p_val < 0.05,
            jarque_bera_p=round(float(jb_p), 4),
            skewness=round(skew, 3),
            kurtosis=round(kurt, 3),
            autocorrelation=round(ac, 3) if not math.isnan(ac) else 0.0,
        )

    def validate_batch(self, results: list[BacktestMetrics]) -> pd.DataFrame:
        rows = []
        for m in results:
            v = self.validate(m)
            rows.append({
                "strategy":          v.strategy,
                "symbol":            v.symbol,
                "sharpe":            v.sharpe_ratio,
                "dsr":               v.deflated_sharpe,
                "psr":               v.psr,
                "is_significant":    v.is_significant,
                "is_sufficient":     v.is_sufficient,
                "skewness":          v.skewness,
                "kurtosis":          v.kurtosis,
                "autocorrelation":   v.autocorrelation,
                "p_value":           v.p_value_return,
            })
        return pd.DataFrame(rows)

    # ── Core formulas ─────────────────────────────────────────────────

    def _annualized_sharpe(self, returns: np.ndarray, avg_duration_h: float) -> float:
        n = len(returns)
        mu    = returns.mean()
        sigma = returns.std(ddof=1)
        if sigma <= 0:
            return 0.0
        trades_per_year = (365 * 24) / max(avg_duration_h, 1)
        return float(mu / sigma * math.sqrt(trades_per_year))

    def _deflated_sharpe(self, sr: float, n: int) -> float:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
        Corrects for selection bias under multiple trials.
        """
        if n < 5:
            return sr

        V = self.N_TRIALS_FACTOR

        expected_max = (
            (1 - 0.5772) * stats.norm.ppf(1 - 1.0 / V)
            + 0.5772 * stats.norm.ppf(1 - 1.0 / (V * math.e))
        )

        sigma_sr = math.sqrt((1 + 0.5 * sr**2) / (n - 1))
        z = (sr - expected_max) / (sigma_sr + 1e-10)
        return float(stats.norm.cdf(z))

    def _probabilistic_sharpe(self, sr: float, n: int, sr_star: float = 0.0) -> float:
        """Probability that the true Sharpe > sr_star given observed sr and n trades."""
        if n < 5:
            return 0.5
        sigma_sr = math.sqrt((1 + 0.5 * sr**2) / (n - 1))
        z = (sr - sr_star) / (sigma_sr + 1e-10)
        return float(stats.norm.cdf(z))

    def _min_track_record_length(self, sr: float, n: int, freq_per_year: float = 252) -> float:
        """
        Minimum Track Record Length (Bailey 2014).
        Returns years of data needed to confirm the SR at 95% confidence.
        """
        if sr <= 0:
            return 99.0
        z95 = 1.6449
        mtrl_trades = (z95 / sr) ** 2 * (1 + 0.5 * sr**2)
        return float(mtrl_trades / freq_per_year)

    def _empty(self, m: BacktestMetrics) -> ValidationResult:
        return ValidationResult(
            strategy=m.strategy, symbol=m.symbol,
            sharpe_ratio=0, deflated_sharpe=0, psr=0,
            min_track_record=99, actual_track_record=0, is_sufficient=False,
            t_stat_return=0, p_value_return=1, is_significant=False,
            jarque_bera_p=1, skewness=0, kurtosis=0, autocorrelation=0,
        )
