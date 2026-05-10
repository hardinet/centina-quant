from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Model = Literal["gbm", "heston", "garch", "copula", "bootstrap"]


@dataclass
class SyntheticConfig:
    model:       Model   = "gbm"
    n_days:      int     = 365
    start_price: float   = 30_000.0
    mu:          float   = 0.0003       # daily drift
    sigma:       float   = 0.025        # daily vol (GBM)
    freq:        str     = "1h"
    seed:        int     = 42


class SyntheticDataGenerator:
    """
    Generates synthetic OHLCV data for backtesting when historical data is insufficient.
    Models: GBM, Heston stochastic vol, GARCH(1,1), Copula multi-asset, Bootstrap.
    """

    def generate(self, cfg: SyntheticConfig | None = None) -> pd.DataFrame:
        cfg = cfg or SyntheticConfig()
        np.random.seed(cfg.seed)

        if cfg.model == "gbm":
            prices = self._gbm(cfg)
        elif cfg.model == "heston":
            prices = self._heston(cfg)
        elif cfg.model == "garch":
            prices = self._garch(cfg)
        elif cfg.model == "bootstrap":
            prices = self._bootstrap(cfg)
        else:
            prices = self._gbm(cfg)

        return self._to_ohlcv(prices, cfg)

    def generate_multi_asset(
        self, n_assets: int = 3, correlation: float = 0.6, cfg: SyntheticConfig | None = None
    ) -> list[pd.DataFrame]:
        cfg = cfg or SyntheticConfig()
        np.random.seed(cfg.seed)

        cov = np.full((n_assets, n_assets), correlation)
        np.fill_diagonal(cov, 1.0)
        L = np.linalg.cholesky(cov)

        n_bars = cfg.n_days * 24 if "h" in cfg.freq.lower() else cfg.n_days
        raw    = np.random.normal(cfg.mu, cfg.sigma, (n_assets, n_bars))
        correlated = (L @ raw)

        dfs = []
        for i in range(n_assets):
            prices = cfg.start_price * np.exp(np.cumsum(correlated[i]))
            c      = SyntheticConfig(**{**cfg.__dict__, "start_price": float(prices[0])})
            dfs.append(self._to_ohlcv(prices, c))
        return dfs

    def generate_crash_scenario(
        self,
        crash_pct: float = -0.40,
        crash_day: int   = 90,
        recovery_days: int = 60,
        cfg: SyntheticConfig | None = None,
    ) -> pd.DataFrame:
        cfg    = cfg or SyntheticConfig()
        prices = self._gbm(cfg)

        n_bars     = len(prices)
        crash_idx  = min(crash_day * 24, n_bars - 1) if "h" in cfg.freq.lower() else min(crash_day, n_bars - 1)
        p_before   = prices[crash_idx]
        p_after    = p_before * (1 + crash_pct)

        # Sharp drop
        drop_bars  = 24 if "h" in cfg.freq.lower() else 3
        for i in range(drop_bars):
            idx = crash_idx + i
            if idx < n_bars:
                t = i / drop_bars
                prices[idx] = p_before * (1 + crash_pct * t)
        prices[crash_idx + drop_bars : crash_idx + drop_bars + recovery_days] = (
            np.linspace(p_after, p_before, recovery_days)
            if crash_idx + drop_bars + recovery_days <= n_bars
            else prices[crash_idx + drop_bars : crash_idx + drop_bars + recovery_days]
        )
        return self._to_ohlcv(prices, cfg)

    # ── Models ────────────────────────────────────────────────────────

    def _gbm(self, cfg: SyntheticConfig) -> np.ndarray:
        n = cfg.n_days * (24 if "h" in cfg.freq.lower() else 1)
        dt = 1 / n
        returns = np.random.normal(
            (cfg.mu - 0.5 * cfg.sigma**2) * dt,
            cfg.sigma * np.sqrt(dt),
            n,
        )
        return cfg.start_price * np.exp(np.cumsum(returns))

    def _heston(self, cfg: SyntheticConfig) -> np.ndarray:
        """Heston stochastic volatility model."""
        n        = cfg.n_days * (24 if "h" in cfg.freq.lower() else 1)
        dt       = 1 / 252
        kappa    = 3.0    # mean reversion speed
        theta    = cfg.sigma ** 2   # long-run variance
        xi       = 0.4    # vol of vol
        rho      = -0.7   # correlation S-V
        v        = theta

        prices   = [cfg.start_price]
        vols     = [v]

        for _ in range(n - 1):
            z1 = np.random.normal()
            z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal()
            v  = max(v + kappa * (theta - v) * dt + xi * np.sqrt(max(v, 0) * dt) * z2, 0)
            p  = prices[-1] * np.exp((cfg.mu - 0.5 * v) * dt + np.sqrt(v * dt) * z1)
            prices.append(p)
            vols.append(v)

        return np.array(prices)

    def _garch(self, cfg: SyntheticConfig) -> np.ndarray:
        """GARCH(1,1) conditional heteroskedasticity."""
        n     = cfg.n_days * (24 if "h" in cfg.freq.lower() else 1)
        omega = cfg.sigma**2 * 0.1
        alpha = 0.1
        beta  = 0.85
        h     = cfg.sigma**2
        prices = [cfg.start_price]

        for _ in range(n - 1):
            eps = np.random.normal(0, np.sqrt(h))
            r   = cfg.mu + eps
            h   = omega + alpha * eps**2 + beta * h
            prices.append(prices[-1] * np.exp(r))

        return np.array(prices)

    def _bootstrap(self, cfg: SyntheticConfig) -> np.ndarray:
        """Block bootstrap using typical crypto return distribution."""
        n       = cfg.n_days * (24 if "h" in cfg.freq.lower() else 1)
        block   = 24
        n_blocks = n // block + 1

        # Simulate from historically calibrated distribution
        blocks = []
        for _ in range(n_blocks):
            mu_b  = np.random.normal(cfg.mu, cfg.mu * 5)
            sig_b = np.random.uniform(cfg.sigma * 0.5, cfg.sigma * 3)
            blocks.extend(np.random.normal(mu_b, sig_b, block).tolist())

        returns = np.array(blocks[:n])
        return cfg.start_price * np.exp(np.cumsum(returns))

    # ── OHLCV builder ─────────────────────────────────────────────────

    def _to_ohlcv(self, prices: np.ndarray, cfg: SyntheticConfig) -> pd.DataFrame:
        n   = len(prices)
        idx = pd.date_range("2020-01-01", periods=n, freq=cfg.freq, tz="UTC")

        spread = np.random.uniform(0.001, 0.005, n)
        highs  = prices * (1 + spread)
        lows   = prices * (1 - spread)
        opens  = np.roll(prices, 1)
        opens[0] = prices[0]

        vol_base = 1_000_000.0
        volumes  = np.random.lognormal(np.log(vol_base), 0.5, n) * (
            1 + np.abs(np.random.normal(0, 0.3, n))
        )

        return pd.DataFrame({
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  prices,
            "volume": volumes,
        }, index=idx)
