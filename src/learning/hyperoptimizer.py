from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    best_params:    dict[str, Any]
    best_value:     float          # Sharpe ratio of best trial
    n_trials:       int
    study_name:     str


class HyperOptimizer:
    """
    Bayesian parameter optimisation via Optuna.

    Optimises the CENTINA strategy parameters to maximise Sharpe ratio.
    Safe to run in a background task (nightly self-correction cycle).
    """

    PARAM_SPACE = {
        "ema_fast":     (5,   15),
        "ema_slow":     (20,  35),
        "rsi_low":      (35,  50),
        "rsi_high":     (60,  75),
        "atr_sl_mult":  (0.5, 2.0),
        "atr_tp_mult":  (1.0, 4.0),
    }

    def __init__(self, n_trials: int = 100, timeout_sec: int = 300):
        self.n_trials    = n_trials
        self.timeout_sec = timeout_sec

    def optimise(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        capital: float = 1000.0,
        study_name: str | None = None,
    ) -> OptimizationResult:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise RuntimeError("optuna is required for hyperoptimisation: pip install optuna")

        from src.learning.backtest_engine import BacktestEngine
        engine = BacktestEngine(initial_capital=capital)
        name   = study_name or f"centina_{symbol}_{timeframe}"

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "ema_fast":    trial.suggest_int("ema_fast",    *self.PARAM_SPACE["ema_fast"]),
                "ema_slow":    trial.suggest_int("ema_slow",    *self.PARAM_SPACE["ema_slow"]),
                "rsi_low":     trial.suggest_float("rsi_low",   *self.PARAM_SPACE["rsi_low"]),
                "rsi_high":    trial.suggest_float("rsi_high",  *self.PARAM_SPACE["rsi_high"]),
                "atr_sl_mult": trial.suggest_float("atr_sl_mult", *self.PARAM_SPACE["atr_sl_mult"]),
                "atr_tp_mult": trial.suggest_float("atr_tp_mult", *self.PARAM_SPACE["atr_tp_mult"]),
            }
            if params["ema_fast"] >= params["ema_slow"]:
                return -999.0
            if params["rsi_low"] >= params["rsi_high"]:
                return -999.0

            try:
                result = engine.run(df, symbol, timeframe, **params)
                if result.total_trades < 5:
                    return -999.0
                return result.sharpe_ratio
            except Exception:
                return -999.0

        study = optuna.create_study(
            direction="maximize",
            study_name=name,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_sec,
            show_progress_bar=False,
            n_jobs=1,
        )

        best = study.best_trial
        logger.info(
            "HyperOptimizer: %s best Sharpe=%.3f params=%s",
            name, best.value, best.params,
        )
        return OptimizationResult(
            best_params=best.params,
            best_value=round(best.value, 4),
            n_trials=len(study.trials),
            study_name=name,
        )
