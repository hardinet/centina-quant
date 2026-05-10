from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .historical_backtest_engine import BacktestMetrics, HistoricalBacktestEngine

logger = logging.getLogger(__name__)

WFMethod = Literal["rolling", "expanding", "anchored"]

PARAM_SPACE = {
    "ema_fast":     [5, 7, 9],
    "ema_slow":     [20, 25, 30],
    "ema_long":     [75, 99, 120],
    "rsi_low":      [45.0, 50.0, 55.0],
    "rsi_high":     [68.0, 72.0, 76.0],
    "atr_sl_mult":  [0.8, 1.0, 1.2],
    "vol_spike":    [1.3, 1.5, 2.0],
}


@dataclass
class WalkForwardFold:
    fold_id:       int
    train_start:   str
    train_end:     str
    test_start:    str
    test_end:      str
    best_params:   dict
    train_sharpe:  float
    test_sharpe:   float
    test_return:   float
    test_max_dd:   float
    overfitting:   bool       # test_sharpe < train_sharpe * 0.5


@dataclass
class WalkForwardResult:
    symbol:        str
    strategy:      str
    method:        WFMethod
    n_folds:       int
    folds:         list[WalkForwardFold] = field(default_factory=list)
    mean_test_sharpe:  float = 0.0
    mean_train_sharpe: float = 0.0
    efficiency_ratio:  float = 0.0    # mean_test / mean_train
    stable_params:     dict = field(default_factory=dict)
    recommended_params: dict = field(default_factory=dict)


class WalkForwardOptimizer:
    """
    Walk-forward analysis: train on in-sample, validate on out-of-sample.
    Supports rolling, expanding, and anchored windows.
    Identifies stable parameter sets across folds.
    """

    def __init__(self, capital: float = 200.0):
        self.capital = capital
        self.engine  = HistoricalBacktestEngine(capital=capital)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy: str = "B",
        method: WFMethod = "rolling",
        train_months: int = 12,
        test_months: int = 3,
        n_folds: int = 8,
        top_n_params: int = 5,
    ) -> WalkForwardResult:
        folds_defs = self._build_folds(df, method, train_months, test_months, n_folds)
        if not folds_defs:
            return WalkForwardResult(symbol=symbol, strategy=strategy, method=method, n_folds=0)

        folds: list[WalkForwardFold] = []
        for fold_id, (tr_s, tr_e, te_s, te_e) in enumerate(folds_defs):
            train_df = df.loc[tr_s:tr_e]
            test_df  = df.loc[te_s:te_e]

            if len(train_df) < 300 or len(test_df) < 30:
                continue

            best_p, train_sharpe = self._optimize_in_sample(train_df, symbol, strategy, top_n_params)
            test_metrics = self.engine.run_backtest(test_df, symbol, "1d", strategy, params=best_p)

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=str(tr_s.date()), train_end=str(tr_e.date()),
                test_start=str(te_s.date()),  test_end=str(te_e.date()),
                best_params=best_p,
                train_sharpe=round(train_sharpe, 3),
                test_sharpe=round(test_metrics.sharpe, 3),
                test_return=round(test_metrics.total_return_pct, 2),
                test_max_dd=round(test_metrics.max_drawdown_pct, 2),
                overfitting=test_metrics.sharpe < train_sharpe * 0.5,
            )
            folds.append(fold)

        if not folds:
            return WalkForwardResult(symbol=symbol, strategy=strategy, method=method, n_folds=0)

        mean_train = float(np.mean([f.train_sharpe for f in folds]))
        mean_test  = float(np.mean([f.test_sharpe  for f in folds]))
        eff        = mean_test / mean_train if mean_train > 0 else 0.0

        stable = self._find_stable_params(folds)
        recommended = self._recommend_params(folds)

        return WalkForwardResult(
            symbol=symbol, strategy=strategy, method=method,
            n_folds=len(folds), folds=folds,
            mean_test_sharpe=round(mean_test, 3),
            mean_train_sharpe=round(mean_train, 3),
            efficiency_ratio=round(eff, 3),
            stable_params=stable,
            recommended_params=recommended,
        )

    def get_report(self, result: WalkForwardResult) -> pd.DataFrame:
        rows = []
        for f in result.folds:
            rows.append({
                "fold":         f.fold_id,
                "train_period": f"{f.train_start} → {f.train_end}",
                "test_period":  f"{f.test_start} → {f.test_end}",
                "train_sharpe": f.train_sharpe,
                "test_sharpe":  f.test_sharpe,
                "test_return":  f.test_return,
                "test_max_dd":  f.test_max_dd,
                "overfitting":  f.overfitting,
            })
        return pd.DataFrame(rows)

    # ── Private ───────────────────────────────────────────────────────

    def _build_folds(
        self,
        df: pd.DataFrame,
        method: WFMethod,
        train_months: int,
        test_months: int,
        n_folds: int,
    ) -> list[tuple]:
        if df.empty:
            return []

        dates = df.index
        total_days   = (dates[-1] - dates[0]).days
        train_days   = train_months * 30
        test_days    = test_months  * 30
        step_days    = test_days

        folds = []
        for i in range(n_folds):
            te_end   = dates[-1] - pd.Timedelta(days=i * step_days)
            te_start = te_end   - pd.Timedelta(days=test_days)

            if method == "rolling":
                tr_end   = te_start - pd.Timedelta(days=1)
                tr_start = tr_end   - pd.Timedelta(days=train_days)
            elif method == "expanding":
                tr_end   = te_start - pd.Timedelta(days=1)
                tr_start = dates[0]
            else:  # anchored
                tr_start = dates[0]
                tr_end   = te_start - pd.Timedelta(days=1)

            if tr_start < dates[0] or te_end > dates[-1]:
                continue
            folds.append((tr_start, tr_end, te_start, te_end))

        return list(reversed(folds))

    def _optimize_in_sample(
        self, train_df: pd.DataFrame, symbol: str, strategy: str, top_n: int
    ) -> tuple[dict, float]:
        best_sharpe = -999.0
        best_params: dict = {}

        param_combos = self._sample_param_space(top_n * 3)

        for params in param_combos:
            try:
                m = self.engine.run_backtest(train_df, symbol, "1d", strategy, params=params)
                if m.sharpe > best_sharpe and m.nb_trades >= 5:
                    best_sharpe = m.sharpe
                    best_params = params
            except Exception:
                continue

        if not best_params:
            best_params = {k: v[len(v)//2] for k, v in PARAM_SPACE.items()}
            best_sharpe = 0.0

        return best_params, best_sharpe

    def _sample_param_space(self, n: int) -> list[dict]:
        combos = []
        for _ in range(n):
            p = {k: v[np.random.randint(len(v))] for k, v in PARAM_SPACE.items()}
            combos.append(p)
        return combos

    def _find_stable_params(self, folds: list[WalkForwardFold]) -> dict:
        if not folds:
            return {}
        all_params = [f.best_params for f in folds if not f.overfitting]
        if not all_params:
            all_params = [f.best_params for f in folds]

        stable: dict = {}
        for key in PARAM_SPACE:
            vals = [p.get(key) for p in all_params if key in p]
            if not vals:
                continue
            if isinstance(vals[0], float):
                stable[key] = round(float(np.median(vals)), 2)
            else:
                from collections import Counter
                stable[key] = Counter(vals).most_common(1)[0][0]
        return stable

    def _recommend_params(self, folds: list[WalkForwardFold]) -> dict:
        # Weight by test_sharpe when positive
        best_folds = sorted(folds, key=lambda f: f.test_sharpe, reverse=True)[:3]
        if not best_folds:
            return {}
        return best_folds[0].best_params
