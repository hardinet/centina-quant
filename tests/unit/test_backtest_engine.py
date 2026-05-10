"""Unit tests for HistoricalBacktestEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.historical_simulation.backtest.historical_backtest_engine import (
    BacktestMetrics,
    HistoricalBacktestEngine,
)
from src.historical_simulation.synthetic.synthetic_data_generator import (
    SyntheticConfig,
    SyntheticDataGenerator,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    gen = SyntheticDataGenerator()
    cfg = SyntheticConfig(model="gbm", n_days=365, start_price=40_000.0, seed=42)
    return gen.generate(cfg)


@pytest.fixture
def engine() -> HistoricalBacktestEngine:
    return HistoricalBacktestEngine(capital=200.0)


class TestBacktestEngine:

    def test_returns_metrics_object(self, engine, synthetic_df):
        m = engine.run_backtest(synthetic_df, "BTCUSDT", "1H", strategy="B")
        assert isinstance(m, BacktestMetrics)

    def test_capital_initial_matches(self, engine, synthetic_df):
        m = engine.run_backtest(synthetic_df, "BTCUSDT", "1H")
        assert m.capital_initial == 200.0

    def test_metrics_have_required_fields(self, engine, synthetic_df):
        m = engine.run_backtest(synthetic_df, "BTCUSDT", "1H")
        for field in ["sharpe", "sortino", "calmar", "max_drawdown_pct",
                      "win_rate", "profit_factor", "total_return_pct", "cagr"]:
            assert hasattr(m, field), f"Missing field: {field}"

    def test_win_rate_between_0_and_1(self, engine, synthetic_df):
        m = engine.run_backtest(synthetic_df, "BTCUSDT", "1H")
        if m.nb_trades > 0:
            assert 0 <= m.win_rate <= 1

    def test_max_drawdown_nonnegative(self, engine, synthetic_df):
        m = engine.run_backtest(synthetic_df, "BTCUSDT", "1H")
        assert m.max_drawdown_pct >= 0

    def test_empty_df_returns_empty_metrics(self, engine):
        empty = pd.DataFrame()
        m = engine.run_backtest(empty, "BTCUSDT", "1H")
        assert m.nb_trades == 0
        assert m.capital_final == 200.0

    def test_short_df_returns_empty(self, engine, synthetic_df):
        short_df = synthetic_df.iloc[:50]
        m = engine.run_backtest(short_df, "BTCUSDT", "1H")
        assert m.nb_trades == 0

    def test_monte_carlo_structure(self, engine, synthetic_df):
        m = engine.run_backtest(synthetic_df, "BTCUSDT", "1H")
        if m.nb_trades >= 5:
            mc = engine.run_monte_carlo_simulation(m, n=100)
            assert "mean" in mc
            assert "p5" in mc
            assert "p95" in mc
            assert "prob_profit" in mc
            assert 0 <= mc["prob_profit"] <= 1

    def test_monte_carlo_empty_trades(self, engine):
        m = BacktestMetrics(
            symbol="X", strategy="B", timeframe="1H",
            start_date="", end_date="",
            capital_initial=200, capital_final=200,
            total_return_pct=0, cagr=0, sharpe=0, sortino=0, calmar=0,
            max_drawdown_pct=0, max_dd_duration_days=0,
            profit_factor=0, win_rate=0, avg_win_pct=0, avg_loss_pct=0,
            expectancy=0, nb_trades=0, avg_duration_h=0,
            var_95=0, cvar_95=0, recovery_factor=0, ulcer_index=0,
        )
        mc = engine.run_monte_carlo_simulation(m, n=50)
        assert mc == {}

    def test_strategy_param_override(self, engine, synthetic_df):
        params = {"ema_fast": 5, "ema_slow": 20, "rsi_low": 55}
        m = engine.run_backtest(synthetic_df, "BTCUSDT", "1H", strategy="A", params=params)
        assert isinstance(m, BacktestMetrics)
        assert m.params.get("ema_fast") == 5
