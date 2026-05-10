"""Unit tests for StressTestEngine."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.historical_simulation.backtest.stress_test_engine import (
    StressTestEngine,
    StressTestResult,
    HISTORICAL_SCENARIOS,
    SYNTHETIC_SCENARIOS,
)
from src.historical_simulation.synthetic.synthetic_data_generator import (
    SyntheticConfig,
    SyntheticDataGenerator,
)


@pytest.fixture
def df() -> pd.DataFrame:
    gen = SyntheticDataGenerator()
    return gen.generate(SyntheticConfig(n_days=60, seed=99))


@pytest.fixture
def engine() -> StressTestEngine:
    return StressTestEngine(capital=200.0)


class TestStressTestEngine:

    def test_historical_scenario_returns_result(self, engine, df):
        result = engine.run_stress_test("covid_crash_2020", df, strategy="B")
        assert isinstance(result, StressTestResult)

    def test_synthetic_scenario_returns_result(self, engine, df):
        result = engine.run_stress_test("flash_crash_30pct", df, strategy="B")
        assert isinstance(result, StressTestResult)

    def test_unknown_scenario_raises(self, engine, df):
        with pytest.raises(ValueError):
            engine.run_stress_test("nonexistent_scenario", df)

    def test_capital_initial_preserved(self, engine, df):
        result = engine.run_stress_test("covid_crash_2020", df)
        assert result.capital_initial == 200.0

    def test_survived_field_is_bool(self, engine, df):
        result = engine.run_stress_test("flash_crash_30pct", df)
        assert isinstance(result.survived, bool)

    def test_bear_market_may_not_survive(self, engine, df):
        result = engine.run_stress_test("bear_market_80pct", df)
        # Bear market should test survival logic
        assert result.capital_surviving <= 200.0

    def test_alt_season_increases_capital(self, engine, df):
        result = engine.run_stress_test("alt_season_200pct", df)
        assert result.capital_surviving >= 200.0

    def test_run_all_returns_list(self, engine, df):
        results = engine.run_all_stress_tests(df)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_all_covers_all_scenarios(self, engine, df):
        results = engine.run_all_stress_tests(df)
        names = {r.scenario for r in results}
        total = len(HISTORICAL_SCENARIOS) + len(SYNTHETIC_SCENARIOS)
        assert len(names) == total

    def test_get_stress_test_report(self, engine, df):
        results = engine.run_all_stress_tests(df)
        report = engine.get_stress_test_report(results)
        assert isinstance(report, pd.DataFrame)
        assert "scenario" in report.columns
        assert "survived" in report.columns
        assert "max_loss_pct" in report.columns

    def test_sl_triggered_nonnegative(self, engine, df):
        result = engine.run_stress_test("covid_crash_2020", df)
        assert result.sl_triggered >= 0

    def test_max_loss_nonnegative(self, engine, df):
        result = engine.run_stress_test("terra_luna_2022", df)
        assert result.max_loss_pct >= 0

    def test_recovery_days_nonnegative(self, engine, df):
        result = engine.run_stress_test("ftx_collapse_2022", df)
        assert result.recovery_days >= 0
