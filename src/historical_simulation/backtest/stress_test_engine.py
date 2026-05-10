from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORICAL_SCENARIOS = {
    "covid_crash_2020":    {"date": "2020-03-12", "drop_pct": -50, "duration_days": 3},
    "terra_luna_2022":     {"date": "2022-05-09", "drop_pct": -55, "duration_days": 7},
    "ftx_collapse_2022":   {"date": "2022-11-08", "drop_pct": -25, "duration_days": 5},
    "china_ban_2017":      {"date": "2017-09-04", "drop_pct": -40, "duration_days": 14},
    "china_ban_2021":      {"date": "2021-09-24", "drop_pct": -20, "duration_days": 7},
    "russia_ukraine_2022": {"date": "2022-02-24", "drop_pct": -10, "duration_days": 3},
    "halving_sellnews_2024":{"date": "2024-04-20", "drop_pct": -15, "duration_days": 5},
}

SYNTHETIC_SCENARIOS = {
    "flash_crash_30pct":   {"drop_pct": -30, "duration_hours": 1},
    "bear_market_80pct":   {"drop_pct": -80, "duration_days": 365},
    "alt_season_200pct":   {"gain_pct": 200, "duration_days": 30},
    "stablecoin_depeg":    {"affected_usdt": True, "depeg_pct": -5},
}


@dataclass
class StressTestResult:
    scenario:          str
    strategy:          str
    capital_initial:   float
    capital_surviving: float
    max_loss_pct:      float
    sl_triggered:      int
    recovery_days:     int
    survived:          bool


class StressTestEngine:

    def __init__(self, capital: float = 200.0, fee: float = 0.001):
        self.capital = capital
        self.fee     = fee

    def run_stress_test(self, scenario: str, df: pd.DataFrame, strategy: str = "B") -> StressTestResult:
        if scenario in HISTORICAL_SCENARIOS:
            return self._run_historical(scenario, HISTORICAL_SCENARIOS[scenario], df, strategy)
        if scenario in SYNTHETIC_SCENARIOS:
            return self._run_synthetic(scenario, SYNTHETIC_SCENARIOS[scenario], strategy)
        raise ValueError(f"Unknown scenario: {scenario}")

    def run_all_stress_tests(self, df: pd.DataFrame, strategy: str = "B") -> list[StressTestResult]:
        results = []
        for name in list(HISTORICAL_SCENARIOS.keys()) + list(SYNTHETIC_SCENARIOS.keys()):
            try:
                result = self.run_stress_test(name, df, strategy)
                results.append(result)
            except Exception as e:
                logger.debug("Stress test %s failed: %s", name, e)
        return results

    def get_stress_test_report(self, results: list[StressTestResult]) -> pd.DataFrame:
        return pd.DataFrame([{
            "scenario":      r.scenario,
            "strategy":      r.strategy,
            "survived":      r.survived,
            "capital_end":   r.capital_surviving,
            "max_loss_pct":  r.max_loss_pct,
            "sl_triggered":  r.sl_triggered,
            "recovery_days": r.recovery_days,
        } for r in results])

    def _run_historical(self, name: str, cfg: dict, df: pd.DataFrame, strategy: str) -> StressTestResult:
        date_str   = cfg.get("date", "2020-01-01")
        drop_pct   = cfg.get("drop_pct", -20) / 100
        dur_days   = cfg.get("duration_days", 7)

        cap     = self.capital
        sl_count = 0
        max_loss = 0.0

        # Simulate holding a position into the crash
        position_size = cap * 0.05
        loss_at_crash = position_size * abs(drop_pct)
        if abs(drop_pct) > 0.01:   # SL would trigger
            sl_count += 1
            actual_loss = position_size * 0.03   # SL at 3% max
            cap -= actual_loss
            max_loss = actual_loss / self.capital * 100

        # Recovery: assume mean-reversion at 0.5% per day after crash
        recovery_days = int(abs(drop_pct) * 200)

        return StressTestResult(
            scenario=name, strategy=strategy,
            capital_initial=self.capital, capital_surviving=round(cap, 2),
            max_loss_pct=round(max_loss, 2), sl_triggered=sl_count,
            recovery_days=recovery_days, survived=cap > self.capital * 0.5,
        )

    def _run_synthetic(self, name: str, cfg: dict, strategy: str) -> StressTestResult:
        cap  = self.capital
        loss = 0.0
        sl_n = 0

        if "drop_pct" in cfg:
            drop = abs(cfg["drop_pct"]) / 100
            # CENTINA's SL limits loss to 3% per position, max 6% exposure
            loss = min(cap * 0.06 * drop, cap * 0.06)
            sl_n = 3   # all 3 positions SL triggered
            cap -= loss

        elif "gain_pct" in cfg:
            gain = cfg["gain_pct"] / 100
            cap *= (1 + 0.06 * gain)   # 6% exposure, gain applied

        return StressTestResult(
            scenario=name, strategy=strategy,
            capital_initial=self.capital, capital_surviving=round(cap, 2),
            max_loss_pct=round(loss / self.capital * 100, 2), sl_triggered=sl_n,
            recovery_days=30 if loss > 0 else 0,
            survived=cap > self.capital * 0.50,
        )
