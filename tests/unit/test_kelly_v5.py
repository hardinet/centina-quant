"""Unit tests for v5.0 KellyCalculator — dynamic caps and consecutive-loss override."""
from __future__ import annotations

import pytest
from src.brain.kelly_calculator import KellyCalculator, KellyResult


@pytest.fixture
def kelly():
    return KellyCalculator()


class TestKellyV5DynamicCaps:

    def test_zero_positions_cap_is_5pct(self, kelly):
        r = kelly.calculate(0.60, 0.03, 0.015, 1000, 0.015, open_positions=0)
        assert r.position_pct <= 0.05

    def test_one_position_cap_is_3pct(self, kelly):
        r = kelly.calculate(0.60, 0.03, 0.015, 1000, 0.015, open_positions=1)
        assert r.position_pct <= 0.03

    def test_two_positions_cap_is_2pct(self, kelly):
        r = kelly.calculate(0.60, 0.03, 0.015, 1000, 0.015, open_positions=2)
        assert r.position_pct <= 0.02

    def test_override_low_winrate_cap_is_1pct(self, kelly):
        r = kelly.calculate(0.35, 0.03, 0.015, 1000, 0.015, open_positions=0)
        assert r.position_pct <= 0.01
        assert r.reduced is True

    def test_override_three_consecutive_losses(self, kelly):
        r = kelly.calculate(0.55, 0.03, 0.015, 1000, 0.015,
                            open_positions=0, consecutive_losses=3)
        assert r.position_pct <= 0.01
        assert r.reduced is True

    def test_trending_strong_boost_within_cap(self, kelly):
        r_normal = kelly.calculate(0.55, 0.03, 0.015, 1000, 0.015,
                                   open_positions=0, trending_strong=False)
        r_strong = kelly.calculate(0.55, 0.03, 0.015, 1000, 0.015,
                                   open_positions=0, trending_strong=True)
        assert r_strong.position_pct >= r_normal.position_pct
        assert r_strong.position_pct <= 0.05

    def test_trending_strong_no_boost_when_override(self, kelly):
        r = kelly.calculate(0.35, 0.03, 0.015, 1000, 0.015,
                            open_positions=0, consecutive_losses=3, trending_strong=True)
        assert r.position_pct <= 0.01
        assert r.reduced is True

    def test_kelly_result_fields(self, kelly):
        r = kelly.calculate(0.55, 0.03, 0.015, 200, 0.015)
        assert isinstance(r, KellyResult)
        assert r.kelly_fraction >= 0
        assert r.position_usdt > 0
        assert r.risk_usdt > 0

    def test_from_signal_score_propagates_positions(self, kelly):
        r0 = kelly.from_signal_score(80, 200, 0.015, open_positions=0)
        r2 = kelly.from_signal_score(80, 200, 0.015, open_positions=2)
        assert r0.position_pct >= r2.position_pct

    def test_invalid_win_rate_raises(self, kelly):
        with pytest.raises(ValueError):
            kelly.calculate(0, 0.03, 0.015, 1000, 0.015)

    def test_invalid_avg_pct_raises(self, kelly):
        with pytest.raises(ValueError):
            kelly.calculate(0.55, 0, 0.015, 1000, 0.015)
