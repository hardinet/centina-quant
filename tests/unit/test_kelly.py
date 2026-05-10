"""Unit tests for KellyCalculator."""
import pytest

from src.brain.kelly_calculator import KellyCalculator, KellyResult


@pytest.fixture
def kelly():
    return KellyCalculator()


class TestKellyCalculator:

    def test_basic_positive_edge(self, kelly):
        result = kelly.calculate(
            win_rate=0.60, avg_win_pct=0.04, avg_loss_pct=0.02,
            capital_usdt=1000.0, sl_distance_pct=0.02,
        )
        assert result.kelly_fraction > 0
        assert result.fractional_kelly == pytest.approx(result.kelly_fraction / 8, rel=1e-3)

    def test_position_capped_at_5pct_zero_open(self, kelly):
        # v5.0: cap is 5% when 0 positions open
        result = kelly.calculate(
            win_rate=0.85, avg_win_pct=0.10, avg_loss_pct=0.01,
            capital_usdt=1000.0, sl_distance_pct=0.01, open_positions=0,
        )
        assert result.position_pct <= 0.05

    def test_zero_edge_returns_zero_position(self, kelly):
        # win_rate × win = loss_rate × loss → no edge
        result = kelly.calculate(
            win_rate=0.50, avg_win_pct=0.02, avg_loss_pct=0.02,
            capital_usdt=1000.0, sl_distance_pct=0.02,
        )
        assert result.position_pct == 0.0

    def test_negative_edge_clamped_to_zero(self, kelly):
        result = kelly.calculate(
            win_rate=0.30, avg_win_pct=0.01, avg_loss_pct=0.05,
            capital_usdt=1000.0, sl_distance_pct=0.05,
        )
        assert result.position_pct == 0.0

    def test_position_usdt_proportional_to_capital(self, kelly):
        r1 = kelly.calculate(0.60, 0.04, 0.02, 1000.0, 0.02)
        r2 = kelly.calculate(0.60, 0.04, 0.02, 2000.0, 0.02)
        assert r2.position_usdt == pytest.approx(r1.position_usdt * 2, rel=1e-3)

    def test_risk_usdt_equals_position_times_sl(self, kelly):
        result = kelly.calculate(0.60, 0.04, 0.02, 1000.0, 0.02)
        expected_risk = result.position_usdt * 0.02
        assert result.risk_usdt == pytest.approx(expected_risk, rel=1e-3)

    def test_invalid_win_rate_raises(self, kelly):
        with pytest.raises(ValueError):
            kelly.calculate(1.0, 0.04, 0.02, 1000.0, 0.02)
        with pytest.raises(ValueError):
            kelly.calculate(0.0, 0.04, 0.02, 1000.0, 0.02)

    def test_invalid_pct_raises(self, kelly):
        with pytest.raises(ValueError):
            kelly.calculate(0.60, 0.0, 0.02, 1000.0, 0.02)

    def test_from_signal_score_high_score(self, kelly):
        result = kelly.from_signal_score(score=90.0, capital_usdt=1000.0, sl_distance_pct=0.02)
        assert result.position_pct > 0
        assert result.position_pct <= 0.05

    def test_from_signal_score_low_score(self, kelly):
        result_low  = kelly.from_signal_score(score=20.0,  capital_usdt=1000.0, sl_distance_pct=0.02)
        result_high = kelly.from_signal_score(score=90.0,  capital_usdt=1000.0, sl_distance_pct=0.02)
        # Higher score → bigger position
        assert result_high.position_pct >= result_low.position_pct
