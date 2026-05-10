"""Unit tests for v5.0 OCOCalculator — fixed-pct TP1/TP2 structure."""
from __future__ import annotations

import pytest
from src.brain.oco_calculator import OCOCalculator, OCOLevels


@pytest.fixture
def calc():
    return OCOCalculator()


class TestOCOV5Structure:

    def test_tp1_is_2_5_pct(self, calc):
        levels = calc.calculate(entry=40_000, atr=500)
        assert abs(levels.tp1 - 40_000 * 1.025) < 0.01

    def test_tp2_is_5_2_pct(self, calc):
        levels = calc.calculate(entry=40_000, atr=500)
        assert abs(levels.tp2 - 40_000 * 1.052) < 0.01

    def test_sl_is_1x_atr(self, calc):
        entry, atr = 40_000, 500
        levels = calc.calculate(entry, atr)
        assert abs(levels.sl - (entry - atr)) < 0.01

    def test_qty_split_50_30_20(self, calc):
        levels = calc.calculate(40_000, 500)
        assert levels.qty_tp1 == pytest.approx(0.50)
        assert levels.qty_tp2 == pytest.approx(0.30)
        assert levels.qty_tp3 == pytest.approx(0.20)

    def test_tp3_trail_atr_is_1_5x(self, calc):
        levels = calc.calculate(40_000, 500)
        assert abs(levels.tp3_trail_atr - 500 * 1.5) < 0.01

    def test_net_gain_positive(self, calc):
        levels = calc.calculate(40_000, 500)
        assert levels.net_gain_pct > 0

    def test_tp1_rr_positive(self, calc):
        levels = calc.calculate(40_000, 500)
        assert levels.tp1_rr > 0

    def test_tp2_rr_greater_than_tp1_rr(self, calc):
        levels = calc.calculate(40_000, 500)
        assert levels.tp2_rr > levels.tp1_rr

    def test_zero_atr_raises(self, calc):
        with pytest.raises(ValueError):
            calc.calculate(40_000, 0)

    def test_zero_entry_raises(self, calc):
        with pytest.raises(ValueError):
            calc.calculate(0, 500)

    def test_sl_pct_field(self, calc):
        levels = calc.calculate(40_000, 500)
        expected_sl_pct = (500 / 40_000) * 100
        assert abs(levels.sl_pct - expected_sl_pct) < 0.01

    def test_display_returns_string(self, calc):
        levels = calc.calculate(40_000, 500)
        s = calc.display(levels)
        assert "ENTRY" in s
        assert "TP1" in s
        assert "TP2" in s
        assert "NET" in s

    def test_step_size_rounding(self, calc):
        price = calc.adjust_for_step_size(40_123.456, 0.01)
        assert round(price, 2) == price
