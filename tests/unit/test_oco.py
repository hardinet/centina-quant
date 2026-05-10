"""Unit tests for OCOCalculator."""
import pytest

from src.brain.oco_calculator import OCOCalculator, OCOLevels


@pytest.fixture
def oco():
    return OCOCalculator()


class TestOCOCalculator:

    def test_sl_below_entry(self, oco):
        levels = oco.calculate(entry=100.0, atr=2.0)
        assert levels.sl < levels.entry

    def test_tp1_above_entry(self, oco):
        levels = oco.calculate(entry=100.0, atr=2.0)
        assert levels.tp1 > levels.entry

    def test_tp2_above_tp1(self, oco):
        levels = oco.calculate(entry=100.0, atr=2.0)
        assert levels.tp2 > levels.tp1

    def test_sl_distance_equals_1x_atr(self, oco):
        entry, atr = 100.0, 2.0
        levels = oco.calculate(entry, atr)
        assert levels.sl == pytest.approx(entry - 1.0 * atr, rel=1e-6)

    def test_tp1_is_2pt5_pct(self, oco):
        # v5.0: TP1 = entry × 1.025
        entry, atr = 100.0, 2.0
        levels = oco.calculate(entry, atr)
        assert levels.tp1 == pytest.approx(entry * 1.025, rel=1e-6)

    def test_tp2_is_5pt2_pct(self, oco):
        # v5.0: TP2 = entry × 1.052
        entry, atr = 100.0, 2.0
        levels = oco.calculate(entry, atr)
        assert levels.tp2 == pytest.approx(entry * 1.052, rel=1e-6)

    def test_tp1_rr_positive(self, oco):
        levels = oco.calculate(entry=100.0, atr=2.0)
        assert levels.tp1_rr > 0

    def test_tp2_rr_greater_than_tp1(self, oco):
        levels = oco.calculate(entry=100.0, atr=2.0)
        assert levels.tp2_rr > levels.tp1_rr

    def test_qty_split_50_30_20(self, oco):
        # v5.0: 50/30/20 split
        levels = oco.calculate(entry=100.0, atr=2.0)
        assert levels.qty_tp1 == pytest.approx(0.50)
        assert levels.qty_tp2 == pytest.approx(0.30)
        assert levels.qty_tp3 == pytest.approx(0.20)

    def test_qty_split_sums_to_100pct(self, oco):
        levels = oco.calculate(entry=100.0, atr=2.0)
        total = levels.qty_tp1 + levels.qty_tp2 + levels.qty_tp3
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_trail_atr_is_1pt5x(self, oco):
        # v5.0: trailing ATR = 1.5×ATR
        levels = oco.calculate(entry=100.0, atr=2.0)
        assert levels.tp3_trail_atr == pytest.approx(3.0, rel=1e-6)

    def test_net_gain_positive(self, oco):
        levels = oco.calculate(entry=100.0, atr=2.0)
        assert levels.net_gain_pct > 0

    def test_invalid_entry_raises(self, oco):
        with pytest.raises(ValueError):
            oco.calculate(entry=0.0, atr=1.0)

    def test_invalid_atr_raises(self, oco):
        with pytest.raises(ValueError):
            oco.calculate(entry=100.0, atr=0.0)

    def test_scales_with_atr(self, oco):
        # ATR affects SL distance and trailing stop, NOT TP1/TP2 (fixed % targets)
        l_small = oco.calculate(entry=100.0, atr=0.5)
        l_large = oco.calculate(entry=100.0, atr=5.0)
        assert l_large.sl < l_small.sl           # larger ATR → wider SL
        assert l_large.tp3_trail_atr > l_small.tp3_trail_atr  # trailing distance scales
        assert l_large.tp1 == pytest.approx(l_small.tp1)      # TP1 fixed at +2.5%

    def test_adjust_for_step_size(self, oco):
        result = oco.adjust_for_step_size(0.12345678, 0.01)
        assert result == pytest.approx(0.12, rel=1e-6)

    def test_adjust_for_step_size_zero(self, oco):
        result = oco.adjust_for_step_size(0.12345, 0.0)
        assert result == 0.12345
