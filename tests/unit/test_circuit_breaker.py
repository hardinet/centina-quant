"""Unit tests for CircuitBreaker."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.security.circuit_breaker import CBLevel, CircuitBreaker


@pytest.fixture
def tmp_state_file(tmp_path):
    state_path = tmp_path / "cb_state.json"
    with patch("src.security.circuit_breaker.STATE_FILE", state_path):
        yield state_path


@pytest.fixture
def cb(tmp_state_file):
    return CircuitBreaker(capital_baseline=1000.0)


class TestCircuitBreakerLevels:

    def test_initial_level_is_normal(self, cb):
        assert cb.level == CBLevel.NORMAL

    def test_alert_at_2pct_drawdown(self, cb):
        level = cb.record_pnl(-20.0)   # 2% of 1000
        assert level == CBLevel.ALERT

    def test_restricted_at_3pt5pct(self, cb):
        level = cb.record_pnl(-35.0)
        assert level == CBLevel.RESTRICTED

    def test_stopped_at_5pct(self, cb):
        level = cb.record_pnl(-50.0)
        assert level == CBLevel.STOPPED

    def test_terminated_at_15pct(self, cb):
        level = cb.record_pnl(-150.0)
        assert level == CBLevel.TERMINATED

    def test_no_new_position_when_restricted(self, cb):
        cb.record_pnl(-35.0)
        assert not cb.can_open_new_position

    def test_can_open_position_when_alert(self, cb):
        cb.record_pnl(-20.0)
        assert cb.can_open_new_position

    def test_position_size_halved_on_alert(self, cb):
        cb.record_pnl(-20.0)
        assert cb.position_size_multiplier == 0.5

    def test_full_size_when_normal(self, cb):
        assert cb.position_size_multiplier == 1.0

    def test_level_monotonically_increases_intraday(self, cb):
        cb.record_pnl(-20.0)   # ALERT
        assert cb.level == CBLevel.ALERT
        cb.record_pnl(-15.0)   # cumulative 35 → RESTRICTED
        assert cb.level == CBLevel.RESTRICTED
        cb.record_pnl(10.0)    # profit should NOT lower level
        assert cb.level == CBLevel.RESTRICTED

    def test_manual_reset_clears_level(self, cb):
        cb.record_pnl(-150.0)
        assert cb.level == CBLevel.TERMINATED
        cb.manual_reset()
        assert cb.level == CBLevel.NORMAL

    def test_daily_reset_clears_non_terminated(self, cb):
        # Daily reset clears the daily PnL and level, but cumulative PnL persists.
        # A small intraday loss that doesn't trigger on cumulative basis resets properly.
        cb.record_pnl(-20.0)   # 2.0% of 1000 → ALERT
        assert cb.level == CBLevel.ALERT
        # Simulate new day
        cb._state["date"] = "2000-01-01"
        cb._state["cumulative_pnl"] = 0.0  # fresh start on cumulative too
        cb.record_pnl(0.0)
        assert cb.level == CBLevel.NORMAL

    def test_terminated_survives_daily_reset(self, cb):
        cb.record_pnl(-150.0)
        cb._state["date"] = "2000-01-01"
        cb.record_pnl(0.0)
        assert cb.level == CBLevel.TERMINATED

    def test_notify_callback_called_on_level_change(self):
        messages = []
        with patch("src.security.circuit_breaker.STATE_FILE", Path(tempfile.mktemp())):
            cb2 = CircuitBreaker(capital_baseline=1000.0, notify_callback=messages.append)
            cb2.record_pnl(-20.0)
        assert any("ALERT" in m for m in messages)
