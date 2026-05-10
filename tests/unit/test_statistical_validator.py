"""Unit tests for StatisticalValidator and OverfittingDetector."""
from __future__ import annotations

import math
import pytest

from src.historical_simulation.validators.statistical_validator import (
    StatisticalValidator,
    ValidationResult,
)
from src.historical_simulation.validators.overfitting_detector import (
    OverfittingDetector,
    OverfittingReport,
)
from src.historical_simulation.backtest.historical_backtest_engine import (
    BacktestMetrics,
    TradeRecord,
)


def _make_metrics(n_trades: int = 50, win_rate: float = 0.55,
                  avg_gain: float = 2.0, avg_loss: float = -1.0,
                  strategy: str = "B") -> BacktestMetrics:
    trades = []
    for i in range(n_trades):
        if i / n_trades < win_rate:
            pnl_pct = avg_gain
            pnl = avg_gain * 2
        else:
            pnl_pct = avg_loss
            pnl = avg_loss * 2
        trades.append(TradeRecord(
            entry_price=40000, exit_price=40000 * (1 + pnl_pct / 100),
            qty=0.001, pnl=pnl, pnl_pct=pnl_pct,
            duration_h=4.0, tp1_hit=pnl_pct > 0,
            tp2_hit=False, exit_reason="tp1" if pnl_pct > 0 else "sl",
            strategy=strategy, regime="Bull",
        ))

    wins  = [t.pnl for t in trades if t.pnl > 0]
    losses= [t.pnl for t in trades if t.pnl <= 0]
    wr    = len(wins) / max(len(trades), 1)
    total = sum(t.pnl for t in trades)
    cap   = 200.0

    return BacktestMetrics(
        symbol="BTCUSDT", strategy=strategy, timeframe="4H",
        start_date="2022-01-01", end_date="2024-01-01",
        capital_initial=cap, capital_final=cap + total,
        total_return_pct=total / cap * 100,
        cagr=10.0, sharpe=1.5, sortino=2.0, calmar=1.0,
        max_drawdown_pct=10.0, max_dd_duration_days=30,
        profit_factor=abs(sum(wins) / sum(losses)) if losses else 99,
        win_rate=wr,
        avg_win_pct=avg_gain, avg_loss_pct=avg_loss,
        expectancy=sum(t.pnl for t in trades) / max(len(trades), 1),
        nb_trades=len(trades),
        avg_duration_h=4.0,
        var_95=-2.0, cvar_95=-3.0,
        recovery_factor=2.0, ulcer_index=0.5,
        trades=trades,
    )


class TestStatisticalValidator:

    def test_returns_validation_result(self):
        v = StatisticalValidator()
        m = _make_metrics(50)
        result = v.validate(m)
        assert isinstance(result, ValidationResult)

    def test_empty_trades_returns_empty(self):
        v = StatisticalValidator()
        m = _make_metrics(0)
        result = v.validate(m)
        assert result.sharpe_ratio == 0
        assert result.is_significant is False

    def test_few_trades_returns_empty(self):
        v = StatisticalValidator()
        m = _make_metrics(3)
        result = v.validate(m)
        assert result.is_significant is False

    def test_sharpe_positive_for_winning_strategy(self):
        v = StatisticalValidator()
        m = _make_metrics(100, win_rate=0.65, avg_gain=3.0, avg_loss=-1.0)
        result = v.validate(m)
        assert result.sharpe_ratio > 0

    def test_psr_between_0_and_1(self):
        v = StatisticalValidator()
        m = _make_metrics(50)
        result = v.validate(m)
        assert 0 <= result.psr <= 1

    def test_dsr_between_0_and_1(self):
        v = StatisticalValidator()
        m = _make_metrics(50)
        result = v.validate(m)
        assert 0 <= result.deflated_sharpe <= 1

    def test_min_track_record_positive(self):
        v = StatisticalValidator()
        m = _make_metrics(50)
        result = v.validate(m)
        assert result.min_track_record >= 0

    def test_batch_returns_dataframe(self):
        v = StatisticalValidator()
        metrics = [_make_metrics(50, strategy=s) for s in ["A", "B", "C"]]
        df = v.validate_batch(metrics)
        assert len(df) == 3
        assert "sharpe" in df.columns
        assert "psr" in df.columns


class TestOverfittingDetector:

    def test_returns_overfitting_report(self):
        det = OverfittingDetector()
        m = _make_metrics(50)
        result = det.detect(m)
        assert isinstance(result, OverfittingReport)

    def test_pbo_between_0_and_1(self):
        det = OverfittingDetector()
        m = _make_metrics(30)
        result = det.detect(m)
        assert 0 <= result.pbo <= 1

    def test_verdict_not_empty(self):
        det = OverfittingDetector()
        m = _make_metrics(50)
        result = det.detect(m)
        assert len(result.verdict) > 0

    def test_is_overfit_flag(self):
        det = OverfittingDetector()
        m = _make_metrics(50)
        result = det.detect(m)
        assert isinstance(result.is_overfit, bool)

    def test_batch_returns_dataframe(self):
        det = OverfittingDetector()
        ms = [_make_metrics(40, strategy=s) for s in ["A", "B"]]
        df = det.detect_batch(ms)
        assert len(df) == 2
        assert "pbo" in df.columns
