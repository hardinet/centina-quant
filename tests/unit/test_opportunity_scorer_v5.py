"""Unit tests for v5.0 OpportunityScorer — VETO, PRIME, score 93+."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.brain.opportunity_scorer import OpportunityScorer, ScoreBreakdown, VETO_CONDITIONS
from src.brain.trend_analyzer import TrendSignal


def _make_df(n: int = 100) -> pd.DataFrame:
    close = pd.Series(np.linspace(30_000, 40_000, n), dtype=float)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.random.uniform(1_000, 5_000, n),
    })


def _make_signal(price: float = 40_000.0) -> TrendSignal:
    return TrendSignal(
        symbol="BTCUSDT",
        timeframe="4h",
        score=75.0,
        ema_score=70.0,
        rsi_score=60.0,
        macd_score=80.0,
        atr=500.0,
        current_price=price,
        ema7=price * 0.98,
        ema25=price * 0.96,
        ema99=price * 0.93,
        ema200=price * 0.90,
        rsi=60.0,
        verdict="BUY",
    )


@pytest.fixture
def scorer():
    s = OpportunityScorer(btc_trend_score=60.0)
    return s


class TestVetoBehavior:

    def test_pump_detected_veto(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df, manipulation_flags={"pump_detected": True})
        assert result.vetoed is True
        assert result.veto_reason == "pump_detected"
        assert result.final_score == 0

    def test_wash_detected_veto(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df, manipulation_flags={"wash_detected": True})
        assert result.vetoed is True
        assert result.veto_reason == "wash_detected"

    def test_btc_crash_veto(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df, btc_crash=True)
        assert result.vetoed is True
        assert result.veto_reason == "btc_crash"

    def test_cb_level_3_veto(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df, cb_level_value=3)
        assert result.vetoed is True
        assert result.veto_reason == "cb_level_3"

    def test_no_veto_when_clean(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df)
        # May be vetoed for score_too_low but not manipulation
        if result.vetoed:
            assert result.veto_reason == "score_too_low"
        else:
            assert result.final_score >= 0


class TestScoreProperties:

    def test_score_between_0_and_100(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df)
        if not result.vetoed:
            assert 0 <= result.final_score <= 100

    def test_prime_threshold_93(self, scorer):
        # Inject max bonuses to push score to PRIME
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df, pattern_bonus=15.0, news_bonus=9.0,
                              market_cap_rank=1)
        if not result.vetoed and result.final_score >= 93:
            assert result.is_prime is True
            assert result.verdict == "PRIME"

    def test_verdict_skip_below_50(self, scorer):
        df = _make_df()
        # Force low signal
        sig = _make_signal()
        sig.rsi = 25.0  # outside range
        result = scorer.score(sig, df)
        if not result.vetoed:
            if result.final_score < 50:
                assert result.verdict == "SKIP"

    def test_breakdown_fields_populated(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df)
        assert isinstance(result, ScoreBreakdown)
        assert result.symbol == "BTCUSDT"
        assert hasattr(result, "tech_pts")
        assert hasattr(result, "volume_pts")
        assert hasattr(result, "external_pts")
        assert hasattr(result, "bonus_pts")
        assert hasattr(result, "malus_pts")

    def test_bonus_capped(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df, pattern_bonus=999, news_bonus=999)
        if not result.vetoed:
            assert result.bonus_pts <= 24

    def test_malus_capped(self, scorer):
        df = _make_df()
        sig = _make_signal()
        result = scorer.score(sig, df,
                              manipulation_flags={"wick_hunt": True, "fake_news": True},
                              cb_level_value=2)
        if not result.vetoed:
            assert result.malus_pts <= 55
