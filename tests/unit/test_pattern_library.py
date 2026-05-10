"""Unit tests for PatternLibrary and RegimeLearner."""
from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from src.learning.pattern_library import Pattern, PatternLibrary
from src.learning.regime_learner import RegimeLearner


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"


class TestPatternLibrary:

    def test_add_and_find(self, tmp_db):
        lib = PatternLibrary(db_path=tmp_db)
        p = Pattern(
            pattern_id="test-001",
            symbol="BTCUSDT",
            timeframe="4h",
            regime="Bull",
            features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            outcome_pct=3.5,
            strategy="B",
            recorded_at=datetime.utcnow().isoformat(),
        )
        lib.add(p)
        results = lib.find_similar([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                   min_similarity=0.9)
        assert len(results) >= 1
        assert results[0]["pattern_id"] == "test-001"

    def test_similarity_ordering(self, tmp_db):
        lib = PatternLibrary(db_path=tmp_db)
        for i, outcome in enumerate([5.0, 1.0, 3.0]):
            scale = 1.0 - i * 0.05
            lib.add(Pattern(
                pattern_id=f"p-{i}",
                symbol="BTCUSDT",
                timeframe="4h",
                regime="Bull",
                features=[scale * j for j in range(1, 9)],
                outcome_pct=outcome,
                strategy="B",
                recorded_at=datetime.utcnow().isoformat(),
            ))
        query = [1.0 * j for j in range(1, 9)]
        results = lib.find_similar(query, min_similarity=0.5)
        if len(results) >= 2:
            assert results[0]["similarity"] >= results[1]["similarity"]

    def test_predict_outcome_returns_dict(self, tmp_db):
        lib = PatternLibrary(db_path=tmp_db)
        lib.add(Pattern(
            pattern_id="x",
            symbol="BTCUSDT",
            timeframe="4h",
            regime="Bull",
            features=[1.0] * 8,
            outcome_pct=2.5,
            strategy="B",
            recorded_at=datetime.utcnow().isoformat(),
        ))
        pred = lib.predict_outcome([1.0] * 8)
        assert "expected_return" in pred
        assert "confidence" in pred
        assert "n_samples" in pred

    def test_predict_outcome_no_data(self, tmp_db):
        lib = PatternLibrary(db_path=tmp_db)
        pred = lib.predict_outcome([1.0] * 8)
        assert pred["n_samples"] == 0

    def test_get_stats(self, tmp_db):
        lib = PatternLibrary(db_path=tmp_db)
        lib.add(Pattern("s1", "ETHUSDT", "4h", "Bull",
                        [0.5]*8, 2.0, "A", datetime.utcnow().isoformat()))
        stats = lib.get_stats()
        assert stats["total_patterns"] >= 1
        assert "avg_outcome" in stats


class TestRegimeLearner:

    def test_record_and_retrieve(self, tmp_db):
        rl = RegimeLearner(db_path=tmp_db)
        rl.record_outcome(regime=2, strategy="B", pnl_pct=3.5, symbol="BTCUSDT")
        rl.record_outcome(regime=2, strategy="B", pnl_pct=-1.0, symbol="BTCUSDT")
        rl.record_outcome(regime=2, strategy="A", pnl_pct=5.0, symbol="BTCUSDT")
        perf = rl.get_regime_performance(regime=2)
        assert len(perf) >= 1

    def test_get_best_strategy_returns_string(self, tmp_db):
        rl = RegimeLearner(db_path=tmp_db)
        rl.record_outcome(2, "B", 4.0)
        rl.record_outcome(2, "A", 1.0)
        best = rl.get_best_strategy(regime=2)
        assert best in ["A", "B", "C", "D", "E"]

    def test_unknown_regime_returns_default(self, tmp_db):
        rl = RegimeLearner(db_path=tmp_db)
        best = rl.get_best_strategy(regime=99)
        assert best == "B"

    def test_strategy_weights_sum_to_1(self, tmp_db):
        rl = RegimeLearner(db_path=tmp_db)
        for strat, pnl in [("A", 3.0), ("B", 2.0), ("C", -1.0)]:
            rl.record_outcome(0, strat, pnl)
        weights = rl.get_strategy_weights_for_regime(0)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_export_returns_dataframe(self, tmp_db):
        rl = RegimeLearner(db_path=tmp_db)
        rl.record_outcome(1, "B", 2.0)
        df = rl.export_performance_table()
        assert "strategy" in df.columns
