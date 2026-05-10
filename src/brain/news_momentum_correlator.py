from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path("data/news_correlations.db")
MIN_EVENTS_FOR_MODEL = 50


class NewsMomentumCorrelator:
    """
    Correlates news events with subsequent price reactions.
    After 50 events, trains a sklearn classifier to predict pump probability.
    Boosts opportunity score by +25 pts if pump probability > 70%.
    """

    def __init__(self):
        self._model = None
        self._event_count = self._init_db()

    def record_news(self, event: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO news_events
                   (timestamp, symbol, news_category, sentiment, source, model_prediction)
                   VALUES (?,?,?,?,?,?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    event.get("symbol", ""),
                    event.get("signal_type", "news"),
                    event.get("sentiment", 0.0),
                    event.get("source", ""),
                    None,
                ),
            )
        self._event_count += 1
        if self._event_count >= MIN_EVENTS_FOR_MODEL and self._event_count % 10 == 0:
            self._train_model()

    def record_outcome(self, news_id: int, price_reaction_pct: float, volume_spike: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE news_events SET price_reaction_pct=?, volume_spike=? WHERE id=?",
                (price_reaction_pct, volume_spike, news_id),
            )

    def predict_pump_probability(self, event: dict[str, Any]) -> float:
        """Returns estimated probability of >3% pump within 30 minutes."""
        if self._model is None:
            return 0.5
        try:
            import numpy as np
            hour_utc = datetime.now(timezone.utc).hour
            features = [[
                event.get("sentiment", 0.0),
                hour_utc,
                event.get("volume_spike", 1.0),
                1 if event.get("urgency") == "high" else 0,
            ]]
            prob = self._model.predict_proba(features)[0][1]
            return float(prob)
        except Exception:
            return 0.5

    def get_score_boost(self, event: dict[str, Any]) -> float:
        """Returns score bonus (+25 if pump_prob > 70%, else 0)."""
        prob = self.predict_pump_probability(event)
        return 25.0 if prob >= 0.70 else 0.0

    def _train_model(self) -> None:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            import numpy as np

            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT sentiment, strftime('%H', timestamp) as hour,
                              volume_spike, price_reaction_pct
                       FROM news_events
                       WHERE price_reaction_pct IS NOT NULL"""
                ).fetchall()

            if len(rows) < MIN_EVENTS_FOR_MODEL:
                return

            X = np.array([[r[0], int(r[1] or 0), r[2] or 1.0, 0] for r in rows])
            y = np.array([1 if (r[3] or 0) > 3.0 else 0 for r in rows])

            self._model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            self._model.fit(X[:, :3], y)
            logger.info("NewsMomentumCorrelator: model trained on %d events", len(rows))
        except Exception as e:
            logger.error("Model training failed: %s", e)

    def _init_db(self) -> int:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, news_category TEXT,
                    sentiment REAL, source TEXT, volume_spike REAL,
                    price_reaction_pct REAL, model_prediction REAL,
                    actual_outcome INTEGER, accuracy REAL
                )
            """)
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM news_events").fetchone()[0]
        return count

    def _conn(self):
        import contextlib
        @contextlib.contextmanager
        def _ctx():
            c = sqlite3.connect(DB_PATH)
            try:
                yield c
                c.commit()
            finally:
                c.close()
        return _ctx()
