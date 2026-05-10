from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = Path("data/pattern_library.db")


@dataclass
class Pattern:
    pattern_id:   str
    symbol:       str
    timeframe:    str
    regime:       str
    features:     list[float]    # 8-dim vector
    outcome_pct:  float          # actual trade return
    strategy:     str
    recorded_at:  str
    similarity_threshold: float = 0.80


class PatternLibrary:
    """
    Persistent library of historical trade setups with outcome tracking.
    Enables similarity-based lookup to predict expected returns.
    """

    def __init__(self, db_path: Path | None = None):
        self.db = db_path or DB_PATH
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add(self, pattern: Pattern) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO patterns
                   (pattern_id, symbol, timeframe, regime, features,
                    outcome_pct, strategy, recorded_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    pattern.pattern_id,
                    pattern.symbol,
                    pattern.timeframe,
                    pattern.regime,
                    json.dumps(pattern.features),
                    pattern.outcome_pct,
                    pattern.strategy,
                    pattern.recorded_at,
                ),
            )

    def find_similar(
        self,
        features: list[float],
        symbol: str | None = None,
        regime: str | None = None,
        top_k: int = 10,
        min_similarity: float = 0.75,
    ) -> list[dict]:
        where = []
        args: list[Any] = []
        if symbol:
            where.append("symbol = ?")
            args.append(symbol)
        if regime:
            where.append("regime = ?")
            args.append(regime)

        sql = "SELECT * FROM patterns"
        if where:
            sql += " WHERE " + " AND ".join(where)

        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()

        if not rows:
            return []

        q = np.array(features, dtype=float)
        results = []
        for row in rows:
            try:
                f = np.array(json.loads(row["features"]), dtype=float)
                sim = self._cosine_similarity(q, f)
                if sim >= min_similarity:
                    results.append({
                        "pattern_id":  row["pattern_id"],
                        "symbol":      row["symbol"],
                        "outcome_pct": row["outcome_pct"],
                        "strategy":    row["strategy"],
                        "similarity":  round(sim, 3),
                        "regime":      row["regime"],
                    })
            except Exception:
                continue

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def predict_outcome(
        self,
        features: list[float],
        symbol: str | None = None,
        top_k: int = 5,
    ) -> dict:
        similar = self.find_similar(features, symbol=symbol, top_k=top_k)
        if not similar:
            return {"expected_return": 0.0, "confidence": 0.0, "n_samples": 0}

        weights = np.array([s["similarity"] for s in similar])
        outcomes = np.array([s["outcome_pct"] for s in similar])
        w_sum = weights.sum()

        if w_sum <= 0:
            return {"expected_return": float(outcomes.mean()), "confidence": 0.0, "n_samples": len(similar)}

        weighted_return = float((weights * outcomes).sum() / w_sum)
        confidence = float(weights.mean())

        return {
            "expected_return": round(weighted_return, 3),
            "confidence":      round(confidence, 3),
            "n_samples":       len(similar),
            "win_rate_similar": round(sum(1 for o in outcomes if o > 0) / len(outcomes), 3),
        }

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            by_sym  = conn.execute(
                "SELECT symbol, COUNT(*) n FROM patterns GROUP BY symbol ORDER BY n DESC LIMIT 10"
            ).fetchall()
            avg_out = conn.execute("SELECT AVG(outcome_pct) FROM patterns").fetchone()[0] or 0.0
        return {
            "total_patterns": total,
            "avg_outcome":    round(float(avg_out), 3),
            "by_symbol":      {r["symbol"]: r["n"] for r in by_sym},
        }

    def delete_old(self, days: int = 365) -> int:
        cutoff = datetime.utcnow().strftime("%Y-%m-%d")
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM patterns WHERE recorded_at < ?", (cutoff,)
            )
            return cur.rowcount

    # ── Private ───────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id   TEXT PRIMARY KEY,
                    symbol       TEXT,
                    timeframe    TEXT,
                    regime       TEXT,
                    features     TEXT,
                    outcome_pct  REAL,
                    strategy     TEXT,
                    recorded_at  TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON patterns(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_regime ON patterns(regime)")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) != len(b):
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
