from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path("data/cortex.db")


@dataclass
class RegimePerformance:
    regime:       int
    strategy:     str
    n_trades:     int
    win_rate:     float
    avg_pnl_pct:  float
    best_strategy: str


class RegimeLearner:
    """
    Learns which strategies perform best in each market regime.
    Updates regime→strategy weights from live trade outcomes.
    Persists regime performance history to SQLite.
    """

    def __init__(self, db_path: Path | None = None):
        self.db = db_path or DB_PATH
        self._init_db()
        self._cache: dict[str, dict] = {}

    def record_outcome(
        self,
        regime: int,
        strategy: str,
        pnl_pct: float,
        symbol: str = "",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO regime_outcomes
                   (regime, strategy, pnl_pct, symbol, recorded_at)
                   VALUES (?,?,?,?,?)""",
                (regime, strategy, pnl_pct, symbol, datetime.utcnow().isoformat()),
            )
        self._cache.clear()

    def get_best_strategy(self, regime: int) -> str:
        perf = self.get_regime_performance(regime)
        if not perf:
            return "B"  # default golden cross
        best = max(perf, key=lambda p: p.avg_pnl_pct * p.win_rate)
        return best.strategy

    def get_regime_performance(self, regime: int) -> list[RegimePerformance]:
        cache_key = f"perf_{regime}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        with self._conn() as conn:
            rows = conn.execute(
                """SELECT strategy,
                          COUNT(*) n,
                          AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) wr,
                          AVG(pnl_pct) avg_pnl
                   FROM regime_outcomes
                   WHERE regime = ?
                   GROUP BY strategy""",
                (regime,),
            ).fetchall()

        if not rows:
            return []

        result = [
            RegimePerformance(
                regime=regime,
                strategy=r["strategy"],
                n_trades=r["n"],
                win_rate=round(float(r["wr"]), 3),
                avg_pnl_pct=round(float(r["avg_pnl"]), 3),
                best_strategy="",
            )
            for r in rows
        ]
        best = max(result, key=lambda p: p.avg_pnl_pct * p.win_rate).strategy
        for p in result:
            p.best_strategy = best

        self._cache[cache_key] = result
        return result

    def get_strategy_weights_for_regime(self, regime: int) -> dict[str, float]:
        perf = self.get_regime_performance(regime)
        if not perf:
            return {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}

        scores = {p.strategy: max(p.avg_pnl_pct * p.win_rate, 0) for p in perf}
        total  = sum(scores.values())
        if total <= 0:
            n = len(scores)
            return {s: 1/n for s in scores}

        weights: dict[str, float] = {}
        for s in ["A", "B", "C", "D", "E"]:
            weights[s] = round(scores.get(s, 0) / total, 3)
        return weights

    def export_performance_table(self) -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql(
                """SELECT regime, strategy,
                          COUNT(*) n_trades,
                          AVG(pnl_pct) avg_pnl,
                          AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) win_rate
                   FROM regime_outcomes
                   GROUP BY regime, strategy
                   ORDER BY regime, avg_pnl DESC""",
                conn,
            )
        return df

    # ── Private ───────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regime_outcomes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime      INTEGER,
                    strategy    TEXT,
                    pnl_pct     REAL,
                    symbol      TEXT,
                    recorded_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_regime_strat ON regime_outcomes(regime, strategy)")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn
