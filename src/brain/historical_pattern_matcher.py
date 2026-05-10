from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

DB_PATH = Path("data/pattern_library.db")
MIN_SIMILARITY = 0.80
TOP_K = 3


@dataclass
class PatternMatch:
    similarity: float
    win_rate:   float
    avg_gain:   float
    duration_h: float
    confidence: float   # 0-1


class HistoricalPatternMatcher:
    """
    Vectorises current market state into 10 features and finds the
    most similar historical patterns using cosine similarity.
    Returns bonus score +15 pts if Top-K matches > 80% similar.
    """

    FEATURES = [
        "ema_ratio", "rsi", "volume_spike", "atr_ratio",
        "hour_utc", "obv_slope", "macd_hist_norm", "price_vs_ema200",
    ]

    def __init__(self):
        self._patterns: np.ndarray | None = None
        self._meta: list[dict] = []
        self._init_db()
        self._load_patterns()

    def get_score_bonus(self, df_4h: pd.DataFrame, symbol: str) -> tuple[float, list[PatternMatch]]:
        """Returns (bonus_pts, top_matches)."""
        vec = self._featurise(df_4h)
        if vec is None or self._patterns is None or len(self._patterns) < 10:
            return 0.0, []

        matches = self._find_matches(vec)
        if not matches:
            return 0.0, []

        best = matches[0]
        if best.similarity >= MIN_SIMILARITY:
            bonus = 15.0
        elif best.similarity >= 0.60:
            bonus = 7.0
        else:
            bonus = 0.0

        return bonus, matches

    def store_pattern(self, df_4h: pd.DataFrame, outcome_win: bool,
                      gain_pct: float, duration_h: float) -> None:
        vec = self._featurise(df_4h)
        if vec is None:
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO patterns (features_json, win, gain_pct, duration_h) VALUES (?,?,?,?)",
                (str(vec.tolist()), int(outcome_win), gain_pct, duration_h),
            )
        self._load_patterns()

    def _featurise(self, df: pd.DataFrame) -> np.ndarray | None:
        try:
            close  = df["close"].astype(float)
            high   = df["high"].astype(float)
            low    = df["low"].astype(float)
            volume = df["volume"].astype(float)

            ema7   = ta.ema(close, 7).iloc[-1]
            ema25  = ta.ema(close, 25).iloc[-1]
            ema200 = ta.ema(close, 200).iloc[-1]
            rsi    = ta.rsi(close, 14).iloc[-1]
            atr    = ta.atr(high, low, close, 14)
            macd_d = ta.macd(close)
            obv    = ta.obv(close, volume)

            price     = close.iloc[-1]
            ema_ratio = ema7 / ema25 if ema25 > 0 else 1.0
            vol_spike = volume.iloc[-1] / volume.iloc[-20:].mean() if volume.iloc[-20:].mean() > 0 else 1.0
            atr_ratio = atr.iloc[-1] / atr.iloc[-20:].mean() if atr.iloc[-20:].mean() > 0 else 1.0
            obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / (abs(obv.iloc[-5]) + 1)
            macd_hist = float(macd_d["MACDh_12_26_9"].iloc[-1])
            p_ema200  = price / ema200 if ema200 > 0 else 1.0

            import datetime
            hour_utc = datetime.datetime.utcnow().hour / 24.0

            return np.array([
                ema_ratio, rsi / 100, vol_spike, atr_ratio,
                hour_utc, obv_slope, macd_hist, p_ema200,
            ], dtype=float)
        except Exception as e:
            logger.debug("Featurise error: %s", e)
            return None

    def _find_matches(self, vec: np.ndarray) -> list[PatternMatch]:
        sims = self._cosine_similarity_batch(vec, self._patterns)
        top_idx = np.argsort(sims)[::-1][:TOP_K]
        results = []
        for idx in top_idx:
            if idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            results.append(PatternMatch(
                similarity=float(sims[idx]),
                win_rate=meta.get("win_rate", 0.5),
                avg_gain=meta.get("avg_gain", 0.0),
                duration_h=meta.get("duration_h", 2.0),
                confidence=float(sims[idx]),
            ))
        return results

    def _cosine_similarity_batch(self, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        norm_v = np.linalg.norm(vec)
        if norm_v == 0:
            return np.zeros(len(matrix))
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-10
        return matrix.dot(vec) / (norms * norm_v)

    def _load_patterns(self) -> None:
        with self._conn() as conn:
            rows = conn.execute("SELECT features_json, win, gain_pct, duration_h FROM patterns").fetchall()
        if not rows:
            return
        vecs, meta = [], []
        for r in rows:
            try:
                vecs.append(np.array(eval(r[0]), dtype=float))
                meta.append({"win": r[1], "gain_pct": r[2], "duration_h": r[3],
                             "win_rate": 0.5, "avg_gain": r[2]})
            except Exception:
                pass
        if vecs:
            self._patterns = np.array(vecs)
            self._meta = meta

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    features_json TEXT, win INTEGER,
                    gain_pct REAL, duration_h REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

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
