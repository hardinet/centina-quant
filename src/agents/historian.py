from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.brain.regime_detector import RegimeDetector
from src.brain.strategy_selector import Strategy
from src.core.event_bus import Event, EventBus, EventType
from src.connectors.binance_rest import BinanceREST
from src.database.cortex_db import CortexDB

logger = logging.getLogger(__name__)

PATTERN_DB = Path("data/historian_patterns.db")
TOP_SYMBOLS = 30
HISTORY_DAYS = 90
NIGHTLY_HOUR_UTC = 23
WEEKLY_DAY = 6   # Sunday


class Historian:
    """
    Sub-agent 3: Pattern learning and real-time validation.

    a) Bootstrap: downloads 90d OHLCV, identifies setups, stores patterns
    b) Real-time: cosine-similarity matching for Analyst
    c) Nightly: learns from today's data, updates weights
    d) Weekly: launches Optuna hyperoptimisation
    """

    def __init__(self, rest: BinanceREST, db: CortexDB, bus: EventBus):
        self._rest    = rest
        self._db      = db
        self._bus     = bus
        self._regime  = RegimeDetector()
        self._bootstrapped = False
        self._init_db()

    async def run(self) -> None:
        if not self._bootstrapped:
            await self._bootstrap()
            self._bootstrapped = True

        last_nightly = None
        last_weekly  = None

        while True:
            now = datetime.now(timezone.utc)

            if last_nightly != now.date() and now.hour == NIGHTLY_HOUR_UTC:
                await self._nightly_learning()
                last_nightly = now.date()

            if last_weekly != now.isocalendar().week and now.weekday() == WEEKLY_DAY and now.hour == 2:
                await self._weekly_hyperopt()
                last_weekly = now.isocalendar().week

            await asyncio.sleep(60)

    async def validate_pattern(
        self, symbol: str, features: dict
    ) -> dict:
        """Called synchronously by Analyst. Returns similarity stats."""
        vec = self._dict_to_vector(features)
        rows = self._get_recent_patterns(limit=1000)

        if len(rows) < 10:
            return {"confidence": "LOW", "win_rate": 0.5, "nb_patterns": len(rows)}

        vecs   = np.array([r["vec"] for r in rows])
        metas  = [r["meta"] for r in rows]
        sims   = self._cosine_batch(vec, vecs)
        top_idx = np.argsort(sims)[::-1][:20]

        win_rates = []
        gains     = []
        durations = []
        for i in top_idx:
            if sims[i] < 0.50:
                break
            m = metas[i]
            win_rates.append(float(m.get("win", 0)))
            gains.append(float(m.get("gain_max_pct", 0)))
            durations.append(float(m.get("duree_h", 2)))

        nb = len(win_rates)
        return {
            "confidence":          "HIGH" if nb >= 20 else ("MEDIUM" if nb >= 10 else "LOW"),
            "win_rate_similaire":  round(sum(win_rates) / max(nb, 1), 3),
            "gain_moyen_pct":      round(sum(gains) / max(nb, 1), 3),
            "duree_mediane_h":     round(sorted(durations)[len(durations) // 2] if durations else 2.0, 2),
            "nb_patterns":         nb,
        }

    # ── Bootstrap ────────────────────────────────────────────────────

    async def _bootstrap(self) -> None:
        logger.info("Historian: bootstrapping 90d data for %d symbols", TOP_SYMBOLS)
        symbols = await self._rest.get_top_usdt_pairs(TOP_SYMBOLS)
        tasks   = [self._process_symbol(sym) for sym in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Historian: bootstrap complete")
        self._bus.emit_nowait(Event(EventType.LEARNING_UPDATE,
                                   {"msg": "Historian bootstrap complete", "symbols": len(symbols)}))

    async def _process_symbol(self, symbol: str) -> None:
        try:
            df = await self._rest.get_klines(symbol, "1h", limit=HISTORY_DAYS * 24)
            self._replay_history(df, symbol)
        except Exception as e:
            logger.debug("Historian: skip %s — %s", symbol, e)

    def _replay_history(self, df: pd.DataFrame, symbol: str) -> None:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)

        ema7   = ta.ema(close, 7)
        ema25  = ta.ema(close, 25)
        ema99  = ta.ema(close, 99)
        ema200 = ta.ema(close, 200)
        rsi    = ta.rsi(close, 14)
        atr    = ta.atr(high, low, close, 14)

        for i in range(210, len(df) - 5):
            p = close.iloc[i]
            if pd.isna(ema200.iloc[i]) or p < ema200.iloc[i]:
                continue
            if pd.isna(rsi.iloc[i]):
                continue

            features = {
                "ema_ratio":     float(ema7.iloc[i] / ema25.iloc[i]) if ema25.iloc[i] > 0 else 1.0,
                "rsi":           float(rsi.iloc[i]) / 100,
                "volume_spike":  float(volume.iloc[i] / volume.iloc[i-20:i].mean()) if volume.iloc[i-20:i].mean() > 0 else 1.0,
                "atr_ratio":     float(atr.iloc[i] / atr.iloc[i-20:i].mean()) if atr.iloc[i-20:i].mean() > 0 else 1.0,
                "ema7_25_aligned": 1.0 if ema7.iloc[i] > ema25.iloc[i] else 0.0,
            }

            future_high  = high.iloc[i+1:i+6].max()
            future_close = close.iloc[i+5]
            gain_max     = (future_high - p) / p * 100
            win          = gain_max >= 5.2

            self._store_pattern(symbol, features, {
                "win": int(win), "gain_max_pct": round(gain_max, 3),
                "duree_h": 5.0, "regime": "UNKNOWN",
            })

    # ── Nightly ───────────────────────────────────────────────────────

    async def _nightly_learning(self) -> None:
        logger.info("Historian: nightly learning cycle started")
        trades = self._db.get_recent_trades(50)
        wins   = [t for t in trades if t.closed_at and (t.pnl_usdt or 0) > 0]
        total  = [t for t in trades if t.closed_at]

        win_rate = len(wins) / len(total) if total else 0
        learned  = len(total)

        if win_rate < 0.50 and len(total) >= 20:
            self._bus.emit_nowait(Event(EventType.LEARNING_UPDATE, {
                "msg": f"Win rate basse ({win_rate:.1%}) sur {len(total)} trades — mode prudent activé",
                "win_rate": win_rate,
                "alert": True,
            }))

        logger.info("Historian: nightly done — WR=%.1f%% (%d trades)", win_rate * 100, learned)
        self._bus.emit_nowait(Event(EventType.LEARNING_UPDATE, {
            "msg": f"Historian a appris {learned} patterns aujourd'hui",
            "win_rate": win_rate,
        }))

    async def _weekly_hyperopt(self) -> None:
        logger.info("Historian: weekly hyperoptimisation (Sunday 02:00 UTC)")
        try:
            from src.learning.hyperoptimizer import HyperOptimizer
            optimizer = HyperOptimizer(n_trials=500, timeout_sec=600)
            df = await self._rest.get_klines("BTCUSDT", "4h", limit=500)
            result = optimizer.optimise(df, "BTCUSDT", "4h")
            logger.info("Historian: weekly hyperopt → Sharpe=%.3f params=%s",
                        result.best_value, result.best_params)
        except Exception as e:
            logger.error("Weekly hyperopt failed: %s", e)

    # ── DB helpers ────────────────────────────────────────────────────

    def _store_pattern(self, symbol: str, features: dict, meta: dict) -> None:
        ph = hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()[:16]
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO patterns (pattern_hash, symbol, features_json, meta_json, created_at) VALUES (?,?,?,?,?)",
                (ph, symbol, json.dumps(features), json.dumps(meta), datetime.utcnow().isoformat()),
            )

    def _get_recent_patterns(self, limit: int = 1000) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT features_json, meta_json FROM patterns ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            try:
                feat = json.loads(r[0])
                meta = json.loads(r[1])
                result.append({"vec": self._dict_to_vector(feat), "meta": meta})
            except Exception:
                pass
        return result

    def _dict_to_vector(self, d: dict) -> np.ndarray:
        keys = ["ema_ratio", "rsi", "volume_spike", "atr_ratio", "ema7_25_aligned"]
        return np.array([float(d.get(k, 0)) for k in keys], dtype=float)

    def _cosine_batch(self, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        nv = np.linalg.norm(vec)
        if nv == 0:
            return np.zeros(len(matrix))
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-10
        return matrix.dot(vec) / (norms * nv)

    def _init_db(self) -> None:
        PATTERN_DB.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT UNIQUE, symbol TEXT,
                    features_json TEXT, meta_json TEXT, created_at TEXT
                )
            """)

    def _conn(self):
        import contextlib
        @contextlib.contextmanager
        def _ctx():
            c = sqlite3.connect(PATTERN_DB)
            try:
                yield c
                c.commit()
            finally:
                c.close()
        return _ctx()
