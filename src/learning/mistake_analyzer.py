from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path("data/cortex.db")


@dataclass
class MistakePattern:
    mistake_type:  str
    count:         int
    avg_loss_pct:  float
    total_loss:    float
    common_regime: str
    common_hour:   int
    recommendation: str


class MistakeAnalyzer:
    """
    Analyzes losing trades to identify systematic mistake patterns.
    Categories: false breakout, overleveraged, bad timing, trend reversal,
    news-driven stop, manipulation victim, poor exit.
    """

    MISTAKE_RULES = {
        "false_breakout": {
            "condition": lambda t: t.get("exit_reason") == "sl" and t.get("duration_h", 99) < 4,
            "recommendation": "Confirm breakout with 2 candles above resistance before entry",
        },
        "bad_timing": {
            "condition": lambda t: t.get("exit_reason") == "sl" and t.get("entry_hour", 12) in [0, 1, 2, 3, 4, 5],
            "recommendation": "Avoid entries between 00:00-06:00 UTC (low liquidity)",
        },
        "quick_reversal": {
            "condition": lambda t: t.get("exit_reason") == "sl" and t.get("duration_h", 99) < 2,
            "recommendation": "Increase ATR multiplier for SL to 1.5× to survive noise",
        },
        "no_tp1_hit": {
            "condition": lambda t: not t.get("tp1_hit") and t.get("pnl_pct", 0) < -1,
            "recommendation": "Review entry quality score — ensure score >= 78 before entry",
        },
        "trailing_stop_premature": {
            "condition": lambda t: t.get("exit_reason") == "trailing" and t.get("pnl_pct", 0) < 1.5,
            "recommendation": "Widen trailing stop ATR multiplier from 1.0 to 1.5",
        },
    }

    def __init__(self, db_path: Path | None = None):
        self.db = db_path or DB_PATH
        self._init_db()

    def record_trade(self, trade: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO mistakes_log
                   (symbol, strategy, pnl_pct, exit_reason, duration_h,
                    tp1_hit, entry_hour, regime, recorded_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    trade.get("symbol", ""),
                    trade.get("strategy", "B"),
                    trade.get("pnl_pct", 0),
                    trade.get("exit_reason", ""),
                    trade.get("duration_h", 0),
                    1 if trade.get("tp1_hit") else 0,
                    trade.get("entry_hour", 12),
                    trade.get("regime", ""),
                    datetime.utcnow().isoformat(),
                ),
            )

    def analyze(self, min_trades: int = 5) -> list[MistakePattern]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM mistakes_log WHERE pnl_pct < 0 ORDER BY recorded_at DESC LIMIT 500"
            ).fetchall()

        if len(rows) < min_trades:
            return []

        trades_list = [dict(r) for r in rows]
        patterns: list[MistakePattern] = []

        for name, rule in self.MISTAKE_RULES.items():
            matching = [t for t in trades_list if self._safe_check(rule["condition"], t)]
            if len(matching) < 3:
                continue

            losses    = [t["pnl_pct"] for t in matching]
            regimes   = [t.get("regime", "") for t in matching]
            hours     = [t.get("entry_hour", 12) for t in matching]

            from collections import Counter
            patterns.append(MistakePattern(
                mistake_type=name,
                count=len(matching),
                avg_loss_pct=round(float(np.mean(losses)), 2),
                total_loss=round(float(sum(losses)), 2),
                common_regime=Counter(regimes).most_common(1)[0][0] if regimes else "",
                common_hour=Counter(hours).most_common(1)[0][0]    if hours   else 12,
                recommendation=rule["recommendation"],
            ))

        patterns.sort(key=lambda p: abs(p.total_loss), reverse=True)
        return patterns

    def get_report(self) -> pd.DataFrame:
        patterns = self.analyze()
        if not patterns:
            return pd.DataFrame()
        return pd.DataFrame([{
            "mistake":        p.mistake_type,
            "count":          p.count,
            "avg_loss":       p.avg_loss_pct,
            "total_loss":     p.total_loss,
            "common_regime":  p.common_regime,
            "peak_hour":      p.common_hour,
            "recommendation": p.recommendation,
        } for p in patterns])

    def get_improvement_summary(self) -> str:
        patterns = self.analyze()
        if not patterns:
            return "No significant mistake patterns detected yet."

        lines = ["MISTAKE ANALYSIS — Top Issues:"]
        for i, p in enumerate(patterns[:3], 1):
            lines.append(
                f"{i}. {p.mistake_type.upper()} ({p.count} trades, "
                f"avg loss {p.avg_loss_pct:.1f}%): {p.recommendation}"
            )
        return "\n".join(lines)

    # ── Private ───────────────────────────────────────────────────────

    def _safe_check(self, condition, trade: dict) -> bool:
        try:
            return bool(condition(trade))
        except Exception:
            return False

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mistakes_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT,
                    strategy    TEXT,
                    pnl_pct     REAL,
                    exit_reason TEXT,
                    duration_h  REAL,
                    tp1_hit     INTEGER,
                    entry_hour  INTEGER,
                    regime      TEXT,
                    recorded_at TEXT
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        return conn
