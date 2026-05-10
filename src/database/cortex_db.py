from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator

DB_PATH = Path("data/cortex.db")

DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,
    strategy        TEXT    DEFAULT 'B',
    entry_price     REAL    NOT NULL,
    exit_price      REAL,
    sl_price        REAL,
    tp1_price       REAL,
    tp2_price       REAL,
    qty             REAL    NOT NULL,
    pnl_usdt        REAL,
    pnl_pct         REAL,
    exit_reason     TEXT,
    timeframe       TEXT,
    score_at_entry  REAL,
    regime          TEXT,
    tp1_hit         INTEGER DEFAULT 0,
    tp2_hit         INTEGER DEFAULT 0,
    fee_usdt        REAL    DEFAULT 0,
    paper           INTEGER DEFAULT 0,
    opened_at       TEXT    NOT NULL,
    closed_at       TEXT
);

CREATE TABLE IF NOT EXISTS performance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at   TEXT    NOT NULL,
    total_trades  INTEGER NOT NULL,
    win_trades    INTEGER NOT NULL,
    loss_trades   INTEGER NOT NULL,
    win_rate      REAL    NOT NULL,
    total_pnl     REAL    NOT NULL,
    avg_win       REAL,
    avg_loss      REAL,
    profit_factor REAL,
    max_drawdown  REAL,
    capital_usdt  REAL
);

CREATE TABLE IF NOT EXISTS historical_simulations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT,
    strategy        TEXT,
    timeframe       TEXT,
    start_date      TEXT,
    end_date        TEXT,
    total_return    REAL,
    cagr            REAL,
    sharpe          REAL,
    max_drawdown    REAL,
    win_rate        REAL,
    nb_trades       INTEGER,
    quality_score   INTEGER,
    params          TEXT,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS correlation_matrices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT,
    variable    TEXT,
    window      INTEGER,
    correlation REAL,
    method      TEXT,
    computed_at TEXT
);

CREATE TABLE IF NOT EXISTS news_correlations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword       TEXT,
    symbol        TEXT,
    pump_prob     REAL,
    actual_gain   REAL,
    event_date    TEXT,
    recorded_at   TEXT
);

CREATE TABLE IF NOT EXISTS whale_alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,
    amount_usd    REAL,
    direction     TEXT,
    exchange      TEXT,
    alert_at      TEXT
);

CREATE TABLE IF NOT EXISTS pattern_matches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT,
    similarity    REAL,
    outcome_pct   REAL,
    matched_at    TEXT
);

CREATE TABLE IF NOT EXISTS web_knowledge (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    query         TEXT,
    source_url    TEXT,
    params        TEXT,
    credibility   REAL,
    test_sharpe   REAL,
    learned_at    TEXT
);

CREATE TABLE IF NOT EXISTS macro_data_cache (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id     TEXT,
    date          TEXT,
    value         REAL,
    cached_at     TEXT,
    UNIQUE(series_id, date)
);
"""


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    qty: float
    timeframe: str = ""
    score_at_entry: float = 0.0
    opened_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    exit_price: float | None = None
    pnl_usdt: float | None = None
    pnl_pct: float | None = None
    exit_reason: str | None = None
    closed_at: str | None = None
    id: int | None = None


class CortexDB:

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def insert_trade(self, trade: TradeRecord) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (symbol, side, entry_price, qty, timeframe, score_at_entry, opened_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (trade.symbol, trade.side, trade.entry_price, trade.qty,
                 trade.timeframe, trade.score_at_entry, trade.opened_at),
            )
            return cur.lastrowid

    def save_simulation(self, sim: dict) -> None:
        import json
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO historical_simulations
                   (symbol, strategy, timeframe, start_date, end_date,
                    total_return, cagr, sharpe, max_drawdown, win_rate,
                    nb_trades, quality_score, params, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sim.get("symbol"), sim.get("strategy"), sim.get("timeframe"),
                    sim.get("start_date"), sim.get("end_date"),
                    sim.get("total_return"), sim.get("cagr"), sim.get("sharpe"),
                    sim.get("max_drawdown"), sim.get("win_rate"), sim.get("nb_trades"),
                    sim.get("quality_score"), json.dumps(sim.get("params", {})),
                    datetime.utcnow().isoformat(),
                ),
            )

    def save_correlation(self, symbol: str, variable: str, window: int,
                         correlation: float, method: str = "pearson") -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO correlation_matrices
                   (symbol, variable, window, correlation, method, computed_at)
                   VALUES (?,?,?,?,?,?)""",
                (symbol, variable, window, correlation, method, datetime.utcnow().isoformat()),
            )

    def get_latest_simulations(self, symbol: str | None = None, limit: int = 20) -> list[dict]:
        sql = "SELECT * FROM historical_simulations"
        args: list = []
        if symbol:
            sql += " WHERE symbol = ?"
            args.append(symbol)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        pnl_usdt: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        closed_at = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades
                   SET exit_price=?, pnl_usdt=?, pnl_pct=?, exit_reason=?, closed_at=?
                   WHERE id=?""",
                (exit_price, pnl_usdt, pnl_pct, exit_reason, closed_at, trade_id),
            )

    def close_trade_by_symbol(
        self,
        symbol: str,
        exit_price: float,
        pnl_usdt: float,
        pnl_pct: float,
        exit_reason: str,
        tp1_hit: bool = False,
        tp2_hit: bool = False,
    ) -> None:
        closed_at = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE trades
                   SET exit_price=?, pnl_usdt=?, pnl_pct=?, exit_reason=?,
                       closed_at=?, tp1_hit=?, tp2_hit=?
                   WHERE symbol=? AND closed_at IS NULL""",
                (exit_price, pnl_usdt, pnl_pct, exit_reason,
                 closed_at, int(tp1_hit), int(tp2_hit), symbol),
            )

    def insert_trade_full(
        self,
        symbol: str,
        entry_price: float,
        qty: float,
        strategy: str = "B",
        score: float = 0.0,
        sl_price: float = 0.0,
        tp1_price: float = 0.0,
        tp2_price: float = 0.0,
        regime: str = "",
        paper: bool = False,
        timeframe: str = "4h",
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (symbol, side, strategy, entry_price, qty, sl_price, tp1_price, tp2_price,
                    score_at_entry, regime, paper, timeframe, opened_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, "BUY", strategy, entry_price, qty, sl_price, tp1_price, tp2_price,
                 score, regime, int(paper), timeframe, datetime.utcnow().isoformat()),
            )
            return cur.lastrowid

    def get_open_trades(self) -> list[TradeRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE closed_at IS NULL"
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_recent_trades(self, limit: int = 50) -> list[TradeRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    # ------------------------------------------------------------------
    # Performance snapshot
    # ------------------------------------------------------------------

    def snapshot_performance(self, capital_usdt: float) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pnl_usdt FROM trades WHERE closed_at IS NOT NULL"
            ).fetchall()

        pnls = [r[0] for r in rows if r[0] is not None]
        total = len(pnls)
        if total == 0:
            return {}

        wins  = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / total
        avg_win  = sum(wins)  / len(wins)  if wins   else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        pf = abs(sum(wins) / sum(losses)) if losses else float("inf")

        # Simplified drawdown: cumulative peak-to-trough
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = (peak - equity) / max(peak, 1)
            if dd > max_dd:
                max_dd = dd

        snap = {
            "recorded_at":   datetime.utcnow().isoformat(),
            "total_trades":  total,
            "win_trades":    len(wins),
            "loss_trades":   len(losses),
            "win_rate":      round(win_rate, 4),
            "total_pnl":     round(sum(pnls), 2),
            "avg_win":       round(avg_win, 2),
            "avg_loss":      round(avg_loss, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown":  round(max_dd, 4),
            "capital_usdt":  capital_usdt,
        }

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO performance
                   (recorded_at,total_trades,win_trades,loss_trades,win_rate,
                    total_pnl,avg_win,avg_loss,profit_factor,max_drawdown,capital_usdt)
                   VALUES (:recorded_at,:total_trades,:win_trades,:loss_trades,:win_rate,
                    :total_pnl,:avg_win,:avg_loss,:profit_factor,:max_drawdown,:capital_usdt)""",
                snap,
            )
        return snap

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
        d = dict(row)
        return TradeRecord(
            id=d["id"],
            symbol=d["symbol"],
            side=d["side"],
            entry_price=d["entry_price"],
            qty=d["qty"],
            timeframe=d["timeframe"] or "",
            score_at_entry=d["score_at_entry"] or 0.0,
            opened_at=d["opened_at"],
            exit_price=d["exit_price"],
            pnl_usdt=d["pnl_usdt"],
            pnl_pct=d["pnl_pct"],
            exit_reason=d["exit_reason"],
            closed_at=d["closed_at"],
        )
