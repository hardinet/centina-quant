from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

SNAPSHOTS_FILE = Path("data/capital_snapshots.json")


class Compounder:
    """
    Tracks capital growth and compounds Kelly sizing after each winning trade.
    Generates projections: 30d / 90d / 180d / 365d at current win-rate.
    """

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self._capital = initial_capital
        self._snapshots: list[dict] = self._load_snapshots()

    @property
    def capital(self) -> float:
        return self._capital

    def update(self, pnl_usdt: float) -> float:
        """Called after every closed trade. Returns new capital."""
        self._capital += pnl_usdt
        self._capital = max(0.0, self._capital)

        growth_pct = (self._capital - self.initial_capital) / self.initial_capital * 100
        colour = "green" if pnl_usdt >= 0 else "red"
        console.print(
            f"[bold {colour}]Capital: {self.initial_capital:.2f} USDT "
            f"→ {self._capital:.2f} USDT "
            f"({growth_pct:+.2f}%)[/bold {colour}]"
        )

        self._save_snapshot()
        return self._capital

    def project(self, daily_win_rate: float = 0.60, avg_gain_net_pct: float = 4.8, trades_per_day: float = 3.0) -> dict:
        """Projects capital growth over 30/90/180/365 days."""
        daily_edge = trades_per_day * daily_win_rate * (avg_gain_net_pct / 100)
        cap = self._capital
        projections = {}
        for days in (30, 90, 180, 365):
            projected = cap * (1 + daily_edge) ** days
            projections[f"{days}d"] = round(projected, 2)
        return projections

    def print_projections(self, win_rate: float = 0.60, avg_gain: float = 4.8, tpd: float = 3.0) -> None:
        proj = self.project(win_rate, avg_gain, tpd)
        table = Table(title="Capital Projections", border_style="cyan")
        table.add_column("Horizon", style="bold")
        table.add_column("Projected USDT", justify="right")
        table.add_column("× growth", justify="right")
        for label, val in proj.items():
            mult = val / self._capital if self._capital > 0 else 1
            table.add_row(label, f"{val:,.2f}", f"×{mult:.1f}")
        console.print(table)

    def days_to_target(self, target: float = 10000.0, win_rate: float = 0.60,
                       avg_gain: float = 4.8, tpd: float = 3.0) -> int:
        daily_edge = tpd * win_rate * (avg_gain / 100)
        if daily_edge <= 0 or self._capital <= 0:
            return -1
        import math
        return math.ceil(math.log(target / self._capital) / math.log(1 + daily_edge))

    def _save_snapshot(self) -> None:
        SNAPSHOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        self._snapshots = [s for s in self._snapshots if s["date"] != today]
        self._snapshots.append({"date": today, "capital": self._capital})
        SNAPSHOTS_FILE.write_text(json.dumps(self._snapshots[-365:], indent=2))

    def _load_snapshots(self) -> list[dict]:
        if SNAPSHOTS_FILE.exists():
            try:
                return json.loads(SNAPSHOTS_FILE.read_text())
            except Exception:
                pass
        return []
