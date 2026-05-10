from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


class DailyTargetCalculator:
    """
    Computes and displays the morning brief with daily target and projections.
    """

    TARGET_LOW  = 10.0   # €/day minimum
    TARGET_HIGH = 30.0   # €/day maximum
    GOAL_CAPITAL = 10_000.0

    def compute(
        self,
        capital_usdt: float,
        win_rate_30d: float,
        avg_gain_net_pct: float,
        trades_per_day: float,
    ) -> dict:
        # Expected daily PnL
        pnl_expected = trades_per_day * win_rate_30d * (avg_gain_net_pct / 100) * capital_usdt

        import math
        daily_edge = trades_per_day * win_rate_30d * (avg_gain_net_pct / 100)
        if daily_edge > 0 and capital_usdt > 0:
            days_to_goal = math.ceil(math.log(self.GOAL_CAPITAL / capital_usdt) / math.log(1 + daily_edge))
        else:
            days_to_goal = -1

        return {
            "capital_usdt":     round(capital_usdt, 2),
            "win_rate_30d":     round(win_rate_30d, 4),
            "avg_gain_net_pct": round(avg_gain_net_pct, 2),
            "trades_per_day":   round(trades_per_day, 1),
            "pnl_expected":     round(pnl_expected, 2),
            "days_to_goal":     days_to_goal,
            "target_low":       self.TARGET_LOW,
            "target_high":      self.TARGET_HIGH,
        }

    def print_brief(
        self,
        capital_usdt: float,
        win_rate_30d: float = 0.60,
        avg_gain_net_pct: float = 4.8,
        trades_per_day: float = 3.0,
    ) -> None:
        d = self.compute(capital_usdt, win_rate_30d, avg_gain_net_pct, trades_per_day)
        pnl_ok = self.TARGET_LOW <= d["pnl_expected"] <= self.TARGET_HIGH
        pnl_colour = "green" if pnl_ok else "yellow"

        text = Text()
        text.append(f"  Capital      : {d['capital_usdt']:.2f} USDT\n",        style="bold")
        text.append(f"  Objectif     : {self.TARGET_LOW:.0f}-{self.TARGET_HIGH:.0f}€/jour\n")
        text.append(f"  Win rate 30j : {d['win_rate_30d']:.1%}\n")
        text.append(f"  Gain moy NET : {d['avg_gain_net_pct']:.1f}%/trade\n")
        text.append(f"  Trades/jour  : {d['trades_per_day']:.1f}\n")
        text.append(f"  PnL attendu  : ~{d['pnl_expected']:.0f}€\n",           style=pnl_colour)
        if d["days_to_goal"] > 0:
            text.append(f"  Jours pour 10 000€ : ~{d['days_to_goal']}\n",      style="cyan")
        else:
            text.append(f"  Jours pour 10 000€ : N/A (edge négatif)\n",        style="red")

        console.print(Panel(text, title="[bold cyan]BRIEF MATINAL CENTINA v5.0[/bold cyan]", border_style="cyan"))
