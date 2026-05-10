"""Rich terminal dashboard for the CENTINA orchestrator."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from src.agents.guardian import Guardian
    from src.agents.scout import Scout


logger = logging.getLogger(__name__)
REFRESH_INTERVAL = 2.0


class TerminalUI:
    """Live terminal UI plus threaded command input."""

    def __init__(
        self,
        mode: str,
        capital_usdt: float,
        scout: "Scout | None" = None,
        guardian: "Guardian | None" = None,
        daily_target: float = 30.0,
        on_command: Callable[[str], None] | None = None,
    ):
        self._mode = mode
        self._capital = capital_usdt
        self._scout = scout
        self._guardian = guardian
        self._target = daily_target
        self._on_command = on_command

        self._daily_pnl = 0.0
        self._total_pnl = 0.0
        self._cb_level = 0
        self._status_msg = "Demarrage..."
        self._last_cmd = ""
        self._console = Console(legacy_windows=False)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="terminal-ui")
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()

    def update_pnl(self, daily: float, total: float) -> None:
        self._daily_pnl = daily
        self._total_pnl = total

    def update_cb_level(self, level: int) -> None:
        self._cb_level = level

    def update_mode(self, mode: str) -> None:
        self._mode = mode

    def set_status(self, msg: str) -> None:
        self._status_msg = msg

    async def run(self) -> None:
        input_task = asyncio.create_task(self._input_loop(), name="terminal_ui_input")
        with Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=0.5,
            screen=False,
        ) as live:
            while True:
                try:
                    await self._drain_commands()
                    self._refresh_runtime_state()
                    live.update(self._build_layout())
                    await asyncio.sleep(REFRESH_INTERVAL)
                except asyncio.CancelledError:
                    input_task.cancel()
                    self._executor.shutdown(wait=False)
                    raise
                except Exception as e:
                    logger.debug("TerminalUI render error: %s", e)
                    await asyncio.sleep(REFRESH_INTERVAL)

    async def _input_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            try:
                raw = await loop.run_in_executor(self._executor, input, "CENTINA> ")
                raw = raw.strip()
                if raw:
                    await self._input_queue.put(raw)
            except asyncio.CancelledError:
                raise
            except EOFError:
                return
            except Exception as e:
                logger.debug("TerminalUI input error: %s", e)
                await asyncio.sleep(1)

    async def _drain_commands(self) -> None:
        while not self._input_queue.empty():
            raw = await self._input_queue.get()
            self._last_cmd = raw
            self._status_msg = f"Derniere commande : {raw}"
            if self._on_command:
                self._on_command(raw)

    def _refresh_runtime_state(self) -> None:
        if not self._guardian:
            return
        try:
            if hasattr(self._guardian, "get_daily_pnl"):
                self._daily_pnl = float(self._guardian.get_daily_pnl())
            if hasattr(self._guardian, "get_total_pnl"):
                self._total_pnl = float(self._guardian.get_total_pnl())
        except Exception:
            pass

    def _build_layout(self):
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        cb_colours = {0: "green", 1: "yellow", 2: "orange3", 3: "red", 4: "bold red"}
        cb_col = cb_colours.get(self._cb_level, "white")
        mode_col = {"ADVISOR": "cyan", "AUTO": "yellow", "PAPER": "dim", "SEMI": "blue"}.get(
            self._mode, "white"
        )
        pnl_col = "green" if self._daily_pnl >= 0 else "red"

        top_bar = Panel(
            f"[bold cyan]CENTINA v5.0[/bold cyan] | "
            f"[{mode_col}]{self._mode}[/{mode_col}] | "
            f"Capital: [bold]{self._capital:.0f}[/bold] USDT | "
            f"PnL jour: [{pnl_col}]{self._daily_pnl:+.2f}[/{pnl_col}] EUR | "
            f"CB: [{cb_col}]L{self._cb_level}[/{cb_col}] | "
            f"[dim]{now}[/dim]",
            style="on #0a0a0a",
            padding=(0, 1),
        )

        columns = Columns(
            [
                Panel(self._opportunity_table(), border_style="cyan", padding=(0, 1)),
                Panel(self._positions_table(), border_style="green", padding=(0, 1)),
            ],
            equal=True,
        )

        pct = min(1.0, max(0.0, self._daily_pnl / self._target)) if self._target > 0 else 0.0
        filled = int(pct * 40)
        bar = "#" * filled + "-" * (40 - filled)
        progress_panel = Panel(
            f"Objectif jour : [cyan]{bar}[/cyan] "
            f"[bold]{self._daily_pnl:.2f}[/bold] / {self._target:.0f} EUR "
            f"({pct*100:.0f}%)",
            border_style="dim",
            padding=(0, 1),
        )

        cmd_panel = Panel(
            f"[dim]{self._status_msg}[/dim]\n"
            "[bold cyan]>[/bold cyan] GO [SYM]  PASSE  STATUT  RAPPORT  OPPS  STRATS  HIST  "
            "AUTO ON/OFF  PAPER ON/OFF  VOICE ON/OFF  BACKTEST  OPTIMISE  EXIT",
            title="[bold]Commandes[/bold]",
            border_style="dim",
            padding=(0, 1),
        )

        return Group(top_bar, columns, progress_panel, cmd_panel)

    def _opportunity_table(self) -> Table:
        table = Table(
            "Score",
            "Symbole",
            "Strategie",
            "Prix",
            "Signal",
            title="[bold]TOP OPPORTUNITES[/bold]",
            style="cyan",
            header_style="bold cyan",
            min_width=40,
        )
        if not self._scout:
            table.add_row("[dim]-", "initialisation", "-", "-", "-")
            return table
        try:
            opps = self._scout.get_last_scan() or []
            for opp in sorted(opps, key=lambda o: getattr(o, "composite_score", 0), reverse=True)[:8]:
                score = float(getattr(opp, "composite_score", 0) or 0)
                score_col = "green" if score >= 93 else "yellow" if score >= 78 else "dim"
                table.add_row(
                    f"[{score_col}]{score:.1f}[/{score_col}]",
                    f"[bold]{getattr(opp, 'symbol', '?')}[/bold]",
                    str(getattr(opp, "suggested_strategy", "?")),
                    f"{float(getattr(opp, 'best_price', 0) or 0):.6g}",
                    "PRIME" if score >= 93 else "GO" if score >= 78 else "-",
                )
        except Exception as e:
            logger.debug("TerminalUI opportunities error: %s", e)
        if table.row_count == 0:
            table.add_row("[dim]-", "aucune opportunite", "-", "-", "-")
        return table

    def _positions_table(self) -> Table:
        table = Table(
            "Symbole",
            "Entree",
            "TP1",
            "TP2",
            "SL",
            title="[bold]POSITIONS OUVERTES[/bold]",
            style="green",
            header_style="bold green",
            min_width=40,
        )
        if self._guardian:
            try:
                for sym, pos in self._guardian.get_positions().items():
                    levels = getattr(pos, "oco_levels", None)
                    table.add_row(
                        f"[bold]{sym}[/bold]",
                        f"{float(getattr(pos, 'entry_price', 0) or 0):.6g}",
                        f"{float(getattr(levels, 'tp1', 0) or 0):.6g}",
                        f"{float(getattr(levels, 'tp2', 0) or 0):.6g}",
                        f"{float(getattr(levels, 'sl', 0) or 0):.6g}",
                    )
            except Exception as e:
                logger.debug("TerminalUI positions error: %s", e)
        if table.row_count == 0:
            table.add_row("[dim]-", "aucune", "-", "-", "-")
        return table
