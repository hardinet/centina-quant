from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.brain.daily_target_calculator import DailyTargetCalculator
from src.connectors.binance_rest import BinanceREST
from src.connectors.binance_ws import BinanceWebSocket
from src.core.event_bus import Event, EventBus, EventType
from src.core.lifecycle import graceful_shutdown, register_shutdown, setup_signal_handlers, wait_for_shutdown
from src.database.cortex_db import CortexDB
from src.notifications.apprise_notifier import AppriseNotifier
from src.security.audit_logger import AuditLogger
from src.security.circuit_breaker import CBLevel, CircuitBreaker

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console(legacy_windows=False)

ACTIVATION_FILE    = Path("data/.activation_date")
PAPER_DAYS_REQUIRED = 7
VERSION            = "5.0"

BANNER = f"""[bold cyan]
  ██████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗ █████╗
 ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔══██╗
 ██║     █████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║███████║
 ██║     ██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══██║
 ╚██████╗███████╗██║ ╚████║   ██║   ██║██║ ╚████║██║  ██║
  ╚═════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝[/bold cyan]
                  [bold white]OMNI-QUANT v{VERSION}[/bold white]"""

HELP_TEXT = """
[bold cyan]Commandes disponibles :[/bold cyan]

  [bold white]── TRADING ──[/bold white]
  [green]GO[/green]           → Exécuter la prochaine opportunité dans la file
  [green]GO BTCUSDT[/green]   → Forcer l'exécution sur un symbole spécifique
  [yellow]PASSE[/yellow]        → Ignorer l'opportunité courante

  [bold white]── INFOS ──[/bold white]
  [cyan]STATUT[/cyan]       → Positions ouvertes + PnL + Circuit Breaker
  [cyan]SOLDE[/cyan]        → Solde Binance (USDT + actifs ouverts)
  [cyan]RAPPORT[/cyan]      → Rapport quotidien complet + audio
  [cyan]OPPS[/cyan]         → Opportunités scannées (top 10)
  [cyan]STRATS[/cyan]       → Stratégies A/B/C + poids par régime
  [cyan]HIST[/cyan]         → 10 derniers trades clôturés

  [bold white]── MODES ──[/bold white]
  [blue]ADVISOR[/blue]      → Mode ADVISOR : chaque signal demande confirmation GO/PASSE
  [blue]SEMI[/blue]         → Mode SEMI-AUTO : PRIME (≥85) auto, les autres demandent GO
  [blue]AUTO[/blue]         → Mode AUTO : tout exécuter automatiquement (score ≥ 78)
  [blue]PAPER[/blue]        → Basculer simulation PAPER on/off
  [blue]VOICE[/blue]        → Activer/désactiver interface vocale

  [bold white]── OUTILS ──[/bold white]
  [magenta]BACKTEST[/magenta]    → Backtest rapide 30 jours sur données synthétiques
  [magenta]OPTIMISE[/magenta]    → Optimisation Optuna 50 itérations
  [dim]AIDE[/dim]         → Cette aide
  [dim]EXIT[/dim]         → Arrêter CENTINA proprement
"""


class CentinaOrchestrator:
    """
    Coordinateur central CENTINA v5.0.

    Modes :
      PAPER  — ordres simulés (7 jours obligatoires)
      ADVISOR — scan + proposition → confirmation manuelle GO/PASSE
      AUTO   — exécution automatique dès score ≥ 78

    Interface interactive via stdin (thread séparé → asyncio.Queue).
    """

    def __init__(self, mode: str = "ADVISOR", enable_voice: bool = False, enable_ui: bool = True):
        self.mode          = mode
        self.auto_mode     = (mode == "AUTO")
        self.semi_mode     = (mode == "SEMI")
        self.paper_mode    = True   # forced until gate clears
        self.enable_voice  = enable_voice
        self.enable_ui     = enable_ui
        self._voice        = None   # VoiceOrchestrator, set in run()
        self._paused       = False
        self._loop: asyncio.AbstractEventLoop | None = None

        self.bus    = EventBus()
        self.db     = CortexDB()
        self.notif  = AppriseNotifier()
        self.audit  = AuditLogger()

        api_key    = os.getenv("BINANCE_API_KEY",    "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        capital    = float(os.getenv("CAPITAL_USDT", "200"))

        self.rest = BinanceREST(api_key, api_secret)
        self.ws   = BinanceWebSocket(api_key, api_secret)
        self.cb   = CircuitBreaker(
            capital_baseline=capital,
            notify_callback=lambda m: self.notif.notify(m),
        )
        self.capital = capital

        self._pending_opps: list[dict] = []      # queue of scored opportunities
        self._cmd_queue: asyncio.Queue[str] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cli")
        self._tasks: list[asyncio.Task] = []
        self._guardian = None
        self._scout    = None
        self._terminal_ui = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._print_banner()
        setup_signal_handlers()

        await self.rest.connect()
        await self.ws.connect()
        register_shutdown(self._shutdown)

        self._paper_gate_check()
        self._print_daily_brief()

        from src.agents.scout    import Scout
        from src.agents.analyst  import Analyst
        from src.agents.historian import Historian
        from src.agents.guardian import Guardian
        from src.agents.sentinel import Sentinel

        # ── Voice Agent (écoute active + parle) ──────────────────────
        if self.enable_voice:
            from src.voice.voice_agent import VoiceAgent
            self._voice = VoiceAgent(orchestrator=self)
            self._voice.start()

        from src.interface.web_bridge import WebBridge
        self._bridge = WebBridge(
            scout=None, guardian=None, cb=self.cb,
            mode=self.mode, capital=self.capital,
        )

        self._scout   = Scout(self.rest, self.bus)
        historian     = Historian(self.rest, self.db, self.bus)
        analyst       = Analyst(self.rest, self.db, self.bus, historian)
        self._analyst = analyst   # kept for post-trade feedback
        self._guardian = Guardian(
            self.rest, self.db, self.bus, self.audit, self.cb, self.notif,
            voice=self._voice,
        )
        sentinel      = Sentinel(self.rest, self.bus, self.notif)

        # Update bridge refs after agents created
        self._bridge._scout    = self._scout
        self._bridge._guardian = self._guardian
        if self.enable_voice:
            self._bridge.set_voice_active(True)

        if self.enable_ui:
            from src.interface.terminal_ui import TerminalUI

            def _terminal_command(raw: str) -> None:
                if not self._loop:
                    return
                asyncio.run_coroutine_threadsafe(
                    self._dispatch_cmd(raw.upper().strip(), raw.strip()),
                    self._loop,
                )

            self._terminal_ui = TerminalUI(
                mode=self.mode,
                capital_usdt=self.capital,
                scout=self._scout,
                guardian=self._guardian,
                daily_target=float(os.getenv("TARGET_EUR_MAX", "30")),
                on_command=_terminal_command,
            )

        # Subscribe to validated opportunities
        self._opp_queue = self.bus.subscribe(
            EventType.OPPORTUNITY_VALIDATED,
            EventType.OPPORTUNITY_PRIME,
            EventType.TRADE_CLOSED,
            EventType.BTC_DANGER,
            EventType.TP1_HIT,
            EventType.TP2_HIT,
        )

        tasks_list = [
            asyncio.create_task(self._scout.run(),         name="scout"),
            asyncio.create_task(historian.run(),            name="historian"),
            asyncio.create_task(analyst.run(),              name="analyst"),
            asyncio.create_task(self._guardian.run(),       name="guardian"),
            asyncio.create_task(sentinel.run(),             name="sentinel"),
            asyncio.create_task(self._event_handler_loop(), name="event_handler"),
            asyncio.create_task(self._bridge.run(),         name="web_bridge"),
            asyncio.create_task(self._cmd_queue_loop(),     name="cmd_queue"),
        ]
        if self._terminal_ui:
            tasks_list.append(asyncio.create_task(self._terminal_ui.run(), name="terminal_ui"))
        else:
            tasks_list.append(asyncio.create_task(self._cli_loop(), name="cli"))
        self._tasks = tasks_list

        self.audit.log("SYSTEM_START", {"mode": self.mode, "version": VERSION})
        logger.info("Orchestrator: all agents started — mode=%s paper=%s", self.mode, self.paper_mode)

        await wait_for_shutdown()
        await graceful_shutdown(timeout=30.0)

    # ------------------------------------------------------------------
    # Event handler loop
    # ------------------------------------------------------------------

    async def _event_handler_loop(self) -> None:
        while True:
            try:
                event = await asyncio.wait_for(self._opp_queue.get(), timeout=1.0)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise

    async def _handle_event(self, event: Event) -> None:
        p = event.payload

        if event.type in (EventType.OPPORTUNITY_VALIDATED, EventType.OPPORTUNITY_PRIME):
            self._pending_opps.append(p)
            if len(self._pending_opps) > 20:
                self._pending_opps = self._pending_opps[-20:]

            is_prime = event.type == EventType.OPPORTUNITY_PRIME
            prime = "🔥 PRIME" if is_prime else ""
            console.print(
                f"\n[bold green]▶ SIGNAL {prime}[/bold green] "
                f"[cyan]{p['symbol']}[/cyan]  "
                f"score=[yellow]{p['score']:.1f}[/yellow]  "
                f"stratégie=[bold]{p.get('strategy','B')}[/bold]  "
                f"[dim]tape GO pour exécuter[/dim]"
            )
            # Push full analyst result to dashboard
            self._bridge.update_opportunity(p)
            ev_type = "PRIME" if is_prime else "INFO"
            sat_b   = p.get("satellite_bonus", 0)
            council = p.get("council_text", "")
            msg = (f"{'🔥 PRIME' if is_prime else '▶ Signal'} {p['symbol']} "
                   f"{p['score']:.1f}pts sat{sat_b:+d} {council[:40]}")
            self._bridge.add_event(msg, ev_type)

            if cmd == "VOICE ON":
                self._set_voice_enabled(True)
            elif self._voice:
                if is_prime:
                    from src.brain.oco_calculator import OCOCalculator
                    try:
                        _oco = OCOCalculator()
                        _lv  = _oco.calculate(p.get("entry_price", 0), p.get("atr", 0))
                        self._voice.announce_prime(
                            p["symbol"], int(p.get("score", 0)),
                            p.get("strategy", "B"), _lv.tp1, _lv.sl,
                        )
                    except Exception:
                        self._voice.announce_opportunity(
                            p["symbol"], int(p.get("score", 0)), p.get("strategy", "B")
                        )
                else:
                    self._voice.announce_opportunity(
                        p["symbol"], int(p.get("score", 0)), p.get("strategy", "B")
                    )

            is_auto_exec = (
                self.auto_mode
                or (self.semi_mode and is_prime and p.get("score", 0) >= 85)
            )
            if is_auto_exec and self.cb.level not in (CBLevel.STOPPED, CBLevel.TERMINATED):
                await self._execute_opportunity(p)
            elif self.semi_mode and not is_prime:
                console.print("[dim]  [SEMI] signal standard — tape GO pour exécuter[/dim]")

        elif event.type == EventType.TRADE_CLOSED:
            pnl = p.get("pnl_usdt", 0)
            colour = "green" if pnl >= 0 else "red"
            console.print(
                f"[{colour}]● CLOSE {p['symbol']} {p.get('reason','')} "
                f"{pnl:+.2f} USDT ({p.get('pnl_pct',0):+.2f}%)[/{colour}]"
            )
            self._bridge.record_pnl(pnl)
            icon = "✅" if pnl >= 0 else "❌"
            self._bridge.add_event(
                f"{icon} CLOSE {p['symbol']} {p.get('reason','')} "
                f"{pnl:+.2f}€ ({p.get('pnl_pct',0):+.2f}%)", "INFO"
            )
            # Remove from validated opps
            self._bridge._validated_opps = [
                o for o in self._bridge._validated_opps if o.get("symbol") != p.get("symbol")
            ]
            # Feed LLM Council accuracy
            try:
                council_verdict = p.get("council_verdict", "")
                if council_verdict and hasattr(self, "_analyst"):
                    was_correct = pnl > 0 and council_verdict in ("BUY", "PRIME")
                    for model in ("deepseek", "grok", "kimi", "openrouter", "gemini", "openai"):
                        self._analyst._llm_council.update_accuracy(model, was_correct)
            except Exception:
                pass

        elif event.type == EventType.BTC_DANGER:
            console.print(f"[bold red]⚠ BTC DANGER {p.get('change_pct',0):.2f}% — trading suspendu[/bold red]")
            self._bridge.add_event(f"⚠ BTC DANGER {p.get('change_pct',0):.2f}% — trading suspendu", "WARN")

        elif event.type == EventType.TP1_HIT:
            console.print(f"[green]✅ TP1 {p['symbol']} +{p.get('pnl_pct',0):.1f}% — SL→BE activé[/green]")
            self._bridge.add_event(f"✅ TP1 {p['symbol']} +{p.get('pnl_pct',0):.1f}% SL→BE", "INFO")

        elif event.type == EventType.TP2_HIT:
            console.print(f"[bold green]🚀 TP2 {p['symbol']} +{p.get('pnl_pct',0):.1f}% — trailing ACTIF[/bold green]")
            self._bridge.add_event(f"🚀 TP2 {p['symbol']} +{p.get('pnl_pct',0):.1f}% trailing actif", "INFO")

    # ------------------------------------------------------------------
    # cmd_queue_loop — lit data/cmd_queue.json (envoyé par dashboard/app.py)
    # ------------------------------------------------------------------

    async def _cmd_queue_loop(self) -> None:
        import pathlib
        queue_file = pathlib.Path("data/cmd_queue.json")
        while True:
            try:
                await asyncio.sleep(1.0)
                if not queue_file.exists():
                    continue
                try:
                    raw = queue_file.read_text(encoding="utf-8").strip()
                    if not raw:
                        continue
                    cmds = json.loads(raw)
                except Exception:
                    continue
                queue_file.write_text("[]", encoding="utf-8")  # vide immédiatement
                for entry in cmds:
                    await self._dispatch_web_cmd(entry)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("cmd_queue_loop error: %s", e)

    async def _dispatch_web_cmd(self, entry: dict) -> None:
        cmd = entry.get("cmd", "")
        if cmd == "SET_MODE":
            val = entry.get("value", "ADVISOR").upper()
            self._set_mode(val)
            self._bridge.add_event(f"Mode → {val}", "USER_CMD")
            console.print(f"[cyan][WEB] Mode → {val}[/cyan]")
        elif cmd == "SET_PARAMS":
            score_min = entry.get("score_min")
            if score_min:
                console.print(f"[cyan][WEB] Score min → {score_min}[/cyan]")
        elif cmd == "CB_RESET":
            self.cb.manual_reset()
            self._bridge.add_event("Circuit Breaker réinitialisé", "USER_CMD")
            console.print("[green][WEB] Circuit Breaker réinitialisé.[/green]")
        elif cmd == "PAUSE":
            self._paused = True
            self._bridge.add_event("⏸ Trading en PAUSE", "USER_CMD")
            console.print("[yellow][WEB] Trading en PAUSE.[/yellow]")
        elif cmd == "RESUME":
            self._paused = False
            self._bridge.add_event("▶ Trading REPRIS", "USER_CMD")
            console.print("[green][WEB] Trading REPRIS.[/green]")
        elif cmd == "CLOSE_ALL":
            self._bridge.add_event("🛑 Fermeture de toutes les positions", "USER_CMD")
            console.print("[bold red][WEB] Fermeture de toutes les positions…[/bold red]")
            if self._guardian:
                for sym in list(self._guardian.get_positions().keys()):
                    try:
                        await self._guardian.close_position(sym, reason="WEB_CLOSE_ALL")
                    except Exception as e:
                        logger.error("close_all error %s: %s", sym, e)
        elif cmd == "CLOSE_POSITION":
            sym = entry.get("symbol", "")
            if sym and self._guardian:
                try:
                    await self._guardian.close_position(sym, reason="WEB_CLOSE")
                    console.print(f"[yellow][WEB] Fermeture {sym}.[/yellow]")
                except Exception as e:
                    logger.error("close_position error %s: %s", sym, e)
        elif cmd == "EXECUTE":
            sym = entry.get("symbol", "")
            if sym:
                await self._cmd_go(sym)
        elif cmd == "FORCE_SCAN":
            self._bridge.add_event("Scan manuel demande", "USER_CMD")
            console.print("[cyan][WEB] Scan manuel demande.[/cyan]")
            if self._scout:
                try:
                    opps = await self._scout.scan()
                    self._scout._last_scan = opps
                    for opp in opps:
                        await self.bus.emit(Event(
                            EventType.OPPORTUNITY_DETECTED,
                            self._scout._opp_to_payload(opp),
                            source="web_force_scan",
                        ))
                    self._bridge.add_event(f"Scan termine: {len(opps)} opportunites", "INFO")
                except Exception as e:
                    logger.error("force_scan error: %s", e)
                    self._bridge.add_event(f"Scan erreur: {e}", "WARN")
        elif cmd == "RUN_BACKTEST":
            self._bridge.add_event("Backtest 30j lance", "USER_CMD")
            console.print("[cyan][WEB] Backtest rapide lance.[/cyan]")
            asyncio.create_task(self._run_backtest())
        elif cmd == "RUN_SIMULATION":
            self._bridge.add_event("Simulation historique lancee", "USER_CMD")
            console.print("[cyan][WEB] Simulation historique lancee.[/cyan]")
            asyncio.create_task(self._run_simulation())
        elif cmd == "USER_INPUT":
            text = entry.get("text", "").strip()
            if text:
                console.print(f"[dim][WEB] > {text}[/dim]")
                await self._dispatch_cmd(text.upper(), text)

    # ------------------------------------------------------------------
    # CLI loop (stdin via executor)
    # ------------------------------------------------------------------

    async def _cli_loop(self) -> None:
        loop = asyncio.get_event_loop()
        console.print(f"\n[dim]Tape [bold]AIDE[/bold] pour la liste des commandes.[/dim]\n")
        while True:
            try:
                raw = await loop.run_in_executor(self._executor, self._read_line, "CENTINA> ")
                cmd = raw.strip().upper()
                if cmd:
                    await self._dispatch_cmd(cmd, raw.strip())
            except asyncio.CancelledError:
                raise
            except EOFError:
                break
            except Exception as e:
                logger.debug("CLI error: %s", e)

    @staticmethod
    def _read_line(prompt: str) -> str:
        return input(prompt)

    def _set_mode(self, mode: str) -> None:
        mode = mode.upper()
        if mode not in {"ADVISOR", "SEMI", "AUTO", "PAPER"}:
            mode = "ADVISOR"
        self.auto_mode = (mode == "AUTO")
        self.semi_mode = (mode == "SEMI")
        self.mode = mode
        if mode == "PAPER":
            self.paper_mode = True
        if hasattr(self, "_bridge"):
            self._bridge.set_mode(mode)
        if self._terminal_ui:
            self._terminal_ui.update_mode(mode)

    def _set_paper_mode(self, enabled: bool) -> None:
        if not enabled and self._paper_gate_remaining_days() > 0:
            remaining = self._paper_gate_remaining_days()
            console.print(f"[yellow]LIVE bloque: encore {remaining} jour(s) de paper gate.[/yellow]")
            self.paper_mode = True
            return
        self.paper_mode = enabled

    def _paper_gate_remaining_days(self) -> int:
        if not ACTIVATION_FILE.exists():
            return PAPER_DAYS_REQUIRED
        try:
            activated_at = datetime.fromisoformat(ACTIVATION_FILE.read_text().strip())
            if activated_at.tzinfo is None:
                activated_at = activated_at.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - activated_at).days
            return max(0, PAPER_DAYS_REQUIRED - elapsed)
        except Exception:
            return PAPER_DAYS_REQUIRED

    def _set_voice_enabled(self, enabled: bool) -> None:
        if enabled and not self._voice:
            from src.voice.voice_agent import VoiceAgent
            self._voice = VoiceAgent(orchestrator=self)
            self._voice.start()
            if hasattr(self, "_bridge"):
                self._bridge.set_voice_active(True)
            console.print("[green]Voix activée.[/green]")
        elif not enabled and self._voice:
            self._voice.stop()
            self._voice = None
            if hasattr(self, "_bridge"):
                self._bridge.set_voice_active(False)
            console.print("[dim]Voix désactivée.[/dim]")

    async def _dispatch_cmd(self, cmd: str, raw: str) -> None:
        if cmd == "AIDE" or cmd == "HELP":
            console.print(HELP_TEXT)

        elif cmd.startswith("GO"):
            parts = raw.split()
            symbol = parts[1].upper() if len(parts) > 1 else None
            await self._cmd_go(symbol)

        elif cmd == "PASSE":
            if self._pending_opps:
                skipped = self._pending_opps.pop(0)
                console.print(f"[yellow]PASSE — {skipped['symbol']} ignoré.[/yellow]")
            else:
                console.print("[dim]Aucune opportunité en attente.[/dim]")

        elif cmd == "STATUT":
            self._print_status()

        elif cmd == "OPPS":
            self._print_pending_opps()

        elif cmd == "SOLDE":
            asyncio.create_task(self._print_solde())

        elif cmd == "ADVISOR":
            self.auto_mode  = False
            self.semi_mode  = False
            self.mode       = "ADVISOR"
            self._bridge.set_mode("ADVISOR")
            self._bridge.add_event("Mode → ADVISOR", "USER_CMD")
            console.print("[yellow]Mode ADVISOR[/yellow] — chaque signal demande confirmation GO/PASSE.")

        elif cmd == "SEMI":
            self.auto_mode  = False
            self.semi_mode  = True
            self.mode       = "SEMI"
            self._bridge.set_mode("SEMI")
            self._bridge.add_event("Mode → SEMI-AUTO", "USER_CMD")
            console.print("[cyan]Mode SEMI-AUTO[/cyan] — PRIME (≥85) exécutés automatiquement, les autres demandent GO.")

        elif cmd == "AUTO OFF":
            self._set_mode("ADVISOR")
            self._bridge.add_event("Mode -> ADVISOR", "USER_CMD")
            console.print("[yellow]Mode AUTO desactive - retour ADVISOR.[/yellow]")

        elif cmd in ("AUTO", "AUTO ON"):
            self.auto_mode  = True
            self.semi_mode  = False
            self.mode       = "AUTO"
            self._bridge.set_mode("AUTO")
            if self._terminal_ui:
                self._terminal_ui.update_mode("AUTO")
            self._bridge.add_event("Mode → AUTO", "USER_CMD")
            console.print("[bold green]Mode AUTO[/bold green] — tous les signaux ≥78 exécutés automatiquement.")

        elif cmd == "PAPER OFF":
            self._set_paper_mode(False)
            state = "[yellow]PAPER[/yellow]" if self.paper_mode else "[bold green]LIVE[/bold green]"
            console.print(f"Mode : {state}")

        elif cmd in ("PAPER", "PAPER ON"):
            if cmd == "PAPER ON":
                self._set_paper_mode(True)
            else:
                self.paper_mode = not self.paper_mode
            state = "[yellow]PAPER[/yellow]" if self.paper_mode else "[bold green]LIVE[/bold green]"
            console.print(f"Mode : {state}")

        elif cmd == "RAPPORT":
            self._print_rapport()

        elif cmd == "STRATS":
            self._print_strategies()

        elif cmd == "HIST":
            self._print_history()

        elif cmd in ("PAUSE",):
            self._paused = True
            console.print("[yellow]Trading en PAUSE.[/yellow]")

        elif cmd in ("RESUME", "REPRISE"):
            self._paused = False
            console.print("[green]Trading REPRIS.[/green]")

        elif cmd == "VOICE OFF":
            self._set_voice_enabled(False)

        elif cmd in ("VOICE", "VOICE ON"):
            if self._voice:
                self._voice.stop()
                self._voice = None
                console.print("[dim]Voix désactivée.[/dim]")
            else:
                from src.voice.voice_agent import VoiceAgent
                self._voice = VoiceAgent(orchestrator=self)
                self._voice.start()
                console.print("[green]Voix activée.[/green]")

        elif cmd == "BACKTEST":
            console.print("[cyan]Lancement backtest 30j…[/cyan]")
            asyncio.create_task(self._run_backtest())

        elif cmd == "OPTIMISE":
            console.print("[cyan]Lancement optimisation Optuna…[/cyan]")
            asyncio.create_task(self._run_optimise())

        elif cmd == "EXIT" or cmd == "QUIT":
            console.print("[bold red]Arrêt CENTINA…[/bold red]")
            for task in self._tasks:
                task.cancel()

        else:
            console.print(f"[dim]Commande inconnue '{cmd}'. Tape AIDE.[/dim]")

    # ------------------------------------------------------------------
    # GO command
    # ------------------------------------------------------------------

    async def _cmd_go(self, symbol: str | None) -> None:
        if symbol:
            opp = next((o for o in self._pending_opps if o["symbol"] == symbol), None)
            if not opp:
                # Try to find in last scan
                if self._scout:
                    opps = self._scout.get_last_scan()
                    raw_opp = next((o for o in opps if o.symbol == symbol), None)
                    if raw_opp:
                        opp = self._scout._opp_to_payload(raw_opp)
            if not opp:
                console.print(f"[red]Symbole {symbol} non trouvé dans les opportunités.[/red]")
                return
        elif self._pending_opps:
            opp = self._pending_opps.pop(0)
        else:
            console.print("[yellow]Aucune opportunité en attente. Attends le prochain scan.[/yellow]")
            return

        await self._execute_opportunity(opp)

    async def _execute_opportunity(self, opp: dict) -> None:
        if self._paused:
            console.print("[yellow]Trading en PAUSE — ordre bloqué. Tape REPRISE ou RESUME.[/yellow]")
            return

        symbol      = opp["symbol"]
        score       = opp.get("score", opp.get("composite_score", 0))
        strategy    = opp.get("strategy", opp.get("suggested_strategy", "B"))
        entry_price = opp.get("entry_price", opp.get("best_price", 0))
        atr         = opp.get("atr", opp.get("best_atr", entry_price * 0.01))

        if self.cb.level in (CBLevel.STOPPED, CBLevel.TERMINATED):
            console.print(f"[bold red]Circuit breaker {self.cb.level.name} — ordre bloqué.[/bold red]")
            return

        if self._guardian and self._guardian.open_positions_count() >= 3:
            console.print("[yellow]MAX 3 positions ouvertes — ordre bloqué.[/yellow]")
            return

        try:
            capital = await self.rest.get_usdt_balance()
        except Exception:
            capital = self.capital

        from src.brain.kelly_calculator import KellyCalculator
        from src.brain.oco_calculator import OCOCalculator
        kelly  = KellyCalculator()
        oco    = OCOCalculator()
        levels = oco.calculate(entry_price, atr)
        sl_pct = levels.sl_pct / 100
        kelly_r = kelly.from_signal_score(
            score, capital, sl_pct,
            open_positions=self._guardian.open_positions_count() if self._guardian else 0,
        )
        final_usdt = kelly_r.position_usdt * self.cb.position_size_multiplier
        qty        = round(final_usdt / entry_price, 6) if entry_price > 0 else 0

        if qty <= 0:
            console.print(f"[red]Taille de position trop petite pour {symbol}.[/red]")
            return

        self._print_trade_proposal(symbol, score, strategy, levels, kelly_r, final_usdt, capital)

        if self.paper_mode:
            console.print(f"[yellow][PAPER] Ordre simulé {symbol} qty={qty:.6f} @ {entry_price:.8f}[/yellow]")
            order_id = 0
        else:
            try:
                order  = await self.rest.place_limit_buy(symbol, qty, entry_price)
                order_id = order.get("orderId", 0)
                console.print(f"[bold green]✓ Ordre Binance placé {symbol} qty={qty:.6f} @ {entry_price:.8f}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]✗ Ordre échoué {symbol}: {e}[/bold red]")
                return

        # Record in DB
        trade_id = self.db.insert_trade_full(
            symbol=symbol,
            entry_price=entry_price,
            qty=qty,
            strategy=strategy,
            score=score,
            sl_price=levels.sl,
            tp1_price=levels.tp1,
            tp2_price=levels.tp2,
            paper=self.paper_mode,
        )

        # Notify Guardian via event
        await self.bus.emit(Event(EventType.TRADE_EXECUTED, {
            "symbol":      symbol,
            "entry_price": entry_price,
            "qty":         qty,
            "atr":         atr,
            "order_id":    order_id,
            "strategy":    strategy,
            "score":       score,
            "paper":       self.paper_mode,
        }, source="orchestrator"))

        self.audit.log("TRADE_EXECUTED", {
            "symbol": symbol, "entry": entry_price, "qty": qty,
            "strategy": strategy, "score": score, "paper": self.paper_mode,
        })
        paper_tag = " [PAPER]" if self.paper_mode else ""
        self._bridge.add_event(
            f"🟢 TRADE{paper_tag} {symbol} strat={strategy} {score:.1f}pts "
            f"@ {entry_price:.6g} SL={levels.sl:.6g} TP1={levels.tp1:.6g}",
            "TRADE"
        )
        if self._voice:
            self._voice.announce_trade_executed(
                symbol, qty, entry_price,
                levels.tp1, levels.sl, self.paper_mode,
            )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_trade_proposal(self, symbol, score, strategy, levels, kelly, final_usdt, capital) -> None:
        from rich.text import Text
        text = Text()
        text.append(f"  Symbole    : {symbol}\n",           style="bold")
        text.append(f"  Score      : {score:.1f}/100\n")
        text.append(f"  Stratégie  : {strategy}\n")
        text.append(f"  Entrée     : {levels.entry:.8f}\n")
        text.append(f"  SL         : {levels.sl:.8f}  (-{levels.sl_pct:.2f}%)\n", style="red")
        text.append(f"  TP1        : {levels.tp1:.8f}  (+2.5%) — 50%\n",          style="green")
        text.append(f"  TP2        : {levels.tp2:.8f}  (+5.2%) — 30%\n",          style="green")
        text.append(f"  TP3        : trailing ATR×1.5 — 20%\n",                   style="green")
        text.append(f"  Gain NET   : {levels.net_gain_pct:.2f}%\n",               style="cyan")
        text.append(f"  Taille     : {final_usdt:.2f} USDT  ({final_usdt/capital*100:.1f}%)\n")
        text.append(f"  Risque     : {kelly.risk_usdt:.2f} USDT\n",               style="yellow")
        mode_str = "[PAPER]" if self.paper_mode else "[LIVE]"
        console.print(Panel(text,
            title=f"[bold cyan]{mode_str} TRADE — {symbol}[/bold cyan]",
            border_style="cyan",
        ))

    def _print_status(self) -> None:
        snap = self.db.snapshot_performance(self.capital)
        cb   = self.cb

        table = Table(title="STATUT CENTINA", box=box.ROUNDED, border_style="cyan")
        table.add_column("Indicateur",  style="dim")
        table.add_column("Valeur",      style="bold")

        if self.paper_mode:
            mode_disp = "[yellow]PAPER[/yellow]"
        elif self.auto_mode:
            mode_disp = "[bold green]AUTO[/bold green]"
        elif self.semi_mode:
            mode_disp = "[cyan]SEMI-AUTO[/cyan]"
        else:
            mode_disp = "[white]ADVISOR[/white]"
        table.add_row("Mode",           mode_disp)
        table.add_row("Auto",           "[green]ON[/green]"  if self.auto_mode else "[dim]OFF[/dim]")
        table.add_row("Circuit Breaker", f"[{'red' if cb.level.value >= 3 else 'green'}]{cb.level.name}[/]")
        table.add_row("Positions ouvertes", str(self._guardian.open_positions_count() if self._guardian else 0))
        table.add_row("Signaux en attente", str(len(self._pending_opps)))

        if snap:
            pnl_col = "green" if snap.get("total_pnl", 0) >= 0 else "red"
            table.add_row("Total trades",    str(snap.get("total_trades", 0)))
            table.add_row("Win rate",        f"{snap.get('win_rate', 0):.1%}")
            table.add_row("PnL total",       f"[{pnl_col}]{snap.get('total_pnl', 0):+.2f} USDT[/]")
            table.add_row("Profit Factor",   f"{snap.get('profit_factor', 0):.2f}")
            table.add_row("Max Drawdown",    f"{snap.get('max_drawdown', 0):.1%}")

        console.print(table)

        # Open positions
        if self._guardian:
            positions = self._guardian.get_positions()
            if positions:
                ptable = Table(title="Positions ouvertes", box=box.SIMPLE, border_style="green")
                ptable.add_column("Symbole")
                ptable.add_column("Stratégie")
                ptable.add_column("Entrée")
                ptable.add_column("TP1")
                ptable.add_column("SL")
                ptable.add_column("Score")
                for sym, pos in positions.items():
                    ptable.add_row(
                        sym,
                        pos.strategy,
                        f"{pos.entry_price:.8f}",
                        f"{pos.oco_levels.tp1:.8f}",
                        f"{pos.oco_levels.sl:.8f}",
                        f"{pos.score:.1f}",
                    )
                console.print(ptable)

    async def _print_solde(self) -> None:
        table = Table(title="SOLDE BINANCE (testnet)", box=box.ROUNDED, border_style="yellow")
        table.add_column("Actif",    style="bold")
        table.add_column("Disponible", justify="right")

        try:
            balances = await self.rest.get_account_balance()
            total_usdt = balances.get("USDT", 0.0)

            for asset, free in sorted(balances.items(), key=lambda x: -x[1]):
                col = "yellow" if asset == "USDT" else "white"
                table.add_row(f"[{col}]{asset}[/{col}]", f"[{col}]{free:.6f}[/{col}]")

            mode_str = "[yellow]PAPER[/yellow]" if self.paper_mode else "[green]LIVE[/green]"
            console.print(table)
            console.print(
                f"  Mode : {mode_str}  |  "
                f"[bold yellow]{total_usdt:.2f} USDT[/bold yellow] disponible  |  "
                f"Capital initial configuré : {self.capital:.2f} USDT"
            )
        except Exception as e:
            console.print(f"[red]Erreur solde Binance : {e}[/red]")
            console.print(f"  Capital configuré : [yellow]{self.capital:.2f} USDT[/yellow]")

    def _print_pending_opps(self) -> None:
        opps = self._pending_opps
        if not opps:
            # Show last scan instead
            if self._scout:
                raw = self._scout.get_last_scan()
                if raw:
                    opps = [self._scout._opp_to_payload(o) for o in raw]

        if not opps:
            console.print("[dim]Aucune opportunité disponible.[/dim]")
            return

        table = Table(title="Opportunités", box=box.ROUNDED, border_style="cyan")
        table.add_column("#",          style="dim",   width=3)
        table.add_column("Symbole",    style="bold")
        table.add_column("Score",      justify="right")
        table.add_column("Stratégie",  justify="center")
        table.add_column("Prix",       justify="right")
        table.add_column("RSI",        justify="right")

        for i, opp in enumerate(opps[:10], 1):
            score = opp.get("score", opp.get("composite_score", 0))
            col   = "green" if score >= 78 else "yellow"
            table.add_row(
                str(i),
                opp.get("symbol", ""),
                f"[{col}]{score:.1f}[/{col}]",
                opp.get("strategy", opp.get("suggested_strategy", "B")),
                f"{opp.get('entry_price', opp.get('best_price', 0)):.6f}",
                f"{opp.get('rsi', 0):.1f}",
            )
        console.print(table)

    # ------------------------------------------------------------------
    # RAPPORT — résumé quotidien complet
    # ------------------------------------------------------------------

    def _print_rapport(self) -> None:
        snap = self.db.snapshot_performance(self.capital) or {}
        dtc  = DailyTargetCalculator()

        table = Table(title="RAPPORT QUOTIDIEN CENTINA", box=box.ROUNDED, border_style="cyan")
        table.add_column("Indicateur", style="dim")
        table.add_column("Valeur",     style="bold")

        total_pnl = snap.get("total_pnl", 0)
        table.add_row("Trades totaux",      str(snap.get("total_trades", 0)))
        table.add_row("Trades gagnants",    str(snap.get("winning_trades", 0)))
        table.add_row("Win rate",           f"{snap.get('win_rate', 0):.1%}")
        table.add_row("PnL total",          f"[{'green' if total_pnl>=0 else 'red'}]{total_pnl:+.2f} USDT[/]")
        table.add_row("Profit Factor",      f"{snap.get('profit_factor', 0):.2f}")
        table.add_row("Max Drawdown",       f"{snap.get('max_drawdown', 0):.1%}")
        table.add_row("Sharpe (est.)",      f"{snap.get('sharpe', 0):.2f}")
        table.add_row("Circuit Breaker",    self.cb.level.name)

        # 10-30 EUR progress
        progress = min(1.0, total_pnl / 30) if total_pnl > 0 else 0
        bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
        table.add_row("Objectif jour",     f"{bar} {total_pnl:.2f}/30 EUR")

        console.print(table)

        # Recent closed trades
        try:
            recent = self.db.get_recent_trades(limit=5)
            if recent:
                rtable = Table(title="5 derniers trades", box=box.SIMPLE, border_style="dim")
                rtable.add_column("Symbole")
                rtable.add_column("Stratégie")
                rtable.add_column("PnL %")
                rtable.add_column("Raison")
                for t in recent:
                    pnl_pct = getattr(t, "pnl_pct", 0) or 0
                    col = "green" if pnl_pct >= 0 else "red"
                    rtable.add_row(
                        getattr(t, "symbol", "?"),
                        getattr(t, "strategy", "?"),
                        f"[{col}]{pnl_pct:+.2f}%[/{col}]",
                        getattr(t, "exit_reason", "?"),
                    )
                console.print(rtable)
        except Exception:
            pass

        if self._voice:
            self._voice.announce_nightly_report(
                day_pnl=total_pnl,
                day_trades=snap.get("total_trades", 0),
                win_rate=snap.get("win_rate", 0) * 100,
                total_pnl=total_pnl,
            )

    # ------------------------------------------------------------------
    # STRATS — affiche les 3+2 stratégies
    # ------------------------------------------------------------------

    def _print_strategies(self) -> None:
        from src.brain.strategy_selector import STRATEGY_CONFIGS, REGIME_WEIGHTS
        from src.brain.regime_detector import Regime

        table = Table(title="Stratégies CENTINA v5.0", box=box.ROUNDED, border_style="cyan")
        table.add_column("ID",   width=3)
        table.add_column("Nom",  style="bold")
        table.add_column("Signal")
        table.add_column("TP1",  justify="right")
        table.add_column("TP2",  justify="right")
        table.add_column("SL",   justify="right")
        table.add_column("Max",  justify="right")
        table.add_column("Desc", overflow="fold")

        colours = {"A": "green", "B": "cyan", "C": "yellow", "D": "dim", "E": "dim"}
        for strat, cfg in STRATEGY_CONFIGS.items():
            col = colours.get(strat.value, "white")
            table.add_row(
                f"[{col}]{strat.value}[/{col}]",
                cfg.name.value if hasattr(cfg.name, "value") else str(cfg.name),
                "Breakout 48h+vol" if strat.value == "A"
                else "Golden Cross EMA7/25" if strat.value == "B"
                else "News+RSI 55-70" if strat.value == "C"
                else "Volume Profile" if strat.value == "D"
                else "Wick Reversal",
                f"+{cfg.tp1_pct:.1f}%",
                f"+{cfg.tp2_pct:.1f}%",
                f"{cfg.sl_mult:.1f}×ATR",
                f"{cfg.max_hours:.0f}h",
                cfg.description,
            )
        console.print(table)

        console.print("\n[dim]Poids par régime :[/dim]")
        rtable = Table(box=box.SIMPLE, border_style="dim")
        rtable.add_column("Régime")
        for s in ["A", "B", "C", "D", "E"]:
            rtable.add_column(s, justify="right")
        for regime in Regime:
            weights = REGIME_WEIGHTS.get(regime, {})
            row = [regime.value]
            for s_val in ["A", "B", "C", "D", "E"]:
                from src.brain.strategy_selector import Strategy
                try:
                    strat = Strategy(s_val)
                    w = weights.get(strat, 0)
                    row.append(f"{w:.2f}")
                except Exception:
                    row.append("—")
            rtable.add_row(*row)
        console.print(rtable)

    # ------------------------------------------------------------------
    # HIST — 10 derniers trades
    # ------------------------------------------------------------------

    def _print_history(self) -> None:
        try:
            trades = self.db.get_recent_trades(limit=10)
        except Exception:
            console.print("[dim]Aucun historique disponible.[/dim]")
            return

        if not trades:
            console.print("[dim]Aucun trade enregistré.[/dim]")
            return

        table = Table(title="10 derniers trades", box=box.ROUNDED, border_style="cyan")
        table.add_column("Symbole", style="bold")
        table.add_column("Strat",   justify="center")
        table.add_column("Entrée",  justify="right")
        table.add_column("Sortie",  justify="right")
        table.add_column("PnL %",   justify="right")
        table.add_column("PnL USDT", justify="right")
        table.add_column("TP1", justify="center")
        table.add_column("TP2", justify="center")
        table.add_column("Raison")
        table.add_column("Paper", justify="center")

        for t in trades:
            pnl_pct  = getattr(t, "pnl_pct",  0) or 0
            pnl_usdt = getattr(t, "pnl_usdt", 0) or 0
            col = "green" if pnl_usdt >= 0 else "red"
            table.add_row(
                getattr(t, "symbol",      "?"),
                getattr(t, "strategy",    "?"),
                f"{getattr(t, 'entry_price', 0):.6f}",
                f"{getattr(t, 'exit_price', 0) or 0:.6f}",
                f"[{col}]{pnl_pct:+.2f}%[/{col}]",
                f"[{col}]{pnl_usdt:+.2f}[/{col}]",
                "✅" if getattr(t, "tp1_hit", False) else "○",
                "✅" if getattr(t, "tp2_hit", False) else "○",
                getattr(t, "exit_reason", "?") or "open",
                "📄" if getattr(t, "paper", True) else "💰",
            )
        console.print(table)

    # ------------------------------------------------------------------
    # BACKTEST / OPTIMISE async tasks
    # ------------------------------------------------------------------

    async def _run_backtest(self) -> None:
        try:
            import subprocess, sys
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [sys.executable, "scripts/quick_backtest.py", "--days", "30"],
                    capture_output=True, text=True, cwd=".",
                ),
            )
            if result.returncode == 0:
                console.print(result.stdout[:2000])
            else:
                console.print(f"[red]Backtest erreur:\n{result.stderr[:500]}[/red]")
        except Exception as e:
            console.print(f"[red]Backtest failed: {e}[/red]")

    async def _run_optimise(self) -> None:
        try:
            from src.learning.hyperoptimizer import HyperOptimizer
            opt = HyperOptimizer()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, opt.run, 50)
            console.print(f"[green]Optimisation terminée. Meilleurs params: {result}[/green]")
        except Exception as e:
            console.print(f"[red]Optimisation failed: {e}[/red]")

    async def _run_simulation(self) -> None:
        try:
            import subprocess
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        sys.executable,
                        "scripts/run_historical_simulation.py",
                        "--symbol",
                        "BTCUSDT",
                        "--strategy",
                        "all",
                        "--years",
                        "1",
                    ],
                    capture_output=True,
                    text=True,
                    cwd=".",
                ),
            )
            if result.returncode == 0:
                console.print(result.stdout[:2000])
            else:
                console.print(f"[red]Simulation erreur:\n{result.stderr[:800]}[/red]")
        except Exception as e:
            console.print(f"[red]Simulation failed: {e}[/red]")

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------

    def _print_daily_brief(self) -> None:
        snap = self.db.snapshot_performance(self.capital)
        wr   = snap.get("win_rate", 0.60) if snap else 0.60
        dtc  = DailyTargetCalculator()
        dtc.print_brief(self.capital, win_rate_30d=wr)
        if self._voice:
            try:
                opps_count = len(self._pending_opps)
                self._voice.announce_brief_matinal(
                    self.capital, 30.0, "TRENDING_NORMAL", opps_count
                )
            except Exception:
                pass

    def _paper_gate_check(self) -> None:
        if not ACTIVATION_FILE.exists():
            ACTIVATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            ACTIVATION_FILE.write_text(datetime.now(timezone.utc).isoformat())
            console.print(Panel(
                "[bold yellow]PAPER MODE ACTIVÉ[/bold yellow]\n"
                "[dim]7 jours de paper trading obligatoires avant le live.[/dim]",
                title="Gate 7 jours", border_style="yellow",
            ))
            self.paper_mode = True
            return

        activated_at  = datetime.fromisoformat(ACTIVATION_FILE.read_text().strip())
        if activated_at.tzinfo is None:
            activated_at = activated_at.replace(tzinfo=timezone.utc)
        elapsed       = (datetime.now(timezone.utc) - activated_at).days
        remaining     = max(0, PAPER_DAYS_REQUIRED - elapsed)

        if remaining > 0:
            console.print(Panel(
                f"[bold yellow]PAPER MODE[/bold yellow] — {remaining} jour(s) restants avant le live.",
                border_style="yellow",
            ))
            self.paper_mode = True
        else:
            self.paper_mode = (self.mode == "PAPER")
            if not self.paper_mode:
                console.print("[bold green]Gate 7 jours dégagé — live autorisé.[/bold green]")

    def _print_banner(self) -> None:
        console.print(BANNER)
        mode_label = "AUTO" if self.mode == "AUTO" else "ADVISOR"
        console.print(Panel.fit(
            f"  Mode: [bold]{mode_label}[/bold]  |  "
            f"Capital: [cyan]{self.capital:.0f} USDT[/cyan]  |  "
            f"Score min: [yellow]78/100[/yellow]  |  "
            f"Max positions: [green]3[/green]  |  "
            f"Objectif: [bold cyan]10-30 EUR/jour[/bold cyan]",
            title="[bold white]CENTINA OMNI-QUANT v5.0[/bold white]",
            border_style="cyan",
        ))

    async def _shutdown(self) -> None:
        logger.info("Orchestrator: arrêt des agents")
        for task in self._tasks:
            task.cancel()
        self._executor.shutdown(wait=False)
        await self.rest.disconnect()
        await self.ws.disconnect()
        self.audit.log("SYSTEM_STOP", {"mode": self.mode})
