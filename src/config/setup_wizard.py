"""
MODULE 1 — Premier démarrage : wizard interactif Rich.
Lance si .env absent ou BINANCE_API_KEY vide.
Écrit le .env et teste la connexion Binance.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

console = Console()

ENV_PATH = pathlib.Path(".env")
TEMPLATE = """\
# CENTINA OMNI-QUANT v5.0 — Configuration
BINANCE_API_KEY={api_key}
BINANCE_API_SECRET={api_secret}
BINANCE_TESTNET={testnet}
BINANCE_TESTNET_API_KEY={testnet_key}
BINANCE_TESTNET_API_SECRET={testnet_secret}

CAPITAL_USDT={capital}
DEFAULT_MODE={mode}
ALLOW_LIVE_AUTO={allow_live}

NOTIFY_URL={notify_url}
OPENAI_API_KEY=
CRYPTOPANIC_API_KEY=

LOG_LEVEL=INFO
"""


def _needs_wizard() -> bool:
    if not ENV_PATH.exists():
        return True
    content = ENV_PATH.read_text(encoding="utf-8")
    for line in content.splitlines():
        if line.startswith("BINANCE_API_KEY=") and line.split("=", 1)[1].strip():
            return False
    return True


async def _test_binance(api_key: str, api_secret: str, testnet: bool) -> tuple[bool, str]:
    try:
        from binance import AsyncClient
        client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
        bal = await client.get_asset_balance(asset="USDT")
        await client.close_connection()
        usdt = float(bal.get("free", 0)) if bal else 0.0
        return True, f"{usdt:.2f} USDT disponibles"
    except Exception as e:
        return False, str(e)


def run_wizard() -> None:
    """Lance le wizard si nécessaire (TTY interactif uniquement)."""
    if not _needs_wizard():
        return

    # Non-interactive context (pipe, CI, subprocess) — skip silently
    try:
        if not sys.stdin.isatty():
            console.print("[yellow]⚠  .env absent — wizard ignoré (non-interactif). Créez un .env.[/yellow]")
            return
    except Exception:
        return

    console.print()
    console.print(Panel(
        Text("CENTINA OMNI-QUANT v5.0\nPremier démarrage — Configuration", justify="center", style="bold cyan"),
        border_style="cyan",
    ))
    console.print()
    console.print("[yellow]Aucune clé Binance trouvée. Configurons ensemble.[/yellow]\n")

    try:
        # ── Capital ──────────────────────────────────────────────────
        capital_str = Prompt.ask("[bold]Capital USDT[/bold]", default="200")
        try:
            capital = float(capital_str)
        except ValueError:
            capital = 200.0

        # ── Mode par défaut ──────────────────────────────────────────
        console.print("\n[bold]Mode de trading :[/bold]")
        console.print("  [cyan]PAPER[/cyan]   — Simulation (recommandé pour débuter)")
        console.print("  [yellow]ADVISOR[/yellow] — Scan + proposition manuelle")
        console.print("  [red]AUTO[/red]    — Exécution automatique (nécessite ALLOW_LIVE_AUTO)")
        mode = Prompt.ask("Mode", choices=["PAPER", "ADVISOR", "AUTO"], default="PAPER")

        # ── Testnet ──────────────────────────────────────────────────
        use_testnet = Confirm.ask(
            "\nUtiliser le [cyan]testnet Binance[/cyan] (recommandé) ?", default=True
        )

        api_key = api_secret = testnet_key = testnet_secret = ""

        if use_testnet:
            console.print("\n[dim]Clés testnet : https://testnet.binance.vision/[/dim]")
            testnet_key    = Prompt.ask("[bold]TESTNET API Key[/bold]", password=False)
            testnet_secret = Prompt.ask("[bold]TESTNET API Secret[/bold]", password=True)
            test_key, test_secret = testnet_key, testnet_secret
        else:
            console.print("\n[dim]Clés live : https://www.binance.com/fr/my/settings/api-management[/dim]")
            api_key    = Prompt.ask("[bold]API Key[/bold]", password=False)
            api_secret = Prompt.ask("[bold]API Secret[/bold]", password=True)
            test_key, test_secret = api_key, api_secret

        allow_live = "True" if (mode == "AUTO" and not use_testnet) else "False"

        # ── Telegram optionnel ───────────────────────────────────────
        want_notif = Confirm.ask("\nActiver les [cyan]notifications Telegram[/cyan] ?", default=False)
        notify_url = ""
        if want_notif:
            console.print("[dim]Format : tgram://BOT_TOKEN/CHAT_ID[/dim]")
            notify_url = Prompt.ask("NOTIFY_URL", default="")

    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Wizard interrompu — créez un .env manuellement.[/yellow]")
        return

    # ── Test connexion ───────────────────────────────────────────────
    console.print("\n[cyan]Test de connexion Binance…[/cyan]")
    try:
        ok, msg = asyncio.run(_test_binance(test_key, test_secret, use_testnet))
        if ok:
            console.print(f"[green]OK Connexion — {msg}[/green]")
        else:
            console.print(f"[yellow]Connexion échouée : {msg}[/yellow]")
            try:
                if not Confirm.ask("Continuer quand même ?", default=True):
                    console.print("[red]Configuration annulée.[/red]")
                    return
            except (EOFError, KeyboardInterrupt):
                return
    except Exception as e:
        console.print(f"[yellow]Impossible de tester : {e}[/yellow]")

    # ── Écriture .env ────────────────────────────────────────────────
    env_content = TEMPLATE.format(
        api_key=api_key,
        api_secret=api_secret,
        testnet="True" if use_testnet else "False",
        testnet_key=testnet_key,
        testnet_secret=testnet_secret,
        capital=capital,
        mode=mode,
        allow_live=allow_live,
        notify_url=notify_url,
    )
    ENV_PATH.write_text(env_content, encoding="utf-8")
    console.print(f"\n[green]OK .env créé ({ENV_PATH.absolute()})[/green]")

    # ── Paper gate ───────────────────────────────────────────────────
    gate = pathlib.Path("data/.activation_date")
    if not gate.exists():
        gate.parent.mkdir(parents=True, exist_ok=True)
        from datetime import date
        gate.write_text(date.today().isoformat(), encoding="utf-8")
        console.print("[dim]Paper gate initialisé (7 jours de simulation obligatoires).[/dim]")

    console.print("\n[bold green]Configuration terminée — démarrage de CENTINA…[/bold green]\n")
