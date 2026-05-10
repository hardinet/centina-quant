"""
CENTINA OMNI-QUANT v5.0 — Point d'entrée principal
Usage :
    python -m src.main                   # mode ADVISOR (défaut)
    python -m src.main --mode AUTO       # exécution automatique
    python -m src.main --mode PAPER      # forcer paper trading
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
from enum import Enum

# Windows : force UTF-8 sur stdout/stderr (caractères Rich)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Créer les dossiers data/ dès le départ
pathlib.Path("data/logs").mkdir(parents=True, exist_ok=True)
pathlib.Path("data/backtest").mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("data/logs/centina.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Réduire le bruit des libs tierces
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("binance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("centina.main")


class Mode(str, Enum):
    ADVISOR = "ADVISOR"
    SEMI    = "SEMI"
    AUTO    = "AUTO"
    PAPER   = "PAPER"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="CENTINA OMNI-QUANT v5.0 — Agent de trading quantitatif",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commandes interactives (pendant l'exécution) :
  GO              → Exécuter la prochaine opportunité
  GO BTCUSDT      → Exécuter un symbole spécifique
  PASSE           → Ignorer l'opportunité courante
  STATUT          → Afficher positions + PnL
  RAPPORT         → Rapport quotidien complet
  OPPS            → Lister les opportunités scannées
  STRATS          → Afficher les stratégies disponibles
  HIST            → Historique des derniers trades
  AUTO            → Basculer mode automatique ON/OFF
  PAPER           → Basculer simulation ON/OFF
  VOICE           → Activer/désactiver interface vocale
  BACKTEST        → Lancer un backtest rapide
  OPTIMISE        → Optimisation Optuna des paramètres
  AIDE            → Aide complète
  EXIT            → Arrêter l'agent
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["ADVISOR", "SEMI", "AUTO", "PAPER"],
        default=os.getenv("DEFAULT_MODE", "PAPER"),
        help="Mode de trading (défaut: PAPER)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Capital en USDT (remplace CAPITAL_USDT dans .env)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activer les logs DEBUG",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Désactiver l'interface vocale",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Désactiver le terminal Rich UI (logs bruts)",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.capital is not None:
        os.environ["CAPITAL_USDT"] = str(args.capital)

    # ── Setup wizard (seulement si .env absent / clé vide) ───────────
    from src.config.setup_wizard import run_wizard
    run_wizard()
    load_dotenv(override=True)  # recharge après éventuelle écriture wizard

    mode = Mode(args.mode)
    logger.info(
        "CENTINA démarrage — mode=%s capital=%s USDT voice=%s ui=%s",
        mode.value,
        os.getenv("CAPITAL_USDT", "200"),
        not args.no_voice,
        not args.no_ui,
    )

    from src.core.orchestrator import CentinaOrchestrator
    orchestrator = CentinaOrchestrator(
        mode=mode.value,
        enable_voice=not args.no_voice,
        enable_ui=not args.no_ui,
    )

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Arrêt manuel (Ctrl+C)")
    except Exception as e:
        logger.critical("Erreur fatale : %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
