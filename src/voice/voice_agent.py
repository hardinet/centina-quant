"""
VoiceAgent — orchestrateur vocal complet CENTINA v5.0.
Point d'entrée unique pour écoute active + synthèse + dispatch commandes.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.voice.active_listener import ActiveListener

if TYPE_CHECKING:
    from src.core.orchestrator import CentinaOrchestrator
    from src.brain.llm_council import LLMCouncil

logger = logging.getLogger(__name__)

# Mapping mots reconnus → commandes orchestrateur
VOICE_COMMAND_MAP = {
    # Exécution
    "go":       "GO",
    "exécute":  "GO",
    "execute":  "GO",
    "fais-le":  "GO",
    "lance":    "GO",
    # Ignorer
    "passe":    "PASSE",
    "ignore":   "PASSE",
    "non":      "PASSE",
    "skip":     "PASSE",
    # Info
    "statut":   "STATUT",
    "état":     "STATUT",
    "status":   "STATUT",
    "solde":    "SOLDE",
    "rapport":  "RAPPORT",
    "résumé":   "RAPPORT",
    "resume":   "RAPPORT",
    "opportunités": "OPPS",
    "signaux":  "OPPS",
    "opps":     "OPPS",
    # Modes
    "mode auto":    "AUTO",
    "automatique":  "AUTO",
    "mode conseil": "ADVISOR",
    "mode advisor": "ADVISOR",
    "semi":         "SEMI",
    "aide":         "AIDE",
    # Fermeture globale (avec confirmation 2FA)
    "ferme tout": "__CLOSE_ALL__",
    "stop tout":  "__CLOSE_ALL__",
    "arrête tout": "__CLOSE_ALL__",
}


class VoiceAgent:
    """
    Agent vocal complet : écoute + parle + intègre le LLMCouncil pour les analyses.
    Reçoit les commandes vocales, les mappe vers les commandes orchestrateur,
    et dispatche via asyncio.run_coroutine_threadsafe.
    """

    def __init__(self, orchestrator: "CentinaOrchestrator", llm_council: "LLMCouncil | None" = None):
        self._orch    = orchestrator
        self._council = llm_council
        self._pending_close_all = False  # 2FA fermeture
        self._listener = ActiveListener(self._on_voice_command)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._listener.start()
        self.speak("CENTINA en écoute. Dites Centina suivi de votre commande.")

    def stop(self) -> None:
        self._listener.stop()

    # ── TTS ───────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        self._listener.speak(text)

    # ── High-level announces ──────────────────────────────────────────

    def announce_prime(
        self, symbol: str, score: float, strategy: str,
        tp1_eur: float, sl_pct: float, duration_h: float,
        council_verdict: str = "",
    ) -> None:
        council_part = f"Conseil des modèles : {council_verdict}. " if council_verdict else ""
        self.speak(
            f"Alerte prime. Stratégie {strategy} sur {symbol}. "
            f"Score {score:.0f} sur 100. "
            f"{council_part}"
            f"Gain potentiel {tp1_eur:.1f} euros. "
            f"Durée estimée {duration_h:.0f} heures. "
            f"Dis GO pour exécuter."
        )

    def announce_trade_result(self, symbol: str, pnl_eur: float, reason: str) -> None:
        sign = "plus" if pnl_eur >= 0 else "moins"
        self.speak(
            f"Position {symbol} fermée. Raison : {reason}. "
            f"Résultat : {sign} {abs(pnl_eur):.2f} euros."
        )

    def announce_daily_brief(
        self, capital: float, pnl_day: float,
        win_rate: float, target_min: float, target_max: float,
    ) -> None:
        pnl_sign = "plus" if pnl_day >= 0 else "moins"
        self.speak(
            f"Bonjour. Brief matinal CENTINA. "
            f"Capital actuel : {capital:.0f} euros. "
            f"Gain hier : {pnl_sign} {abs(pnl_day):.2f} euros. "
            f"Win rate 30 jours : {win_rate:.0f} pour cent. "
            f"Objectif du jour : entre {target_min:.0f} et {target_max:.0f} euros. "
            f"Je commence le scan."
        )

    def announce_opportunity(self, symbol: str, score: float, strategy: str) -> None:
        self.speak(
            f"Signal détecté. {symbol}, score {score:.0f}. "
            f"Stratégie {strategy}. Dis GO pour exécuter."
        )

    def announce_trade_executed(
        self, symbol: str, entry: float, tp1: float, sl: float, paper: bool,
    ) -> None:
        mode = "simulation" if paper else "réel"
        self.speak(
            f"Trade lancé sur {symbol} en mode {mode}. "
            f"Entrée à {entry:.4f}. TP1 à {tp1:.4f}. Stop loss à {sl:.4f}."
        )

    def announce_circuit_breaker(self, level_name: str, loss_pct: float) -> None:
        self.speak(
            f"Circuit breaker {level_name} activé. "
            f"Perte de {loss_pct:.1f} pour cent. "
            f"Consultez le dashboard."
        )

    def announce_status(self, capital: float, positions: int, pnl: float, cb: str) -> None:
        sign = "plus" if pnl >= 0 else "moins"
        self.speak(
            f"Statut CENTINA. Capital : {capital:.0f} euros. "
            f"Positions ouvertes : {positions}. "
            f"PnL du jour : {sign} {abs(pnl):.2f} euros. "
            f"Circuit breaker : {cb}."
        )

    def announce_opps(self, opps: list[dict]) -> None:
        if not opps:
            self.speak("Aucune opportunité disponible pour le moment.")
            return
        best = opps[0]
        self.speak(
            f"J'ai {len(opps)} opportunité{'s' if len(opps) > 1 else ''}. "
            f"La meilleure est {best.get('symbol', '?')} "
            f"avec un score de {best.get('score', 0):.0f}."
        )

    # ── Voice command dispatch ────────────────────────────────────────

    def _on_voice_command(self, text: str) -> None:
        text_lower = text.lower().strip()
        logger.info("VoiceAgent commande: %s", text_lower)
        self.speak("Reçu.")

        # Fermeture totale — 2FA
        if "__CLOSE_ALL__" in self._resolve_cmd(text_lower):
            if self._pending_close_all:
                self._pending_close_all = False
                self._dispatch("CLOSE_ALL")
                self.speak("Fermeture de toutes les positions en cours.")
            else:
                self._pending_close_all = True
                self.speak("Confirmez : dites à nouveau ferme tout pour clôturer toutes les positions.")
            return
        self._pending_close_all = False

        # GO avec symbole : "go bitcoin" / "go BTCUSDT"
        if text_lower.startswith("go ") or text_lower.startswith("execute "):
            parts = text_lower.split()
            symbol = parts[-1].upper()
            if not symbol.endswith("USDT"):
                symbol += "USDT"
            self._dispatch(f"GO {symbol}")
            return

        cmd = self._resolve_cmd(text_lower)
        if cmd:
            self._dispatch(cmd)
        else:
            self.speak("Commande non reconnue. Essayez : statut, rapport, go, passe, ou solde.")

    def _resolve_cmd(self, text: str) -> str:
        for trigger, cmd in sorted(VOICE_COMMAND_MAP.items(), key=lambda x: -len(x[0])):
            if trigger in text:
                return cmd
        return ""

    def _dispatch(self, cmd: str) -> None:
        loop = getattr(self._orch, "_loop", None)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._orch._dispatch_cmd(cmd.upper(), cmd),
                loop,
            )
        else:
            logger.warning("VoiceAgent: pas de loop asyncio — commande '%s' ignorée", cmd)
