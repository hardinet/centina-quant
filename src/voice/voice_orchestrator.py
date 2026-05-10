"""
MODULE 4 — VoiceOrchestrator v5.0
7 templates vocaux FR : brief matinal, PRIME, BUY, trade exécuté,
TP1/TP2, clôture, circuit breaker, rapport nocturne.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

WAKE_WORDS = ["centina", "hey centina", "alerte", "alert", "stop tout"]

COMMANDS = {
    "statut":    "status",
    "status":    "status",
    "rapport":   "report",
    "report":    "report",
    "pause":     "pause",
    "reprendre": "resume",
    "resume":    "resume",
    "briefing":  "briefing",
    "go":        "go",
    "passe":     "pass",
    "auto":      "auto",
    "paper":     "paper",
    "backtest":  "backtest",
    "optimise":  "optimise",
    "optimiser": "optimise",
    "stop":      "stop",
    "stop tout": "emergency_stop",
    "sortir":    "exit",
    "exit":      "exit",
}

# ── Templates FR ─────────────────────────────────────────────────────────

def tmpl_brief_matinal(capital: float, daily_target: float, regime: str, n_opp: int) -> str:
    hour = datetime.now(timezone.utc).strftime("%Hh%M")
    return (
        f"Bonjour. Briefing CENTINA du matin, {hour} UTC. "
        f"Capital disponible : {capital:.0f} euros. "
        f"Objectif du jour : {daily_target:.0f} euros. "
        f"Régime de marché actuel : {regime}. "
        f"Opportunités détectées dans la file : {n_opp}. "
        f"CENTINA est opérationnel. Bon trading."
    )


def tmpl_prime_alert(symbol: str, score: int, strategy: str, tp1: float, sl: float) -> str:
    return (
        f"ALERTE PRIME ! {symbol}, score {score} sur cent. "
        f"Stratégie {strategy}. "
        f"TP1 à {tp1:.2f}, stop loss à {sl:.2f}. "
        f"Confirmez avec GO ou ignorez."
    )


def tmpl_buy_alert(symbol: str, score: int, strategy: str) -> str:
    return (
        f"Opportunité détectée. {symbol}, score {score} sur cent. "
        f"Stratégie {strategy}. "
        f"Dites GO pour exécuter ou PASSE pour ignorer."
    )


def tmpl_trade_executed(symbol: str, qty: float, entry: float, tp1: float, sl: float, paper: bool) -> str:
    mode = "simulation" if paper else "réel"
    return (
        f"Trade exécuté en mode {mode}. "
        f"{symbol}, quantité {qty:.4f}. "
        f"Prix d'entrée {entry:.4f}. "
        f"TP1 à {tp1:.4f}, stop loss à {sl:.4f}. "
        f"OCO placé. Surveillance active."
    )


def tmpl_tp_hit(symbol: str, tp_num: int, pnl_pct: float, next_action: str) -> str:
    return (
        f"Target {tp_num} touché sur {symbol} ! "
        f"Plus {pnl_pct:.1f} pourcent. "
        f"{next_action}"
    )


def tmpl_trade_closed(symbol: str, pnl_usdt: float, pnl_pct: float, reason: str) -> str:
    if pnl_usdt >= 0:
        return (
            f"Trade {symbol} clôturé en profit. "
            f"Plus {pnl_pct:.1f} pourcent, soit {pnl_usdt:.2f} euros. "
            f"Raison : {reason}. Excellent travail."
        )
    return (
        f"Trade {symbol} clôturé en perte. "
        f"Moins {abs(pnl_pct):.1f} pourcent, soit {abs(pnl_usdt):.2f} euros. "
        f"Raison : {reason}. Stop loss respecté."
    )


def tmpl_circuit_breaker(level: int, loss_pct: float) -> str:
    msgs = {
        1: (
            f"Attention. Circuit breaker niveau 1 activé. "
            f"Perte de {loss_pct:.1f} pourcent atteinte. "
            f"Taille des positions réduite de moitié."
        ),
        2: (
            f"Alerte. Circuit breaker niveau 2. "
            f"Perte de {loss_pct:.1f} pourcent. "
            f"Mode advisor uniquement. Aucune nouvelle position autorisée."
        ),
        3: (
            f"Arrêt d'urgence. Circuit breaker niveau 3. "
            f"Perte de {loss_pct:.1f} pourcent. "
            f"Trading suspendu pour 24 heures."
        ),
        4: (
            f"TERMINAISON. Circuit breaker final activé. "
            f"Perte dépassant 15 pourcent. "
            f"Intervention manuelle immédiate requise."
        ),
    }
    return msgs.get(level, f"Circuit breaker niveau {level} activé. Perte de {loss_pct:.1f} pourcent.")


def tmpl_nightly_report(
    day_pnl: float, day_trades: int, win_rate: float,
    total_pnl: float, best_trade: str, worst_trade: str,
) -> str:
    pnl_word = "bénéfice" if day_pnl >= 0 else "perte"
    return (
        f"Rapport nocturne CENTINA. "
        f"Journée terminée avec un {pnl_word} de {abs(day_pnl):.2f} euros. "
        f"{day_trades} trades exécutés, taux de réussite {win_rate:.0f} pourcent. "
        f"PnL total depuis le lancement : {total_pnl:+.2f} euros. "
        f"Meilleur trade du jour : {best_trade}. "
        f"Trade le plus difficile : {worst_trade}. "
        f"Bonne nuit. CENTINA continue la surveillance."
    )


# ── VoiceOrchestrator ─────────────────────────────────────────────────────

class VoiceOrchestrator:
    """
    Coordinateur vocal CENTINA v5.0.
    Écoute le mot de réveil, parse les commandes, route vers les handlers.
    TTS : ElevenLabs si clé dispo, sinon pyttsx3 via SimpleVoice.
    STT : Whisper si installé, sinon SpeechRecognition.
    """

    def __init__(self, use_elevenlabs: bool = False, use_whisper: bool = False):
        self._running   = False
        self._handlers: dict[str, Callable] = {}
        self._speaker   = self._init_speaker(use_elevenlabs)
        self._listener  = self._init_listener(use_whisper)

    # ── Registration ──────────────────────────────────────────────────

    def register(self, command: str, handler: Callable) -> None:
        self._handlers[command] = handler

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self.speak("CENTINA voice interface activée. Dites Centina suivi de votre commande.")
        logger.info("VoiceOrchestrator démarré")

        loop = asyncio.get_event_loop()
        while self._running:
            try:
                text = await loop.run_in_executor(None, self._listener.listen_once, 8)
                if text:
                    await self._process(text.lower())
            except Exception as e:
                logger.debug("VoiceOrchestrator loop error: %s", e)
                await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False

    # ── TTS public API ────────────────────────────────────────────────

    def speak(self, text: str, async_: bool = True) -> None:
        try:
            if async_:
                self._speaker.speak_async(text)
            else:
                self._speaker.speak(text)
        except Exception as e:
            logger.debug("speak error: %s", e)

    # ── High-level announce methods ───────────────────────────────────

    def announce_morning_brief(self, capital: float, daily_target: float, regime: str, n_opp: int) -> None:
        self.speak(tmpl_brief_matinal(capital, daily_target, regime, n_opp))

    def announce_brief_matinal(self, capital: float, daily_target: float, regime: str, n_opp: int) -> None:
        self.announce_morning_brief(capital, daily_target, regime, n_opp)

    def announce_prime(self, symbol: str, score: int, strategy: str, tp1: float, sl: float) -> None:
        self.speak(tmpl_prime_alert(symbol, score, strategy, tp1, sl))

    def announce_opportunity(self, symbol: str, score: int, strategy: str) -> None:
        self.speak(tmpl_buy_alert(symbol, score, strategy))

    def announce_trade_executed(self, symbol: str, qty: float, entry: float,
                                tp1: float, sl: float, paper: bool = False) -> None:
        self.speak(tmpl_trade_executed(symbol, qty, entry, tp1, sl, paper))

    def announce_tp1_hit(self, symbol: str, pnl_pct: float) -> None:
        self.speak(tmpl_tp_hit(symbol, 1, pnl_pct, "Stop loss déplacé au breakeven. TP2 en attente."))

    def announce_tp2_hit(self, symbol: str, pnl_pct: float) -> None:
        self.speak(tmpl_tp_hit(symbol, 2, pnl_pct, "Trailing stop activé sur les 20 pourcent restants."))

    def announce_trade_result(self, symbol: str, pnl_pct: float, exit_reason: str,
                              pnl_usdt: float = 0.0) -> None:
        self.speak(tmpl_trade_closed(symbol, pnl_usdt, pnl_pct, exit_reason))

    def announce_tp1(self, symbol: str, pnl_pct: float) -> None:
        self.announce_tp1_hit(symbol, pnl_pct)

    def announce_tp2(self, symbol: str, pnl_pct: float) -> None:
        self.announce_tp2_hit(symbol, pnl_pct)

    def announce_trade_closed(self, symbol: str, pnl_pct: float, exit_reason: str,
                              pnl_usdt: float = 0.0) -> None:
        self.announce_trade_result(symbol, pnl_pct, exit_reason, pnl_usdt)

    def announce_circuit_breaker(self, level: int, loss_pct: float) -> None:
        self.speak(tmpl_circuit_breaker(level, loss_pct))

    def announce_nightly_report(
        self, day_pnl: float, day_trades: int, win_rate: float,
        total_pnl: float, best_trade: str = "N/A", worst_trade: str = "N/A",
    ) -> None:
        self.speak(tmpl_nightly_report(
            day_pnl, day_trades, win_rate, total_pnl, best_trade, worst_trade
        ))

    # ── STT processing ────────────────────────────────────────────────

    async def _process(self, text: str) -> None:
        if not any(w in text for w in WAKE_WORDS):
            return
        for trigger, cmd in COMMANDS.items():
            if trigger in text:
                logger.info("Voice command: %s → %s", trigger, cmd)
                handler = self._handlers.get(cmd)
                if handler:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler()
                        else:
                            await asyncio.get_event_loop().run_in_executor(None, handler)
                    except Exception as e:
                        logger.debug("Handler error for %s: %s", cmd, e)
                        self.speak(f"Erreur lors de l'exécution de la commande {cmd}.")
                else:
                    self.speak(f"Commande {cmd} reconnue mais non enregistrée.")
                return
        self.speak("Commande non reconnue. Dites statut, rapport, go, stop, ou briefing.")

    # ── Init helpers ──────────────────────────────────────────────────

    def _init_speaker(self, use_elevenlabs: bool):
        if use_elevenlabs and os.getenv("ELEVENLABS_API_KEY"):
            try:
                from .elevenlabs_speaker import ElevenLabsSpeaker
                return ElevenLabsSpeaker()
            except Exception as e:
                logger.debug("ElevenLabs init error: %s", e)
        from .simple_voice import SimpleVoice
        return SimpleVoice()

    def _init_listener(self, use_whisper: bool):
        if use_whisper:
            try:
                from .whisper_listener import WhisperListener
                return WhisperListener()
            except Exception as e:
                logger.debug("Whisper init error: %s", e)
        from .simple_voice import SimpleVoice
        return SimpleVoice()
