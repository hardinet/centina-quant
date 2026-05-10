"""
ActiveListener — écoute le micro en continu, détecte le wake word "CENTINA",
puis écoute la commande suivante et la dispatche via cmd_callback.
100% local : pyttsx3 TTS + SpeechRecognition STT (Google gratuit).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

WAKE_WORD       = "centina"
COMMAND_TIMEOUT = 8    # secondes max pour dicter la commande
PHRASE_LIMIT    = 6    # secondes max par phrase


class ActiveListener:
    """
    Tourne dans un thread daemon.
    Dès que "centina" est entendu → répond "Oui ?" et écoute la commande.
    Dispatche via cmd_callback(text: str).
    """

    def __init__(self, cmd_callback: Callable[[str], None]):
        self._cmd_callback = cmd_callback
        self._running      = False
        self._thread: threading.Thread | None = None
        self._tts_lock     = threading.Lock()

        self._recognizer = None
        self._microphone = None
        self._tts        = None
        self._ready      = False
        self._init_engines()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._listen_loop, daemon=True, name="active-listener")
        self._thread.start()
        logger.info("ActiveListener: démarré (wake word = '%s')", WAKE_WORD)

    def stop(self) -> None:
        self._running = False
        logger.info("ActiveListener: arrêté")

    # ── TTS public ────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthèse vocale non-bloquante."""
        threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

    def _speak_sync(self, text: str) -> None:
        with self._tts_lock:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 165)
                engine.setProperty("volume", 0.9)
                self._set_french_voice(engine)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.debug("TTS error: %s", e)
                print(f"[VOICE] {text}")

    # ── Init ──────────────────────────────────────────────────────────

    def _init_engines(self) -> None:
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.pause_threshold   = 0.8
            self._recognizer.energy_threshold  = 300
            self._microphone = sr.Microphone()
            # Calibration ambiante initiale
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
            self._ready = True
            logger.info("ActiveListener: microphone initialisé")
        except Exception as e:
            logger.warning("ActiveListener: SpeechRecognition non disponible — %s", e)

    @staticmethod
    def _set_french_voice(engine) -> None:
        try:
            for v in engine.getProperty("voices"):
                name_id = (v.name + v.id).lower()
                if "french" in name_id or "fr_" in name_id or "fr-" in name_id:
                    engine.setProperty("voice", v.id)
                    return
        except Exception:
            pass

    # ── Main loop ─────────────────────────────────────────────────────

    def _listen_loop(self) -> None:
        if not self._ready:
            logger.warning("ActiveListener: pas de micro — boucle inactive")
            return

        import speech_recognition as sr

        while self._running:
            try:
                with self._microphone as source:
                    audio = self._recognizer.listen(source, timeout=2, phrase_time_limit=4)
                text = self._recognizer.recognize_google(audio, language="fr-FR").lower()
                logger.debug("Entendu : %s", text)

                if WAKE_WORD in text:
                    self.speak("Oui ?")
                    # Écoute la commande suivante
                    with self._microphone as source:
                        audio2 = self._recognizer.listen(
                            source,
                            timeout=COMMAND_TIMEOUT,
                            phrase_time_limit=PHRASE_LIMIT,
                        )
                    cmd_text = self._recognizer.recognize_google(audio2, language="fr-FR")
                    logger.info("Commande vocale reçue : %s", cmd_text)
                    try:
                        self._cmd_callback(cmd_text)
                    except Exception as e:
                        logger.debug("Erreur callback commande vocale: %s", e)

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                logger.debug("ActiveListener loop error: %s", e)
                time.sleep(0.5)

    # ── listen_once compat ────────────────────────────────────────────

    def listen_once(self, timeout: int = 8) -> str | None:
        """Compat avec VoiceOrchestrator._listener.listen_once()."""
        if not self._ready:
            return None
        import speech_recognition as sr
        try:
            with self._microphone as source:
                audio = self._recognizer.listen(source, timeout=timeout, phrase_time_limit=PHRASE_LIMIT)
            return self._recognizer.recognize_google(audio, language="fr-FR").lower()
        except Exception:
            return None
