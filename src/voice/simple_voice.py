from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class SimpleVoice:
    """
    Lightweight voice I/O using SpeechRecognition (STT) and pyttsx3 (TTS).
    Falls back gracefully if packages are unavailable.
    """

    def __init__(self):
        self._tts_engine = None
        self._recognizer = None
        self._microphone = None
        self._ready      = False
        self._init()

    def _init(self) -> None:
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", 160)
            self._tts_engine.setProperty("volume", 0.9)
            logger.debug("pyttsx3 TTS initialized")
        except Exception as e:
            logger.debug("pyttsx3 unavailable: %s", e)

        try:
            import speech_recognition as sr
            self._recognizer  = sr.Recognizer()
            self._microphone  = sr.Microphone()
            self._recognizer.energy_threshold = 300
            self._ready = True
            logger.debug("SpeechRecognition initialized")
        except Exception as e:
            logger.debug("SpeechRecognition unavailable: %s", e)

    def speak(self, text: str) -> None:
        if self._tts_engine is None:
            print(f"[VOICE] {text}")
            return
        try:
            self._tts_engine.say(text)
            self._tts_engine.runAndWait()
        except Exception as e:
            logger.debug("TTS speak error: %s", e)
            print(f"[VOICE] {text}")

    def speak_async(self, text: str) -> None:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()

    def listen(self, timeout: int = 5, phrase_timeout: int = 10) -> str | None:
        if not self._ready or self._recognizer is None:
            return None
        try:
            import speech_recognition as sr
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._recognizer.listen(source, timeout=timeout,
                                                phrase_time_limit=phrase_timeout)
            text = self._recognizer.recognize_google(audio, language="fr-FR")
            logger.info("STT heard: %s", text)
            return text.lower().strip()
        except Exception as e:
            logger.debug("STT error: %s", e)
            return None

    def listen_for_command(self, commands: list[str], timeout: int = 5) -> str | None:
        text = self.listen(timeout=timeout)
        if text is None:
            return None
        for cmd in commands:
            if cmd.lower() in text:
                return cmd
        return None

    def set_rate(self, wpm: int) -> None:
        if self._tts_engine:
            self._tts_engine.setProperty("rate", wpm)

    def set_voice(self, gender: str = "male") -> None:
        if self._tts_engine is None:
            return
        try:
            voices = self._tts_engine.getProperty("voices")
            for v in voices:
                if gender.lower() in v.name.lower() or gender.lower() in (v.id or "").lower():
                    self._tts_engine.setProperty("voice", v.id)
                    return
        except Exception:
            pass
