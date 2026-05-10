from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
DEFAULT_VOICE_ID   = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel


class ElevenLabsSpeaker:
    """
    Premium TTS using ElevenLabs API with local pyttsx3 fallback.
    Caches audio files to reduce API calls.
    """

    CACHE_DIR = Path("data/voice_cache")

    def __init__(self, api_key: str = "", voice_id: str = ""):
        self.api_key  = api_key or ELEVENLABS_API_KEY
        self.voice_id = voice_id or DEFAULT_VOICE_ID
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._fallback = None
        if not self.api_key:
            self._init_fallback()

    def _init_fallback(self) -> None:
        try:
            import pyttsx3
            self._fallback = pyttsx3.init()
            self._fallback.setProperty("rate", 155)
            logger.debug("ElevenLabs: no API key, using pyttsx3 fallback")
        except Exception as e:
            logger.debug("pyttsx3 fallback unavailable: %s", e)

    def speak(self, text: str, cache: bool = True) -> bool:
        if not self.api_key:
            return self._speak_fallback(text)

        cache_key = Path(self.CACHE_DIR) / f"{hash(text)}.mp3"
        if cache and cache_key.exists():
            return self._play_file(cache_key)

        audio = self._generate(text)
        if audio is None:
            return self._speak_fallback(text)

        if cache:
            cache_key.write_bytes(audio)
            return self._play_file(cache_key)

        return self._play_bytes(audio)

    def speak_async(self, text: str) -> None:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()

    def get_voices(self) -> list[dict]:
        if not self.api_key:
            return []
        try:
            import requests
            r = requests.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": self.api_key},
                timeout=10,
            )
            return r.json().get("voices", [])
        except Exception as e:
            logger.debug("get_voices error: %s", e)
            return []

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id

    # ── Private ───────────────────────────────────────────────────────

    def _generate(self, text: str) -> bytes | None:
        try:
            import requests
            url  = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
            body = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            r = requests.post(
                url,
                json=body,
                headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
                timeout=30,
            )
            if r.status_code == 200:
                return r.content
            logger.debug("ElevenLabs API error %d: %s", r.status_code, r.text[:100])
            return None
        except Exception as e:
            logger.debug("ElevenLabs generate error: %s", e)
            return None

    def _play_file(self, path: Path) -> bool:
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                import time; time.sleep(0.1)
            return True
        except Exception:
            pass
        try:
            import subprocess
            subprocess.run(["ffplay", "-nodisp", "-autoexit", str(path)],
                           capture_output=True, check=False)
            return True
        except Exception as e:
            logger.debug("_play_file error: %s", e)
            return False

    def _play_bytes(self, audio: bytes) -> bool:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio)
            return self._play_file(Path(f.name))

    def _speak_fallback(self, text: str) -> bool:
        if self._fallback:
            try:
                self._fallback.say(text)
                self._fallback.runAndWait()
                return True
            except Exception:
                pass
        print(f"[VOICE] {text}")
        return False
