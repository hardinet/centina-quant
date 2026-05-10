from __future__ import annotations

import os
from typing import Literal

try:
    from pydantic import field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseSettings, validator  # type: ignore
    PYDANTIC_V2 = False


class Settings(BaseSettings):
    # ── Binance ──────────────────────────────────────────────────────
    binance_api_key:    str = ""
    binance_api_secret: str = ""
    binance_testnet:    bool = False
    binance_testnet_api_key:    str = ""
    binance_testnet_api_secret: str = ""

    # ── Capital & risk ───────────────────────────────────────────────
    capital_usdt:         float = 200.0
    max_positions:        int   = 3
    max_exposure_pct:     float = 0.06   # 6% total
    score_min_entry:      int   = 78
    score_prime:          int   = 93

    # ── Mode ────────────────────────────────────────────────────────
    default_mode: Literal["ADVISOR", "AUTO", "PAPER"] = "PAPER"
    allow_live_auto: bool = False

    # ── Redis ───────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Notifications ───────────────────────────────────────────────
    notify_url:   str = ""
    notify_url_2: str = ""
    notify_url_3: str = ""

    # ── External APIs ───────────────────────────────────────────────
    openai_api_key: str = ""
    fred_api_key:   str = ""
    glassnode_api_key: str = ""
    cryptopanic_api_key: str = ""

    # ── Logging ─────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file:  str = "data/logs/centina.log"

    if PYDANTIC_V2:
        model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, frozen=True)

        @model_validator(mode="after")
        def validate_live_auto(self) -> "Settings":
            if not self.binance_testnet and self.default_mode == "AUTO" and not self.allow_live_auto:
                raise ValueError(
                    "ALLOW_LIVE_AUTO=True is required to run AUTO mode on live Binance. "
                    "Use BINANCE_TESTNET=True or set DEFAULT_MODE=PAPER."
                )
            return self
    else:
        class Config:
            env_file = ".env"

        @validator("allow_live_auto", always=True)
        def check_live_auto(cls, v, values):  # type: ignore
            if (not values.get("binance_testnet")
                    and values.get("default_mode") == "AUTO"
                    and not v):
                raise ValueError("Set ALLOW_LIVE_AUTO=True for live AUTO mode.")
            return v


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
