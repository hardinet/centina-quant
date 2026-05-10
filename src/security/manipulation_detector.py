from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds
PUMP_CANDLE_BODY_MULT  = 3.0    # body > 3× avg body → pump candle
PUMP_PRICE_SPIKE_PCT   = 0.08   # 8% single-candle move
WASH_VOL_MULT          = 4.0    # volume spike × 4 with price movement < 0.5%
WICK_HUNT_RATIO        = 0.65   # wick > 65% of total candle range
FAKE_NEWS_LOOKBACK     = 3      # candles to look back for sudden reversal after spike


class ManipulationDetector:
    """
    Detects 4 manipulation patterns:
    - Pump & dump: sudden spike with abnormal candle body
    - Wash trading: volume spike with no real price movement
    - Wick hunting: long wicks designed to trigger stop-losses
    - Fake-news pump: spike followed by rapid reversal within a few candles
    """

    def check(self, df: pd.DataFrame, symbol: str = "") -> dict[str, bool]:
        flags: dict[str, bool] = {
            "pump_detected": False,
            "wash_detected": False,
            "wick_hunt":     False,
            "fake_news":     False,
        }
        if len(df) < 20:
            return flags

        try:
            flags["pump_detected"] = self._detect_pump(df)
            flags["wash_detected"] = self._detect_wash(df)
            flags["wick_hunt"]     = self._detect_wick_hunt(df)
            flags["fake_news"]     = self._detect_fake_news(df)
        except Exception as e:
            logger.warning("ManipulationDetector error for %s: %s", symbol, e)

        if any(flags.values()):
            logger.warning(
                "Manipulation flags for %s: %s",
                symbol,
                {k: v for k, v in flags.items() if v},
            )
        return flags

    # ── Anti-pump ─────────────────────────────────────────────────────

    def _detect_pump(self, df: pd.DataFrame) -> bool:
        close  = df["close"].astype(float)
        open_  = df["open"].astype(float)

        bodies     = (close - open_).abs()
        avg_body   = bodies.iloc[-20:-1].mean()
        last_body  = bodies.iloc[-1]
        last_move  = abs(close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]

        body_spike = avg_body > 0 and last_body > avg_body * PUMP_CANDLE_BODY_MULT
        price_spike = last_move > PUMP_PRICE_SPIKE_PCT
        return body_spike and price_spike

    # ── Anti-wash trading ─────────────────────────────────────────────

    def _detect_wash(self, df: pd.DataFrame) -> bool:
        vol   = df["volume"].astype(float)
        close = df["close"].astype(float)

        vol_mean   = vol.iloc[-20:-1].mean()
        last_vol   = vol.iloc[-1]
        price_move = abs(close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]

        vol_spike   = vol_mean > 0 and last_vol > vol_mean * WASH_VOL_MULT
        no_movement = price_move < 0.005
        return vol_spike and no_movement

    # ── Anti-wick hunting ─────────────────────────────────────────────

    def _detect_wick_hunt(self, df: pd.DataFrame) -> bool:
        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)

        total_range = high.iloc[-1] - low.iloc[-1]
        if total_range == 0:
            return False

        body       = abs(close.iloc[-1] - open_.iloc[-1])
        lower_wick = min(close.iloc[-1], open_.iloc[-1]) - low.iloc[-1]
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_.iloc[-1])
        max_wick   = max(lower_wick, upper_wick)

        return max_wick / total_range > WICK_HUNT_RATIO and body / total_range < 0.20

    # ── Anti-fake news ────────────────────────────────────────────────

    def _detect_fake_news(self, df: pd.DataFrame) -> bool:
        """
        Pattern: large bullish candle followed by rapid mean-reversion
        within FAKE_NEWS_LOOKBACK candles (spike then dump).
        """
        if len(df) < FAKE_NEWS_LOOKBACK + 2:
            return False

        close = df["close"].astype(float)
        # Was there a spike candle?
        for i in range(-FAKE_NEWS_LOOKBACK - 1, -1):
            move = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
            if move > PUMP_PRICE_SPIKE_PCT:
                # Check if price reverted > 60% of spike within next candles
                spike_high = close.iloc[i]
                base       = close.iloc[i - 1]
                current    = close.iloc[-1]
                retracement = (spike_high - current) / (spike_high - base) if spike_high != base else 0
                if retracement > 0.60:
                    return True
        return False
