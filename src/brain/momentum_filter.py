from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta


@dataclass
class FilterResult:
    passed:       bool
    reason:       str = ""   # rejection reason if not passed


class MomentumFilter:
    """
    Hard gate: ALL conditions must pass for a trade to be considered.
    Returns FilterResult(passed=True) or FilterResult(passed=False, reason=...).
    """

    RSI_MIN = 50.0
    RSI_MAX = 72.0
    VOL_MULT = 1.5
    RECENT_RUN_PCT = 3.0   # % — reject if pair already ran this much in 2h
    ATR_MAX_MULT = 2.0

    def check(self, df: pd.DataFrame) -> FilterResult:
        if len(df) < 25:
            return FilterResult(False, "Insufficient candles")

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        vol    = df["volume"].astype(float)

        ema7  = ta.ema(close, length=7).iloc[-1]
        ema25 = ta.ema(close, length=25).iloc[-1]
        rsi   = ta.rsi(close, length=14).iloc[-1]
        atr   = ta.atr(high, low, close, length=14)
        price = close.iloc[-1]

        # 1. EMA alignment
        if not (price > ema7 > ema25):
            return FilterResult(False, f"EMA not aligned: price={price:.4f} ema7={ema7:.4f} ema25={ema25:.4f}")

        # 2. RSI sweet spot
        if not (self.RSI_MIN <= rsi <= self.RSI_MAX):
            return FilterResult(False, f"RSI {rsi:.1f} outside [{self.RSI_MIN},{self.RSI_MAX}]")

        # 3. Volume confirmation
        vol_mean = vol.iloc[-20:-1].mean()
        if vol_mean > 0 and vol.iloc[-1] < vol_mean * self.VOL_MULT:
            return FilterResult(False, f"Volume too low ({vol.iloc[-1]:.0f} < {vol_mean * self.VOL_MULT:.0f})")

        # 4. Recent run check (no +3% in last 2 candles on hourly = 2h)
        if len(close) >= 3:
            run_2h = (close.iloc[-1] - close.iloc[-3]) / close.iloc[-3] * 100
            if run_2h > self.RECENT_RUN_PCT:
                return FilterResult(False, f"Already ran +{run_2h:.1f}% in 2h (max {self.RECENT_RUN_PCT}%)")

        # 5. ATR not excessive
        atr_now  = atr.iloc[-1]
        atr_mean = atr.iloc[-20:].mean()
        if atr_mean > 0 and atr_now > atr_mean * self.ATR_MAX_MULT:
            return FilterResult(False, f"ATR spike: {atr_now:.6f} > {atr_mean * self.ATR_MAX_MULT:.6f}")

        return FilterResult(True)
