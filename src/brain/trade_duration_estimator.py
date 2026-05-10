from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta


@dataclass
class DurationEstimate:
    estimated_hours: float
    atr_pct_per_hour: float
    viable: bool          # False if estimated_hours > MAX_HOURS


MAX_HOURS = 5.0
TARGET_GAIN_PCT = 5.2   # TP2 net target


class TradeDurationEstimator:
    """
    Estimates how many hours it will take to reach +5.2% (TP2) based on
    the average hourly ATR as a percentage of price.

    Formula: estimated_hours = TARGET_GAIN_PCT / (ATR_pct_per_hour × 0.8)
    Factor 0.8 accounts for non-directional movement within each ATR range.
    """

    def estimate(self, df_1h: pd.DataFrame) -> DurationEstimate:
        if len(df_1h) < 14:
            return DurationEstimate(estimated_hours=999.0, atr_pct_per_hour=0.0, viable=False)

        close = df_1h["close"].astype(float)
        high  = df_1h["high"].astype(float)
        low   = df_1h["low"].astype(float)

        atr       = ta.atr(high, low, close, length=14).iloc[-1]
        price     = close.iloc[-1]
        atr_pct   = atr / price * 100      # ATR as % of price per 1h candle

        if atr_pct <= 0:
            return DurationEstimate(estimated_hours=999.0, atr_pct_per_hour=0.0, viable=False)

        estimated = TARGET_GAIN_PCT / (atr_pct * 0.8)

        return DurationEstimate(
            estimated_hours=round(estimated, 2),
            atr_pct_per_hour=round(atr_pct, 4),
            viable=estimated <= MAX_HOURS,
        )
