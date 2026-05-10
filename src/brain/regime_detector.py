from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import pandas_ta as ta


class Regime(str, Enum):
    TRENDING_STRONG = "TRENDING_STRONG"   # v5.0: ADX > 35, Supertrend bullish, ATR stable
    TRENDING_NORMAL = "TRENDING_NORMAL"   # ADX 18-35, Supertrend bullish
    RANGING         = "RANGING"           # ADX < 18, low ATR vs recent avg
    VOLATILE        = "VOLATILE"          # ATR spike > 1.5x 20-period avg


@dataclass
class RegimeResult:
    regime: Regime
    adx: float
    adx_pos: float          # +DI
    adx_neg: float          # -DI
    supertrend_bullish: bool
    atr: float
    atr_ratio: float        # current ATR / 20-period ATR mean
    trend_score: float      # 0-100 composite regime quality for bull trades


ADX_STRONG   = 35.0    # v5.0 raised threshold for TRENDING_STRONG
ADX_MODERATE = 18.0
ATR_SPIKE    = 1.5

KELLY_MULTIPLIER_TRENDING_STRONG = 1.2   # Kelly × 1.2 in TRENDING_STRONG


class RegimeDetector:
    """
    Classifies the current market regime using ADX, Supertrend, and ATR.
    Used by OpportunityScorer to scale confidence and by Guardian to adapt stops.
    """

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        if len(df) < 50:
            raise ValueError(f"Need at least 50 candles, got {len(df)}")

        high  = df["high"].astype(float)
        low   = df["low"].astype(float)
        close = df["close"].astype(float)

        # ADX (14)
        adx_df = ta.adx(high, low, close, length=14)
        adx     = adx_df["ADX_14"].iloc[-1]
        adx_pos = adx_df["DMP_14"].iloc[-1]
        adx_neg = adx_df["DMN_14"].iloc[-1]

        # Supertrend (7, 3.0)
        st_df = ta.supertrend(high, low, close, length=7, multiplier=3.0)
        st_dir_col = [c for c in st_df.columns if c.startswith("SUPERTd_")][0]
        st_bullish = int(st_df[st_dir_col].iloc[-1]) == 1

        # ATR ratio
        atr_series = ta.atr(high, low, close, length=14)
        atr        = atr_series.iloc[-1]
        atr_mean   = atr_series.iloc[-20:].mean()
        atr_ratio  = atr / atr_mean if atr_mean > 0 else 1.0

        regime = self._classify(adx, st_bullish, atr_ratio)
        trend_score = self._trend_score(regime, adx, adx_pos, adx_neg, st_bullish, atr_ratio)

        return RegimeResult(
            regime=regime,
            adx=round(adx, 2),
            adx_pos=round(adx_pos, 2),
            adx_neg=round(adx_neg, 2),
            supertrend_bullish=st_bullish,
            atr=round(atr, 8),
            atr_ratio=round(atr_ratio, 3),
            trend_score=round(trend_score, 2),
        )

    def _classify(self, adx: float, st_bullish: bool, atr_ratio: float) -> Regime:
        if atr_ratio >= ATR_SPIKE:
            return Regime.VOLATILE
        if adx >= ADX_STRONG and st_bullish:
            return Regime.TRENDING_STRONG
        if adx >= ADX_MODERATE and st_bullish:
            return Regime.TRENDING_NORMAL
        return Regime.RANGING

    def _trend_score(
        self,
        regime: Regime,
        adx: float,
        adx_pos: float,
        adx_neg: float,
        st_bullish: bool,
        atr_ratio: float,
    ) -> float:
        """Quality score for bullish trades in this regime (0-100)."""
        if regime == Regime.VOLATILE:
            return 20.0
        if regime == Regime.RANGING:
            return 35.0

        base = 50.0
        # ADX strength bonus (up to +25)
        base += min(25.0, (adx - ADX_MODERATE) * 1.5)
        # +DI dominance bonus (up to +15)
        if adx_pos > adx_neg:
            base += min(15.0, (adx_pos - adx_neg) * 0.5)
        # Supertrend confirmation (+10)
        if st_bullish:
            base += 10.0
        # Penalise very high ATR (starts at ratio 1.2)
        if atr_ratio > 1.2:
            base -= (atr_ratio - 1.2) * 20

        return max(0.0, min(100.0, base))
