from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass


@dataclass
class TrendSignal:
    symbol: str
    timeframe: str
    score: float          # 0-100 bullish score
    ema_score: float
    rsi_score: float
    macd_score: float
    atr: float
    current_price: float
    ema7: float
    ema25: float
    ema99: float
    ema200: float
    rsi: float
    verdict: str          # STRONG_BUY / BUY / NEUTRAL / SKIP


class TrendAnalyzer:
    """
    Scores bullish momentum on a 0-100 scale.
    Combines EMA alignment, RSI sweet-spot, and MACD momentum.
    """

    RSI_SWEET_LOW = 45.0
    RSI_SWEET_HIGH = 65.0
    EMA_PERIODS = (7, 25, 99, 200)

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> TrendSignal:
        if len(df) < 210:
            raise ValueError(f"Need at least 210 candles, got {len(df)}")

        close = df["close"].astype(float)

        ema7   = ta.ema(close, length=7).iloc[-1]
        ema25  = ta.ema(close, length=25).iloc[-1]
        ema99  = ta.ema(close, length=99).iloc[-1]
        ema200 = ta.ema(close, length=200).iloc[-1]
        rsi    = ta.rsi(close, length=14).iloc[-1]
        atr    = ta.atr(df["high"].astype(float), df["low"].astype(float), close, length=14).iloc[-1]

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_line   = macd_df["MACD_12_26_9"].iloc[-1]
        macd_signal = macd_df["MACDs_12_26_9"].iloc[-1]
        macd_hist   = macd_df["MACDh_12_26_9"].iloc[-1]

        price = close.iloc[-1]

        ema_score  = self._score_ema(price, ema7, ema25, ema99, ema200)
        rsi_score  = self._score_rsi(rsi)
        macd_score = self._score_macd(macd_line, macd_signal, macd_hist)

        total = ema_score * 0.45 + rsi_score * 0.30 + macd_score * 0.25

        return TrendSignal(
            symbol=symbol,
            timeframe=timeframe,
            score=round(total, 2),
            ema_score=round(ema_score, 2),
            rsi_score=round(rsi_score, 2),
            macd_score=round(macd_score, 2),
            atr=round(atr, 8),
            current_price=round(price, 8),
            ema7=round(ema7, 8),
            ema25=round(ema25, 8),
            ema99=round(ema99, 8),
            ema200=round(ema200, 8),
            rsi=round(rsi, 2),
            verdict=self._verdict(total),
        )

    def _score_ema(
        self, price: float, e7: float, e25: float, e99: float, e200: float
    ) -> float:
        """Full bullish stack: price > e7 > e25 > e99 > e200 = 100."""
        score = 0.0
        # Each step of the stack is worth 20 points
        if price > e7:   score += 20
        if e7    > e25:  score += 20
        if e25   > e99:  score += 20
        if e99   > e200: score += 20
        # Bonus: price comfortably above e200
        if price > e200 * 1.02:
            score += 20
        return min(score, 100.0)

    def _score_rsi(self, rsi: float) -> float:
        """Sweet-spot 45-65 = 100.  Outside = penalised linearly."""
        if self.RSI_SWEET_LOW <= rsi <= self.RSI_SWEET_HIGH:
            return 100.0
        if rsi < self.RSI_SWEET_LOW:
            # Below 45 – momentum not confirmed yet
            return max(0.0, (rsi - 30) / (self.RSI_SWEET_LOW - 30) * 70)
        # Above 65 – overbought risk
        return max(0.0, 100 - (rsi - self.RSI_SWEET_HIGH) * 4)

    def _score_macd(self, line: float, signal: float, hist: float) -> float:
        """MACD above signal + positive/growing histogram."""
        score = 0.0
        if line > signal:
            score += 50
        if hist > 0:
            score += 30
        # Histogram expanding (momentum accelerating)
        if hist > 0 and line > 0:
            score += 20
        return score

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 80:
            return "STRONG_BUY"
        if score >= 65:
            return "BUY"
        if score >= 45:
            return "NEUTRAL"
        return "SKIP"
