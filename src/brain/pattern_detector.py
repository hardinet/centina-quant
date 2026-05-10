"""
PatternDetector — détection automatique de patterns chartistes haussiers.
Inspiré TrendSpider. scipy.signal + régression linéaire numpy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bonus score par pattern
PATTERN_BONUS = {
    "ascending_triangle":  12,
    "bull_flag":           12,
    "cup_and_handle":      10,
    "double_bottom":       12,
    "resistance_breakout": 18,   # avec volume
    "hammer":              8,
    "morning_star":        10,
    "engulfing_bullish":   8,
}

VOLUME_SPIKE_BONUS = 6   # bonus supplémentaire si volume > 2.5×MA


@dataclass
class PatternResult:
    pattern_name:  str   = ""
    detected:      bool  = False
    confidence:    float = 0.0        # 0-100
    target_price:  float = 0.0        # objectif prix estimé
    stop_price:    float = 0.0        # niveau stop suggéré
    duration_est:  float = 2.0        # durée estimée en heures
    bonus:         int   = 0          # pts à ajouter au score CENTINA
    details:       dict  = field(default_factory=dict)


class PatternDetector:
    """
    Détecte les 8 patterns haussiers principaux sur OHLCV.
    Retourne le meilleur pattern détecté.
    """

    LOOKBACK = 50   # bougies analysées

    def detect(self, df: pd.DataFrame, symbol: str = "") -> PatternResult:
        if len(df) < self.LOOKBACK:
            return PatternResult()
        try:
            df = df.tail(self.LOOKBACK).reset_index(drop=True)
            closes  = df["close"].astype(float).values
            highs   = df["high"].astype(float).values
            lows    = df["low"].astype(float).values
            volumes = df["volume"].astype(float).values

            vol_ma20 = np.mean(volumes[-20:])
            vol_spike = volumes[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
            rsi = _simple_rsi(closes, 14)

            results: list[PatternResult] = []

            r = self._ascending_triangle(closes, highs, lows, vol_spike, rsi)
            if r.detected: results.append(r)

            r = self._bull_flag(closes, highs, lows, volumes, vol_ma20)
            if r.detected: results.append(r)

            r = self._double_bottom(closes, lows, vol_spike)
            if r.detected: results.append(r)

            r = self._resistance_breakout(closes, highs, volumes, vol_ma20)
            if r.detected: results.append(r)

            r = self._hammer(closes, highs, lows, volumes[-3:])
            if r.detected: results.append(r)

            r = self._morning_star(closes, highs, lows)
            if r.detected: results.append(r)

            r = self._cup_and_handle(closes, lows, vol_spike)
            if r.detected: results.append(r)

            r = self._engulfing_bullish(closes, highs, lows)
            if r.detected: results.append(r)

            if not results:
                return PatternResult()

            best = max(results, key=lambda x: x.bonus + x.confidence / 10)
            if vol_spike >= 2.5:
                best.bonus += VOLUME_SPIKE_BONUS
                best.details["vol_spike"] = round(vol_spike, 2)

            logger.info("PatternDetector %s: %s conf=%.0f%% bonus=%d",
                        symbol, best.pattern_name, best.confidence, best.bonus)
            return best

        except Exception as e:
            logger.debug("PatternDetector error: %s", e)
            return PatternResult()

    # ── Pattern methods ───────────────────────────────────────────────

    def _ascending_triangle(self, closes, highs, lows, vol_spike, rsi) -> PatternResult:
        n = len(closes)
        # Résistance quasi-horizontale (plafond) + plancher montant
        try:
            recent_highs = highs[-20:]
            resistance   = np.max(recent_highs)
            high_std     = np.std(recent_highs)
            if high_std / resistance > 0.015:     # resistance trop variable
                return PatternResult()
            # Tendance des bas : régression linéaire ascendante
            x = np.arange(20)
            coef = np.polyfit(x, lows[-20:], 1)
            if coef[0] <= 0:
                return PatternResult()
            # Prix près de la résistance
            proximity = (closes[-1] - resistance) / resistance
            if proximity < -0.03:   # trop loin de la résistance
                return PatternResult()
            conf = min(100, 60 + vol_spike * 5 + (45 <= rsi <= 70) * 10)
            return PatternResult(
                pattern_name="ascending_triangle", detected=True,
                confidence=round(conf), target_price=resistance * 1.03,
                stop_price=lows[-1] * 0.99, duration_est=2.0,
                bonus=PATTERN_BONUS["ascending_triangle"],
            )
        except Exception:
            return PatternResult()

    def _bull_flag(self, closes, highs, lows, volumes, vol_ma20) -> PatternResult:
        n = len(closes)
        if n < 30:
            return PatternResult()
        try:
            # Pole : forte hausse sur 5-10 bougies
            pole_gain = (closes[-20] - closes[-30]) / closes[-30] if closes[-30] > 0 else 0
            if pole_gain < 0.04:     # pole doit faire +4% min
                return PatternResult()
            # Drapeau : consolidation légèrement descendante
            flag_closes = closes[-10:]
            coef = np.polyfit(np.arange(10), flag_closes, 1)
            if coef[0] > 0:          # consolidation doit être plate ou légèrement baissière
                return PatternResult()
            # Volume pendant le drapeau < pole
            vol_flag = np.mean(volumes[-10:])
            vol_pole = np.mean(volumes[-30:-20])
            if vol_flag > vol_pole:   # volume doit baisser pendant le drapeau
                return PatternResult()
            conf = min(100, 65 + pole_gain * 100)
            return PatternResult(
                pattern_name="bull_flag", detected=True,
                confidence=round(conf), target_price=closes[-1] * (1 + pole_gain),
                stop_price=np.min(closes[-10:]) * 0.99, duration_est=1.5,
                bonus=PATTERN_BONUS["bull_flag"],
            )
        except Exception:
            return PatternResult()

    def _double_bottom(self, closes, lows, vol_spike) -> PatternResult:
        try:
            from scipy.signal import find_peaks
            # Trouve les creux locaux
            inv_lows = -lows[-30:]
            peaks, props = find_peaks(inv_lows, distance=5)
            if len(peaks) < 2:
                return PatternResult()
            b1, b2 = peaks[-2], peaks[-1]
            l1, l2 = lows[-30 + b1], lows[-30 + b2]
            # Les deux creux doivent être proches (< 2%)
            if abs(l1 - l2) / max(l1, l2) > 0.02:
                return PatternResult()
            # Prix actuel doit être au-dessus des deux creux
            if closes[-1] < max(l1, l2) * 1.005:
                return PatternResult()
            conf = min(100, 65 + vol_spike * 3)
            return PatternResult(
                pattern_name="double_bottom", detected=True,
                confidence=round(conf), target_price=closes[-1] * 1.05,
                stop_price=min(l1, l2) * 0.99, duration_est=3.0,
                bonus=PATTERN_BONUS["double_bottom"],
            )
        except Exception:
            return PatternResult()

    def _resistance_breakout(self, closes, highs, volumes, vol_ma20) -> PatternResult:
        try:
            # Résistance = plus haut des 20 dernières bougies (sauf la dernière)
            resistance = np.max(highs[-20:-1])
            if closes[-1] <= resistance * 1.001:   # pas encore cassé
                return PatternResult()
            # Confirmation volume
            vol_ratio = volumes[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
            if vol_ratio < 2.0:
                return PatternResult()
            conf = min(100, 70 + min(vol_ratio - 2, 3) * 10)
            return PatternResult(
                pattern_name="resistance_breakout", detected=True,
                confidence=round(conf), target_price=closes[-1] * 1.05,
                stop_price=resistance * 0.995, duration_est=2.0,
                bonus=PATTERN_BONUS["resistance_breakout"],
                details={"vol_ratio": round(vol_ratio, 2)},
            )
        except Exception:
            return PatternResult()

    def _hammer(self, closes, highs, lows, last_vols) -> PatternResult:
        try:
            o = closes[-2]; c = closes[-1]; h = highs[-1]; l = lows[-1]
            body   = abs(c - o)
            lower_wick = o - l if o > l else c - l
            upper_wick = h - max(o, c)
            total  = h - l
            if total == 0:
                return PatternResult()
            # Hammer : corps petit, mèche basse longue, petite mèche haute
            if lower_wick < total * 0.55:
                return PatternResult()
            if upper_wick > total * 0.15:
                return PatternResult()
            if body > total * 0.35:
                return PatternResult()
            # Tendance préalable baissière (le hammer marque un creux)
            if closes[-1] >= closes[-5]:
                return PatternResult()
            return PatternResult(
                pattern_name="hammer", detected=True, confidence=65,
                target_price=closes[-1] * 1.03, stop_price=l * 0.995, duration_est=1.5,
                bonus=PATTERN_BONUS["hammer"],
            )
        except Exception:
            return PatternResult()

    def _morning_star(self, closes, highs, lows) -> PatternResult:
        try:
            if len(closes) < 3:
                return PatternResult()
            c1, c2, c3 = closes[-3], closes[-2], closes[-1]
            # Bougie 1 : grande bearish
            if c2 >= c1 * 0.995:
                return PatternResult()
            # Bougie 2 : doji ou petite
            body2 = abs(c2 - closes[-4] if len(closes) >= 4 else c1)
            # Bougie 3 : grande bullish qui remonte > milieu de bougie 1
            if c3 <= (c1 + c2) / 2:
                return PatternResult()
            return PatternResult(
                pattern_name="morning_star", detected=True, confidence=70,
                target_price=c3 * 1.04, stop_price=lows[-2] * 0.998, duration_est=2.0,
                bonus=PATTERN_BONUS["morning_star"],
            )
        except Exception:
            return PatternResult()

    def _cup_and_handle(self, closes, lows, vol_spike) -> PatternResult:
        try:
            n = len(closes)
            if n < 40:
                return PatternResult()
            # Forme en U sur 30 bougies
            cup = closes[-40:-10]
            rim_left  = cup[0]
            rim_right = cup[-1]
            cup_bottom = np.min(cup)
            depth = (min(rim_left, rim_right) - cup_bottom) / min(rim_left, rim_right)
            if depth < 0.03 or depth > 0.30:
                return PatternResult()
            # Symétrie approximative
            if abs(rim_left - rim_right) / rim_left > 0.03:
                return PatternResult()
            # Handle : légère consolidation dans les 10 dernières bougies
            handle = closes[-10:]
            handle_drop = (np.max(handle) - np.min(handle)) / np.max(handle)
            if handle_drop > 0.08:
                return PatternResult()
            conf = min(100, 60 + vol_spike * 4)
            return PatternResult(
                pattern_name="cup_and_handle", detected=True,
                confidence=round(conf), target_price=rim_right * (1 + depth),
                stop_price=np.min(handle) * 0.99, duration_est=3.0,
                bonus=PATTERN_BONUS["cup_and_handle"],
            )
        except Exception:
            return PatternResult()

    def _engulfing_bullish(self, closes, highs, lows) -> PatternResult:
        try:
            if len(closes) < 3:
                return PatternResult()
            # Bougie N-1 : bearish
            if closes[-2] >= closes[-3]:
                return PatternResult()
            # Bougie N : bullish ET englobe la précédente
            if closes[-1] <= closes[-2]:
                return PatternResult()
            if lows[-1] > lows[-2]:
                return PatternResult()
            if highs[-1] < highs[-2]:
                return PatternResult()
            return PatternResult(
                pattern_name="engulfing_bullish", detected=True, confidence=62,
                target_price=closes[-1] * 1.025, stop_price=lows[-1] * 0.997, duration_est=1.5,
                bonus=PATTERN_BONUS["engulfing_bullish"],
            )
        except Exception:
            return PatternResult()


def _simple_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-period - 1:])
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) + 1e-10
    avg_loss = np.mean(losses) + 1e-10
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)
