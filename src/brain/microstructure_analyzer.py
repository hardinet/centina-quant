"""
MicrostructureAnalyzer — VPVR, VWAP, Delta Volume, CVD.
Inspiré Bookmap / Algorithmic Trading Pro.
Toutes les données calculées depuis OHLCV — aucune API externe nécessaire.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DELTA_LOOKBACK = 10   # bougies pour le delta cumulé
CVD_LOOKBACK   = 20


@dataclass
class MicrostructureResult:
    vwap:           float = 0.0
    poc:            float = 0.0        # Point of Control (VPVR)
    price_vs_vwap:  float = 0.0        # % above/below VWAP
    delta_pts:      float = 0.0        # delta volume positif pts
    cvd_slope:      float = 0.0        # pente CVD normalisée
    vwap_bullish:   bool  = False
    poc_bullish:    bool  = False
    cvd_bullish:    bool  = False
    veto:           bool  = False
    veto_reason:    str   = ""
    bonus:          int   = 0
    details:        dict  = field(default_factory=dict)


class MicrostructureAnalyzer:
    """
    VPVR  → trouve le POC ; si prix > POC + trend haussier → +8 pts
    VWAP  → si prix > VWAP + montée → +8 pts ; prix < VWAP malgré hausse → -10 pts
    Delta → achats - ventes ; delta > 0 sur 10 bougies → +15 pts
    CVD   → pente positive = accumulation → +18 pts ; negative + hausse = VETO
    """

    def analyze(self, df: pd.DataFrame) -> MicrostructureResult:
        if len(df) < 30:
            return MicrostructureResult()
        try:
            closes  = df["close"].astype(float).values
            highs   = df["high"].astype(float).values
            lows    = df["low"].astype(float).values
            volumes = df["volume"].astype(float).values
            opens   = df["open"].astype(float).values

            result = MicrostructureResult()

            # ── VWAP ─────────────────────────────────────────────────
            typical_price = (highs + lows + closes) / 3
            cum_vol_price = np.cumsum(typical_price * volumes)
            cum_vol       = np.cumsum(volumes)
            vwap = cum_vol_price[-1] / cum_vol[-1] if cum_vol[-1] > 0 else closes[-1]
            result.vwap = round(vwap, 8)

            price_vs_vwap = (closes[-1] - vwap) / vwap * 100 if vwap > 0 else 0
            result.price_vs_vwap = round(price_vs_vwap, 3)

            # Prix > VWAP et montée
            if closes[-1] > vwap and closes[-1] > closes[-3]:
                result.vwap_bullish = True
                result.bonus += 8
            # Prix < VWAP malgré hausse → faiblesse
            elif closes[-1] < vwap and closes[-1] > closes[-5]:
                result.bonus -= 10
                result.details["vwap_weakness"] = True

            # ── VPVR (Point of Control) ───────────────────────────────
            poc = self._compute_poc(closes, volumes)
            result.poc = round(poc, 8)
            if poc > 0 and closes[-1] > poc and closes[-1] > closes[-3]:
                result.poc_bullish = True
                result.bonus += 8

            # ── Delta Volume ──────────────────────────────────────────
            # Approximation : close > open → buying ; close < open → selling
            delta = np.array([
                v if closes[i] >= opens[i] else -v
                for i, v in enumerate(volumes[-DELTA_LOOKBACK:])
            ])
            cum_delta = np.sum(delta)
            delta_norm = cum_delta / (np.sum(np.abs(delta)) + 1e-10)
            result.delta_pts = round(float(delta_norm), 3)

            if delta_norm > 0.2:    # forte pression acheteuse
                result.bonus += 15
                result.details["delta_bullish"] = True
            elif delta_norm < -0.2:
                result.bonus -= 10
                result.details["delta_bearish"] = True

            # ── CVD (Cumulative Volume Delta) ─────────────────────────
            delta_all = np.array([
                v if closes[i] >= opens[i] else -v
                for i, v in enumerate(volumes[-CVD_LOOKBACK:])
            ])
            cvd = np.cumsum(delta_all)
            if len(cvd) >= 5:
                coef = np.polyfit(np.arange(len(cvd)), cvd, 1)
                cvd_slope = coef[0] / (np.std(cvd) + 1e-10)
                result.cvd_slope = round(float(cvd_slope), 4)

                price_trend_up = closes[-1] > closes[-CVD_LOOKBACK]

                if cvd_slope > 0.1:         # accumulation
                    result.cvd_bullish = True
                    result.bonus += 18
                    result.details["cvd_accumulation"] = True
                elif cvd_slope < -0.1 and price_trend_up:
                    # Distribution cachée — VETO
                    result.veto        = True
                    result.veto_reason = "CVD_DISTRIBUTION"
                    result.bonus -= 20
                    result.details["cvd_distribution"] = True

            result.bonus = max(-30, min(49, result.bonus))
            return result

        except Exception as e:
            logger.debug("MicrostructureAnalyzer error: %s", e)
            return MicrostructureResult()

    @staticmethod
    def _compute_poc(closes: np.ndarray, volumes: np.ndarray, bins: int = 50) -> float:
        """Volume Profile Visible Range — retourne le prix le plus tradé (POC)."""
        if len(closes) < 2:
            return closes[-1] if len(closes) else 0.0
        try:
            price_min = np.min(closes)
            price_max = np.max(closes)
            if price_max == price_min:
                return float(closes[-1])
            edges = np.linspace(price_min, price_max, bins + 1)
            vol_profile = np.zeros(bins)
            for i, c in enumerate(closes):
                bin_idx = min(int((c - price_min) / (price_max - price_min) * bins), bins - 1)
                vol_profile[bin_idx] += volumes[i]
            poc_bin = np.argmax(vol_profile)
            return float((edges[poc_bin] + edges[poc_bin + 1]) / 2)
        except Exception:
            return float(closes[-1])
