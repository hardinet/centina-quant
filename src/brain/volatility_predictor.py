"""
VolatilityPredictor — prédit la volatilité des 4h suivantes.
Ajuste automatiquement le sizing et les SL.
Inspiré Volatility Lab. GARCH(1,1) simplifié + features temporelles.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ATR_PERIOD = 14
LOOKBACK   = 50


@dataclass
class VolatilityForecast:
    atr_predicted:     float = 0.0    # ATR estimé pour les 4 prochaines heures
    atr_ratio:         float = 1.0    # atr_predicted / atr_mean_20
    volatility_label:  str   = "NORMAL"  # LOW / NORMAL / HIGH / EXTREME
    sl_multiplier:     float = 1.0    # multiplier pour le SL
    sizing_multiplier: float = 1.0    # multiplier pour la taille de position
    veto:              bool  = False   # True si trop volatil
    bonus:             int   = 0       # pts ajustés au score
    details:           dict  = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class VolatilityPredictor:
    """
    Prédit la volatilité des 4h suivantes via GARCH(1,1) simplifié.
    Si volatilité > 2× moyenne → VETO.
    Si volatilité faible → SL serré, sizing normal.
    """

    def predict_4h(self, df: pd.DataFrame) -> VolatilityForecast:
        if len(df) < LOOKBACK:
            return VolatilityForecast()
        try:
            closes = df["close"].astype(float).values
            highs  = df["high"].astype(float).values
            lows   = df["low"].astype(float).values

            # ATR historique
            atr_series = self._compute_atr(highs, lows, closes, ATR_PERIOD)
            atr_current = atr_series[-1]
            atr_mean20  = np.mean(atr_series[-20:])
            atr_std20   = np.std(atr_series[-20:])

            # GARCH(1,1) simplifié
            returns = np.diff(np.log(closes[-LOOKBACK:] + 1e-10))
            omega   = 0.0001
            alpha   = 0.10
            beta    = 0.85
            sigma2  = np.var(returns)
            for r in returns[-10:]:
                sigma2 = omega + alpha * r**2 + beta * sigma2
            predicted_vol = np.sqrt(max(sigma2, 1e-10))

            # Convertit en ATR estimé (vol × 4 bougies × prix courant)
            price        = closes[-1]
            atr_predicted = predicted_vol * np.sqrt(4) * price
            atr_ratio     = atr_predicted / atr_mean20 if atr_mean20 > 0 else 1.0

            # Catégorisation
            if atr_ratio < 0.75:
                label = "LOW"
                sl_mult = 0.9; size_mult = 1.1; bonus = 5
            elif atr_ratio < 1.5:
                label = "NORMAL"
                sl_mult = 1.0; size_mult = 1.0; bonus = 0
            elif atr_ratio < 2.0:
                label = "HIGH"
                sl_mult = 1.5; size_mult = 0.67; bonus = -5
            else:
                label = "EXTREME"
                sl_mult = 2.0; size_mult = 0.5
                return VolatilityForecast(
                    atr_predicted=round(atr_predicted, 6),
                    atr_ratio=round(atr_ratio, 3),
                    volatility_label="EXTREME",
                    sl_multiplier=2.0, sizing_multiplier=0.5,
                    veto=True, bonus=-20,
                    details={"reason": "Volatilité > 2× moyenne → VETO"},
                )
                bonus = -20

            return VolatilityForecast(
                atr_predicted=round(atr_predicted, 6),
                atr_ratio=round(atr_ratio, 3),
                volatility_label=label,
                sl_multiplier=round(sl_mult, 2),
                sizing_multiplier=round(size_mult, 2),
                veto=False,
                bonus=bonus,
                details={
                    "atr_current": round(float(atr_current), 6),
                    "atr_mean20":  round(float(atr_mean20), 6),
                    "atr_ratio":   round(float(atr_ratio), 3),
                },
            )

        except Exception as e:
            logger.debug("VolatilityPredictor error: %s", e)
            return VolatilityForecast()

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                     period: int = 14) -> np.ndarray:
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:]  - closes[:-1]),
            )
        )
        atr = np.zeros(len(tr))
        if len(tr) == 0:
            return atr
        atr[0] = tr[0]
        alpha = 1.0 / period
        for i in range(1, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        return atr
