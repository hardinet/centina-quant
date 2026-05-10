"""
RiskParityAllocator — sizing basé sur risque fixe 1% du capital.
Inspiré Bridgewater All Weather. Alternative au Kelly si RISK_PARITY=True.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskParityResult:
    position_usdt:   float  # taille en USDT
    risk_usdt:       float  # risque en USDT (= capital × risk_pct)
    risk_pct:        float  # % du capital risqué
    sizing_pct:      float  # % du capital investi
    sl_distance_pct: float  # distance SL en %
    method:          str    # "risk_parity" ou "kelly_fallback"
    note:            str    = ""


class RiskParityAllocator:
    """
    Chaque trade risque exactement RISK_PCT du capital.
    Position = risque_usdt / sl_distance.

    Exemple :
      Capital 500 €, risque 1% = 5 €, SL à -3%
      → Position = 5 / 0.03 = 166 €
    """

    ENABLED = os.getenv("RISK_PARITY", "False").lower() == "true"

    def __init__(self, risk_pct: float = 0.01):
        self.risk_pct = risk_pct    # 1% du capital par défaut

    def size(
        self,
        capital_usdt: float,
        sl_distance_pct: float,
        max_pct: float = 0.15,      # plafond 15% du capital
    ) -> RiskParityResult:
        """
        Calcule la taille de position pour risquer exactement risk_pct du capital.

        Args:
            capital_usdt:    Capital disponible en USDT
            sl_distance_pct: Distance au SL en fraction (ex: 0.02 = 2%)
            max_pct:         Plafond de la position en % du capital

        Returns: RiskParityResult avec position_usdt et risk_usdt
        """
        if capital_usdt <= 0 or sl_distance_pct <= 0:
            return RiskParityResult(
                position_usdt=0, risk_usdt=0, risk_pct=self.risk_pct,
                sizing_pct=0, sl_distance_pct=sl_distance_pct,
                method="risk_parity", note="Invalid inputs",
            )

        risk_usdt    = capital_usdt * self.risk_pct
        position_usdt = risk_usdt / sl_distance_pct

        # Plafond absolu
        max_position = capital_usdt * max_pct
        capped       = position_usdt > max_position
        position_usdt = min(position_usdt, max_position)

        # Recompute actual risk after capping
        actual_risk  = position_usdt * sl_distance_pct
        sizing_pct   = position_usdt / capital_usdt

        note = f"{'capped at ' + str(round(max_pct*100)) + '%' if capped else 'ok'}"
        logger.debug(
            "RiskParity: capital=%.0f€  SL=%.2f%%  pos=%.2f€  risk=%.2f€  %s",
            capital_usdt, sl_distance_pct * 100, position_usdt, actual_risk, note,
        )

        return RiskParityResult(
            position_usdt=round(position_usdt, 2),
            risk_usdt=round(actual_risk, 2),
            risk_pct=self.risk_pct,
            sizing_pct=round(sizing_pct, 4),
            sl_distance_pct=sl_distance_pct,
            method="risk_parity",
            note=note,
        )

    def size_from_df(
        self,
        df: pd.DataFrame,
        capital_usdt: float,
        atr_multiplier: float = 1.5,
    ) -> RiskParityResult:
        """
        Calcule le SL distance depuis l'ATR du dataframe, puis size.
        """
        try:
            highs  = df["high"].astype(float).values[-15:]
            lows   = df["low"].astype(float).values[-15:]
            closes = df["close"].astype(float).values[-15:]
            if len(closes) < 2:
                raise ValueError("not enough data")
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1]),
                )
            )
            atr = float(np.mean(tr[-14:]))
            price  = float(closes[-1])
            sl_pct = (atr * atr_multiplier) / price if price > 0 else 0.02
            return self.size(capital_usdt, sl_pct)
        except Exception as e:
            logger.debug("RiskParity size_from_df error: %s", e)
            return self.size(capital_usdt, 0.02)
