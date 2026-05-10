from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import pandas_ta as ta

from src.brain.regime_detector import Regime


class Strategy(str, Enum):
    A_BREAKOUT_MOMENTUM  = "A"
    B_GOLDEN_CROSS       = "B"
    C_NEWS_MOMENTUM      = "C"
    D_VOLUME_PROFILE     = "D"
    E_LIQUIDITY_REVERSAL = "E"


@dataclass
class StrategyConfig:
    name:      Strategy
    tp1_pct:   float   # % gain TP1
    tp2_pct:   float   # % gain TP2
    tp3_trail: float   # ATR×N trailing
    sl_mult:   float   # ATR×N pour SL
    max_hours: float   # durée max trade
    qty_tp1:   float = 0.50
    qty_tp2:   float = 0.30
    qty_tp3:   float = 0.20
    description: str = ""


STRATEGY_CONFIGS: dict[Strategy, StrategyConfig] = {
    Strategy.A_BREAKOUT_MOMENTUM: StrategyConfig(
        name=Strategy.A_BREAKOUT_MOMENTUM,
        tp1_pct=2.5, tp2_pct=5.5, tp3_trail=2.0, sl_mult=1.2, max_hours=3.0,
        description="Breakout de résistance + volume ×2.5 — trend fort",
    ),
    Strategy.B_GOLDEN_CROSS: StrategyConfig(
        name=Strategy.B_GOLDEN_CROSS,
        tp1_pct=2.0, tp2_pct=5.2, tp3_trail=1.5, sl_mult=1.0, max_hours=5.0,
        description="Golden Cross EMA7/EMA25 — trend normal",
    ),
    Strategy.C_NEWS_MOMENTUM: StrategyConfig(
        name=Strategy.C_NEWS_MOMENTUM,
        tp1_pct=3.0, tp2_pct=6.0, tp3_trail=1.5, sl_mult=0.8, max_hours=2.0,
        description="News bullish + RSI momentum 55-70",
    ),
    Strategy.D_VOLUME_PROFILE: StrategyConfig(
        name=Strategy.D_VOLUME_PROFILE,
        tp1_pct=1.8, tp2_pct=4.0, tp3_trail=1.0, sl_mult=0.6, max_hours=2.0,
        description="Volume profile scalp — ranging market",
    ),
    Strategy.E_LIQUIDITY_REVERSAL: StrategyConfig(
        name=Strategy.E_LIQUIDITY_REVERSAL,
        tp1_pct=2.0, tp2_pct=5.0, tp3_trail=1.5, sl_mult=1.0, max_hours=4.0,
        description="Wick hunt reversal — volatile market",
    ),
}

# ── Poids par régime ──────────────────────────────────────────────────
# A/B/C prioritaires ; D/E actifs mais rang inférieur

REGIME_WEIGHTS: dict[Regime, dict[Strategy, float]] = {
    Regime.TRENDING_STRONG: {
        Strategy.A_BREAKOUT_MOMENTUM:  1.00,   # ← champion en trend fort
        Strategy.B_GOLDEN_CROSS:       0.85,
        Strategy.C_NEWS_MOMENTUM:      0.55,
        Strategy.D_VOLUME_PROFILE:     0.30,
        Strategy.E_LIQUIDITY_REVERSAL: 0.20,
    },
    Regime.TRENDING_NORMAL: {
        Strategy.A_BREAKOUT_MOMENTUM:  0.70,
        Strategy.B_GOLDEN_CROSS:       1.00,   # ← champion en trend normal
        Strategy.C_NEWS_MOMENTUM:      0.65,
        Strategy.D_VOLUME_PROFILE:     0.45,
        Strategy.E_LIQUIDITY_REVERSAL: 0.35,
    },
    Regime.RANGING: {
        Strategy.A_BREAKOUT_MOMENTUM:  0.20,
        Strategy.B_GOLDEN_CROSS:       0.25,
        Strategy.C_NEWS_MOMENTUM:      0.60,   # news crée des moves même en ranging
        Strategy.D_VOLUME_PROFILE:     1.00,
        Strategy.E_LIQUIDITY_REVERSAL: 0.80,
    },
    Regime.VOLATILE: {
        Strategy.A_BREAKOUT_MOMENTUM:  0.25,
        Strategy.B_GOLDEN_CROSS:       0.25,
        Strategy.C_NEWS_MOMENTUM:      0.45,
        Strategy.D_VOLUME_PROFILE:     0.55,
        Strategy.E_LIQUIDITY_REVERSAL: 1.00,
    },
}

# Minimum de confiance pour retourner A/B/C comme "viable"
ABC_MIN_WEIGHT = 0.55


class StrategySelector:
    """
    Sélecteur de stratégie v5.0.

    - Priorité A/B/C (objectif 10-30€/jour, trades courts 2-5h)
    - D et E disponibles en fallback (ranging/volatile)
    - Feedback win-rate : ajuste les poids au fil des trades
    - Détection automatique (breakout, golden cross, volume expansion)
    """

    def select(
        self,
        df: pd.DataFrame,
        regime: Regime,
        has_news_signal: bool = False,
        has_wick_reversal: bool = False,
        historical_win_rates: dict[Strategy, float] | None = None,
    ) -> tuple[Strategy, float]:
        """Retourne (meilleure_stratégie, confiance 0-1)."""
        weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[Regime.TRENDING_NORMAL]).copy()
        hist    = historical_win_rates or {}

        if has_news_signal:
            weights[Strategy.C_NEWS_MOMENTUM] *= 1.6

        if has_wick_reversal:
            weights[Strategy.E_LIQUIDITY_REVERSAL] *= 1.5

        # Feedback win-rate : poids × (0.5 + win_rate)
        for s, wr in hist.items():
            if s in weights:
                weights[s] *= max(0.1, 0.5 + wr)

        # Ajustement par signal de marché
        weights = self._adjust_for_signals(df, weights)

        best    = max(weights, key=lambda s: weights[s])
        max_w   = max(weights.values())
        confidence = weights[best] / max_w if max_w > 0 else 0.5
        return best, round(confidence, 3)

    def select_abc(
        self,
        df: pd.DataFrame,
        regime: Regime,
        has_news_signal: bool = False,
        historical_win_rates: dict[Strategy, float] | None = None,
    ) -> tuple[Strategy, float]:
        """Sélection restreinte A/B/C (objectif 10-30€/jour)."""
        weights = {
            s: REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[Regime.TRENDING_NORMAL]).get(s, 0.1)
            for s in (Strategy.A_BREAKOUT_MOMENTUM, Strategy.B_GOLDEN_CROSS, Strategy.C_NEWS_MOMENTUM)
        }

        if has_news_signal:
            weights[Strategy.C_NEWS_MOMENTUM] *= 1.6

        hist = historical_win_rates or {}
        for s, wr in hist.items():
            if s in weights:
                weights[s] *= max(0.1, 0.5 + wr)

        weights = {
            s: w for s, w in self._adjust_for_signals(df, {**weights}).items()
            if s in weights
        }

        best  = max(weights, key=lambda s: weights[s])
        max_w = max(weights.values())
        return best, round(weights[best] / max_w if max_w > 0 else 0.5, 3)

    def get_all_viable(
        self,
        regime: Regime,
        historical_win_rates: dict[Strategy, float] | None = None,
        min_weight: float = 0.6,
        abc_only: bool = True,
    ) -> list[Strategy]:
        """Toutes les stratégies au-dessus du seuil min_weight."""
        weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[Regime.TRENDING_NORMAL])
        max_w   = max(weights.values())
        all_viable = [s for s, w in weights.items() if w / max_w >= min_weight]
        if abc_only:
            return [s for s in all_viable
                    if s in (Strategy.A_BREAKOUT_MOMENTUM, Strategy.B_GOLDEN_CROSS, Strategy.C_NEWS_MOMENTUM)]
        return all_viable

    def get_config(self, strategy: Strategy, regime: Regime) -> StrategyConfig:
        cfg = STRATEGY_CONFIGS[strategy]
        # En TRENDING_STRONG → TP2 étendu à 7.2%
        if regime == Regime.TRENDING_STRONG and strategy in (
            Strategy.A_BREAKOUT_MOMENTUM, Strategy.B_GOLDEN_CROSS,
        ):
            from dataclasses import replace
            cfg = replace(cfg, tp2_pct=7.2)
        return cfg

    def explain(self, strategy: Strategy, regime: Regime) -> str:
        cfg = STRATEGY_CONFIGS.get(strategy)
        if not cfg:
            return ""
        w = REGIME_WEIGHTS.get(regime, {}).get(strategy, 0)
        return (
            f"{strategy.value}: {cfg.description} | "
            f"TP1={cfg.tp1_pct}% TP2={cfg.tp2_pct}% SL={cfg.sl_mult}×ATR | "
            f"poids_régime={w:.2f}"
        )

    # ------------------------------------------------------------------
    # Signal-based adjustments
    # ------------------------------------------------------------------

    def _adjust_for_signals(self, df: pd.DataFrame, weights: dict) -> dict:
        try:
            close  = df["close"].astype(float)
            high   = df["high"].astype(float)
            low    = df["low"].astype(float)
            volume = df["volume"].astype(float)

            ema7  = ta.ema(close, length=7)
            ema25 = ta.ema(close, length=25)

            # A: Breakout 48h high + volume spike
            if len(high) > 50:
                is_breakout = close.iloc[-1] >= high.iloc[-48:-1].max()
                vol_spike   = volume.iloc[-1] > volume.iloc[-20:-1].mean() * 2.5
                if is_breakout and vol_spike and Strategy.A_BREAKOUT_MOMENTUM in weights:
                    weights[Strategy.A_BREAKOUT_MOMENTUM] *= 1.40

            # B: Golden Cross (EMA7 a croisé EMA25 dans les 3 dernières bougies)
            if len(ema7) > 5 and Strategy.B_GOLDEN_CROSS in weights:
                if (ema7.iloc[-1] > ema25.iloc[-1]
                        and (ema7.iloc[-3] < ema25.iloc[-3] or ema7.iloc[-2] < ema25.iloc[-2])):
                    weights[Strategy.B_GOLDEN_CROSS] *= 1.35

            # C: RSI momentum + expansion volume 3 bougies
            rsi = ta.rsi(close, length=14)
            if len(rsi) > 5 and Strategy.C_NEWS_MOMENTUM in weights:
                if 55 <= rsi.iloc[-1] <= 70:
                    if volume.iloc[-1] > volume.iloc[-2] > volume.iloc[-3]:
                        weights[Strategy.C_NEWS_MOMENTUM] *= 1.25

            # E: Wick de retournement (lower wick > 50% range)
            if len(low) > 3 and Strategy.E_LIQUIDITY_REVERSAL in weights:
                total_range = high.iloc[-1] - low.iloc[-1]
                open_c      = df["open"].astype(float).iloc[-1]
                lower_wick  = min(close.iloc[-1], open_c) - low.iloc[-1]
                if total_range > 0 and lower_wick / total_range > 0.5:
                    weights[Strategy.E_LIQUIDITY_REVERSAL] *= 1.30

        except Exception:
            pass

        return weights
