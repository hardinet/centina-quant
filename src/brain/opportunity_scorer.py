from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandas_ta as ta

from src.brain.regime_detector import Regime, RegimeDetector, RegimeResult
from src.brain.trend_analyzer import TrendSignal


# ── Score ceilings per category (v5.0) ──────────────────────────────
TECH_MAX     = 83    # Technical indicators
VOLUME_MAX   = 28    # Volume signals
EXTERNAL_MAX = 25    # BTC + market cap + macro
BONUS_MAX    = 24    # Pattern match +15, news pump +25, PRIME +10
MALUS_MAX    = 55    # Manipulation, CB level, bad regime
RAW_POSITIVE_MAX = TECH_MAX + VOLUME_MAX + EXTERNAL_MAX + BONUS_MAX  # 160

# v5.0 VETO conditions — instant SKIP regardless of score
VETO_CONDITIONS = [
    "pump_detected",
    "wash_detected",
    "btc_crash",       # BTC -2% in 1h
    "cb_level_3",      # Circuit breaker level 3+
    "score_too_low",   # raw < 40 before normalisation
]


@dataclass
class ScoreBreakdown:
    symbol: str
    raw_score: float          # before normalisation
    final_score: float        # 0-100 normalised
    tech_pts: float
    volume_pts: float
    external_pts: float
    bonus_pts: float
    malus_pts: float
    regime: Regime
    verdict: str
    vetoed: bool = False
    veto_reason: str = ""
    is_prime: bool = False    # score >= 93
    details: dict = field(default_factory=dict)


class OpportunityScorer:
    """
    Master scoring engine — combines all signals into a single 0-100 score.

    Technical   max +83 pts
    Volume      max +28 pts
    External    max +25 pts
    Bonus       max +24 pts  (pattern +15, news +9)
    Malus       up to -55 pts
    ─────────────────────────────────────────────
    Raw positive max = 160 → normalised to 100
    Score 78+  = STRONG_BUY (entry allowed)
    Score 93+  = PRIME (maximum conviction)
    """

    def __init__(self, btc_trend_score: float = 50.0):
        self._regime = RegimeDetector()
        self._btc_score = btc_trend_score   # updated externally via set_btc_score()

    def set_btc_score(self, score: float) -> None:
        self._btc_score = score

    def score(
        self,
        signal_4h: TrendSignal,
        df_4h: pd.DataFrame,
        manipulation_flags: dict | None = None,
        cb_level_value: int = 0,
        market_cap_rank: int | None = None,
        pattern_bonus: float = 0.0,
        news_bonus: float = 0.0,
        btc_crash: bool = False,
    ) -> ScoreBreakdown:
        manipulation_flags = manipulation_flags or {}
        regime_r = self._regime.detect(df_4h)

        # ── VETO check ────────────────────────────────────────────────
        if manipulation_flags.get("pump_detected"):
            return self._vetoed(signal_4h.symbol, regime_r.regime, "pump_detected")
        if manipulation_flags.get("wash_detected"):
            return self._vetoed(signal_4h.symbol, regime_r.regime, "wash_detected")
        if btc_crash:
            return self._vetoed(signal_4h.symbol, regime_r.regime, "btc_crash")
        if cb_level_value >= 3:
            return self._vetoed(signal_4h.symbol, regime_r.regime, "cb_level_3")

        tech   = self._technical(signal_4h, regime_r)
        volume = self._volume(df_4h)
        extern = self._external(regime_r, market_cap_rank)
        bonus  = self._bonus(signal_4h, regime_r, volume, pattern_bonus, news_bonus)
        malus  = self._malus(manipulation_flags, cb_level_value, regime_r)

        raw   = tech + volume + extern + bonus - malus
        if raw < 40:
            return self._vetoed(signal_4h.symbol, regime_r.regime, "score_too_low")

        final    = max(0.0, min(100.0, raw / RAW_POSITIVE_MAX * 100))
        is_prime = final >= 93

        return ScoreBreakdown(
            symbol=signal_4h.symbol,
            raw_score=round(raw, 2),
            final_score=round(final, 2),
            tech_pts=round(tech, 2),
            volume_pts=round(volume, 2),
            external_pts=round(extern, 2),
            bonus_pts=round(bonus, 2),
            malus_pts=round(malus, 2),
            regime=regime_r.regime,
            verdict=self._verdict(final),
            vetoed=False,
            veto_reason="",
            is_prime=is_prime,
            details={
                "regime_score":     regime_r.trend_score,
                "adx":              regime_r.adx,
                "supertrend_bull":  regime_r.supertrend_bullish,
                "atr_ratio":        regime_r.atr_ratio,
                "btc_score":        self._btc_score,
                "pattern_bonus":    pattern_bonus,
                "news_bonus":       news_bonus,
            },
        )

    def _vetoed(self, symbol: str, regime: Regime, reason: str) -> ScoreBreakdown:
        return ScoreBreakdown(
            symbol=symbol, raw_score=0, final_score=0,
            tech_pts=0, volume_pts=0, external_pts=0, bonus_pts=0, malus_pts=0,
            regime=regime, verdict="VETO", vetoed=True, veto_reason=reason, is_prime=False,
        )

    # ── Technical (+83) ──────────────────────────────────────────────

    def _technical(self, sig: TrendSignal, regime: RegimeResult) -> float:
        pts = 0.0

        # EMA full alignment — 5 steps × 5pts = 25
        if sig.current_price > sig.ema7:   pts += 5
        if sig.ema7  > sig.ema25:          pts += 5
        if sig.ema25 > sig.ema99:          pts += 5
        if sig.ema99 > sig.ema200:         pts += 5
        if sig.current_price > sig.ema200 * 1.01: pts += 5

        # Price above EMA200 (trend health) — max +8
        if sig.current_price > sig.ema200:
            gap_pct = (sig.current_price - sig.ema200) / sig.ema200
            pts += min(8.0, gap_pct * 200)

        # RSI sweet-spot 45-65 — +10; near-sweet 40-70 — +5
        if 45 <= sig.rsi <= 65:
            pts += 10
        elif 40 <= sig.rsi <= 70:
            pts += 5

        # MACD — max +15
        pts += min(15.0, sig.macd_score * 0.15)

        # Regime quality — max +20
        pts += regime.trend_score * 0.20

        # ADX extra bonus — max +5
        if regime.adx > 30:
            pts += min(5.0, (regime.adx - 30) * 0.25)

        return min(float(TECH_MAX), pts)  # cap at 83

    # ── Volume (+28) ─────────────────────────────────────────────────

    def _volume(self, df: pd.DataFrame) -> float:
        pts  = 0.0
        vol  = df["volume"].astype(float)
        last = vol.iloc[-1]
        ma20 = vol.iloc[-20:].mean()

        # Volume vs 20-period MA — max +15
        if ma20 > 0:
            ratio = last / ma20
            pts += min(15.0, (ratio - 1.0) * 10) if ratio > 1.0 else 0.0

        # 3-candle volume expansion — +8
        if len(vol) >= 4:
            v_now, v1, v2 = vol.iloc[-1], vol.iloc[-2], vol.iloc[-3]
            if v_now > v1 > v2:
                pts += 8

        # Quote volume (turnover proxy) — max +5
        if "quote_volume" in df.columns:
            qvol = df["quote_volume"].astype(float)
            qv_last = qvol.iloc[-1]
            qv_mean = qvol.iloc[-20:].mean()
            if qv_mean > 0 and qv_last / qv_mean > 1.5:
                pts += 5

        return min(float(VOLUME_MAX), pts)

    # ── External (+25) ────────────────────────────────────────────────

    def _external(self, regime: RegimeResult, market_cap_rank: int | None) -> float:
        pts = 0.0

        # BTC not in downtrend — +10
        if self._btc_score >= 50:
            pts += 10
        elif self._btc_score >= 40:
            pts += 5

        # Market cap rank — +10 (top 20), +7 (top 50), +3 (top 100)
        if market_cap_rank is not None:
            if market_cap_rank <= 20:
                pts += 10
            elif market_cap_rank <= 50:
                pts += 7
            elif market_cap_rank <= 100:
                pts += 3

        # Global regime bullish — +5
        if regime.regime in (Regime.TRENDING_STRONG, Regime.TRENDING_NORMAL):
            pts += 5

        return min(float(EXTERNAL_MAX), pts)

    # ── Bonus (+24) ───────────────────────────────────────────────────

    def _bonus(
        self,
        sig: TrendSignal,
        regime: RegimeResult,
        volume_pts: float,
        pattern_bonus: float = 0.0,
        news_bonus: float = 0.0,
    ) -> float:
        pts = 0.0

        # Historical pattern match — max +15 (from historian agent)
        pts += min(15.0, pattern_bonus)

        # News momentum correlation — max +9 (pump prob >70% → +9, >50% → +5)
        pts += min(9.0, news_bonus)

        # Perfect setup: strong trend + sweet RSI + full EMA stack — +6
        if (
            regime.regime == Regime.TRENDING_STRONG
            and 50 <= sig.rsi <= 72
            and sig.current_price > sig.ema7 > sig.ema25 > sig.ema99 > sig.ema200
        ):
            pts += 6

        # Volume + regime confluence — +5
        if volume_pts >= 20 and regime.supertrend_bullish:
            pts += 5

        # Near 99 EMA support (pullback entry) — +4
        if sig.ema99 > 0:
            gap = (sig.current_price - sig.ema99) / sig.ema99
            if 0.0 < gap < 0.02:
                pts += 4

        return min(float(BONUS_MAX), pts)

    # ── Malus (up to -55) ─────────────────────────────────────────────

    def _malus(
        self,
        flags: dict,
        cb_level: int,
        regime: RegimeResult,
    ) -> float:
        pts = 0.0

        # Manipulation detected — up to -20
        if flags.get("pump_detected"):    pts += 20
        if flags.get("wash_detected"):    pts += 10
        if flags.get("wick_hunt"):        pts += 8
        if flags.get("fake_news"):        pts += 7

        # Circuit breaker — up to -15
        pts += cb_level * 5   # level 1→-5, 2→-10, 3→-15

        # Volatile regime — -10
        if regime.regime == Regime.VOLATILE:
            pts += 10

        # Ranging regime (lower confidence) — -5
        if regime.regime == Regime.RANGING:
            pts += 5

        return min(float(MALUS_MAX), pts)

    @staticmethod
    def _verdict(score: float) -> str:
        if score >= 93:  return "PRIME"        # 93+ = PRIME alert
        if score >= 78:  return "STRONG_BUY"   # 78+ = enter position
        if score >= 65:  return "BUY"
        if score >= 50:  return "WEAK_BUY"
        return "SKIP"
