from __future__ import annotations

import asyncio
import logging
from binance.enums import KLINE_INTERVAL_1HOUR, KLINE_INTERVAL_4HOUR

from src.brain.momentum_filter import MomentumFilter
from src.brain.opportunity_scorer import OpportunityScorer
from src.brain.strategy_selector import Strategy, StrategySelector
from src.brain.trade_duration_estimator import TradeDurationEstimator
from src.brain.trend_analyzer import TrendAnalyzer
from src.core.event_bus import Event, EventBus, EventType
from src.connectors.binance_rest import BinanceREST
from src.database.cortex_db import CortexDB
from src.security.manipulation_detector import ManipulationDetector

logger = logging.getLogger(__name__)

FEE_PCT = 0.002   # 0.1% maker + 0.1% taker round-trip
SCORE_MIN = 78
SCORE_PRIME = 93
MAX_CONCURRENT = 3


class Analyst:
    """
    Sub-agent 2 (replaces quant.py).
    Receives OpportunityDetected events from Scout, runs full v5.0 scoring pipeline,
    and emits OpportunityValidated or OpportunityPRIME.
    Intègre le LLMCouncil pour bonus/malus consensus multi-modèles.
    """

    def __init__(self, rest: BinanceREST, db: CortexDB, bus: EventBus, historian):
        self._rest      = rest
        self._db        = db
        self._bus       = bus
        self._historian = historian
        self._scorer    = OpportunityScorer()
        self._analyzer  = TrendAnalyzer()
        self._filter    = MomentumFilter()
        self._selector  = StrategySelector()
        self._estimator = TradeDurationEstimator()
        self._manip     = ManipulationDetector()
        self._queue     = bus.subscribe(EventType.OPPORTUNITY_DETECTED, EventType.NEWS_RECEIVED)
        self._open_positions: int = 0
        self._last_news: dict = {}
        self._btc_crash: bool = False
        from src.brain.llm_council import LLMCouncil
        self._llm_council = LLMCouncil()
        # ── Satellite modules (10 professional trading tools) ─────────
        from src.brain.pattern_detector       import PatternDetector
        from src.brain.microstructure_analyzer import MicrostructureAnalyzer
        from src.brain.volatility_predictor   import VolatilityPredictor
        from src.brain.adaptive_scorer        import AdaptiveScorer
        from src.brain.risk_parity_allocator  import RiskParityAllocator
        from src.brain.liquidity_heatmap      import LiquidityHeatmap
        from src.brain.orderflow_analyzer     import OrderflowAnalyzer
        from src.brain.cross_asset_correlator import CrossAssetCorrelator
        from src.brain.social_sentiment       import SocialSentiment
        from src.brain.smart_money_detector   import SmartMoneyDetector
        from src.learning.deep_predictor import DeepPredictor
        self._deep_predictor = DeepPredictor()
        self._pattern    = PatternDetector()
        self._micro      = MicrostructureAnalyzer()
        self._vol_pred   = VolatilityPredictor()
        self._adaptive   = AdaptiveScorer()
        self._risk_parity = RiskParityAllocator()
        self._liquidity  = LiquidityHeatmap()
        self._orderflow  = OrderflowAnalyzer()
        self._cross      = CrossAssetCorrelator()
        self._sentiment  = SocialSentiment()
        self._smart      = SmartMoneyDetector()

    async def run(self) -> None:
        logger.info("Analyst: started")
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                if event.type == EventType.NEWS_RECEIVED:
                    self._last_news = event.payload
                    self._btc_crash = event.payload.get("btc_crash", False)
                elif event.type == EventType.OPPORTUNITY_DETECTED:
                    await self._process(event.payload)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Analyst error: %s", e)

    async def _process(self, payload: dict) -> None:
        symbol = payload.get("symbol", "")
        if not symbol:
            return

        try:
            df_4h = await self._rest.get_klines(symbol, KLINE_INTERVAL_4HOUR, limit=300)
            df_1h = await self._rest.get_klines(symbol, KLINE_INTERVAL_1HOUR, limit=100)
        except Exception as e:
            logger.debug("Analyst: data fetch failed for %s: %s", symbol, e)
            return

        # 1. Veto checks (mandatory — score = 0 if failed)
        signal_4h = self._analyzer.analyze(df_4h, symbol, KLINE_INTERVAL_4HOUR)
        if not self._passes_veto(signal_4h):
            return

        # 2. Momentum filter (ALL conditions)
        filt = self._filter.check(df_4h)
        if not filt.passed:
            logger.debug("Analyst: momentum filter rejected %s — %s", symbol, filt.reason)
            return

        # 3. Duration estimate
        dur = self._estimator.estimate(df_1h)
        if not dur.viable:
            logger.debug("Analyst: %s duration estimate %.1fh > 5h, skip", symbol, dur.estimated_hours)
            return

        # 4. Manipulation check
        manip_flags = self._manip.check(df_4h, symbol)

        # 5. Historian validation
        features = {
            "ema_ratio":    signal_4h.ema7 / signal_4h.ema25 if signal_4h.ema25 > 0 else 1.0,
            "rsi":          signal_4h.rsi / 100,
            "volume_spike": 1.0,
            "atr_ratio":    1.0,
            "ema7_25_aligned": 1.0,
        }
        historian_data = await self._historian.validate_pattern(symbol, features)

        # 6. Derive bonus inputs from historian and news
        hist_wr      = historian_data.get("win_rate_similaire", 0.5)
        pattern_bonus: float
        if hist_wr >= 0.80:
            pattern_bonus = 15.0
        elif hist_wr >= 0.60:
            pattern_bonus = 8.0
        else:
            pattern_bonus = 0.0

        pump_prob = self._last_news.get("pump_probability", 0.0)
        news_bonus: float
        if pump_prob > 0.70:
            news_bonus = 9.0
        elif pump_prob > 0.50:
            news_bonus = 5.0
        else:
            news_bonus = 0.0

        market_cap_rank: int | None = payload.get("market_cap_rank")

        # 7. Full score (v5.0 — pattern_bonus and news_bonus baked in)
        breakdown = self._scorer.score(
            signal_4h=signal_4h,
            df_4h=df_4h,
            manipulation_flags=manip_flags,
            cb_level_value=0,
            market_cap_rank=market_cap_rank,
            pattern_bonus=pattern_bonus,
            news_bonus=news_bonus,
            btc_crash=self._btc_crash,
        )

        if breakdown.vetoed or breakdown.final_score < SCORE_MIN:
            return

        score = breakdown.final_score

        # 8. LLM Council — bonus/malus consensus multi-modèles
        council: dict = {}
        try:
            from src.brain.regime_detector import RegimeDetector as _RD
            _regime_r2 = _RD().detect(df_4h)
            council = await self._llm_council.consult({
                "symbol":        symbol,
                "strategy":      payload.get("strategy", "B"),
                "score":         score,
                "regime":        _regime_r2.regime.value,
                "rsi":           signal_4h.rsi,
                "volume_spike":  max(1.0, breakdown.volume_pts / 5) if breakdown.volume_pts > 0 else 1.0,
                "ema_aligned":   signal_4h.ema7 > signal_4h.ema25 > signal_4h.ema99,
                "news_sentiment": self._last_news.get("score_bullish", 50),
                "duration_h":    dur.estimated_hours,
                "atr_pct":       signal_4h.atr / signal_4h.current_price * 100 if signal_4h.current_price > 0 else 2.0,
            })
            score_bonus = council.get("score_bonus", 0)
            score = max(0.0, min(100.0, score + score_bonus))
            if council.get("models_voted", 0) > 0:
                logger.info("LLM Council %s: %s", symbol, council.get("council_text", ""))
            # Si conseil unanime PASS → rejeter même si score ≥ 78
            if council.get("verdict") == "PASS" and council.get("agreement_pct", 0) >= 75:
                logger.info("Analyst: %s rejeté par LLM Council (PASS unanime)", symbol)
                return
            # Si conseil PRIME avec accord fort → forcer score prime
            if council.get("verdict") in ("PRIME", "BUY") and score >= 85 and council.get("agreement_pct", 0) >= 75:
                breakdown = type(breakdown)(
                    **{**breakdown.__dict__, "is_prime": True, "final_score": score}
                )
        except Exception as e:
            logger.debug("LLM Council error for %s: %s", symbol, e)

        if score < SCORE_MIN:
            return

        # 9. Satellite analysis — 10 professional trading modules
        sat = await self._satellite_analysis(symbol, df_4h, df_1h, signal_4h)
        if sat.get("veto"):
            logger.info("Analyst: %s vetoed by satellite (%s)", symbol, sat.get("veto_reason", ""))
            return
        sat_bonus = sat.get("total_bonus", 0)
        score = max(0.0, min(100.0, score + sat_bonus))
        if score < SCORE_MIN:
            return

        # 10. Kelly sizing (or Risk Parity if enabled)
        kelly_pct: float
        if self._risk_parity.ENABLED:
            from src.brain.risk_parity_allocator import RiskParityAllocator
            _rp = self._risk_parity.size_from_df(df_4h, capital_usdt=1000.0)
            kelly_pct = _rp.sizing_pct
        else:
            kelly_pct = {0: 0.05, 1: 0.03, 2: 0.02}.get(min(self._open_positions, 2), 0.02)

        # Apply sizing multiplier from volatility predictor
        vol_sizing = sat.get("vol_sizing_mult", 1.0)
        kelly_pct  = round(kelly_pct * vol_sizing, 4)

        # 11. OCO levels (sl_multiplier from volatility predictor)
        from src.brain.oco_calculator import OCOCalculator
        oco    = OCOCalculator()
        sl_mult = sat.get("vol_sl_mult", 1.0)
        levels = oco.calculate(
            signal_4h.current_price,
            signal_4h.atr * sl_mult,
        )

        # 12. Strategy selection
        from src.brain.regime_detector import RegimeDetector
        regime_r = RegimeDetector().detect(df_4h)
        has_news = self._last_news.get("urgency") == "high"
        strategy, conf = self._selector.select(df_4h, regime_r.regime, has_news_signal=has_news)

        # 13. DeepPredictor probability
        deep_proba = 0.5
        try:
            deep_features = self._build_deep_features_window(df_1h, signal_4h, breakdown)
            deep_proba = self._deep_predictor.predict(deep_features)
        except Exception:
            deep_proba = 0.5

        result_payload = {
            "symbol":            symbol,
            "score":             round(score, 2),
            "is_prime":          breakdown.is_prime,
            "strategy":          strategy.value,
            "regime":            regime_r.regime.value,
            "entry_price":       signal_4h.current_price,
            "sl":                levels.sl,
            "tp1":               levels.tp1,
            "tp2":               levels.tp2,
            "tp3_trail_atr":     levels.tp3_trail_atr,
            "sl_pct":            levels.sl_pct,
            "net_gain_pct":      levels.net_gain_pct,
            "kelly_pct":         kelly_pct,
            "estimated_hours":   dur.estimated_hours,
            "historian":         historian_data,
            "manip_flags":       manip_flags,
            "pattern_bonus":     pattern_bonus,
            "news_bonus":        news_bonus,
            "score_breakdown":   breakdown.details,
            "council_text":      council.get("council_text", "") if council else "",
            "council_verdict":   council.get("verdict", "") if council else "",
            "council_agreement": council.get("agreement_pct", 0) if council else 0,
            "council_models":    council.get("models_voted", 0) if council else 0,
            # ── Satellite summary ─────────────────────────────────────
            "satellite_bonus":      sat_bonus,
            "satellite_details":    sat.get("details", {}),
            "pattern_signal":       sat.get("pattern_signal", ""),
            "patterns_found":       sat.get("patterns_found", []),
            "pattern_confidence":   sat.get("pattern_confidence", 0.0),
            "pattern_target":       sat.get("pattern_target", 0.0),
            "orderflow_signal":     sat.get("orderflow_signal", ""),
            "bid_ask_imbalance":    sat.get("bid_ask_imbalance", 1.0),
            "buy_pressure":         sat.get("buy_pressure", 50.0),
            "absorption":           sat.get("absorption", False),
            "stop_hunt":            sat.get("stop_hunt", False),
            "iceberg":              sat.get("iceberg", False),
            "smart_money":          sat.get("smart_signal", ""),
            "whale_buys":           sat.get("whale_buys", 0),
            "whale_sells":          sat.get("whale_sells", 0),
            "net_flow":             sat.get("net_flow", 0.0),
            "large_buy_ratio":      sat.get("large_buy_ratio", 0.5),
            "social_score":         sat.get("social_score", 50),
            "reddit_trend":         sat.get("reddit_trend", "NEUTRAL"),
            "fear_greed":           sat.get("fear_greed", 50.0),
            "cryptopanic_score":    sat.get("cryptopanic_score", 50.0),
            "volatility_label":     sat.get("vol_label", "NORMAL"),
            "atr_ratio":            sat.get("atr_ratio", 1.0),
            "atr_predicted":        sat.get("atr_predicted", 0.0),
            "btc_dominance":        sat.get("btc_dominance", 50.0),
            "eth_btc_ratio":        sat.get("eth_btc_ratio", 0.0),
            "eth_btc_trend":        sat.get("eth_btc_trend", "NEUTRAL"),
            "funding_rate":         sat.get("funding_rate", 0.0),
            "funding_signal":       sat.get("funding_signal", "NEUTRAL"),
            "vwap_signal":          sat.get("vwap_signal", "NEUTRAL"),
            "cvd_signal":           sat.get("cvd_signal", "NEUTRAL"),
            "buy_delta_pct":        sat.get("buy_delta_pct", 50.0),
            "liq_above_pct":        sat.get("liq_above_pct", 0.0),
            "liq_below_pct":        sat.get("liq_below_pct", 0.0),
            "liq_above_price":      sat.get("liq_above_price", 0.0),
            "liq_below_price":      sat.get("liq_below_price", 0.0),
            "liq_above_size":       sat.get("liq_above_size", 0.0),
            "liq_below_size":       sat.get("liq_below_size", 0.0),
            # DeepPredictor
            "deep_proba":           round(deep_proba, 3),
            # Sizing method
            "risk_method":          "RiskParity" if self._risk_parity.ENABLED else "Kelly",
            # Microstructure summary
            "microstructure_signal": sat.get("microstructure_signal", "NEUTRAL"),
        }

        event_type = EventType.OPPORTUNITY_PRIME if score >= SCORE_PRIME else EventType.OPPORTUNITY_VALIDATED
        await self._bus.emit(Event(event_type, result_payload, source="analyst"))
        logger.info("Analyst: %s score=%.1f strategy=%s → %s",
                    symbol, score, strategy.value, event_type.value)

    def _build_deep_features_window(self, df_1h, signal, breakdown) -> list[dict]:
        """Build the 24-candle feature window expected by DeepPredictor."""
        try:
            import pandas as pd

            close = df_1h["close"].astype(float)
            volume = df_1h["volume"].astype(float)
            ema7 = close.ewm(span=7, adjust=False).mean()
            ema25 = close.ewm(span=25, adjust=False).mean()
            ema99 = close.ewm(span=99, adjust=False).mean()
            vol_avg = volume.rolling(20, min_periods=1).mean()
            atr_pct = (
                signal.atr / signal.current_price * 100
                if signal.current_price > 0
                else 2.0
            )

            rows: list[dict] = []
            for idx, row in df_1h.tail(24).iterrows():
                price = float(row.get("close", signal.current_price) or signal.current_price)
                timestamp = row.get("open_time")
                if not hasattr(timestamp, "hour"):
                    timestamp = pd.Timestamp.utcnow()
                v_avg = float(vol_avg.loc[idx]) if idx in vol_avg.index else float(volume.mean() or 1.0)
                rows.append({
                    "close": price,
                    "rsi": signal.rsi,
                    "ema7_ratio": float(ema7.loc[idx] / price) if price > 0 and idx in ema7.index else 1.0,
                    "ema25_ratio": float(ema25.loc[idx] / price) if price > 0 and idx in ema25.index else 1.0,
                    "ema99_ratio": float(ema99.loc[idx] / price) if price > 0 and idx in ema99.index else 1.0,
                    "volume_ratio": float(row.get("volume", 0) or 0) / v_avg if v_avg > 0 else 1.0,
                    "atr_pct": atr_pct,
                    "macd_hist": 0.0,
                    "obv_slope": 0.0,
                    "hour_utc": int(timestamp.hour),
                    "day_of_week": int(timestamp.dayofweek if hasattr(timestamp, "dayofweek") else 0),
                    "btc_pct_1h": 0.0,
                    "spread_pct": 0.001,
                })

            if len(rows) >= 24:
                return rows
        except Exception as e:
            logger.debug("DeepPredictor feature window error: %s", e)

        fallback = {
            "close": signal.current_price,
            "rsi": signal.rsi,
            "ema7_ratio": signal.ema7 / signal.current_price if signal.current_price > 0 else 1.0,
            "ema25_ratio": signal.ema25 / signal.current_price if signal.current_price > 0 else 1.0,
            "ema99_ratio": signal.ema99 / signal.current_price if signal.current_price > 0 else 1.0,
            "volume_ratio": max(1.0, breakdown.volume_pts / 5) if breakdown.volume_pts > 0 else 1.0,
            "atr_pct": signal.atr / signal.current_price * 100 if signal.current_price > 0 else 2.0,
            "macd_hist": 0.0,
            "obv_slope": 0.0,
            "hour_utc": 12,
            "day_of_week": 3,
            "btc_pct_1h": 0.0,
            "spread_pct": 0.001,
        }
        return [fallback.copy() for _ in range(24)]

    def _passes_veto(self, sig) -> bool:
        """VETO absolus — any failure = score 0."""
        if sig.ema7 < sig.ema25:   return False
        if sig.ema25 < sig.ema99:  return False
        if sig.current_price < sig.ema200: return False
        return True

    async def _satellite_analysis(self, symbol: str, df_4h, df_1h, signal) -> dict:
        """
        Runs all 10 satellite modules in parallel, aggregates bonuses and vetoes.
        Returns a dict with total_bonus, veto, veto_reason, and per-module details.
        """
        price = signal.current_price

        # ── Parallel execution ────────────────────────────────────────
        (
            pat_r, micro_r, vol_r,
            liq_r, of_r, cross_r,
            social_r, smart_r,
        ) = await asyncio.gather(
            self._pattern.analyze(df_4h),
            self._micro.analyze(df_4h),
            self._vol_pred.predict(df_4h),
            self._liquidity.analyze(symbol, price),
            self._orderflow.analyze(symbol, price),
            self._cross.analyze(symbol),
            self._sentiment.analyze(symbol),
            self._smart.analyze(symbol, price),
            return_exceptions=True,
        )

        total_bonus   = 0
        veto          = False
        veto_reason   = ""
        details: dict = {}

        # ── PatternDetector ───────────────────────────────────────────
        if hasattr(pat_r, "bonus"):
            total_bonus += pat_r.bonus
            if pat_r.patterns_found:
                details["patterns"] = pat_r.patterns_found

        # ── MicrostructureAnalyzer ────────────────────────────────────
        if hasattr(micro_r, "bonus"):
            if micro_r.veto:
                veto = True
                veto_reason = micro_r.veto_reason
            else:
                total_bonus += micro_r.bonus
            if micro_r.vwap_signal != "NEUTRAL":
                details["micro"] = f"VWAP={micro_r.vwap_signal} CVD={micro_r.cvd_signal}"

        # ── VolatilityPredictor ───────────────────────────────────────
        vol_sl_mult      = 1.0
        vol_sizing_mult  = 1.0
        vol_label        = "NORMAL"
        if hasattr(vol_r, "bonus"):
            if vol_r.veto:
                veto = True
                veto_reason = vol_r.veto_reason
            else:
                total_bonus  += vol_r.bonus
                vol_sl_mult   = vol_r.sl_multiplier
                vol_sizing_mult = vol_r.sizing_multiplier
                vol_label     = vol_r.volatility_label
            details["volatility"] = vol_r.volatility_label

        # ── LiquidityHeatmap ─────────────────────────────────────────
        if hasattr(liq_r, "bonus"):
            total_bonus += liq_r.bonus
            if liq_r.details:
                details["liquidity"] = liq_r.details

        # ── OrderflowAnalyzer ─────────────────────────────────────────
        orderflow_signal = "NEUTRAL"
        if hasattr(of_r, "bonus"):
            if of_r.veto:
                veto = True
                veto_reason = of_r.veto_reason
            else:
                total_bonus += of_r.bonus
            orderflow_signal = (
                "BULLISH" if of_r.bid_ask_imbalance >= 1.5 else
                "BEARISH" if of_r.bid_ask_imbalance <= 0.67 else "NEUTRAL"
            )
            if of_r.details:
                details["orderflow"] = of_r.details

        # ── CrossAssetCorrelator ──────────────────────────────────────
        if hasattr(cross_r, "bonus"):
            total_bonus += cross_r.bonus
            if cross_r.details:
                details["cross_asset"] = cross_r.details

        # ── SocialSentiment ───────────────────────────────────────────
        social_score = 50.0
        if hasattr(social_r, "bonus"):
            if social_r.veto:
                veto = True
                veto_reason = "EXTREME_FEAR"
            else:
                total_bonus  += social_r.bonus
                social_score  = social_r.combined_score
            if social_r.details:
                details["sentiment"] = social_r.details

        # ── SmartMoneyDetector ────────────────────────────────────────
        smart_signal = "NEUTRAL"
        if hasattr(smart_r, "bonus"):
            if smart_r.veto:
                veto = True
                veto_reason = "WHALE_DUMP"
            else:
                total_bonus  += smart_r.bonus
                smart_signal  = smart_r.smart_signal
            if smart_r.details:
                details["smart_money"] = smart_r.details

        # ── AdaptiveScorer (weight-based adjustment) ──────────────────
        adaptive_adj = self._adaptive.get_bonus_for_score(details)
        total_bonus += adaptive_adj

        # ── Cap total satellite bonus ─────────────────────────────────
        total_bonus = max(-30, min(30, round(total_bonus)))

        pattern_signal = pat_r.patterns_found[0] if (hasattr(pat_r, "patterns_found") and pat_r.patterns_found) else ""

        return {
            "total_bonus":      total_bonus,
            "veto":             veto,
            "veto_reason":      veto_reason,
            "details":          details,
            "vol_sl_mult":      vol_sl_mult,
            "vol_sizing_mult":  vol_sizing_mult,
            "vol_label":        vol_label,
            "orderflow_signal": orderflow_signal,
            "smart_signal":     smart_signal,
            "social_score":     social_score,
            "pattern_signal":   pattern_signal,
            # ── Detailed fields for dashboard ─────────────────────────
            # Pattern
            "pattern_confidence": getattr(pat_r, "confidence", 0.0),
            "pattern_target":     getattr(pat_r, "target_price", 0.0),
            "patterns_found":     getattr(pat_r, "patterns_found", []),
            # Microstructure
            "vwap_signal":        getattr(micro_r, "vwap_signal", "NEUTRAL"),
            "cvd_signal":         getattr(micro_r, "cvd_signal", "NEUTRAL"),
            "buy_delta_pct":      getattr(micro_r, "buy_delta_pct", 50.0),
            # Volatility
            "atr_ratio":          getattr(vol_r, "atr_ratio", 1.0),
            "atr_predicted":      getattr(vol_r, "atr_predicted", 0.0),
            # Orderflow
            "bid_ask_imbalance":  getattr(of_r, "bid_ask_imbalance", 1.0),
            "buy_pressure":       getattr(of_r, "buy_pressure", 50.0),
            "absorption":         getattr(of_r, "absorption_detected", False),
            "stop_hunt":          getattr(of_r, "stop_hunt_detected", False),
            "iceberg":            getattr(of_r, "iceberg_detected", False),
            # Cross-asset
            "btc_dominance":      getattr(cross_r, "btc_dominance", 50.0),
            "eth_btc_ratio":      getattr(cross_r, "eth_btc_ratio", 0.0),
            "eth_btc_trend":      getattr(cross_r, "eth_btc_trend", "NEUTRAL"),
            "funding_rate":       getattr(cross_r, "funding_rate", 0.0),
            "funding_signal":     getattr(cross_r, "funding_signal", "NEUTRAL"),
            # Social sentiment
            "reddit_trend":       getattr(social_r, "reddit_trend", "NEUTRAL"),
            "fear_greed":         getattr(social_r, "fear_greed", 50.0),
            "cryptopanic_score":  getattr(social_r, "cryptopanic_score", 50.0),
            # Smart money
            "whale_buys":         getattr(smart_r, "whale_buys", 0),
            "whale_sells":        getattr(smart_r, "whale_sells", 0),
            "net_flow":           getattr(smart_r, "net_flow", 0.0),
            "large_buy_ratio":    getattr(smart_r, "large_buy_ratio", 0.5),
            # Liquidity
            "liq_above_pct":      getattr(liq_r, "above_distance_pct", 0.0),
            "liq_below_pct":      getattr(liq_r, "below_distance_pct", 0.0),
            "liq_above_price":    getattr(liq_r, "nearest_liq_above", 0.0),
            "liq_below_price":    getattr(liq_r, "nearest_liq_below", 0.0),
            "liq_above_size":     getattr(liq_r, "liq_above_size", 0.0),
            "liq_below_size":     getattr(liq_r, "liq_below_size", 0.0),
            # Microstructure signal summary
            "microstructure_signal": (
                "BULLISH" if getattr(micro_r, "vwap_signal", "NEUTRAL") == "BUY"
                             and getattr(micro_r, "cvd_signal", "NEUTRAL") == "BUY"
                else "BEARISH" if getattr(micro_r, "vwap_signal", "NEUTRAL") == "SELL"
                else "NEUTRAL"
            ),
        }
