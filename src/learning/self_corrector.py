from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.connectors.binance_rest import BinanceREST
from src.database.cortex_db import CortexDB
from src.learning.backtest_engine import BacktestEngine
from src.learning.deep_predictor import DeepPredictor
from src.learning.hyperoptimizer import HyperOptimizer
from src.learning.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

PARAMS_FILE    = Path("config/strategy_params.yaml")
OPTIMISED_FILE = Path("data/optimised_params.json")
RUN_HOUR       = 3    # 03:00 UTC nightly
DEEP_RUN_HOUR  = 2    # 02:00 UTC — réentraînement du DeepPredictor


class SelfCorrector:
    """
    Nightly self-correction loop.

    At RUN_HOUR UTC it:
    1. Reads recent closed trades from CortexDB
    2. Evaluates current performance (Sharpe, win-rate, drawdown)
    3. If performance degraded below thresholds → runs HyperOptimizer
    4. Writes improved params to data/optimised_params.json (hot-reloaded by main)
    5. Logs the decision to audit log
    """

    SHARPE_FLOOR    = 0.5    # trigger re-optimisation below this
    WIN_RATE_FLOOR  = 0.45
    MIN_TRADES      = 20     # need enough trades for meaningful stats
    TOP_SYMBOLS_N   = 5      # optimise on top-N most traded symbols

    def __init__(self, rest: BinanceREST, db: CortexDB):
        self._rest      = rest
        self._db        = db
        self._tracker   = PerformanceTracker()
        self._optim     = HyperOptimizer(n_trials=80, timeout_sec=240)
        self._engine    = BacktestEngine()
        self._deep      = DeepPredictor()

    async def run_forever(self) -> None:
        logger.info("SelfCorrector: nightly loop started (fires at %02d:00 UTC)", RUN_HOUR)
        while True:
            now = datetime.utcnow()

            # Deep model retrain at 02:00 UTC
            deep_target = now.replace(hour=DEEP_RUN_HOUR, minute=0, second=0, microsecond=0)
            if now >= deep_target:
                from datetime import timedelta
                deep_target += timedelta(days=1)

            # Main correction at 03:00 UTC
            target_hour = now.replace(hour=RUN_HOUR, minute=0, second=0, microsecond=0)
            if now >= target_hour:
                from datetime import timedelta
                target_hour += timedelta(days=1)

            # Wait for whichever fires first
            wait = min(
                (deep_target - now).total_seconds(),
                (target_hour - now).total_seconds(),
            )
            logger.info("SelfCorrector: next run in %.0f seconds", wait)
            await asyncio.sleep(wait)

            now2 = datetime.utcnow()
            if abs(now2.hour - DEEP_RUN_HOUR) == 0 and now2.minute < 5:
                await self._retrain_deep_model()
            if abs(now2.hour - RUN_HOUR) == 0 and now2.minute < 5:
                await self._nightly_run()

    async def _nightly_run(self) -> None:
        logger.info("SelfCorrector: starting nightly analysis")
        trades = self._db.get_recent_trades(200)
        closed = [t for t in trades if t.closed_at and t.pnl_usdt is not None]

        if len(closed) < self.MIN_TRADES:
            logger.info("SelfCorrector: only %d trades, skipping (need %d)", len(closed), self.MIN_TRADES)
            return

        pnls    = [t.pnl_usdt for t in closed]
        capital = float(self._get_param("capital", 1000.0))
        report  = self._tracker.compute(pnls, capital)

        logger.info("SelfCorrector: %s", report.summary())

        needs_reopt = (
            report.sharpe_ratio < self.SHARPE_FLOOR
            or report.win_rate   < self.WIN_RATE_FLOOR
        )

        # ── Adaptive scorer morning update ───────────────────────────
        try:
            from src.brain.adaptive_scorer import AdaptiveScorer
            trade_dicts = [
                {
                    "pnl_usdt":       t.pnl_usdt,
                    "score_breakdown": json.loads(t.score_breakdown) if isinstance(t.score_breakdown, str) else (t.score_breakdown or {}),
                    "tp1_hit":        getattr(t, "tp1_hit", False),
                }
                for t in closed[-30:]
            ]
            aw = AdaptiveScorer().update_from_trades(trade_dicts)
            logger.info("SelfCorrector: AdaptiveScorer updated (%d trades, %d changes)",
                        aw.n_trades, len(aw.changes))
        except Exception as e:
            logger.debug("AdaptiveScorer update error: %s", e)

        if not needs_reopt:
            logger.info("SelfCorrector: performance OK – no changes")
            return

        logger.warning("SelfCorrector: performance below thresholds → launching hyperoptimisation")

        symbols = self._top_symbols(closed)
        new_params: dict[str, Any] = {}

        for sym in symbols[:self.TOP_SYMBOLS_N]:
            try:
                df = await self._rest.get_klines(sym, "4h", limit=500)
                result = self._optim.optimise(df, sym, "4h", capital)
                new_params[sym] = result.best_params
                logger.info("SelfCorrector: %s → Sharpe %.3f  params=%s", sym, result.best_value, result.best_params)
            except Exception as e:
                logger.error("SelfCorrector: optimisation failed for %s: %s", sym, e)

        if new_params:
            OPTIMISED_FILE.parent.mkdir(parents=True, exist_ok=True)
            OPTIMISED_FILE.write_text(
                json.dumps({"updated_at": datetime.utcnow().isoformat(), "params": new_params}, indent=2)
            )
            logger.info("SelfCorrector: updated params written to %s", OPTIMISED_FILE)

    async def _retrain_deep_model(self) -> None:
        logger.info("SelfCorrector: réentraînement DeepPredictor (90j BTCUSDT 1h)…")
        try:
            df = await self._rest.get_klines("BTCUSDT", "1h", limit=2160)  # 90 jours
            if df is None or len(df) < 500:
                logger.warning("SelfCorrector: pas assez de données pour DeepPredictor")
                return

            # Prépare dataset avec les features attendues
            import numpy as np
            closes = df["close"].astype(float).values
            volumes = df["volume"].astype(float).values
            vol_mean = np.mean(volumes)

            records: list[dict] = []
            for i, row in enumerate(df.itertuples()):
                close = float(row.close)
                hour  = row.Index.hour if hasattr(row.Index, "hour") else (i % 24)
                dow   = row.Index.dayofweek if hasattr(row.Index, "dayofweek") else (i % 7)
                vol   = float(row.volume) / vol_mean if vol_mean > 0 else 1.0
                records.append({
                    "close":       close,
                    "rsi":         50.0,   # simplified — full features need TA
                    "ema7_ratio":  1.0,
                    "ema25_ratio": 1.0,
                    "ema99_ratio": 1.0,
                    "volume_ratio": vol,
                    "atr_pct":     0.02,
                    "macd_hist":   0.0,
                    "obv_slope":   0.0,
                    "hour_utc":    hour,
                    "day_of_week": dow,
                    "btc_pct_1h":  0.0,
                    "spread_pct":  0.001,
                })

            metrics = self._deep.train(records)
            if "error" in metrics:
                logger.warning("SelfCorrector: DeepPredictor erreur — %s", metrics["error"])
            else:
                auc  = metrics.get("auc_mean", 0)
                n    = metrics.get("n_samples", 0)
                logger.info("SelfCorrector: Deep model réentraîné : AUC = %.3f sur %d exemples", auc, n)
                if auc > 0.65:
                    logger.info("SelfCorrector: AUC > 0.65 — prédicteur activé pour les prochains trades")
        except Exception as e:
            logger.error("SelfCorrector: _retrain_deep_model error: %s", e)

    def _top_symbols(self, trades) -> list[str]:
        from collections import Counter
        counts = Counter(t.symbol for t in trades)
        return [sym for sym, _ in counts.most_common(self.TOP_SYMBOLS_N)]

    def _get_param(self, key: str, default: Any) -> Any:
        try:
            data = yaml.safe_load(PARAMS_FILE.read_text())
            return data.get(key, default)
        except Exception:
            return default
