#!/usr/bin/env python3
"""
CENTINA OMNI-QUANT v5.0 — Full Historical Simulation Runner
Fetches 20 years of data, runs all 5 strategies, generates full report.

Usage:
    python scripts/run_historical_simulation.py --symbol BTCUSDT --strategy all
    python scripts/run_historical_simulation.py --symbol ETHUSDT --strategy B --years 5
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("historical_sim")


async def main(args: argparse.Namespace) -> None:
    from historical_simulation.data_sources.crypto_data_fetcher import CryptoDataFetcher
    from historical_simulation.data_sources.macro_data_fetcher   import MacroDataFetcher
    from historical_simulation.backtest.historical_backtest_engine import HistoricalBacktestEngine
    from historical_simulation.backtest.multi_strategy_backtest   import MultiStrategyBacktest
    from historical_simulation.backtest.stress_test_engine        import StressTestEngine
    from historical_simulation.backtest.walk_forward_optimizer    import WalkForwardOptimizer
    from historical_simulation.validators.statistical_validator   import StatisticalValidator
    from historical_simulation.validators.overfitting_detector    import OverfittingDetector
    from historical_simulation.reports.report_generator           import ReportGenerator

    symbol = args.symbol.upper()
    years  = args.years

    logger.info("=== CENTINA Historical Simulation: %s — %d years ===", symbol, years)

    # ── Fetch data ────────────────────────────────────────────────────
    fetcher = CryptoDataFetcher()
    logger.info("Fetching OHLCV data...")
    df = await fetcher.fetch(symbol, "1d", years=years)

    if df.empty or len(df) < 300:
        logger.error("Insufficient data for %s (%d bars)", symbol, len(df))
        return

    logger.info("Loaded %d bars from %s to %s", len(df), df.index[0].date(), df.index[-1].date())

    # ── Run backtests ─────────────────────────────────────────────────
    engine    = HistoricalBacktestEngine()
    strats    = ["A", "B", "C", "D", "E"] if args.strategy == "all" else [args.strategy.upper()]
    all_metrics = []

    for s in strats:
        logger.info("Running strategy %s...", s)
        m = engine.run_backtest(df, symbol, "1d", strategy=s)
        all_metrics.append(m)
        logger.info(
            "  Strategy %s: Return=%.1f%% Sharpe=%.2f WR=%.1f%% MaxDD=%.1f%% Trades=%d",
            s, m.total_return_pct, m.sharpe, m.win_rate*100, m.max_drawdown_pct, m.nb_trades,
        )

    # ── Multi-strategy portfolio ──────────────────────────────────────
    if len(strats) > 1:
        logger.info("Running multi-strategy portfolio...")
        ms = MultiStrategyBacktest()
        ms_result = ms.run(df, symbol, "1d", allocation="perf_weighted", strategies=strats)
        logger.info(
            "  Combined: Return=%.1f%% Sharpe=%.2f MaxDD=%.1f%% DivRatio=%.2f",
            ms_result.combined_return, ms_result.combined_sharpe,
            ms_result.combined_max_dd, ms_result.diversification_ratio,
        )

    # ── Stress tests ──────────────────────────────────────────────────
    logger.info("Running stress tests...")
    stress_engine = StressTestEngine()
    stress_results = stress_engine.run_all_stress_tests(df, strategy=strats[0])
    survived = sum(1 for r in stress_results if r.survived)
    logger.info("  Stress: %d/%d scenarios survived", survived, len(stress_results))

    # ── Walk-forward ──────────────────────────────────────────────────
    logger.info("Running walk-forward optimization...")
    wf = WalkForwardOptimizer()
    wf_result = wf.run(df, symbol, strategy=strats[0], method="rolling", n_folds=6)
    logger.info(
        "  WF: IS Sharpe=%.2f OOS Sharpe=%.2f Efficiency=%.0f%%",
        wf_result.mean_train_sharpe, wf_result.mean_test_sharpe,
        wf_result.efficiency_ratio * 100,
    )

    # ── Statistical validation ────────────────────────────────────────
    logger.info("Running statistical validation...")
    validator = StatisticalValidator()
    ov_detector = OverfittingDetector()
    validation  = validator.validate(all_metrics[0])
    overfitting = ov_detector.detect(all_metrics[0])
    logger.info(
        "  Validation: DSR=%.3f PSR=%.3f Significant=%s",
        validation.deflated_sharpe, validation.psr, validation.is_significant,
    )
    logger.info("  Overfitting: PBO=%.2f %s", overfitting.pbo, overfitting.verdict)

    # ── Generate report ───────────────────────────────────────────────
    logger.info("Generating report...")
    gen    = ReportGenerator()
    report = gen.generate_full_report(
        metrics=all_metrics[0],
        stress_results=stress_results,
        validation=validation,
        overfitting=overfitting,
        walk_forward=wf_result,
        save=True,
    )

    score = report["summary"]["quality_score"]
    logger.info("=== DONE — Quality Score: %d/100 Deployable: %s ===",
                score, report["summary"]["deployable"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CENTINA Historical Simulation")
    parser.add_argument("--symbol",   default="BTCUSDT", help="Trading pair")
    parser.add_argument("--strategy", default="B",       help="Strategy A-E or 'all'")
    parser.add_argument("--years",    default=5, type=int, help="Years of history")
    asyncio.run(main(parser.parse_args()))
