#!/usr/bin/env python3
"""
CENTINA OMNI-QUANT v5.0 — Quick Backtest (no API, uses cached or synthetic data)

Usage:
    python scripts/quick_backtest.py
    python scripts/quick_backtest.py --symbol ETHUSDT --strategy A --days 365
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("quick_backtest")


def run(args: argparse.Namespace) -> None:
    from historical_simulation.synthetic.synthetic_data_generator import (
        SyntheticDataGenerator, SyntheticConfig,
    )
    from historical_simulation.backtest.historical_backtest_engine import HistoricalBacktestEngine
    from historical_simulation.backtest.stress_test_engine         import StressTestEngine
    from historical_simulation.validators.statistical_validator    import StatisticalValidator

    symbol = args.symbol.upper()
    days   = args.days
    strat  = args.strategy.upper()

    print(f"\n{'='*60}")
    print(f" CENTINA Quick Backtest — {symbol} / Strategy {strat} / {days}d")
    print(f"{'='*60}")

    # Generate synthetic OHLCV — GBM with bullish drift to simulate trending crypto market
    gen  = SyntheticDataGenerator()
    cfg  = SyntheticConfig(
        model="gbm", n_days=days, start_price=40_000.0, freq="1h",
        mu=0.0008,     # ~29% annual drift (bullish crypto bias)
        sigma=0.022,   # ~2.2% daily vol
        seed=42,
    )
    df   = gen.generate(cfg)
    print(f"[DATA] {len(df)} bars @ 1h = {len(df)//24}d (synthetic GBM, bullish)")

    # Backtest
    engine  = HistoricalBacktestEngine(capital=200.0)
    metrics = engine.run_backtest(df, symbol, "1h", strategy=strat)

    print(f"\n[RESULTS]")
    print(f"  Capital:        €{metrics.capital_initial:.0f} → €{metrics.capital_final:.2f}")
    print(f"  Total Return:   {metrics.total_return_pct:+.2f}%")
    print(f"  CAGR:           {metrics.cagr:+.2f}%")
    print(f"  Sharpe:         {metrics.sharpe:.3f}")
    print(f"  Sortino:        {metrics.sortino:.3f}")
    print(f"  Calmar:         {metrics.calmar:.3f}")
    print(f"  Max Drawdown:   {metrics.max_drawdown_pct:.2f}%")
    print(f"  Win Rate:       {metrics.win_rate*100:.1f}%")
    print(f"  Profit Factor:  {metrics.profit_factor:.3f}")
    print(f"  Trades:         {metrics.nb_trades}")
    print(f"  VaR 95%:        {metrics.var_95:.2f}€")
    print(f"  CVaR 95%:       {metrics.cvar_95:.2f}€")
    print(f"  Ulcer Index:    {metrics.ulcer_index:.4f}")

    # Monte Carlo
    mc = engine.run_monte_carlo_simulation(metrics, n=500)
    if mc:
        print(f"\n[MONTE CARLO n=500]")
        print(f"  Mean:           €{mc['mean']:.2f}")
        print(f"  P5 (worst 5%):  €{mc['p5']:.2f}")
        print(f"  P95 (best 5%):  €{mc['p95']:.2f}")
        print(f"  Prob Profit:    {mc['prob_profit']*100:.1f}%")

    # Stress tests
    stress  = StressTestEngine()
    results = stress.run_all_stress_tests(df, strategy=strat)
    survived = sum(1 for r in results if r.survived)
    print(f"\n[STRESS TESTS]  Survived: {survived}/{len(results)}")
    for r in results:
        icon = "✓" if r.survived else "✗"
        print(f"  {icon} {r.scenario:<28} loss={r.max_loss_pct:.1f}%  recovery={r.recovery_days}d")

    # Statistical validation
    val = StatisticalValidator()
    v   = val.validate(metrics)
    print(f"\n[STATISTICAL VALIDATION]")
    print(f"  Sharpe:              {v.sharpe_ratio:.3f}")
    print(f"  Deflated Sharpe:     {v.deflated_sharpe:.3f}")
    print(f"  PSR:                 {v.psr:.4f}")
    print(f"  Significant:         {v.is_significant}")
    print(f"  Track record:        {v.actual_track_record:.2f}y / min {v.min_track_record:.2f}y needed")
    print(f"  Skewness:            {v.skewness:.3f}")
    print(f"  Kurtosis:            {v.kurtosis:.3f}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CENTINA Quick Backtest")
    parser.add_argument("--symbol",   default="BTCUSDT", help="Symbol (display only)")
    parser.add_argument("--strategy", default="B",       help="Strategy A-E")
    parser.add_argument("--days",     default=365, type=int, help="Simulation length in days")
    run(parser.parse_args())
