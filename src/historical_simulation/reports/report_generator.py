from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.historical_backtest_engine import BacktestMetrics
from ..backtest.stress_test_engine import StressTestResult
from ..validators.statistical_validator import ValidationResult
from ..validators.overfitting_detector import OverfittingReport
from ..backtest.walk_forward_optimizer import WalkForwardResult

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/historical")


class ReportGenerator:
    """
    Generates comprehensive HTML, JSON, and CSV reports from simulation results.
    Bundles: backtest metrics, stress tests, statistical validation, overfitting, walk-forward.
    """

    def __init__(self, output_dir: Path | None = None):
        self.out = output_dir or REPORTS_DIR
        self.out.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        metrics: BacktestMetrics,
        stress_results: list[StressTestResult] | None = None,
        validation: ValidationResult | None = None,
        overfitting: OverfittingReport | None = None,
        walk_forward: WalkForwardResult | None = None,
        save: bool = True,
    ) -> dict:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = f"{metrics.symbol}_{metrics.strategy}_{ts}"

        report = {
            "generated_at":  datetime.utcnow().isoformat(),
            "symbol":        metrics.symbol,
            "strategy":      metrics.strategy,
            "timeframe":     metrics.timeframe,
            "period":        f"{metrics.start_date} → {metrics.end_date}",
            "backtest":      self._metrics_dict(metrics),
            "stress_tests":  [self._stress_dict(s) for s in (stress_results or [])],
            "validation":    asdict(validation)  if validation  else {},
            "overfitting":   asdict(overfitting) if overfitting else {},
            "walk_forward":  self._wf_dict(walk_forward) if walk_forward else {},
            "summary":       self._make_summary(metrics, validation, overfitting),
        }

        if save:
            self._save_json(report,  slug)
            self._save_csv(metrics,  slug)
            self._save_html(report,  slug)

        return report

    def generate_comparison_report(
        self,
        results: list[BacktestMetrics],
        save: bool = True,
    ) -> pd.DataFrame:
        rows = []
        for m in results:
            rows.append({
                "symbol":        m.symbol,
                "strategy":      m.strategy,
                "timeframe":     m.timeframe,
                "start":         m.start_date,
                "end":           m.end_date,
                "total_return":  m.total_return_pct,
                "cagr":          m.cagr,
                "sharpe":        m.sharpe,
                "sortino":       m.sortino,
                "calmar":        m.calmar,
                "max_dd":        m.max_drawdown_pct,
                "win_rate":      m.win_rate,
                "profit_factor": m.profit_factor,
                "nb_trades":     m.nb_trades,
                "var_95":        m.var_95,
                "cvar_95":       m.cvar_95,
                "ulcer_index":   m.ulcer_index,
                "recovery_factor": m.recovery_factor,
            })
        df = pd.DataFrame(rows)
        if save:
            ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = self.out / f"comparison_{ts}.csv"
            df.to_csv(path, index=False)
            logger.info("Comparison report saved: %s", path)
        return df

    # ── Private helpers ───────────────────────────────────────────────

    def _metrics_dict(self, m: BacktestMetrics) -> dict:
        return {
            "capital_initial": m.capital_initial,
            "capital_final":   m.capital_final,
            "total_return":    m.total_return_pct,
            "cagr":            m.cagr,
            "sharpe":          m.sharpe,
            "sortino":         m.sortino,
            "calmar":          m.calmar,
            "max_drawdown":    m.max_drawdown_pct,
            "profit_factor":   m.profit_factor,
            "win_rate":        m.win_rate,
            "nb_trades":       m.nb_trades,
            "avg_win_pct":     m.avg_win_pct,
            "avg_loss_pct":    m.avg_loss_pct,
            "expectancy":      m.expectancy,
            "var_95":          m.var_95,
            "cvar_95":         m.cvar_95,
            "ulcer_index":     m.ulcer_index,
            "recovery_factor": m.recovery_factor,
        }

    def _stress_dict(self, s: StressTestResult) -> dict:
        return {
            "scenario":         s.scenario,
            "survived":         s.survived,
            "capital_surviving": s.capital_surviving,
            "max_loss_pct":     s.max_loss_pct,
            "sl_triggered":     s.sl_triggered,
            "recovery_days":    s.recovery_days,
        }

    def _wf_dict(self, wf: WalkForwardResult) -> dict:
        return {
            "method":             wf.method,
            "n_folds":            wf.n_folds,
            "mean_train_sharpe":  wf.mean_train_sharpe,
            "mean_test_sharpe":   wf.mean_test_sharpe,
            "efficiency_ratio":   wf.efficiency_ratio,
            "stable_params":      wf.stable_params,
            "recommended_params": wf.recommended_params,
            "n_overfit_folds":    sum(1 for f in wf.folds if f.overfitting),
        }

    def _make_summary(
        self,
        metrics: BacktestMetrics,
        val: ValidationResult | None,
        ov:  OverfittingReport  | None,
    ) -> dict:
        flags: list[str] = []
        score = 100

        if metrics.sharpe < 0.5:
            flags.append("Low Sharpe (<0.5)")
            score -= 20
        if metrics.max_drawdown_pct > 30:
            flags.append("High max drawdown (>30%)")
            score -= 15
        if metrics.win_rate < 0.4:
            flags.append("Low win rate (<40%)")
            score -= 10
        if val and not val.is_significant:
            flags.append("Returns not statistically significant (p>0.05)")
            score -= 15
        if val and not val.is_sufficient:
            flags.append(f"Insufficient track record (<{val.min_track_record:.1f}y needed)")
            score -= 10
        if ov and ov.is_overfit:
            flags.append(f"Overfitting detected (PBO={ov.pbo:.2f})")
            score -= 20
        if metrics.nb_trades < 30:
            flags.append("Too few trades (<30)")
            score -= 10

        return {
            "quality_score": max(score, 0),
            "flags":         flags,
            "deployable":    score >= 60 and not (ov and ov.is_overfit),
        }

    def _save_json(self, report: dict, slug: str) -> None:
        path = self.out / f"{slug}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("JSON report saved: %s", path)

    def _save_csv(self, metrics: BacktestMetrics, slug: str) -> None:
        if not metrics.trades:
            return
        df = pd.DataFrame([{
            "entry": t.entry_price, "exit": t.exit_price,
            "qty": t.qty, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
            "duration_h": t.duration_h, "tp1_hit": t.tp1_hit,
            "tp2_hit": t.tp2_hit, "exit_reason": t.exit_reason,
        } for t in metrics.trades])
        path = self.out / f"{slug}_trades.csv"
        df.to_csv(path, index=False)
        logger.info("CSV trades saved: %s", path)

    def _save_html(self, report: dict, slug: str) -> None:
        path = self.out / f"{slug}.html"
        b    = report["backtest"]
        s    = report["summary"]
        stress_rows = "".join(
            f"<tr><td>{r['scenario']}</td>"
            f"<td>{'✓' if r['survived'] else '✗'}</td>"
            f"<td>{r['max_loss_pct']:.1f}%</td>"
            f"<td>{r['recovery_days']}d</td></tr>"
            for r in report.get("stress_tests", [])
        )
        flag_items = "".join(f"<li>{f}</li>" for f in s.get("flags", []))
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CENTINA Report — {report['symbol']} {report['strategy']}</title>
<style>
body{{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:24px}}
h1{{color:#58a6ff}} h2{{color:#79c0ff;border-bottom:1px solid #30363d;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #30363d;padding:8px}}
th{{background:#161b22}} .ok{{color:#3fb950}} .warn{{color:#d29922}} .bad{{color:#f85149}}
.badge{{padding:4px 12px;border-radius:4px;font-weight:bold}}
</style></head><body>
<h1>CENTINA Historical Simulation Report</h1>
<p>Symbol: <b>{report['symbol']}</b> | Strategy: <b>{report['strategy']}</b>
| Period: {report['period']} | Generated: {report['generated_at'][:19]} UTC</p>

<h2>Quality Score</h2>
<p class="badge {'ok' if s['quality_score']>=70 else ('warn' if s['quality_score']>=50 else 'bad')}">
{s['quality_score']}/100 — {'DEPLOYABLE' if s['deployable'] else 'NOT DEPLOYABLE'}</p>
<ul>{flag_items}</ul>

<h2>Backtest Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Return</td><td>{b['total_return']:.2f}%</td></tr>
<tr><td>CAGR</td><td>{b['cagr']:.2f}%</td></tr>
<tr><td>Sharpe</td><td>{b['sharpe']:.3f}</td></tr>
<tr><td>Sortino</td><td>{b['sortino']:.3f}</td></tr>
<tr><td>Calmar</td><td>{b['calmar']:.3f}</td></tr>
<tr><td>Max Drawdown</td><td>{b['max_drawdown']:.2f}%</td></tr>
<tr><td>Win Rate</td><td>{b['win_rate']*100:.1f}%</td></tr>
<tr><td>Profit Factor</td><td>{b['profit_factor']:.3f}</td></tr>
<tr><td>Trades</td><td>{b['nb_trades']}</td></tr>
<tr><td>VaR 95%</td><td>{b['var_95']:.2f}</td></tr>
<tr><td>CVaR 95%</td><td>{b['cvar_95']:.2f}</td></tr>
</table>

<h2>Stress Tests</h2>
<table>
<tr><th>Scenario</th><th>Survived</th><th>Max Loss</th><th>Recovery</th></tr>
{stress_rows}
</table>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved: %s", path)
