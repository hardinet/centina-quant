"""
AdaptiveScorer — poids dynamiques ajustés chaque matin selon les 30 derniers trades.
Inspiré Kensho (S&P Global). Améliore le scoring sans API externe.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_FILE = Path("data/adaptive_weights.json")

# Poids par défaut (v5.0)
DEFAULT_WEIGHTS: dict[str, float] = {
    "rsi_sweet_spot":  10.0,
    "ema_alignment":   25.0,
    "macd_score":      15.0,
    "volume_spike":    15.0,
    "price_above_200": 8.0,
    "adx_bonus":       5.0,
    "regime_quality":  20.0,
    "news_bonus":       9.0,
    "pattern_bonus":   15.0,
}

TOTAL_TECH_BUDGET = 83.0   # budget technique max
MIN_WEIGHT        = 2.0    # poids minimal pour un critère
MAX_WEIGHT_FACTOR = 2.0    # un critère ne peut pas dépasser 2× son défaut


@dataclass
class AdaptiveWeights:
    weights:    dict[str, float]
    updated_at: str   = ""
    n_trades:   int   = 0
    changes:    list  = field(default_factory=list)  # log des changements


class AdaptiveScorer:
    """
    Analyse les 30 derniers trades chaque matin et ajuste les poids des critères.
    Un critère avec > 70% de réussite → poids +20%.
    Un critère avec < 40% de réussite → poids -30%.
    Un critère avec < 30% → retiré temporairement (MIN_WEIGHT).
    """

    def __init__(self):
        self._weights = self._load_weights()

    def get_weights(self) -> dict[str, float]:
        """Retourne les poids courants."""
        return dict(self._weights.weights)

    def get_bonus_for_score(self, score_details: dict) -> float:
        """
        Recalcule un bonus/malus basé sur les poids adaptatifs.
        score_details : dictionnaire des features présentes dans le signal.
        Retourne un ajustement entre -15 et +15 pts.
        """
        w = self._weights.weights
        default = DEFAULT_WEIGHTS
        adjustment = 0.0

        for criterion, default_w in default.items():
            current_w = w.get(criterion, default_w)
            delta = current_w - default_w
            if criterion in score_details and score_details[criterion]:
                adjustment += delta * 0.1

        return max(-15.0, min(15.0, adjustment))

    def update_from_trades(self, trades: list[dict]) -> AdaptiveWeights:
        """
        Appelle-moi avec les 30 derniers trades fermés.
        trades : liste de dicts avec keys 'pnl_usdt', 'score_breakdown', 'tp1_hit'.
        """
        if len(trades) < 10:
            return self._weights

        win_rates: dict[str, list[bool]] = {k: [] for k in DEFAULT_WEIGHTS}

        for trade in trades:
            won  = (trade.get("pnl_usdt", 0) or 0) > 0
            bd   = trade.get("score_breakdown") or {}

            # Corrélation critères → victoire
            if bd.get("rsi_sweet"):    win_rates["rsi_sweet_spot"].append(won)
            if bd.get("ema_full"):     win_rates["ema_alignment"].append(won)
            if bd.get("macd_bull"):    win_rates["macd_score"].append(won)
            if bd.get("volume_spike"): win_rates["volume_spike"].append(won)
            if bd.get("above_200"):    win_rates["price_above_200"].append(won)
            if bd.get("pattern_bonus", 0) > 0: win_rates["pattern_bonus"].append(won)
            if bd.get("news_bonus", 0) > 0:    win_rates["news_bonus"].append(won)

        new_weights = dict(DEFAULT_WEIGHTS)
        changes: list[str] = []

        for criterion, results in win_rates.items():
            if len(results) < 5:
                continue
            wr = sum(results) / len(results)
            current = new_weights.get(criterion, DEFAULT_WEIGHTS.get(criterion, 5.0))
            default  = DEFAULT_WEIGHTS.get(criterion, 5.0)

            if wr >= 0.70:
                new_w = min(current * 1.20, default * MAX_WEIGHT_FACTOR)
                if new_w != current:
                    changes.append(f"{criterion} +{new_w - current:.1f}pts (WR={wr:.0%})")
            elif wr < 0.30:
                new_w = MIN_WEIGHT
                if new_w != current:
                    changes.append(f"{criterion} → {new_w:.0f}pts MIN (WR={wr:.0%})")
            elif wr < 0.40:
                new_w = max(current * 0.70, MIN_WEIGHT)
                if new_w != current:
                    changes.append(f"{criterion} -{current - new_w:.1f}pts (WR={wr:.0%})")
            else:
                new_w = current

            new_weights[criterion] = round(new_w, 2)

        # Normalise pour respecter le budget technique
        tech_keys = [k for k in new_weights if k not in ("news_bonus", "pattern_bonus")]
        tech_total = sum(new_weights[k] for k in tech_keys)
        if tech_total > TOTAL_TECH_BUDGET:
            scale = TOTAL_TECH_BUDGET / tech_total
            for k in tech_keys:
                new_weights[k] = round(new_weights[k] * scale, 2)

        self._weights = AdaptiveWeights(
            weights=new_weights,
            updated_at=datetime.now(timezone.utc).isoformat(),
            n_trades=len(trades),
            changes=changes,
        )
        self._save_weights()

        if changes:
            logger.info("AdaptiveScorer: poids ajustés ce matin : %s", " | ".join(changes))
        else:
            logger.info("AdaptiveScorer: poids stables (n=%d trades)", len(trades))

        return self._weights

    # ── Persistence ───────────────────────────────────────────────────

    def _load_weights(self) -> AdaptiveWeights:
        try:
            if WEIGHTS_FILE.exists():
                data = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
                # Vérifie fraîcheur (< 36h)
                updated = datetime.fromisoformat(data.get("updated_at", "2000-01-01"))
                if datetime.now(timezone.utc).replace(tzinfo=None) - updated.replace(tzinfo=None) < timedelta(hours=36):
                    return AdaptiveWeights(
                        weights=data.get("weights", dict(DEFAULT_WEIGHTS)),
                        updated_at=data.get("updated_at", ""),
                        n_trades=data.get("n_trades", 0),
                    )
        except Exception as e:
            logger.debug("AdaptiveScorer load error: %s", e)
        return AdaptiveWeights(weights=dict(DEFAULT_WEIGHTS))

    def _save_weights(self) -> None:
        try:
            WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            WEIGHTS_FILE.write_text(
                json.dumps({
                    "weights":    self._weights.weights,
                    "updated_at": self._weights.updated_at,
                    "n_trades":   self._weights.n_trades,
                    "changes":    self._weights.changes,
                }, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("AdaptiveScorer save error: %s", e)
