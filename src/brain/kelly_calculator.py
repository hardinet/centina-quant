from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KellyResult:
    kelly_fraction: float       # raw Kelly f*
    fractional_kelly: float     # 1/8 Kelly
    position_pct: float         # capped per open-position rules
    position_usdt: float        # in USDT given capital
    risk_usdt: float            # expected risk USDT (stop-loss distance)
    reduced: bool = False       # True when override cap applied


class KellyCalculator:
    """
    Fractional Kelly (1/8) position sizing with v5.0 dynamic caps.

    Caps by open positions:
      0 positions → 5%
      1 position  → 3%
      2 positions → 2%
    Override cap:
      win_rate < 40% OR 3 consecutive losses → 1%
    TRENDING_STRONG regime: multiply by 1.2 (up to cap)
    """

    KELLY_FRACTION = 1 / 8

    # v5.0 position caps
    CAPS_BY_OPEN = {0: 0.05, 1: 0.03, 2: 0.02}
    OVERRIDE_CAP = 0.01    # 1% when conditions are weak
    TRENDING_STRONG_MULT = 1.2

    def calculate(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        capital_usdt: float,
        sl_distance_pct: float,
        open_positions: int = 0,
        consecutive_losses: int = 0,
        trending_strong: bool = False,
    ) -> KellyResult:
        if win_rate <= 0 or win_rate >= 1:
            raise ValueError("win_rate must be strictly between 0 and 1")
        if avg_win_pct <= 0 or avg_loss_pct <= 0:
            raise ValueError("avg_win_pct and avg_loss_pct must be positive")

        loss_rate = 1.0 - win_rate
        b = avg_win_pct / avg_loss_pct

        kelly_f   = max(0.0, (b * win_rate - loss_rate) / b)
        fractional = kelly_f * self.KELLY_FRACTION

        # Determine cap
        override = win_rate < 0.40 or consecutive_losses >= 3
        reduced  = False
        if override:
            cap = self.OVERRIDE_CAP
            reduced = True
        else:
            cap = self.CAPS_BY_OPEN.get(open_positions, 0.02)

        position_pct = min(fractional, cap)

        # TRENDING_STRONG boost (bounded by cap)
        if trending_strong and not override:
            position_pct = min(position_pct * self.TRENDING_STRONG_MULT, cap)

        position_usdt = capital_usdt * position_pct
        risk_usdt     = position_usdt * sl_distance_pct

        return KellyResult(
            kelly_fraction=round(kelly_f, 4),
            fractional_kelly=round(fractional, 4),
            position_pct=round(position_pct, 4),
            position_usdt=round(position_usdt, 2),
            risk_usdt=round(risk_usdt, 2),
            reduced=reduced,
        )

    def from_signal_score(
        self,
        score: float,
        capital_usdt: float,
        sl_distance_pct: float,
        base_win_rate: float = 0.55,
        open_positions: int = 0,
        consecutive_losses: int = 0,
        trending_strong: bool = False,
    ) -> KellyResult:
        boost    = (score - 50) / 200
        win_rate = max(0.30, min(0.75, base_win_rate + boost))
        avg_win  = sl_distance_pct * 2.0
        return self.calculate(
            win_rate, avg_win, sl_distance_pct, capital_usdt, sl_distance_pct,
            open_positions=open_positions,
            consecutive_losses=consecutive_losses,
            trending_strong=trending_strong,
        )
