from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCOLevels:
    entry: float
    sl: float              # stop-loss = entry - 1×ATR
    tp1: float             # take-profit 1 = entry + 2.5% → close 50%
    tp2: float             # take-profit 2 = entry + 5.2% → close 30%
    tp3_trail_atr: float   # ATR distance for trailing stop on remaining 20%
    sl_pct: float          # SL distance as % of entry
    tp1_pct: float         # TP1 gain as % of entry
    tp2_pct: float         # TP2 gain as % of entry
    tp1_rr: float          # reward/risk at TP1
    tp2_rr: float          # reward/risk at TP2
    qty_tp1: float         # 50% closed at TP1
    qty_tp2: float         # 30% closed at TP2
    qty_tp3: float         # 20% trailed to TP3
    net_gain_pct: float    # weighted net expected gain (fees included)


class OCOCalculator:
    """
    v5.0 OCO Structure:
    TP1 = entry × 1.025  → close 50% qty  (+2.5%)
    TP2 = entry × 1.052  → close 30% qty  (+5.2%)
    TP3 = trailing ATR×1.5 → close 20%   (open-ended)
    SL  = entry - 1×ATR  → cut entire remaining position
    """

    TP1_PCT   = 0.025    # +2.5%
    TP2_PCT   = 0.052    # +5.2%
    SL_MULT   = 1.0      # ATR multiplier for SL
    TRAIL_MULT = 1.5     # ATR multiplier for trailing stop

    QTY_TP1 = 0.50       # 50% at TP1
    QTY_TP2 = 0.30       # 30% at TP2
    QTY_TP3 = 0.20       # 20% trailing

    FEE = 0.001          # 0.1% per leg

    def calculate(self, entry: float, atr: float) -> OCOLevels:
        if atr <= 0:
            raise ValueError("ATR must be positive")
        if entry <= 0:
            raise ValueError("Entry price must be positive")

        sl  = entry - self.SL_MULT   * atr
        tp1 = entry * (1 + self.TP1_PCT)
        tp2 = entry * (1 + self.TP2_PCT)

        sl_dist  = entry - sl
        sl_pct   = sl_dist / entry * 100

        net_gain = (
            self.QTY_TP1 * (self.TP1_PCT - 2 * self.FEE) +
            self.QTY_TP2 * (self.TP2_PCT - 2 * self.FEE) +
            self.QTY_TP3 * (self.TP2_PCT * 1.3 - 2 * self.FEE)  # trailing estimate
        ) * 100

        return OCOLevels(
            entry=round(entry, 8),
            sl=round(sl, 8),
            tp1=round(tp1, 8),
            tp2=round(tp2, 8),
            tp3_trail_atr=round(self.TRAIL_MULT * atr, 8),
            sl_pct=round(sl_pct, 4),
            tp1_pct=round(self.TP1_PCT * 100, 2),
            tp2_pct=round(self.TP2_PCT * 100, 2),
            tp1_rr=round((tp1 - entry) / sl_dist, 2),
            tp2_rr=round((tp2 - entry) / sl_dist, 2),
            qty_tp1=self.QTY_TP1,
            qty_tp2=self.QTY_TP2,
            qty_tp3=self.QTY_TP3,
            net_gain_pct=round(net_gain, 3),
        )

    def display(self, levels: OCOLevels) -> str:
        return (
            f"ENTRY={levels.entry:.4f}  "
            f"SL={levels.sl:.4f} (-{levels.sl_pct:.2f}%)  "
            f"TP1={levels.tp1:.4f} (+{levels.tp1_pct:.1f}% × 50%)  "
            f"TP2={levels.tp2:.4f} (+{levels.tp2_pct:.1f}% × 30%)  "
            f"TP3=trail×{self.TRAIL_MULT}ATR × 20%  "
            f"NET≈+{levels.net_gain_pct:.2f}%"
        )

    def adjust_for_step_size(self, price: float, step_size: float) -> float:
        if step_size <= 0:
            return price
        precision = len(str(step_size).rstrip("0").split(".")[-1])
        return round(round(price / step_size) * step_size, precision)
