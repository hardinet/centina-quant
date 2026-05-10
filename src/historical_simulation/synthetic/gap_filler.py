from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FillMethod = Literal["linear", "forward", "spline", "gbm", "mean_revert"]


class GapFiller:
    """
    Fills OHLCV gaps in historical price data using multiple methods.
    Handles exchange downtime, API gaps, and delisted periods.
    """

    def fill(
        self,
        df: pd.DataFrame,
        method: FillMethod = "gbm",
        max_gap_hours: int = 72,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.sort_index()
        full_idx = pd.date_range(df.index[0], df.index[-1], freq=self._infer_freq(df))
        df = df.reindex(full_idx)

        gaps = self._find_gaps(df)
        if not gaps:
            return df.ffill()

        for start, end in gaps:
            gap_len = (end - start).total_seconds() / 3600
            if gap_len > max_gap_hours:
                logger.debug("Gap too large (%dh), skipping fill", int(gap_len))
                continue

            df = self._fill_gap(df, start, end, method)

        return df

    def fill_volume_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns:
            return df
        # Zero volume → fill with rolling mean
        zero_mask = df["volume"] == 0
        if zero_mask.any():
            rolling_mean = df["volume"].replace(0, np.nan).rolling(20, min_periods=1).mean()
            df.loc[zero_mask, "volume"] = rolling_mean[zero_mask]
        return df

    def validate(self, df: pd.DataFrame) -> dict:
        total    = len(df)
        missing  = int(df["close"].isna().sum()) if "close" in df.columns else 0
        gaps     = self._find_gaps(df)
        return {
            "total_rows":   total,
            "missing_rows": missing,
            "completeness": round(1 - missing / max(total, 1), 4),
            "n_gaps":       len(gaps),
            "max_gap_h":    max(
                (int((e - s).total_seconds() / 3600) for s, e in gaps), default=0
            ),
        }

    # ── Private ───────────────────────────────────────────────────────

    def _infer_freq(self, df: pd.DataFrame) -> str:
        if len(df) < 2:
            return "1h"
        delta = df.index[1] - df.index[0]
        h = delta.total_seconds() / 3600
        if h <= 0.25:  return "15min"
        if h <= 1:     return "1h"
        if h <= 4:     return "4h"
        return "1d"

    def _find_gaps(self, df: pd.DataFrame) -> list[tuple]:
        col = "close" if "close" in df.columns else df.columns[0]
        nas = df[col].isna()
        gaps: list[tuple] = []
        in_gap = False
        start  = None
        for ts, is_na in nas.items():
            if is_na and not in_gap:
                in_gap = True
                start  = ts
            elif not is_na and in_gap:
                gaps.append((start, ts))
                in_gap = False
        return gaps

    def _fill_gap(self, df: pd.DataFrame, start, end, method: FillMethod) -> pd.DataFrame:
        before = df.loc[:start].dropna(subset=["close"]) if "close" in df.columns else df.loc[:start].dropna()
        after  = df.loc[end:].dropna(subset=["close"])   if "close" in df.columns else df.loc[end:].dropna()

        if before.empty or after.empty:
            return df.ffill()

        p_start = float(before["close"].iloc[-1])
        p_end   = float(after["close"].iloc[0])
        gap_idx = df.loc[start:end].index
        n       = len(gap_idx)

        if n <= 1:
            return df

        if method == "forward":
            prices = np.full(n, p_start)
        elif method == "linear":
            prices = np.linspace(p_start, p_end, n)
        elif method == "spline":
            prices = np.interp(
                range(n), [0, n-1], [p_start, p_end]
            )
        elif method == "gbm":
            prices = self._gbm_fill(p_start, p_end, n)
        else:  # mean_revert
            prices = self._mean_revert_fill(p_start, p_end, n)

        for i, ts in enumerate(gap_idx):
            p = prices[i]
            df.loc[ts, "open"]  = p * np.random.uniform(0.998, 1.002)
            df.loc[ts, "close"] = p
            df.loc[ts, "high"]  = p * np.random.uniform(1.000, 1.005)
            df.loc[ts, "low"]   = p * np.random.uniform(0.995, 1.000)
            df.loc[ts, "volume"] = float(before["volume"].tail(5).mean()) if "volume" in df.columns else 0.0

        return df

    def _gbm_fill(self, p_start: float, p_end: float, n: int) -> np.ndarray:
        mu    = np.log(p_end / p_start) / n
        sigma = abs(mu) * 2 + 0.001
        path  = [p_start]
        for _ in range(n - 1):
            dW = np.random.normal(0, 1) * sigma
            path.append(path[-1] * np.exp(mu + dW))
        path[-1] = p_end
        return np.array(path)

    def _mean_revert_fill(self, p_start: float, p_end: float, n: int) -> np.ndarray:
        theta = 0.3
        mid   = (p_start + p_end) / 2
        path  = [p_start]
        for _ in range(n - 1):
            revert = theta * (mid - path[-1])
            noise  = np.random.normal(0, abs(p_start - p_end) / (n * 3))
            path.append(path[-1] + revert + noise)
        path[-1] = p_end
        return np.array(path)
