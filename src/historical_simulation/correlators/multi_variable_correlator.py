from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

Method = Literal["pearson", "spearman", "kendall", "cosine"]
WINDOW_SIZES = [7, 14, 30, 90, 180, 365]


class MultiVariableCorrelator:
    """
    Correlates crypto prices with macro/sentiment variables.
    Methods: Pearson, Spearman, Kendall, cosine, Granger causality.
    """

    def correlate(
        self,
        crypto: pd.Series,
        variable: pd.Series,
        window: int = 30,
        method: Method = "pearson",
    ) -> pd.Series:
        """Rolling correlation over time."""
        combined = pd.concat([crypto, variable], axis=1).dropna()
        if len(combined) < window:
            return pd.Series(dtype=float)
        c, v = combined.iloc[:, 0], combined.iloc[:, 1]
        if method == "pearson":
            return c.rolling(window).corr(v)
        if method == "spearman":
            return c.rolling(window).apply(
                lambda x: stats.spearmanr(x, v.loc[x.index])[0], raw=False
            )
        if method == "kendall":
            return c.rolling(window).apply(
                lambda x: stats.kendalltau(x, v.loc[x.index])[0], raw=False
            )
        if method == "cosine":
            def cos_sim(x: pd.Series) -> float:
                a, b = x.values, v.loc[x.index].values
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                return float(a.dot(b) / (na * nb)) if na * nb > 0 else 0.0
            return c.rolling(window).apply(cos_sim, raw=False)
        raise ValueError(f"Unknown method: {method}")

    def find_leading_indicators(
        self,
        crypto: pd.Series,
        target: pd.Series,
        max_lag: int = 30,
        method: Method = "pearson",
    ) -> pd.DataFrame:
        """Find if 'crypto' leads 'target' — returns correlation vs lag."""
        results = []
        for lag in range(0, max_lag + 1):
            shifted = crypto.shift(lag)
            combined = pd.concat([shifted, target], axis=1).dropna()
            if len(combined) < 20:
                results.append({"lag": lag, "correlation": float("nan"), "p_value": float("nan")})
                continue
            if method == "pearson":
                r, p = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
            else:
                r, p = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
            results.append({"lag": lag, "correlation": r, "p_value": p})
        return pd.DataFrame(results).set_index("lag")

    def granger_causality(
        self, cause: pd.Series, effect: pd.Series, max_lag: int = 10
    ) -> dict:
        """Test if 'cause' Granger-causes 'effect'. Returns p-values by lag."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            combined = pd.concat([effect, cause], axis=1).dropna()
            if len(combined) < max_lag * 3:
                return {}
            result = grangercausalitytests(combined.values, maxlag=max_lag, verbose=False)
            return {
                lag: float(vals[0]["ssr_ftest"][1])  # p-value
                for lag, vals in result.items()
            }
        except Exception as e:
            logger.debug("Granger test failed: %s", e)
            return {}

    def cointegration_test(self, x: pd.Series, y: pd.Series) -> dict:
        """Johansen / ADF cointegration test."""
        try:
            from statsmodels.tsa.stattools import coint
            combined = pd.concat([x, y], axis=1).dropna()
            score, pvalue, _ = coint(combined.iloc[:, 0], combined.iloc[:, 1])
            return {"score": float(score), "p_value": float(pvalue), "cointegrated": pvalue < 0.05}
        except Exception as e:
            logger.debug("Cointegration test failed: %s", e)
            return {}

    def get_correlation_report(
        self,
        crypto: pd.Series,
        variables: dict[str, pd.Series],
        date_range: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        results = []
        for name, var in variables.items():
            for window in [30, 90]:
                rolling = self.correlate(crypto, var, window=window)
                latest = rolling.dropna().iloc[-1] if not rolling.dropna().empty else float("nan")
                results.append({
                    "variable": name,
                    "window": window,
                    "correlation": round(latest, 4),
                    "method": "pearson",
                })
        return pd.DataFrame(results)
