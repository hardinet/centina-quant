from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RegimeLabel = Literal["Bull", "Bear", "Accumulation", "Distribution", "Volatile", "Calm"]


class RegimeCorrelator:
    """
    Identifies market regimes via HMM or k-means and correlates with external variables.
    """

    def detect_regimes(
        self,
        prices: pd.Series,
        method: Literal["hmm", "kmeans"] = "kmeans",
        n_regimes: int = 4,
    ) -> pd.Series:
        returns = prices.pct_change().dropna()
        features = pd.DataFrame({
            "return":  returns,
            "vol":     returns.rolling(20).std(),
            "momentum": returns.rolling(10).sum(),
        }).dropna()

        if method == "hmm":
            return self._detect_hmm(features, n_regimes)
        return self._detect_kmeans(features, n_regimes)

    def get_regime_probabilities(self, prices: pd.Series, as_of_date: str) -> dict[str, float]:
        regimes = self.detect_regimes(prices)
        if as_of_date not in regimes.index.strftime("%Y-%m-%d"):
            return {"Bull": 0.25, "Bear": 0.25, "Volatile": 0.25, "Calm": 0.25}
        return {str(int(regimes.iloc[-1])): 1.0}

    def get_regime_transition_matrix(self, prices: pd.Series) -> pd.DataFrame:
        regimes = self.detect_regimes(prices)
        regimes = regimes.dropna().astype(int)
        states = sorted(regimes.unique())
        matrix = pd.DataFrame(0, index=states, columns=states)
        for i in range(len(regimes) - 1):
            matrix.loc[regimes.iloc[i], regimes.iloc[i+1]] += 1
        row_sums = matrix.sum(axis=1)
        return matrix.div(row_sums + 1e-10, axis=0)

    def predict_next_regime(self, current_regime: int, transition_matrix: pd.DataFrame) -> int:
        if current_regime not in transition_matrix.index:
            return current_regime
        row = transition_matrix.loc[current_regime]
        return int(row.idxmax())

    def correlate_regime_with_macro(
        self,
        regimes: pd.Series,
        macro: pd.DataFrame,
    ) -> pd.DataFrame:
        results = []
        for col in macro.columns:
            macro_aligned = macro[col].reindex(regimes.index).ffill()
            for r in regimes.unique():
                mask = regimes == r
                mean_macro = float(macro_aligned[mask].mean())
                results.append({"regime": r, "variable": col, "mean_value": mean_macro})
        return pd.DataFrame(results)

    # ── Private ───────────────────────────────────────────────────────

    def _detect_kmeans(self, features: pd.DataFrame, n: int) -> pd.Series:
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(features.fillna(0))
            labels = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(X)
            return pd.Series(labels, index=features.index)
        except Exception as e:
            logger.debug("KMeans regime detection failed: %s", e)
            return pd.Series(0, index=features.index)

    def _detect_hmm(self, features: pd.DataFrame, n: int) -> pd.Series:
        try:
            from hmmlearn import hmm
            X = features.fillna(0).values
            model = hmm.GaussianHMM(n_components=n, covariance_type="diag",
                                    n_iter=100, random_state=42)
            model.fit(X)
            labels = model.predict(X)
            return pd.Series(labels, index=features.index)
        except Exception as e:
            logger.debug("HMM regime detection failed: %s", e)
            return self._detect_kmeans(features, n)
