"""
DeepPredictor — modèle GradientBoosting pour prédire la probabilité
de hausse > 5% dans les 4h suivantes.
Se réentraîne chaque nuit via SelfCorrector.
Aucun GPU requis — scikit-learn pur.
"""
from __future__ import annotations

import logging
import pathlib
import pickle
from typing import Any

logger = logging.getLogger(__name__)

MODEL_PATH = pathlib.Path("data/deep_model.pkl")

FEATURES = [
    "rsi", "ema7_ratio", "ema25_ratio", "ema99_ratio",
    "volume_ratio", "atr_pct", "macd_hist", "obv_slope",
    "hour_utc", "day_of_week", "btc_pct_1h", "spread_pct",
]
TARGET_GAIN = 0.05  # 5% hausse dans 4h = signal positif
LOOKBACK    = 24    # 24 bougies à 1h


class DeepPredictor:
    """
    Prédit la probabilité de hausse > 5% dans les 4h.
    Utilise GradientBoostingClassifier (scikit-learn) avec validation walk-forward.
    """

    def __init__(self, model_path: str | pathlib.Path = MODEL_PATH):
        self.model_path = pathlib.Path(model_path)
        self._model  = None
        self._scaler = None
        self._trained = False
        self._load_if_exists()

    # ── Public API ────────────────────────────────────────────────────

    def predict(self, features_window: list[dict]) -> float:
        """
        Prédit la probabilité de hausse > TARGET_GAIN dans les 4h.
        features_window : liste d'au moins LOOKBACK dicts avec les FEATURES.
        Retourne float 0.0 - 1.0.
        """
        if not self._trained or self._model is None:
            return 0.5
        try:
            import numpy as np
            feat = self._extract_features(features_window)
            if feat is None:
                return 0.5
            X_scaled = self._scaler.transform([feat])
            prob = self._model.predict_proba(X_scaled)[0][1]
            return float(prob)
        except Exception as e:
            logger.debug("DeepPredictor predict error: %s", e)
            return 0.5

    def train(self, historical_data: list[dict]) -> dict[str, Any]:
        """
        Entraîne sur l'historique. Retourne les métriques AUC.
        historical_data : liste de dicts avec close + les FEATURES + datetime.
        """
        if len(historical_data) < 500:
            return {"error": f"Pas assez de données : {len(historical_data)} < 500"}

        try:
            import numpy as np
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler

            X, y = self._prepare_dataset(historical_data)
            if len(X) == 0:
                return {"error": "Aucun exemple valide dans le dataset"}

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Validation walk-forward (anti-data-leakage)
            tscv      = TimeSeriesSplit(n_splits=5)
            auc_scores: list[float] = []
            for train_idx, val_idx in tscv.split(X_scaled):
                clf = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
                )
                clf.fit(X_scaled[train_idx], y[train_idx])
                prob = clf.predict_proba(X_scaled[val_idx])[:, 1]
                try:
                    auc_scores.append(roc_auc_score(y[val_idx], prob))
                except ValueError:
                    pass  # skip if only one class in fold

            # Entraînement final sur tout le dataset
            self._model = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
            )
            self._model.fit(X_scaled, y)
            self._scaler  = scaler
            self._trained = True

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump({"model": self._model, "scaler": self._scaler}, f)

            auc_mean = float(np.mean(auc_scores)) if auc_scores else 0.0
            auc_std  = float(np.std(auc_scores))  if auc_scores else 0.0
            logger.info(
                "DeepPredictor entraîné : AUC=%.3f±%.3f sur %d exemples",
                auc_mean, auc_std, len(X),
            )
            return {
                "auc_mean":  auc_mean,
                "auc_std":   auc_std,
                "n_samples": int(len(X)),
                "n_features": len(FEATURES),
            }

        except Exception as e:
            logger.error("DeepPredictor train error: %s", e)
            return {"error": str(e)}

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── Data preparation ──────────────────────────────────────────────

    def _prepare_dataset(self, data: list[dict]):
        import numpy as np
        X, y = [], []
        for i in range(LOOKBACK, len(data) - 4):
            window = data[i - LOOKBACK : i]
            feat   = self._extract_features(window)
            if feat is None:
                continue
            # Label : hausse >= TARGET_GAIN dans les 4 prochaines heures
            entry_close  = data[i].get("close", 0)
            future_close = data[i + 3].get("close", 0)
            if entry_close <= 0:
                continue
            future_gain = (future_close - entry_close) / entry_close
            label = 1 if future_gain >= TARGET_GAIN else 0
            X.append(feat)
            y.append(label)
        return np.array(X), np.array(y)

    def _extract_features(self, window: list[dict]) -> list[float] | None:
        if len(window) < LOOKBACK:
            return None
        last = window[-1]
        return [
            last.get("rsi", 50) / 100,
            last.get("ema7_ratio", 1.0),
            last.get("ema25_ratio", 1.0),
            last.get("ema99_ratio", 1.0),
            last.get("volume_ratio", 1.0),
            last.get("atr_pct", 0.02),
            last.get("macd_hist", 0.0),
            last.get("obv_slope", 0.0),
            last.get("hour_utc", 12) / 24,
            last.get("day_of_week", 3) / 7,
            last.get("btc_pct_1h", 0.0),
            last.get("spread_pct", 0.001),
        ]

    # ── Persistence ───────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
            self._model   = saved.get("model")
            self._scaler  = saved.get("scaler")
            self._trained = self._model is not None
            if self._trained:
                logger.info("DeepPredictor: modèle chargé depuis %s", self.model_path)
        except Exception as e:
            logger.warning("DeepPredictor: impossible de charger le modèle — %s", e)
