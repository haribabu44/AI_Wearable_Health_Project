"""
models/ml_model.py
------------------
MLModels is loaded ONCE as an app-level singleton via FastAPI lifespan.
All models are sklearn Pipelines (scaler + model) — no manual scaling needed.

Improvements over v1:
  - Singleton pattern — no re-loading on each request
  - Pipeline wrapping — scaler is baked in
  - Graceful degradation — missing models log a warning, don't crash
  - Human-readable interpretation strings in disease results
  - Uses full DISEASE_FEATURES (age, bmi, sex) for better predictions
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BASE_FEATURES    = ["steps", "temperature", "spo2", "glucose", "bp"]
HR_FEATURES      = ["steps", "temperature", "spo2", "glucose", "bp", "age", "bmi", "sex"]
ANOMALY_FEATURES = ["steps", "temperature", "spo2", "glucose", "bp", "age", "bmi"]
DISEASE_FEATURES = ["steps", "temperature", "spo2", "glucose", "bp",
                    "heart_rate", "bmi", "age"]

DISEASE_MODEL_FILES: dict[str, str] = {
    "Diabetes":        "disease_diabetes_model.pkl",
    "Hypertension":    "disease_hypertension_model.pkl",
    "Hypoxia":         "disease_hypoxia_model.pkl",
    "Fever/Infection": "disease_fever_model.pkl",
    "Cardiac Risk":    "disease_cardiac_model.pkl",
}

_RISK_THRESHOLDS = {
    "Diabetes":        (0.40, 0.65),  # (moderate_risk, high_risk)
    "Hypertension":    (0.40, 0.65),
    "Hypoxia":         (0.35, 0.60),
    "Fever/Infection": (0.35, 0.60),
    "Cardiac Risk":    (0.35, 0.55),
}


def _interpret(disease: str, prob: float) -> str:
    low, high = _RISK_THRESHOLDS.get(disease, (0.40, 0.65))
    if prob < low:
        return "Low risk — vitals within normal range"
    if prob < high:
        return "Moderate risk — monitor closely and consult a physician"
    return "High risk — prompt medical evaluation recommended"


class MLModels:
    """Singleton holder for all trained models."""

    _instance: "MLModels | None" = None

    def __new__(cls) -> "MLModels":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self) -> None:
        """Load all .pkl files from the models directory."""
        if self._loaded:
            return

        models_dir = Path(settings.models_dir)
        self.heart_model   = self._load(models_dir / "heart_model.pkl")
        self.anomaly_model = self._load(models_dir / "anomaly_model.pkl")

        self.disease_models: dict = {}
        for name, filename in DISEASE_MODEL_FILES.items():
            model = self._load(models_dir / filename)
            if model is not None:
                self.disease_models[name] = model

        self._loaded = True
        logger.info(
            "MLModels loaded: heart=%s anomaly=%s diseases=%d",
            self.heart_model is not None,
            self.anomaly_model is not None,
            len(self.disease_models),
        )

    @staticmethod
    def _load(path: Path):
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as exc:
                logger.error("Failed to load %s: %s", path, exc)
        else:
            logger.warning("Model not found: %s", path)
        return None

    # ── Inference helpers ────────────────────────────────────────────────

    def _safe_val(self, data: dict, key: str, default: float) -> float:
        v = data.get(key)
        return float(v) if v is not None else default

    def predict_heart_rate(self, data: dict) -> float:
        if self.heart_model is None:
            return 72.0
        row = pd.DataFrame([{
            "steps":       data["steps"],
            "temperature": data["temperature"],
            "spo2":        data["spo2"],
            "glucose":     data["glucose"],
            "bp":          data["bp"],
            "age":         self._safe_val(data, "age", 35),
            "bmi":         self._safe_val(data, "bmi", 22.0),
            "sex":         self._safe_val(data, "sex", 0),
        }])
        return round(float(self.heart_model.predict(row)[0]), 1)

    def detect_anomaly(self, data: dict) -> str:
        if self.anomaly_model is None:
            return "Unknown"
        row = pd.DataFrame([{
            "steps":       data["steps"],
            "temperature": data["temperature"],
            "spo2":        data["spo2"],
            "glucose":     data["glucose"],
            "bp":          data["bp"],
            "age":         self._safe_val(data, "age", 35),
            "bmi":         self._safe_val(data, "bmi", 22.0),
        }])
        result = self.anomaly_model.predict(row)[0]
        return "Anomaly Detected" if result == -1 else "Normal"

    def predict_diseases(self, data: dict, predicted_hr: float) -> dict:
        results = {}
        row = pd.DataFrame([{
            "steps":       data["steps"],
            "temperature": data["temperature"],
            "spo2":        data["spo2"],
            "glucose":     data["glucose"],
            "bp":          data["bp"],
            "heart_rate":  self._safe_val(data, "heart_rate", predicted_hr),
            "bmi":         self._safe_val(data, "bmi", 22.0),
            "age":         self._safe_val(data, "age", 35),
        }])
        for name, model in self.disease_models.items():
            try:
                prob = float(model.predict_proba(row)[0][1])
                low, _ = _RISK_THRESHOLDS.get(name, (0.40, 0.65))
                results[name] = {
                    "risk":           prob >= low,
                    "probability":    round(prob, 3),
                    "interpretation": _interpret(name, prob),
                }
            except Exception as exc:
                logger.error("Disease predict failed for %s: %s", name, exc)
        return results
