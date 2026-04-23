"""
routes/predict.py
-----------------
Two endpoints:
  POST /api/predict  — run full inference pipeline, persist, return result
  GET  /api/history  — paginated record history
  GET  /api/stats    — aggregate statistics for the dashboard summary
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.db.database import get_record_count, get_records, insert_record
from app.models.ml_model import MLModels
from app.models.schema import HealthInput, HistoryResponse, PredictionResponse
from app.services.data_processing import preprocess
from app.services.health_logic import calculate_health_score, generate_alerts

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predictions"])


def get_ml_models(request: Request) -> MLModels:
    """Dependency — retrieves the singleton from app state."""
    return request.app.state.ml_models


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run health prediction pipeline",
    status_code=status.HTTP_200_OK,
)
def predict(
    payload: HealthInput,
    ml: Annotated[MLModels, Depends(get_ml_models)],
) -> PredictionResponse:
    """
    Accepts wearable vitals and patient metadata.
    Returns heart-rate prediction, anomaly status, health score,
    clinical alerts, and disease risk probabilities.
    """
    try:
        # 1. Preprocess — fill defaults, clip ranges
        data = preprocess(payload.model_dump())

        # 2. ML inference
        predicted_hr  = ml.predict_heart_rate(data)
        anomaly_status = ml.detect_anomaly(data)
        disease_preds  = ml.predict_diseases(data, predicted_hr)

        # 3. Rule-based scoring
        health_score = calculate_health_score(data)
        alerts       = generate_alerts(data, predicted_hr)

        # 4. Persist
        record_id = insert_record({
            **data,
            "predicted_heart_rate": predicted_hr,
            "anomaly_status":       anomaly_status,
            "health_score":         health_score,
            "alerts":               alerts,
            "disease_predictions":  disease_preds,
        })

        logger.info(
            "Prediction complete — id=%d score=%d anomaly=%s",
            record_id, health_score, anomaly_status,
        )

        return PredictionResponse(
            record_id=record_id,
            patient_name=data.get("patient_name"),
            predicted_heart_rate=predicted_hr,
            anomaly_status=anomaly_status,
            health_score=health_score,
            alerts=alerts,
            disease_predictions=disease_preds,
            inputs={k: v for k, v in data.items() if k != "patient_name"},
        )

    except Exception as exc:
        logger.exception("Prediction pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Paginated record history",
)
def history(
    limit:  Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)]         = 0,
) -> HistoryResponse:
    records = get_records(limit=limit, offset=offset)
    total   = get_record_count()
    return HistoryResponse(total=total, records=records)


@router.get("/health", summary="API health check")
def api_health(request: Request) -> dict:
    ml: MLModels = request.app.state.ml_models
    return {
        "status": "ok",
        "models_loaded": ml._loaded,
        "disease_models": list(ml.disease_models.keys()),
    }
