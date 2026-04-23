"""
models/schema.py
----------------
Request and response models with field-level validation.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class HealthInput(BaseModel):
    """Wearable sensor + patient input — all fields validated at the boundary."""

    patient_name: Optional[str] = Field(default=None, max_length=100,
                                         description="Patient display name")
    steps:       float = Field(..., ge=0,    le=50_000, description="Daily step count")
    temperature: float = Field(..., ge=34.0, le=43.0,  description="Body temp °C")
    spo2:        float = Field(..., ge=50.0, le=100.0, description="Blood oxygen %")
    glucose:     float = Field(..., ge=20.0, le=600.0, description="Blood glucose mg/dL")
    bp:          float = Field(..., ge=50.0, le=250.0, description="Systolic BP mmHg")
    heart_rate:  Optional[float] = Field(default=None, ge=20.0, le=250.0)
    bmi:         Optional[float] = Field(default=None, ge=10.0, le=70.0)
    age:         Optional[int]   = Field(default=None, ge=1,    le=120)
    sex:         Optional[int]   = Field(default=None, ge=0,    le=1,
                                          description="0=female 1=male")

    @field_validator("spo2")
    @classmethod
    def spo2_range(cls, v: float) -> float:
        if v > 100:
            raise ValueError("SpO2 cannot exceed 100%")
        return round(v, 1)

    @field_validator("temperature")
    @classmethod
    def temp_range(cls, v: float) -> float:
        return round(v, 2)

    model_config = {"json_schema_extra": {
        "example": {
            "patient_name": "Arjun Sharma",
            "steps": 6800, "temperature": 36.7, "spo2": 98.0,
            "glucose": 95.0, "bp": 118.0, "heart_rate": 72.0,
            "bmi": 24.5, "age": 34, "sex": 1,
        }
    }}


class DiseaseResult(BaseModel):
    risk: bool
    probability: float
    interpretation: str


class PredictionResponse(BaseModel):
    record_id:            int
    patient_name:         Optional[str]
    predicted_heart_rate: float
    anomaly_status:       str
    health_score:         int
    alerts:               list[str]
    disease_predictions:  dict[str, DiseaseResult]
    inputs:               dict


class HistoryResponse(BaseModel):
    total:   int
    records: list[dict]
