"""
services/data_processing.py
----------------------------
Input preprocessing pipeline — now actually used in the predict route.

Responsibilities:
  - Fill optional fields with population-median defaults
  - Clip values to physiologically plausible ranges (belt-and-suspenders
    on top of Pydantic validation)
  - Return a clean dict ready for ML inference
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Population-median defaults (NHANES 2017-18)
_DEFAULTS: dict[str, float] = {
    "heart_rate": 72.0,
    "bmi":        27.5,
    "age":        40.0,
    "sex":        0.0,
}

_CLIP_RANGES: dict[str, tuple[float, float]] = {
    "steps":       (0,     50_000),
    "temperature": (34.0,  43.0),
    "spo2":        (50.0,  100.0),
    "glucose":     (20.0,  600.0),
    "bp":          (50.0,  250.0),
    "heart_rate":  (20.0,  250.0),
    "bmi":         (10.0,  70.0),
    "age":         (1.0,   120.0),
}


def preprocess(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Fill defaults for optional fields and clip values.
    Returns a new dict; does not mutate the input.
    """
    data = dict(raw)

    # Fill missing optional fields
    for field, default in _DEFAULTS.items():
        if data.get(field) is None:
            logger.debug("Filling default for %s = %s", field, default)
            data[field] = default

    # Clip to valid ranges
    for field, (lo, hi) in _CLIP_RANGES.items():
        if field in data and data[field] is not None:
            clipped = max(lo, min(hi, float(data[field])))
            if clipped != data[field]:
                logger.warning(
                    "Clipped %s: %.2f → %.2f", field, data[field], clipped
                )
            data[field] = clipped

    return data
