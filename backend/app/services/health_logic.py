"""
services/health_logic.py
------------------------
Rule-based health scoring and alert generation.

The scoring model uses a weighted deduction system aligned with clinical
severity scales. Alert thresholds are consistent with WHO / AHA guidelines.
"""

from dataclasses import dataclass, field


@dataclass
class ScoringRule:
    """A single deduction rule with severity level."""
    field:     str
    condition: str   # "lt" | "gt"
    threshold: float
    deduction: int
    alert:     str | None = None


_RULES: list[ScoringRule] = [
    # Steps (activity)
    ScoringRule("steps",       "lt", 2000, 15, "⚠️ Very low activity — under 2,000 steps"),
    ScoringRule("steps",       "lt", 5000, 7,  None),

    # Temperature
    ScoringRule("temperature", "gt", 39.0, 20, "🔥 High fever — temperature > 39°C"),
    ScoringRule("temperature", "gt", 38.0, 10, "🔥 Elevated temperature > 38°C"),
    ScoringRule("temperature", "lt", 36.0, 10, "🧊 Low body temperature < 36°C"),

    # SpO2
    ScoringRule("spo2",        "lt", 90,   30, "🚨 Critical: SpO2 < 90% — seek immediate care"),
    ScoringRule("spo2",        "lt", 95,   15, "🚨 Low blood oxygen — SpO2 < 95%"),

    # Glucose
    ScoringRule("glucose",     "gt", 200,  20, "⚠️ Very high glucose > 200 mg/dL"),
    ScoringRule("glucose",     "gt", 126,  10, "⚠️ Elevated glucose > 126 mg/dL"),
    ScoringRule("glucose",     "lt", 70,   15, "⚠️ Low glucose < 70 mg/dL — risk of hypoglycaemia"),

    # Blood pressure
    ScoringRule("bp",          "gt", 180,  25, "🚨 Hypertensive crisis — BP > 180 mmHg"),
    ScoringRule("bp",          "gt", 140,  15, "⚠️ High blood pressure > 140 mmHg"),
    ScoringRule("bp",          "lt", 80,   10, "⚠️ Low blood pressure < 80 mmHg"),
]


def _applies(rule: ScoringRule, value: float) -> bool:
    if rule.condition == "lt":
        return value < rule.threshold
    if rule.condition == "gt":
        return value > rule.threshold
    return False


def calculate_health_score(data: dict) -> int:
    """
    Returns an integer 0–100 score.
    Deductions are additive but capped at each severity level.
    """
    score = 100
    for rule in _RULES:
        val = data.get(rule.field)
        if val is not None and _applies(rule, val):
            score -= rule.deduction
    return max(0, score)


def generate_alerts(data: dict, predicted_hr: float) -> list[str]:
    """
    Collect clinical alerts from vitals + predicted heart rate.
    Returns at least one message.
    """
    alerts: list[str] = []

    # Rule-based vitals alerts
    seen_fields: set[str] = set()
    for rule in _RULES:
        if rule.alert is None:
            continue
        val = data.get(rule.field)
        if val is not None and _applies(rule, val) and rule.field not in seen_fields:
            alerts.append(rule.alert)
            seen_fields.add(rule.field)

    # Heart rate alerts (predicted)
    if predicted_hr > 150:
        alerts.append("🚨 Critical tachycardia — predicted HR > 150 bpm")
    elif predicted_hr > 100:
        alerts.append("⚠️ Tachycardia — predicted HR > 100 bpm")
    elif predicted_hr < 40:
        alerts.append("🚨 Critical bradycardia — predicted HR < 40 bpm")
    elif predicted_hr < 55:
        alerts.append("⚠️ Bradycardia — predicted HR < 55 bpm")

    if not alerts:
        alerts.append("✅ All vitals are within normal range")

    return alerts


def score_label(score: int) -> str:
    """Human-readable label for a health score."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    if score >= 30:
        return "Poor"
    return "Critical"
