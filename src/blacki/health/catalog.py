"""Google Health ingestion contracts, verified against the provider catalog.

Source: https://developers.google.com/health/data-types (2026-11-05).
Keep unsupported types explicit so coverage cannot silently claim completeness.
"""

from dataclasses import dataclass

SCOPE_PREFIX = "https://www.googleapis.com/auth/googlehealth."


@dataclass(frozen=True)
class HealthStream:
    """One provider representation and its required read permission."""

    key: str
    scope: str
    method: str = "list"
    version: str = "v4"


HEALTH_STREAMS = (
    HealthStream("active-energy-burned", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("active-minutes", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("active-zone-minutes", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("activity-level", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("altitude", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream(
        "blood-glucose", "health_metrics_and_measurements.readonly", "list", "v4"
    ),
    HealthStream("body-fat", "health_metrics_and_measurements.readonly", "list", "v4"),
    HealthStream(
        "calories-in-heart-rate-zone", "activity_and_fitness.readonly", "rollUp", "v4"
    ),
    HealthStream(
        "core-body-temperature",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream(
        "daily-heart-rate-variability",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream(
        "daily-heart-rate-zones",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream(
        "daily-oxygen-saturation",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream(
        "daily-respiratory-rate",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream(
        "daily-resting-heart-rate",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream(
        "daily-sleep-temperature-derivations",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream("daily-vo2-max", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("distance", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("electrocardiogram", "ecg.readonly", "list", "v4"),
    HealthStream("exercise", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("floors", "activity_and_fitness.readonly", "reconcile", "v4"),
    HealthStream("food", "nutrition.readonly", "list", "v4"),
    HealthStream("food-measurement-unit", "nutrition.readonly", "list", "v4"),
    HealthStream(
        "heart-rate", "health_metrics_and_measurements.readonly", "list", "v4"
    ),
    HealthStream(
        "heart-rate-variability",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream("height", "health_metrics_and_measurements.readonly", "list", "v4"),
    HealthStream("hydration-log", "nutrition.readonly", "list", "v4"),
    HealthStream("irregular-rhythm-notification", "irn.readonly", "list", "v4"),
    HealthStream("nutrition-log", "nutrition.readonly", "list", "v4"),
    HealthStream(
        "oxygen-saturation", "health_metrics_and_measurements.readonly", "list", "v4"
    ),
    HealthStream(
        "respiratory-rate-sleep-summary",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4",
    ),
    HealthStream("run-vo2-max", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("sedentary-period", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream(
        "skin-temperature-sensors",
        "health_metrics_and_measurements.readonly",
        "list",
        "v4beta",
    ),
    HealthStream("sleep", "sleep.readonly", "list", "v4"),
    HealthStream("steps", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("swim-lengths-data", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream(
        "time-in-heart-rate-zone", "activity_and_fitness.readonly", "list", "v4"
    ),
    HealthStream("total-calories", "activity_and_fitness.readonly", "rollUp", "v4"),
    HealthStream("vo2-max", "activity_and_fitness.readonly", "list", "v4"),
    HealthStream("weight", "health_metrics_and_measurements.readonly", "list", "v4"),
    HealthStream("profile", "profile.readonly", "resource"),
    HealthStream("settings", "settings.readonly", "resource"),
    HealthStream("irnProfile", "irn.readonly", "resource"),
    HealthStream("pairedDevices", "settings.readonly", "devices"),
)

UNREADABLE_DATA_TYPES = ("menstrual-period", "moods", "ovulation-test", "symptoms")
