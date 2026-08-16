"""Vendor-neutral Google Health records used by Blacki summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class HealthWorkout:
    """A completed workout with only summary fields needed by Blacki."""

    type: str
    minutes: int
    calories_kcal: float | None = None
    active_zone_minutes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the normalized workout without provider identifiers."""
        result: dict[str, Any] = {"type": self.type, "minutes": self.minutes}
        if self.calories_kcal is not None:
            result["calories_kcal"] = self.calories_kcal
        if self.active_zone_minutes is not None:
            result["active_zone_minutes"] = self.active_zone_minutes
        return result


@dataclass(frozen=True, slots=True)
class HealthSleep:
    """A completed sleep session with optional stage summaries."""

    minutes: int
    start_time: str | None = None
    end_time: str | None = None
    stages: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize a sleep record while omitting unavailable fields."""
        result: dict[str, Any] = {"minutes": self.minutes}
        if self.start_time is not None:
            result["start_time"] = self.start_time
        if self.end_time is not None:
            result["end_time"] = self.end_time
        if self.stages:
            result["stages"] = [dict(stage) for stage in self.stages]
        return result


@dataclass(frozen=True, slots=True)
class HealthDay:
    """Normalized daily health aggregate persisted by Blacki."""

    date: str
    steps: int | None = None
    distance_meters: float | None = None
    active_calories_kcal: float | None = None
    active_minutes: int | None = None
    active_zone_minutes: int | None = None
    resting_heart_rate_bpm: int | None = None
    weight_kg: float | None = None
    body_fat_percent: float | None = None
    heart_rate_zones: dict[str, int] = field(default_factory=dict)
    heart_rate_zone_thresholds: tuple[dict[str, Any], ...] = ()
    workouts: tuple[HealthWorkout, ...] = ()
    sleep: tuple[HealthSleep, ...] = ()
    source: str = "google_health"

    def to_dict(self) -> dict[str, Any]:
        """Serialize only normalized fields suitable for summary generation."""
        result: dict[str, Any] = {"date": self.date, "source": self.source}
        scalar_fields = {
            "steps": self.steps,
            "distance_meters": self.distance_meters,
            "active_calories_kcal": self.active_calories_kcal,
            "active_minutes": self.active_minutes,
            "active_zone_minutes": self.active_zone_minutes,
            "resting_heart_rate_bpm": self.resting_heart_rate_bpm,
            "weight_kg": self.weight_kg,
            "body_fat_percent": self.body_fat_percent,
        }
        result.update(
            {key: value for key, value in scalar_fields.items() if value is not None}
        )
        if self.heart_rate_zones:
            result["heart_rate_zones"] = dict(sorted(self.heart_rate_zones.items()))
        if self.heart_rate_zone_thresholds:
            result["heart_rate_zone_thresholds"] = [
                dict(threshold) for threshold in self.heart_rate_zone_thresholds
            ]
        if self.workouts:
            result["workouts"] = [workout.to_dict() for workout in self.workouts]
        if self.sleep:
            result["sleep"] = [sleep.to_dict() for sleep in self.sleep]
        return result
