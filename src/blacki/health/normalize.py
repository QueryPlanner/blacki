"""Normalize Google Health data points into daily Blacki records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .models import HealthDay, HealthSleep, HealthWorkout

DATA_COMPONENTS = {
    "steps": "steps",
    "distance": "distance",
    "active-energy-burned": "activeEnergyBurned",
    "active-minutes": "activeMinutes",
    "active-zone-minutes": "activeZoneMinutes",
    "exercise": "exercise",
    "sleep": "sleep",
    "daily-resting-heart-rate": "dailyRestingHeartRate",
    "daily-heart-rate-zones": "dailyHeartRateZones",
    "time-in-heart-rate-zone": "timeInHeartRateZone",
    "weight": "weight",
    "body-fat": "bodyFat",
}

_DURATION_PATTERN = re.compile(r"^(?P<seconds>-?\d+(?:\.\d+)?)s$")


@dataclass
class _DayBuilder:
    """Mutable accumulator used only during one in-memory sync."""

    steps: int = 0
    has_steps: bool = False
    distance_meters: float = 0.0
    has_distance: bool = False
    active_calories_kcal: float = 0.0
    has_active_calories: bool = False
    active_minutes: int = 0
    has_active_minutes: bool = False
    active_zone_minutes: int = 0
    has_active_zone_minutes: bool = False
    resting_heart_rate_bpm: int | None = None
    weight_kg: float | None = None
    body_fat_percent: float | None = None
    heart_rate_zones: dict[str, int] = field(default_factory=dict)
    heart_rate_zone_thresholds: list[dict[str, Any]] = field(default_factory=list)
    workouts: list[HealthWorkout] = field(default_factory=list)
    workout_keys: set[str] = field(default_factory=set)
    sleep: list[HealthSleep] = field(default_factory=list)
    sleep_keys: set[str] = field(default_factory=set)
    latest_weight_time: str = ""
    latest_body_fat_time: str = ""

    def to_day(self, day: str) -> HealthDay:
        """Build an immutable normalized day."""
        return HealthDay(
            date=day,
            steps=self.steps if self.has_steps else None,
            distance_meters=round(self.distance_meters, 3)
            if self.has_distance
            else None,
            active_calories_kcal=round(self.active_calories_kcal, 2)
            if self.has_active_calories
            else None,
            active_minutes=self.active_minutes if self.has_active_minutes else None,
            active_zone_minutes=(
                self.active_zone_minutes if self.has_active_zone_minutes else None
            ),
            resting_heart_rate_bpm=self.resting_heart_rate_bpm,
            weight_kg=round(self.weight_kg, 3) if self.weight_kg is not None else None,
            body_fat_percent=(
                round(self.body_fat_percent, 2)
                if self.body_fat_percent is not None
                else None
            ),
            heart_rate_zones=self.heart_rate_zones,
            heart_rate_zone_thresholds=tuple(self.heart_rate_zone_thresholds),
            workouts=tuple(self.workouts),
            sleep=tuple(self.sleep),
        )


def normalize_data_points(
    data_points_by_type: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[HealthDay]:
    """Normalize API data points and omit records with no usable date."""
    builders: dict[str, _DayBuilder] = defaultdict(_DayBuilder)
    for data_type, points in data_points_by_type.items():
        for point in points:
            component_name = DATA_COMPONENTS.get(data_type)
            if component_name is None:
                continue
            component = point.get(component_name)
            if not isinstance(component, Mapping):
                continue
            day = _date_for_component(component, data_type)
            if day is None:
                continue
            builder = builders[day]
            _apply_component(builder, component, data_type, point)

    return [builders[day].to_day(day) for day in sorted(builders)]


def _apply_component(
    builder: _DayBuilder,
    component: Mapping[str, Any],
    data_type: str,
    point: Mapping[str, Any],
) -> None:
    if data_type == "steps":
        value = _number(component.get("count"))
        if value is not None:
            builder.steps += max(0, int(value))
            builder.has_steps = True
    elif data_type == "distance":
        value = _number(component.get("millimeters"))
        if value is not None:
            builder.distance_meters += max(0.0, value / 1000)
            builder.has_distance = True
    elif data_type == "active-energy-burned":
        value = _number(component.get("kcal"))
        if value is not None:
            builder.active_calories_kcal += max(0.0, value)
            builder.has_active_calories = True
    elif data_type == "active-minutes":
        records = component.get("activeMinutesByActivityLevel")
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            total = sum(
                max(0, int(value))
                for record in records
                if isinstance(record, Mapping)
                for value in [_number(record.get("activeMinutes"))]
                if value is not None
            )
            builder.active_minutes += total
            builder.has_active_minutes = True
    elif data_type == "active-zone-minutes":
        value = _number(component.get("activeZoneMinutes"))
        if value is not None:
            builder.active_zone_minutes += max(0, int(value))
            builder.has_active_zone_minutes = True
    elif data_type == "time-in-heart-rate-zone":
        zone = component.get("heartRateZoneType")
        minutes = _interval_minutes(component.get("interval"))
        if isinstance(zone, str) and minutes is not None:
            builder.heart_rate_zones[zone] = builder.heart_rate_zones.get(
                zone, 0
            ) + max(0, minutes)
    elif data_type == "daily-resting-heart-rate":
        value = _number(component.get("beatsPerMinute"))
        if value is not None:
            builder.resting_heart_rate_bpm = max(0, int(value))
    elif data_type == "daily-heart-rate-zones":
        _add_zone_thresholds(builder, component)
    elif data_type == "exercise":
        _add_workout(builder, component, point)
    elif data_type == "sleep":
        _add_sleep(builder, component, point)
    elif data_type == "weight":
        _set_latest_measurement(
            builder,
            point,
            component,
            value=_number(component.get("weightGrams")),
            field_name="weight_kg",
            scale=1000,
        )
    elif data_type == "body-fat":
        _set_latest_measurement(
            builder,
            point,
            component,
            value=_number(component.get("percentage")),
            field_name="body_fat_percent",
            scale=1,
        )


def _add_workout(
    builder: _DayBuilder,
    component: Mapping[str, Any],
    point: Mapping[str, Any],
) -> None:
    key = _record_key(point)
    if key in builder.workout_keys:
        return
    interval = component.get("interval")
    if not isinstance(interval, Mapping):
        return
    duration = _duration_seconds(component.get("activeDuration"))
    if duration is None:
        duration = _interval_seconds(interval)
    if duration is None:
        return
    exercise_type = component.get("displayName") or component.get("exerciseType")
    if not isinstance(exercise_type, str) or not exercise_type.strip():
        exercise_type = "Workout"
    metrics = component.get("metricsSummary")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    calories = _number(metrics.get("caloriesKcal"))
    active_zone_minutes = _number(metrics.get("activeZoneMinutes"))
    builder.workouts.append(
        HealthWorkout(
            type=exercise_type.strip(),
            minutes=max(0, int(round(duration / 60))),
            calories_kcal=round(max(0.0, calories), 2)
            if calories is not None
            else None,
            active_zone_minutes=(
                max(0, int(active_zone_minutes))
                if active_zone_minutes is not None
                else None
            ),
        )
    )
    builder.workout_keys.add(key)


def _add_sleep(
    builder: _DayBuilder,
    component: Mapping[str, Any],
    point: Mapping[str, Any],
) -> None:
    key = _record_key(point)
    if key in builder.sleep_keys:
        return
    interval = component.get("interval")
    if not isinstance(interval, Mapping):
        return
    minutes = _sleep_minutes(component, interval)
    if minutes is None:
        return
    stages = _sleep_stages(component)
    start_time = _string_value(interval.get("startTime"))
    end_time = _string_value(interval.get("endTime"))
    builder.sleep.append(
        HealthSleep(
            minutes=max(0, int(minutes)),
            start_time=start_time,
            end_time=end_time,
            stages=tuple(stages),
        )
    )
    builder.sleep_keys.add(key)


def _sleep_minutes(
    component: Mapping[str, Any], interval: Mapping[str, Any]
) -> int | None:
    summary = component.get("summary")
    if isinstance(summary, Mapping):
        minutes = _number(summary.get("minutesAsleep"))
        if minutes is not None:
            return int(minutes)
    seconds = _interval_seconds(interval)
    return int(round(seconds / 60)) if seconds is not None else None


def _sleep_stages(component: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = component.get("summary")
    raw_stages = summary.get("stagesSummary") if isinstance(summary, Mapping) else None
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        return []
    stages: list[dict[str, Any]] = []
    for stage in raw_stages:
        if not isinstance(stage, Mapping):
            continue
        stage_type = stage.get("type")
        minutes = _number(stage.get("minutes"))
        if isinstance(stage_type, str) and minutes is not None:
            stages.append({"type": stage_type, "minutes": max(0, int(minutes))})
    return stages


def _add_zone_thresholds(builder: _DayBuilder, component: Mapping[str, Any]) -> None:
    zones = component.get("heartRateZones")
    if not isinstance(zones, Sequence) or isinstance(zones, (str, bytes)):
        return
    for zone in zones:
        if not isinstance(zone, Mapping):
            continue
        zone_type = zone.get("heartRateZoneType")
        minimum = _number(zone.get("minBeatsPerMinute"))
        maximum = _number(zone.get("maxBeatsPerMinute"))
        if isinstance(zone_type, str) and minimum is not None and maximum is not None:
            builder.heart_rate_zone_thresholds.append(
                {
                    "type": zone_type,
                    "min_bpm": max(0, int(minimum)),
                    "max_bpm": max(0, int(maximum)),
                }
            )


def _set_latest_measurement(
    builder: _DayBuilder,
    point: Mapping[str, Any],
    component: Mapping[str, Any],
    *,
    value: float | None,
    field_name: str,
    scale: float,
) -> None:
    if value is None:
        return
    sample_time = component.get("sampleTime")
    timestamp = ""
    if isinstance(sample_time, Mapping):
        timestamp = _string_value(sample_time.get("physicalTime")) or ""
    if not timestamp:
        timestamp = _string_value(point.get("name")) or ""
    latest_field = (
        "latest_weight_time" if field_name == "weight_kg" else "latest_body_fat_time"
    )
    previous = getattr(builder, latest_field)
    if previous and timestamp < previous:
        return
    setattr(builder, field_name, max(0.0, value / scale))
    setattr(builder, latest_field, timestamp)


def _date_for_component(component: Mapping[str, Any], data_type: str) -> str | None:
    if data_type in {"daily-resting-heart-rate", "daily-heart-rate-zones"}:
        direct = _date_from_value(component.get("date"))
        if direct is not None:
            return direct

    interval = component.get("interval")
    if isinstance(interval, Mapping):
        preferred = (
            ("civilEndTime", "endTime", "civilStartTime", "startTime")
            if data_type == "sleep"
            else (
                "civilStartTime",
                "startTime",
                "civilEndTime",
                "endTime",
            )
        )
        for key in preferred:
            value = interval.get(key)
            parsed = _date_from_value(value)
            if parsed is not None:
                return parsed

    sample_time = component.get("sampleTime")
    if isinstance(sample_time, Mapping):
        parsed = _date_from_value(sample_time.get("civilTime"))
        if parsed is not None:
            return parsed
        parsed = _date_from_value(sample_time.get("physicalTime"))
        if parsed is not None:
            return parsed
    return None


def _date_from_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        year = _number(value.get("year"))
        month = _number(value.get("month"))
        day = _number(value.get("day"))
        if year is not None and month is not None and day is not None:
            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                return None
        date_value = value.get("date")
        if isinstance(date_value, Mapping):
            year = _number(date_value.get("year"))
            month = _number(date_value.get("month"))
            day = _number(date_value.get("day"))
            if year is not None and month is not None and day is not None:
                try:
                    return date(int(year), int(month), int(day)).isoformat()
                except ValueError:
                    return None
        return None
    if isinstance(value, str):
        try:
            return (
                datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
            )
        except ValueError:
            return None
    return None


def _interval_minutes(value: Any) -> int | None:
    seconds = _interval_seconds(value)
    return int(round(seconds / 60)) if seconds is not None else None


def _interval_seconds(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    start = value.get("startTime")
    end = value.get("endTime")
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    seconds = (end_dt - start_dt).total_seconds()
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _duration_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    match = _DURATION_PATTERN.fullmatch(value)
    if match is None:
        return None
    seconds = float(match.group("seconds"))
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _string_value(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _record_key(point: Mapping[str, Any]) -> str:
    name = point.get("name")
    if not isinstance(name, str) or not name:
        name = json.dumps(point, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(name.encode("utf-8")).hexdigest()
