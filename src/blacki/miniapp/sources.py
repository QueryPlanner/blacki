"""Named, owner-scoped data sources the Mini App pages can read.

Pages name a source with ``Query("health.sleep", {days: 14})``; the compiler
validates the parameters against ``SOURCES`` and the server resolves the
rows for the authenticated user only. Health data is read from the local
Google Health archive with a read-only SQLite connection, mirroring
``blacki.health.query``. Nothing here writes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from blacki.container import AppContainer
from blacki.health.config import (
    GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION,
    health_user_id_for_telegram_user,
)
from blacki.utils.timezone import get_app_timezone, now_utc

logger = logging.getLogger(__name__)

MAX_ROWS = 400


@dataclass(frozen=True)
class MiniAppUser:
    """The verified Telegram user behind a Mini App request."""

    telegram_user_id: int

    @property
    def chat_user_id(self) -> str:
        """The private-chat identity used by meals, workouts and pages."""
        return f"telegram-chat-{self.telegram_user_id}"

    @property
    def sender_id(self) -> str:
        """The sender identity used by durable user files."""
        return str(self.telegram_user_id)


@dataclass(frozen=True)
class SourceParam:
    kind: str  # int, enum, text
    default: Any = None
    min: int | None = None
    max: int | None = None
    options: tuple[str, ...] = ()
    required: bool = False


@dataclass(frozen=True)
class SourceSpec:
    name: str
    doc: str
    params: dict[str, SourceParam]
    fields: tuple[str, ...] | None  # None: fields are only known at runtime
    public: bool = True


HEALTH_METRICS = (
    "hrv",
    "resting_hr",
    "spo2",
    "breathing",
    "vo2_max",
    "steps",
    "skin_temp",
)

SOURCES: dict[str, SourceSpec] = {
    s.name: s
    for s in (
        SourceSpec(
            "health.days",
            "One row per day: sleep, heart and steps together.",
            {"days": SourceParam("int", 7, 1, 31)},
            ("date", "hrv", "resting_hr", "asleep", "deep", "rem", "steps", "kcal_in"),
        ),
        SourceSpec(
            "health.sleep",
            "Main sleep per night with stage minutes.",
            {"days": SourceParam("int", 14, 1, 90)},
            ("date", "bed", "wake", "asleep", "deep", "rem", "light", "awake"),
        ),
        SourceSpec(
            "health.daily",
            "One daily health metric as date + value.",
            {
                "metric": SourceParam(
                    "enum", None, options=HEALTH_METRICS, required=True
                ),
                "days": SourceParam("int", 30, 1, 90),
            },
            ("date", "value"),
        ),
        SourceSpec(
            "body.weighins",
            "Weigh-ins with body fat when measured.",
            {"days": SourceParam("int", 180, 7, 365)},
            ("date", "kg", "fat"),
        ),
        SourceSpec(
            "training.sessions",
            "Workouts from the tracker and from Blacki's log.",
            {"days": SourceParam("int", 30, 1, 90)},
            ("date", "time", "name", "minutes", "kcal", "km", "source"),
        ),
        SourceSpec(
            "training.weekly",
            "Training totals per week (Monday start).",
            {"weeks": SourceParam("int", 10, 1, 26)},
            ("week", "minutes", "sessions", "kcal"),
        ),
        SourceSpec(
            "food.days",
            "Logged food totals per day (days with logs only).",
            {"days": SourceParam("int", 14, 1, 30)},
            ("date", "kcal", "protein", "carbs", "fat", "entries"),
        ),
        SourceSpec(
            "food.meals",
            "Meals from the most recent days that have logs.",
            {"days": SourceParam("int", 3, 1, 14)},
            (
                "id",
                "date",
                "time",
                "name",
                "kcal",
                "protein",
                "carbs",
                "fat",
                "meal_type",
                "source",
            ),
        ),
        SourceSpec(
            "files.list",
            "Stored files (names and sizes only).",
            {},
            ("id", "name", "kind", "mime", "size", "uploaded"),
        ),
        SourceSpec(
            "targets",
            "The user's saved daily targets as one row.",
            {},
            (
                "calories_kcal",
                "protein_g",
                "steps",
                "sleep_minutes",
                "water_ml",
                "weekly_training_minutes",
            ),
        ),
        SourceSpec(
            "tracker",
            "Rows from one of the user's saved SELECT query templates.",
            {
                "template": SourceParam("text", None, required=True),
                "limit": SourceParam("int", 50, 1, 200),
            },
            None,
        ),
        SourceSpec("cards.recent", "Cards for the feed.", {}, None, public=False),
        SourceSpec("cards.pinned", "Pinned cards.", {}, None, public=False),
    )
}


@dataclass
class SourceContext:
    """Per-request resolution context with a small cache of archive reads."""

    user: MiniAppUser
    container: AppContainer
    store: Any  # MiniAppStore; typed loosely to avoid an import cycle
    page_id: str = ""
    _archive: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _archive_status: str | None = None


Resolver = Callable[[SourceContext, dict[str, Any]], Awaitable[dict[str, Any]]]


def _today() -> date:
    return now_utc().astimezone(get_app_timezone()).date()


def _window(days: int) -> tuple[date, date]:
    end = _today()
    return end - timedelta(days=days - 1), end


def _ok(rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"status": "ok", "rows": rows[-MAX_ROWS:], **extra}


# ---------------------------------------------------------------- archive


def _read_archive(
    path: str, owner: str, account: str, streams: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Read the latest version of each record for the given streams."""
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in streams}
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    ) as conn:
        conn.execute("PRAGMA query_only=ON")
        marks = ",".join("?" for _ in streams)
        rows = conn.execute(
            f"""
            SELECT stream, payload, digest FROM google_health_archive
            WHERE owner=? AND account=? AND stream IN ({marks})
            ORDER BY last_seen ASC, rowid ASC
            """,  # noqa: S608 - placeholders only
            (owner, account, *streams),
        ).fetchall()
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for stream, payload, digest in rows:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        inner = data.get("data") or {}
        key = inner.get("name") or inner.get("exercise") or digest
        if stream == "exercise" and "tcx" not in inner:
            continue
        latest[(stream, str(key))] = inner
    for (stream, _), inner in latest.items():
        out[stream].append(inner)
    return out


async def _archive(
    ctx: SourceContext, *streams: str
) -> dict[str, list[dict[str, Any]]] | None:
    """Load archive streams once per request; None when health isn't readable."""
    missing = [s for s in streams if s not in ctx._archive]
    if missing and ctx._archive_status is None:
        storage = ctx.container.google_health_storage
        owner = health_user_id_for_telegram_user(ctx.user.chat_user_id)
        connection = await storage.get_connection(owner) if owner else None
        if connection is None:
            ctx._archive_status = "not_connected"
        elif connection.status != "connected":
            ctx._archive_status = connection.status
        elif connection.archive_consent_version < GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION:
            ctx._archive_status = "archive_consent_required"
        else:
            path = await storage.archive.database_path()
            if not path:
                ctx._archive_status = "unavailable"
            else:
                try:
                    loaded = await asyncio.to_thread(
                        _read_archive,
                        path,
                        owner or "",
                        connection.health_user_id,
                        missing,
                    )
                    ctx._archive.update(loaded)
                except sqlite3.Error:
                    logger.exception("Mini App archive read failed")
                    ctx._archive_status = "unavailable"
    if ctx._archive_status is not None:
        return None
    return {s: ctx._archive.get(s, []) for s in streams}


def _civil(d: dict[str, Any]) -> date:
    return date(int(d["year"]), int(d["month"]), int(d["day"]))


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _local(value: str, offset: str | None) -> datetime:
    seconds = int((offset or "0s").rstrip("s") or 0)
    return _ts(value) + timedelta(seconds=seconds)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


_DAILY = {
    "hrv": (
        "daily-heart-rate-variability",
        "dailyHeartRateVariability",
        "averageHeartRateVariabilityMilliseconds",
    ),
    "resting_hr": (
        "daily-resting-heart-rate",
        "dailyRestingHeartRate",
        "beatsPerMinute",
    ),
    "spo2": ("daily-oxygen-saturation", "dailyOxygenSaturation", "averagePercentage"),
    "breathing": ("daily-respiratory-rate", "dailyRespiratoryRate", "breathsPerMinute"),
    "vo2_max": ("daily-vo2-max", "dailyVo2Max", "vo2Max"),
    "skin_temp": (
        "daily-sleep-temperature-derivations",
        "dailySleepTemperatureDerivations",
        "nightlyTemperatureCelsius",
    ),
}


def _daily_values(
    records: list[dict[str, Any]], key: str, value_field: str
) -> dict[date, float]:
    out: dict[date, float] = {}
    for rec in records:
        body = rec.get(key) or {}
        try:
            day = _civil(body["date"])
        except (KeyError, TypeError, ValueError):
            continue
        value = _num(body.get(value_field))
        if value is not None:
            out[day] = round(value, 1)
    return out


def _steps_by_day(records: list[dict[str, Any]]) -> dict[date, int]:
    """Sum steps per local day per platform, keeping the larger platform total."""
    per: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        body = rec.get("steps") or {}
        interval = body.get("interval") or {}
        try:
            if "civilStartTime" in interval:
                day = _civil(interval["civilStartTime"]["date"])
            else:
                day = _local(
                    interval["startTime"], interval.get("startUtcOffset")
                ).date()
            count = int(body.get("count", 0))
        except (KeyError, TypeError, ValueError):
            continue
        platform = str((rec.get("dataSource") or {}).get("platform", "?"))
        per[day][platform] += count
    return {d: max(v.values()) for d, v in per.items()}


def _sleep_nights(records: list[dict[str, Any]]) -> dict[date, dict[str, Any]]:
    nights: dict[date, dict[str, Any]] = {}
    for rec in records:
        sleep = rec.get("sleep") or {}
        interval = sleep.get("interval") or {}
        try:
            wake = _local(interval["endTime"], interval.get("endUtcOffset"))
            bed = _local(interval["startTime"], interval.get("startUtcOffset"))
        except (KeyError, ValueError):
            continue
        stages: dict[str, float] = defaultdict(float)
        for stage in sleep.get("stages") or []:
            try:
                minutes = (
                    _ts(stage["endTime"]) - _ts(stage["startTime"])
                ).total_seconds() / 60
            except (KeyError, ValueError):
                continue
            stages[str(stage.get("type", "UNKNOWN"))] += minutes
        asleep = sum(v for k, v in stages.items() if k != "AWAKE")
        if not asleep:
            continue
        night = wake.date()
        if night in nights and nights[night]["asleep"] >= asleep:
            continue
        nights[night] = {
            "date": night.isoformat(),
            "bed": bed.strftime("%H:%M"),
            "wake": wake.strftime("%H:%M"),
            "asleep": round(asleep),
            "deep": round(stages["DEEP"]),
            "rem": round(stages["REM"]),
            "light": round(stages["LIGHT"]),
            "awake": round(stages["AWAKE"]),
        }
    return nights


def _tcx_sessions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sessions = []
    for rec in records:
        tcx = str(rec.get("tcx") or "")
        ident = re.search(r"<Id>([^<]+)</Id>", tcx)
        if not ident:
            continue
        try:
            start = datetime.fromisoformat(ident.group(1))
        except ValueError:
            continue
        sport = re.search(r'Sport="([^"]+)"', tcx)
        seconds = sum(float(x) for x in re.findall(r"<TotalTimeSeconds>([^<]+)<", tcx))
        kcal = sum(int(x) for x in re.findall(r"<Calories>(\d+)<", tcx))
        meters = sum(
            float(x)
            for x in re.findall(
                r"<DistanceMeters>([^<]+)</DistanceMeters>\s*<Calories>", tcx
            )
        )
        name = sport.group(1) if sport else "Workout"
        sessions.append(
            {
                "date": start.date().isoformat(),
                "time": start.strftime("%H:%M"),
                "name": "Workout" if name == "Other" else name,
                "minutes": round(seconds / 60),
                "kcal": kcal,
                "km": round(meters / 1000, 2),
                "source": "tracker",
            }
        )
    return sessions


# ---------------------------------------------------------------- resolvers


def _not_ready(status: str) -> dict[str, Any]:
    return {"status": status, "rows": []}


async def _health_daily(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    metric = params["metric"]
    start, end = _window(params["days"])
    if metric == "steps":
        data = await _archive(ctx, "steps")
        if data is None:
            return _not_ready(ctx._archive_status or "unavailable")
        values: dict[date, float] = {
            k: float(v) for k, v in _steps_by_day(data["steps"]).items()
        }
    else:
        stream, key, value_field = _DAILY[metric]
        data = await _archive(ctx, stream)
        if data is None:
            return _not_ready(ctx._archive_status or "unavailable")
        values = _daily_values(data[stream], key, value_field)
    rows = [
        {"date": d.isoformat(), "value": v}
        for d, v in sorted(values.items())
        if start <= d <= end
    ]
    return _ok(rows)


async def _health_sleep(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    data = await _archive(ctx, "sleep")
    if data is None:
        return _not_ready(ctx._archive_status or "unavailable")
    start, end = _window(params["days"])
    nights = _sleep_nights(data["sleep"])
    return _ok([n for d, n in sorted(nights.items()) if start <= d <= end])


async def _health_days(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    streams = (
        "daily-heart-rate-variability",
        "daily-resting-heart-rate",
        "sleep",
        "steps",
    )
    data = await _archive(ctx, *streams)
    if data is None:
        return _not_ready(ctx._archive_status or "unavailable")
    start, end = _window(params["days"])
    hrv = _daily_values(data[streams[0]], *_DAILY["hrv"][1:])
    rhr = _daily_values(data[streams[1]], *_DAILY["resting_hr"][1:])
    nights = _sleep_nights(data["sleep"])
    steps = _steps_by_day(data["steps"])
    food = {
        r["date"]: r["kcal"]
        for r in (await _food_days(ctx, {"days": params["days"]}))["rows"]
    }
    rows = []
    day = start
    while day <= end:
        night = nights.get(day) or {}
        rows.append(
            {
                "date": day.isoformat(),
                "hrv": hrv.get(day),
                "resting_hr": rhr.get(day),
                "asleep": night.get("asleep"),
                "deep": night.get("deep"),
                "rem": night.get("rem"),
                "steps": steps.get(day),
                "kcal_in": food.get(day.isoformat()),
            }
        )
        day += timedelta(days=1)
    return _ok(rows)


async def _body_weighins(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    data = await _archive(ctx, "weight", "body-fat")
    if data is None:
        return _not_ready(ctx._archive_status or "unavailable")
    start, end = _window(params["days"])
    fat: dict[str, float] = {}
    for rec in data["body-fat"]:
        body = rec.get("bodyFat") or {}
        try:
            fat[_civil(body["sampleTime"]["civilTime"]["date"]).isoformat()] = float(
                body["percentage"]
            )
        except (KeyError, TypeError, ValueError):
            continue
    rows = []
    for rec in data["weight"]:
        body = rec.get("weight") or {}
        try:
            day = _civil(body["sampleTime"]["civilTime"]["date"])
            kg = round(float(body["weightGrams"]) / 1000, 1)
        except (KeyError, TypeError, ValueError):
            continue
        if start <= day <= end:
            rows.append(
                {"date": day.isoformat(), "kg": kg, "fat": fat.get(day.isoformat())}
            )
    rows.sort(key=lambda r: str(r["date"]))
    return _ok(rows)


def _owner_clause(user: MiniAppUser) -> tuple[str, tuple[str, str]]:
    """Match the private chat and its topic threads, as calorie storage does."""
    return "(user_id = ? OR user_id LIKE ?)", (
        user.chat_user_id,
        f"{user.chat_user_id}-thread-%",
    )


async def _fetch(
    ctx: SourceContext, sql: str, params: tuple[Any, ...]
) -> list[dict[str, Any]]:
    async with ctx.container.conn.execute(sql, params) as cursor:
        columns = [c[0] for c in cursor.description or ()]
        return [
            dict(zip(columns, row, strict=False)) for row in await cursor.fetchall()
        ]


def _archive_meals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Meals from Google Health nutrition logs, read-only (no local id)."""
    meals = []
    for rec in records:
        log = rec.get("nutritionLog") or {}
        interval = log.get("interval") or {}
        try:
            civil = interval["civilStartTime"]
            day = _civil(civil["date"])
            clock = civil.get("time") or {}
        except (KeyError, TypeError, ValueError):
            continue
        nutrients = {
            str(n.get("nutrient")): _num((n.get("quantity") or {}).get("grams"))
            for n in log.get("nutrients") or []
        }
        meals.append(
            {
                "id": None,
                "date": day.isoformat(),
                "time": f"{int(clock.get('hours', 0)):02d}:{int(clock.get('minutes', 0)):02d}",  # noqa: E501
                "name": str(log.get("foodDisplayName") or "Meal"),
                "kcal": round(_num((log.get("energy") or {}).get("kcal")) or 0),
                "protein": nutrients.get("PROTEIN"),
                "carbs": nutrients.get("TOTAL_CARBOHYDRATE"),
                "fat": nutrients.get("TOTAL_FAT"),
                "meal_type": None,
                "source": "google_health",
            }
        )
    return meals


async def _all_meals(
    ctx: SourceContext, start: date, end: date
) -> list[dict[str, Any]]:
    """Local meals, plus Google Health logs for days with no local entries.

    Blacki exports its own meals to Google Health, so a day with local entries
    uses only those, which avoids counting the same meal twice.
    """
    await ctx.container.calorie_storage.initialize()
    clause, owner = _owner_clause(ctx.user)
    local = await _fetch(
        ctx,
        f"""
        SELECT id, logged_date AS date, logged_at, description AS name,
               calories AS kcal, protein_g AS protein, carbs_g AS carbs,
               fat_g AS fat, meal_type
        FROM calorie_logs WHERE {clause} AND logged_date >= ? AND logged_date <= ?
        ORDER BY logged_date, logged_at LIMIT 400
        """,  # noqa: S608 - fixed clause
        (*owner, start.isoformat(), end.isoformat()),
    )
    for row in local:
        stamp = str(row.pop("logged_at") or "")
        row["time"] = stamp[11:16] if len(stamp) >= 16 else ""
        row["source"] = "blacki"
    local_days = {row["date"] for row in local}
    data = await _archive(ctx, "nutrition-log")
    remote = [
        m
        for m in (_archive_meals(data["nutrition-log"]) if data else [])
        if start.isoformat() <= m["date"] <= end.isoformat()
        and m["date"] not in local_days
    ]
    meals = local + remote
    meals.sort(key=lambda m: (str(m["date"]), str(m["time"])))
    return meals


async def _food_days(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    start, end = _window(params["days"])
    totals: dict[str, dict[str, Any]] = {}
    for meal in await _all_meals(ctx, start, end):
        day = totals.setdefault(
            meal["date"],
            {
                "date": meal["date"],
                "kcal": 0,
                "protein": None,
                "carbs": None,
                "fat": None,
                "entries": 0,
            },
        )
        day["kcal"] += meal["kcal"] or 0
        day["entries"] += 1
        for key in ("protein", "carbs", "fat"):
            if meal[key] is not None:
                day[key] = (day[key] or 0) + meal[key]
    rows = [totals[d] for d in sorted(totals)]
    for row in rows:
        for key in ("protein", "carbs", "fat"):
            if row[key] is not None:
                row[key] = round(float(row[key]))
    return _ok(rows)


async def _food_meals(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    """Meals from the most recent ``days`` days that have any logs (up to 60 back)."""
    start, end = _window(60)
    meals = await _all_meals(ctx, start, end)
    keep = sorted({m["date"] for m in meals})[-params["days"] :]
    return _ok([m for m in meals if m["date"] in keep])


async def _local_sessions(
    ctx: SourceContext, start: date, end: date
) -> list[dict[str, Any]]:
    await ctx.container.workout_storage.initialize()
    clause, owner = _owner_clause(ctx.user)
    rows = await _fetch(
        ctx,
        f"""
        SELECT workout_date, created_at, split_name, metrics FROM workout_sessions
        WHERE {clause} AND workout_date >= ? AND workout_date <= ?
        ORDER BY workout_date
        """,  # noqa: S608 - fixed clause
        (*owner, start.isoformat(), end.isoformat()),
    )
    out = []
    for row in rows:
        try:
            metrics = json.loads(row["metrics"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metrics = {}
        minutes = (
            _num(metrics.get("duration_min") or metrics.get("duration_minutes")) or 0
        )
        out.append(
            {
                "date": row["workout_date"],
                "time": str(row["created_at"] or "")[11:16],
                "name": str(row["split_name"] or "Workout").title(),
                "minutes": round(minutes),
                "kcal": round(_num(metrics.get("kcal")) or 0),
                "km": round(_num(metrics.get("km")) or 0, 2),
                "source": "blacki",
            }
        )
    return out


async def _training_sessions(
    ctx: SourceContext, params: dict[str, Any]
) -> dict[str, Any]:
    start, end = _window(params["days"])
    data = await _archive(ctx, "exercise")
    tracker = _tcx_sessions(data["exercise"]) if data else []
    rows = [s for s in tracker if start.isoformat() <= s["date"] <= end.isoformat()]
    rows += await _local_sessions(ctx, start, end)
    rows.sort(key=lambda r: (r["date"], r["time"]))
    return _ok(rows)


async def _training_weekly(
    ctx: SourceContext, params: dict[str, Any]
) -> dict[str, Any]:
    weeks = params["weeks"]
    sessions = (await _training_sessions(ctx, {"days": min(weeks * 7, 90)}))["rows"]
    today = _today()
    monday = today - timedelta(days=today.weekday())
    totals = {
        (monday - timedelta(weeks=i)).isoformat(): {
            "minutes": 0,
            "sessions": 0,
            "kcal": 0,
        }
        for i in range(weeks)
    }
    for s in sessions:
        d = date.fromisoformat(s["date"])
        key = (d - timedelta(days=d.weekday())).isoformat()
        if key in totals:
            totals[key]["minutes"] += s["minutes"]
            totals[key]["sessions"] += 1
            totals[key]["kcal"] += s["kcal"]
    return _ok([{"week": k, **v} for k, v in sorted(totals.items())])


async def _files_list(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    from blacki.user_files.service import get_user_file_service

    try:
        files = await get_user_file_service().list_files(ctx.user.sender_id, "", 50)
    except Exception:
        logger.exception("Mini App file listing failed")
        return _not_ready("unavailable")
    return _ok(
        [
            {
                "id": f.object_id,
                "name": f.display_name,
                "kind": f.media_kind,
                "mime": f.mime_type,
                "size": f.size_bytes,
                "uploaded": f.uploaded_at,
            }
            for f in files
        ]
    )


async def _targets(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    targets = await ctx.store.get_pref(ctx.user.chat_user_id, "targets") or {}
    row = {k: targets.get(k) for k in SOURCES["targets"].fields or ()}
    return _ok([row])


async def _tracker(ctx: SourceContext, params: dict[str, Any]) -> dict[str, Any]:
    storage = ctx.container.declarative_db_storage
    await storage.initialize()
    template = await _fetch(
        ctx,
        "SELECT query_type FROM saved_query_templates WHERE user_id = ? AND template_name = ?",  # noqa: E501
        (ctx.user.chat_user_id, params["template"]),
    )
    if not template:
        return {
            "status": "error",
            "rows": [],
            "error": "No saved query template with that name",
        }
    if str(template[0]["query_type"]).upper() != "SELECT":
        return {
            "status": "error",
            "rows": [],
            "error": "Tracker sources must be SELECT templates",
        }
    try:
        result = await storage.execute_query_template(
            ctx.user.chat_user_id, params["template"], {}
        )
    except ValueError as exc:
        return {"status": "error", "rows": [], "error": str(exc)}
    rows = result if isinstance(result, list) else []
    return _ok(rows[: params["limit"]])


async def _cards(
    ctx: SourceContext, params: dict[str, Any], *, pinned: bool
) -> dict[str, Any]:
    cards = await ctx.store.list_cards(ctx.user.chat_user_id, pinned_only=pinned)
    return _ok(cards)


RESOLVERS: dict[str, Resolver] = {
    "health.days": _health_days,
    "health.sleep": _health_sleep,
    "health.daily": _health_daily,
    "body.weighins": _body_weighins,
    "training.sessions": _training_sessions,
    "training.weekly": _training_weekly,
    "food.days": _food_days,
    "food.meals": _food_meals,
    "files.list": _files_list,
    "targets": _targets,
    "tracker": _tracker,
    "cards.recent": lambda ctx, p: _cards(ctx, p, pinned=False),
    "cards.pinned": lambda ctx, p: _cards(ctx, p, pinned=True),
}


def validate_params(
    source: str, raw: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Check query parameters against the source's declared limits."""
    spec = SOURCES[source]
    errors: list[str] = []
    out: dict[str, Any] = {}
    for key in raw:
        if key not in spec.params:
            allowed = ", ".join(spec.params) or "none"
            errors.append(f"{source} has no setting {key!r} (allowed: {allowed})")
    for key, param in spec.params.items():
        value = raw.get(key, param.default)
        if value is None:
            if param.required:
                errors.append(f"{source} needs {key!r}")
            continue
        if param.kind == "int":
            if type(value) is not int or not param.min <= value <= param.max:  # type: ignore[operator]
                errors.append(
                    f"{source} {key} must be a whole number {param.min}..{param.max}"
                )
                continue
        elif param.kind == "enum":
            if value not in param.options:
                errors.append(
                    f"{source} {key} must be one of {', '.join(param.options)}"
                )
                continue
        elif param.kind == "text" and (
            not isinstance(value, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value)
        ):
            errors.append(f"{source} {key} must be a lowercase name")
            continue
        out[key] = value
    return out, errors


async def resolve(
    ctx: SourceContext, source: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Resolve one validated query, never raising to the page."""
    try:
        return await RESOLVERS[source](ctx, params)
    except Exception:
        logger.exception("Mini App source %s failed", source)
        return {"status": "error", "rows": [], "error": "This data could not be loaded"}
