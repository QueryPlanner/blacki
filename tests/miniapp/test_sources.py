from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import AsyncGenerator
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import aiosqlite
import pytest

from blacki.container import AppContainer
from blacki.declarative_db.storage import SqliteDeclarativeDbStorage
from blacki.health.config import (
    GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION,
)
from blacki.health.storage import (
    HealthConnection,
    SqliteGoogleHealthStorage,
)
from blacki.miniapp.sources import (
    SOURCES,
    MiniAppUser,
    SourceContext,
    SourceParam,
    SourceSpec,
    _civil,
    _local,
    _num,
    _owner_clause,
    _read_archive,
    _today,
    _ts,
    _window,
    resolve,
    validate_params,
)
from blacki.miniapp.store import MiniAppStore
from blacki.user_files.service import StoredUserFile, UserFileService


def test_miniapp_user() -> None:
    u = MiniAppUser(12345)
    assert u.telegram_user_id == 12345
    assert u.chat_user_id == "telegram-chat-12345"
    assert u.sender_id == "12345"


def test_helpers() -> None:
    assert isinstance(_today(), date)
    start, end = _window(7)
    assert (end - start).days == 6

    u = MiniAppUser(42)
    clause, args = _owner_clause(u)
    assert clause == "(user_id = ? OR user_id LIKE ?)"
    assert args == ("telegram-chat-42", "telegram-chat-42-thread-%")

    d = _civil({"year": 2026, "month": 5, "day": 12})
    assert d == date(2026, 5, 12)

    dt = _ts("2026-05-12T10:30:00Z")
    assert dt.tzinfo is not None

    loc1 = _local("2026-05-12T10:30:00Z", "3600s")
    loc2 = _local("2026-05-12T10:30:00Z", None)
    assert (loc1 - loc2).total_seconds() == 3600

    assert _num(10) == 10.0
    assert _num("12.5") == 12.5
    assert _num("invalid") is None
    assert _num(None) is None
    assert _num(float("nan")) is None


def test_validate_params() -> None:
    # 1. Unknown param error
    out, errors = validate_params("health.days", {"unknown": 123})
    assert any("has no setting 'unknown'" in e for e in errors)

    # 2. Required param missing
    out, errors = validate_params("tracker", {})
    assert any("needs 'template'" in e for e in errors)

    # 3. Valid defaults
    out, errors = validate_params("health.days", {})
    assert errors == []
    assert out["days"] == 7

    # 4. Int param invalid type and bounds
    out, errors = validate_params("health.days", {"days": "7"})
    assert any("must be a whole number" in e for e in errors)

    out, errors = validate_params("health.days", {"days": 0})
    assert any("must be a whole number" in e for e in errors)

    out, errors = validate_params("health.days", {"days": 32})
    assert any("must be a whole number" in e for e in errors)

    out, errors = validate_params("health.days", {"days": 10})
    assert errors == []
    assert out["days"] == 10

    # 5. Enum param
    out, errors = validate_params("health.daily", {"metric": "invalid", "days": 30})
    assert any("must be one of" in e for e in errors)

    out, errors = validate_params("health.daily", {"metric": "steps", "days": 30})
    assert errors == []
    assert out["metric"] == "steps"

    # 6. Text param
    out, errors = validate_params("tracker", {"template": "Invalid-Name!"})
    assert any("must be a lowercase name" in e for e in errors)

    out, errors = validate_params("tracker", {"template": "my_query"})
    assert errors == []
    assert out["template"] == "my_query"
    assert out["limit"] == 50


@pytest.fixture
async def app_container(tmp_path: Path) -> AsyncGenerator[AppContainer, None]:
    db_path = tmp_path / "blacki_test.db"
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    lock = asyncio.Lock()
    container = AppContainer(conn, lock)

    # Initialize storage tables
    await container.calorie_storage.initialize()
    await container.workout_storage.initialize()
    await container.declarative_db_storage.initialize()

    # Google Health Storage mock on container using an instance for autospec
    real_gh = SqliteGoogleHealthStorage(conn, lock)
    gh_storage = create_autospec(real_gh, spec_set=True)
    container._google_health_storage = gh_storage

    try:
        yield container
    finally:
        await conn.close()


@pytest.fixture
async def miniapp_store(app_container: AppContainer) -> MiniAppStore:
    lock = asyncio.Lock()
    st = MiniAppStore(app_container.conn, lock)
    await st.initialize()
    return st


def setup_archive_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE google_health_archive (
            owner TEXT,
            account TEXT,
            stream TEXT,
            payload TEXT,
            digest TEXT,
            last_seen TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def test_read_archive(tmp_path: Path) -> None:
    db_file = tmp_path / "archive.db"
    setup_archive_db(db_file)

    conn = sqlite3.connect(db_file)
    # Insert some records: valid, invalid json, and exercise with & without tcx
    records = [
        (
            "owner1",
            "acc1",
            "weight",
            json.dumps(
                {
                    "data": {
                        "name": "w1",
                        "weight": {
                            "sampleTime": {
                                "civilTime": {
                                    "date": {"year": 2026, "month": 5, "day": 10}
                                }
                            },
                            "weightGrams": 75000,
                        },
                    }
                }
            ),
            "d1",
            "2026-05-10",
        ),
        ("owner1", "acc1", "weight", "invalid json", "d2", "2026-05-10"),
        (
            "owner1",
            "acc1",
            "exercise",
            json.dumps({"data": {"exercise": "ex1"}}),  # missing tcx
            "d3",
            "2026-05-10",
        ),
        (
            "owner1",
            "acc1",
            "exercise",
            json.dumps(
                {
                    "data": {
                        "exercise": "ex2",
                        "tcx": (
                            '<Activity Sport="Running">'
                            "<Id>2026-05-10T08:00:00Z</Id>"
                            "<TotalTimeSeconds>1800</TotalTimeSeconds>"
                            "<DistanceMeters>5000</DistanceMeters>"
                            "<Calories>300</Calories>"
                            "</Activity>"
                        ),
                    }
                }
            ),
            "d4",
            "2026-05-10",
        ),
    ]
    conn.executemany(
        "INSERT INTO google_health_archive VALUES (?, ?, ?, ?, ?, ?)", records
    )
    conn.commit()
    conn.close()

    res = _read_archive(str(db_file), "owner1", "acc1", ["weight", "exercise"])
    assert len(res["weight"]) == 1
    assert res["weight"][0]["name"] == "w1"
    assert len(res["exercise"]) == 1
    assert res["exercise"][0]["exercise"] == "ex2"


async def test_archive_status_branches(
    app_container: AppContainer, miniapp_store: MiniAppStore
) -> None:
    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)
    gh_storage: Any = app_container.google_health_storage

    # 1. Connection is None -> not_connected
    gh_storage.get_connection.return_value = None
    res = await resolve(ctx, "health.sleep", {"days": 7})
    assert res["status"] == "not_connected"

    # 2. Connection status != "connected" -> connection.status
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)
    conn_mock = create_autospec(HealthConnection, instance=True)
    conn_mock.status = "error_revoked"
    gh_storage.get_connection.return_value = conn_mock
    res = await resolve(ctx, "health.sleep", {"days": 7})
    assert res["status"] == "error_revoked"

    # 3. Archive consent version too low -> archive_consent_required
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)
    conn_mock.status = "connected"
    conn_mock.archive_consent_version = GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION - 1
    res = await resolve(ctx, "health.sleep", {"days": 7})
    assert res["status"] == "archive_consent_required"

    # 4. No archive database path -> unavailable
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)
    conn_mock.archive_consent_version = GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION
    gh_storage.archive.database_path.return_value = None
    res = await resolve(ctx, "health.sleep", {"days": 7})
    assert res["status"] == "unavailable"

    # 5. sqlite3.Error when reading archive -> unavailable
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)
    gh_storage.archive.database_path.return_value = "/nonexistent/path/to/archive.db"
    conn_mock.health_user_id = "health_user_100"
    res = await resolve(ctx, "health.sleep", {"days": 7})
    assert res["status"] == "unavailable"


async def test_health_daily_and_sleep(
    tmp_path: Path, app_container: AppContainer, miniapp_store: MiniAppStore
) -> None:
    db_file = tmp_path / "archive.db"
    setup_archive_db(db_file)

    today = _today()
    d_str = today.isoformat()
    d_dict = {"year": today.year, "month": today.month, "day": today.day}

    conn = sqlite3.connect(db_file)
    records = [
        # steps with civilStartTime
        (
            "telegram-chat-100",
            "acc1",
            "steps",
            json.dumps(
                {
                    "data": {
                        "name": "s1",
                        "steps": {
                            "count": 5000,
                            "interval": {"civilStartTime": {"date": d_dict}},
                        },
                        "dataSource": {"platform": "garmin"},
                    }
                }
            ),
            "d_s1",
            d_str,
        ),
        # steps with startTime and offset
        (
            "telegram-chat-100",
            "acc1",
            "steps",
            json.dumps(
                {
                    "data": {
                        "name": "s2",
                        "steps": {
                            "count": 6000,
                            "interval": {
                                "startTime": f"{d_str}T08:00:00Z",
                                "startUtcOffset": "0s",
                            },
                        },
                        "dataSource": {"platform": "google"},
                    }
                }
            ),
            "d_s2",
            d_str,
        ),
        # steps invalid (continue branch)
        (
            "telegram-chat-100",
            "acc1",
            "steps",
            json.dumps({"data": {"name": "s3", "steps": {}}}),
            "d_s3",
            d_str,
        ),
        # hrv
        (
            "telegram-chat-100",
            "acc1",
            "daily-heart-rate-variability",
            json.dumps(
                {
                    "data": {
                        "name": "hrv1",
                        "dailyHeartRateVariability": {
                            "date": d_dict,
                            "averageHeartRateVariabilityMilliseconds": 65.4,
                        },
                    }
                }
            ),
            "d_hrv",
            d_str,
        ),
        # hrv invalid (continue branch)
        (
            "telegram-chat-100",
            "acc1",
            "daily-heart-rate-variability",
            json.dumps(
                {
                    "data": {
                        "name": "hrv2",
                        "dailyHeartRateVariability": {"date": "invalid"},
                    }
                }
            ),
            "d_hrv2",
            d_str,
        ),
        # sleep valid
        (
            "telegram-chat-100",
            "acc1",
            "sleep",
            json.dumps(
                {
                    "data": {
                        "name": "sl1",
                        "sleep": {
                            "interval": {
                                "startTime": f"{d_str}T22:00:00Z",
                                "endTime": f"{d_str}T06:00:00Z",
                            },
                            "stages": [
                                {
                                    "startTime": f"{d_str}T00:00:00Z",
                                    "endTime": f"{d_str}T01:00:00Z",
                                    "type": "DEEP",
                                },
                                {
                                    "startTime": f"{d_str}T01:00:00Z",
                                    "endTime": f"{d_str}T03:00:00Z",
                                    "type": "LIGHT",
                                },
                                {
                                    "startTime": f"{d_str}T03:00:00Z",
                                    "endTime": f"{d_str}T05:00:00Z",
                                    "type": "REM",
                                },
                                {
                                    "startTime": f"{d_str}T05:00:00Z",
                                    "endTime": f"{d_str}T05:30:00Z",
                                    "type": "AWAKE",
                                },
                                {
                                    "startTime": "invalid",
                                    "endTime": "invalid",
                                    "type": "UNKNOWN",
                                },
                            ],
                        },
                    }
                }
            ),
            "d_sl1",
            d_str,
        ),
        # Sleep item 1: name sl1
        # Sleep item 2 (same night d_str, but different name sl2 and LESS
        # asleep time) to test line 390.
        # In order for both sl1 and sl2 to be in records, their keys must
        # be different in latest.
        # In _read_archive, key = inner.get("name") or inner.get("exercise") or digest.
        # So sl1 has name "sl1", sl2 has name "sl2".
        # sl2 wake time is d_str so night is same!
        (
            "telegram-chat-100",
            "acc1",
            "sleep",
            json.dumps(
                {
                    "data": {
                        "name": "sl2",
                        "sleep": {
                            "interval": {
                                "startTime": f"{d_str}T00:00:00Z",
                                "endTime": f"{d_str}T06:00:00Z",
                            },
                            "stages": [
                                {
                                    "startTime": f"{d_str}T00:00:00Z",
                                    "endTime": f"{d_str}T01:00:00Z",
                                    "type": "DEEP",
                                },
                            ],
                        },
                    }
                }
            ),
            "d_sl2",
            d_str,
        ),
        # sleep with 0 asleep time (continue branch)
        (
            "telegram-chat-100",
            "acc1",
            "sleep",
            json.dumps(
                {
                    "data": {
                        "name": "sl3",
                        "sleep": {
                            "interval": {
                                "startTime": f"{d_str}T22:00:00Z",
                                "endTime": f"{d_str}T06:00:00Z",
                            },
                            "stages": [],
                        },
                    }
                }
            ),
            "d_sl3",
            d_str,
        ),
        # sleep with invalid interval (continue branch)
        (
            "telegram-chat-100",
            "acc1",
            "sleep",
            json.dumps(
                {
                    "data": {
                        "name": "sl4",
                        "sleep": {"interval": {}},
                    }
                }
            ),
            "d_sl4",
            d_str,
        ),
    ]
    conn.executemany(
        "INSERT INTO google_health_archive VALUES (?, ?, ?, ?, ?, ?)", records
    )
    conn.commit()
    conn.close()

    gh_storage: Any = app_container.google_health_storage
    conn_mock = create_autospec(HealthConnection, instance=True)
    conn_mock.status = "connected"
    conn_mock.archive_consent_version = GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION
    conn_mock.health_user_id = "acc1"
    gh_storage.get_connection.return_value = conn_mock
    gh_storage.archive.database_path.return_value = str(db_file)

    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    # Test health.daily with steps
    res_steps = await resolve(ctx, "health.daily", {"metric": "steps", "days": 7})
    assert res_steps["status"] == "ok"
    assert len(res_steps["rows"]) == 1
    assert (
        res_steps["rows"][0]["value"] == 6000.0
    )  # max of garmin(5000) and google(6000)

    # Test health.daily with hrv
    res_hrv = await resolve(ctx, "health.daily", {"metric": "hrv", "days": 7})
    assert res_hrv["status"] == "ok"
    assert len(res_hrv["rows"]) == 1
    assert res_hrv["rows"][0]["value"] == 65.4

    # Test health.sleep
    res_sleep = await resolve(ctx, "health.sleep", {"days": 7})
    assert res_sleep["status"] == "ok"
    assert len(res_sleep["rows"]) == 1
    assert res_sleep["rows"][0]["asleep"] == 300  # 60 deep + 120 light + 120 rem

    # Test health.days
    res_days = await resolve(ctx, "health.days", {"days": 3})
    assert res_days["status"] == "ok"
    assert len(res_days["rows"]) == 3
    # Find row for today
    today_row = next(r for r in res_days["rows"] if r["date"] == d_str)
    assert today_row["hrv"] == 65.4
    assert today_row["steps"] == 6000


async def test_body_weighins(
    tmp_path: Path, app_container: AppContainer, miniapp_store: MiniAppStore
) -> None:
    db_file = tmp_path / "archive.db"
    setup_archive_db(db_file)

    today = _today()
    d_str = today.isoformat()
    d_dict = {"year": today.year, "month": today.month, "day": today.day}

    conn = sqlite3.connect(db_file)
    records = [
        (
            "telegram-chat-100",
            "acc1",
            "body-fat",
            json.dumps(
                {
                    "data": {
                        "name": "fat1",
                        "bodyFat": {
                            "sampleTime": {"civilTime": {"date": d_dict}},
                            "percentage": 15.2,
                        },
                    }
                }
            ),
            "d_fat1",
            d_str,
        ),
        # invalid body-fat
        (
            "telegram-chat-100",
            "acc1",
            "body-fat",
            json.dumps({"data": {"name": "fat2", "bodyFat": {}}}),
            "d_fat2",
            d_str,
        ),
        # valid weight
        (
            "telegram-chat-100",
            "acc1",
            "weight",
            json.dumps(
                {
                    "data": {
                        "name": "wt1",
                        "weight": {
                            "sampleTime": {"civilTime": {"date": d_dict}},
                            "weightGrams": 78200,
                        },
                    }
                }
            ),
            "d_wt1",
            d_str,
        ),
        # invalid weight
        (
            "telegram-chat-100",
            "acc1",
            "weight",
            json.dumps({"data": {"name": "wt2", "weight": {}}}),
            "d_wt2",
            d_str,
        ),
    ]
    conn.executemany(
        "INSERT INTO google_health_archive VALUES (?, ?, ?, ?, ?, ?)", records
    )
    conn.commit()
    conn.close()

    gh_storage: Any = app_container.google_health_storage
    conn_mock = create_autospec(HealthConnection, instance=True)
    conn_mock.status = "connected"
    conn_mock.archive_consent_version = GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION
    conn_mock.health_user_id = "acc1"
    gh_storage.get_connection.return_value = conn_mock
    gh_storage.archive.database_path.return_value = str(db_file)

    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    res = await resolve(ctx, "body.weighins", {"days": 30})
    assert res["status"] == "ok"
    assert len(res["rows"]) == 1
    assert res["rows"][0]["kg"] == 78.2
    assert res["rows"][0]["fat"] == 15.2


async def test_food_and_meals(
    tmp_path: Path, app_container: AppContainer, miniapp_store: MiniAppStore
) -> None:
    db_file = tmp_path / "archive.db"
    setup_archive_db(db_file)

    today = _today()
    d_str = today.isoformat()

    # 1. Insert remote nutrition log for a different day
    conn = sqlite3.connect(db_file)
    records = [
        (
            "telegram-chat-100",
            "acc1",
            "nutrition-log",
            json.dumps(
                {
                    "data": {
                        "name": "nut1",
                        "nutritionLog": {
                            "foodDisplayName": "Apple",
                            "interval": {
                                "civilStartTime": {
                                    "date": {"year": 2026, "month": 1, "day": 1},
                                    "time": {"hours": 12, "minutes": 30},
                                }
                            },
                            "energy": {"kcal": 95},
                            "nutrients": [
                                {"nutrient": "PROTEIN", "quantity": {"grams": 0.5}},
                                {
                                    "nutrient": "TOTAL_CARBOHYDRATE",
                                    "quantity": {"grams": 25},
                                },
                                {"nutrient": "TOTAL_FAT", "quantity": {"grams": 0.3}},
                            ],
                        },
                    }
                }
            ),
            "d_nut1",
            "2026-01-01",
        ),
        # Invalid nutrition log
        (
            "telegram-chat-100",
            "acc1",
            "nutrition-log",
            json.dumps({"data": {"name": "nut2", "nutritionLog": {}}}),
            "d_nut2",
            "2026-01-01",
        ),
    ]
    conn.executemany(
        "INSERT INTO google_health_archive VALUES (?, ?, ?, ?, ?, ?)", records
    )
    conn.commit()
    conn.close()

    # 2. Insert local calorie log for today
    await app_container.conn.execute(
        """
        INSERT INTO calorie_logs
        (user_id, logged_date, logged_at, description,
         calories, protein_g, carbs_g, fat_g, meal_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "telegram-chat-100",
            d_str,
            f"{d_str}T13:45:00",
            "Chicken Rice",
            650,
            45,
            70,
            15,
            "lunch",
        ),
    )
    await app_container.conn.commit()

    gh_storage: Any = app_container.google_health_storage
    conn_mock = create_autospec(HealthConnection, instance=True)
    conn_mock.status = "connected"
    conn_mock.archive_consent_version = GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION
    conn_mock.health_user_id = "acc1"
    gh_storage.get_connection.return_value = conn_mock
    gh_storage.archive.database_path.return_value = str(db_file)

    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    # Food days
    res_days = await resolve(ctx, "food.days", {"days": 14})
    assert res_days["status"] == "ok"
    assert len(res_days["rows"]) == 1
    assert res_days["rows"][0]["kcal"] == 650
    assert res_days["rows"][0]["protein"] == 45
    assert res_days["rows"][0]["entries"] == 1

    # Food meals
    res_meals = await resolve(ctx, "food.meals", {"days": 3})
    assert res_meals["status"] == "ok"
    assert len(res_meals["rows"]) == 1
    assert res_meals["rows"][0]["name"] == "Chicken Rice"
    assert res_meals["rows"][0]["source"] == "blacki"


async def test_training_sessions_and_weekly(
    tmp_path: Path, app_container: AppContainer, miniapp_store: MiniAppStore
) -> None:
    db_file = tmp_path / "archive.db"
    setup_archive_db(db_file)

    today = _today()
    d_str = today.isoformat()

    # 1. Archive exercise
    conn = sqlite3.connect(db_file)
    records = [
        (
            "telegram-chat-100",
            "acc1",
            "exercise",
            json.dumps(
                {
                    "data": {
                        "exercise": "ex1",
                        "tcx": (
                            '<Activity Sport="Running">'
                            f"<Id>{d_str}T07:15:00Z</Id>"
                            "<TotalTimeSeconds>2400</TotalTimeSeconds>"
                            "<DistanceMeters>8000</DistanceMeters>"
                            "<Calories>520</Calories>"
                            "</Activity>"
                        ),
                    }
                }
            ),
            "d_ex1",
            d_str,
        ),
        # Invalid TCX
        (
            "telegram-chat-100",
            "acc1",
            "exercise",
            json.dumps({"data": {"exercise": "ex2", "tcx": "no id here"}}),
            "d_ex2",
            d_str,
        ),
        (
            "telegram-chat-100",
            "acc1",
            "exercise",
            json.dumps({"data": {"exercise": "ex3", "tcx": "<Id>invalid-date</Id>"}}),
            "d_ex3",
            d_str,
        ),
    ]
    conn.executemany(
        "INSERT INTO google_health_archive VALUES (?, ?, ?, ?, ?, ?)", records
    )
    conn.commit()
    conn.close()

    # 2. Local workout session
    await app_container.conn.execute(
        """
        INSERT INTO workout_sessions
        (user_id, workout_date, created_at, split_name, metrics)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "telegram-chat-100",
            d_str,
            f"{d_str}T18:00:00",
            "push day",
            json.dumps({"duration_min": 50, "kcal": 350, "km": 0}),
        ),
    )
    # Another with bad metrics json
    await app_container.conn.execute(
        """
        INSERT INTO workout_sessions
        (user_id, workout_date, created_at, split_name, metrics)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("telegram-chat-100", d_str, f"{d_str}T19:00:00", "legs", "bad json"),
    )
    await app_container.conn.commit()

    gh_storage: Any = app_container.google_health_storage
    conn_mock = create_autospec(HealthConnection, instance=True)
    conn_mock.status = "connected"
    conn_mock.archive_consent_version = GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION
    conn_mock.health_user_id = "acc1"
    gh_storage.get_connection.return_value = conn_mock
    gh_storage.archive.database_path.return_value = str(db_file)

    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    res_sess = await resolve(ctx, "training.sessions", {"days": 7})
    assert res_sess["status"] == "ok"
    assert len(res_sess["rows"]) == 3

    res_week = await resolve(ctx, "training.weekly", {"weeks": 4})
    assert res_week["status"] == "ok"
    assert len(res_week["rows"]) == 4


async def test_files_list(
    app_container: AppContainer,
    miniapp_store: MiniAppStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    service_mock = create_autospec(UserFileService, spec_set=True)
    file_item = StoredUserFile(
        object_id="f123",
        display_name="report.pdf",
        media_kind="document",
        mime_type="application/pdf",
        size_bytes=1024,
        uploaded_at="2026-05-10T12:00:00Z",
        expires_at=None,
    )
    service_mock.list_files.return_value = [file_item]

    monkeypatch.setattr(
        "blacki.user_files.service.get_user_file_service", lambda: service_mock
    )

    res = await resolve(ctx, "files.list", {})
    assert res["status"] == "ok"
    assert len(res["rows"]) == 1
    assert res["rows"][0]["id"] == "f123"
    assert res["rows"][0]["name"] == "report.pdf"

    # Exception branch
    service_mock.list_files.side_effect = RuntimeError("Service down")
    res_err = await resolve(ctx, "files.list", {})
    assert res_err["status"] == "unavailable"


async def test_targets(
    app_container: AppContainer, miniapp_store: MiniAppStore
) -> None:
    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    await miniapp_store.set_pref(
        user.chat_user_id, "targets", {"calories_kcal": 2200, "protein_g": 160}
    )

    res = await resolve(ctx, "targets", {})
    assert res["status"] == "ok"
    assert len(res["rows"]) == 1
    assert res["rows"][0]["calories_kcal"] == 2200
    assert res["rows"][0]["protein_g"] == 160
    assert res["rows"][0]["steps"] is None


async def test_tracker(
    app_container: AppContainer, miniapp_store: MiniAppStore
) -> None:
    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    # 1. Template doesn't exist
    res = await resolve(ctx, "tracker", {"template": "nonexistent", "limit": 10})
    assert res["status"] == "error"
    assert "No saved query template" in res["error"]

    # 2. Template is not SELECT
    await app_container.conn.execute(
        """
        INSERT INTO saved_query_templates
        (user_id, template_name, table_name, query_type)
        VALUES (?, ?, ?, ?)
        """,
        (user.chat_user_id, "delete_stuff", "foo", "DELETE"),
    )
    await app_container.conn.commit()

    res = await resolve(ctx, "tracker", {"template": "delete_stuff", "limit": 10})
    assert res["status"] == "error"
    assert "must be SELECT templates" in res["error"]

    # 3. Valid SELECT template, storage raises ValueError
    decl_mock = create_autospec(SqliteDeclarativeDbStorage, spec_set=True)
    app_container._declarative_db_storage = decl_mock
    await app_container.conn.execute(
        """
        INSERT INTO saved_query_templates
        (user_id, template_name, table_name, query_type)
        VALUES (?, ?, ?, ?)
        """,
        (user.chat_user_id, "my_select", "foo", "SELECT"),
    )
    await app_container.conn.commit()

    decl_mock.execute_query_template.side_effect = ValueError("Query execution failed")
    res = await resolve(ctx, "tracker", {"template": "my_select", "limit": 10})
    assert res["status"] == "error"
    assert res["error"] == "Query execution failed"

    # 4. Valid SELECT template returns list
    decl_mock.execute_query_template.side_effect = None
    decl_mock.execute_query_template.return_value = [{"col": 1}, {"col": 2}]
    res = await resolve(ctx, "tracker", {"template": "my_select", "limit": 1})
    assert res["status"] == "ok"
    assert len(res["rows"]) == 1
    assert res["rows"][0]["col"] == 1


async def test_cards(app_container: AppContainer, miniapp_store: MiniAppStore) -> None:
    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    c1_id = await miniapp_store.create_card(
        user.chat_user_id, title="Card 1", source="Card('c1')", ir={"id": "c1"}
    )
    await miniapp_store.set_card_pinned(user.chat_user_id, c1_id, True)
    await miniapp_store.create_card(
        user.chat_user_id, title="Card 2", source="Card('c2')", ir={"id": "c2"}
    )

    res_rec = await resolve(ctx, "cards.recent", {})
    assert res_rec["status"] == "ok"
    assert len(res_rec["rows"]) == 2

    res_pin = await resolve(ctx, "cards.pinned", {})
    assert res_pin["status"] == "ok"
    assert len(res_pin["rows"]) == 1
    assert res_pin["rows"][0]["id"] == c1_id


async def test_resolve_exception(
    app_container: AppContainer,
    miniapp_store: MiniAppStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = MiniAppUser(100)
    ctx = SourceContext(user=user, container=app_container, store=miniapp_store)

    # Force a resolver to blow up unexpectedly
    monkeypatch.setitem(
        SOURCES,
        "exploding",
        SourceSpec("exploding", "Blows up", {}, None),
    )
    from blacki.miniapp.sources import RESOLVERS

    async def _explode(ctx: Any, params: Any) -> Any:
        raise ZeroDivisionError("boom")

    monkeypatch.setitem(RESOLVERS, "exploding", _explode)

    res = await resolve(ctx, "exploding", {})
    assert res["status"] == "error"
    assert res["error"] == "This data could not be loaded"


async def test_sources_additional_missing_branches(
    tmp_path: Path, app_container: AppContainer, miniapp_store: MiniAppStore
) -> None:
    user = MiniAppUser(100)

    # 1. 844->846: param default is None and required is False
    import blacki.miniapp.sources as src_module
    from blacki.miniapp.sources import SourceSpec

    custom_spec = SourceSpec(
        "custom.test",
        "Test",
        {"opt": SourceParam("int", default=None, required=False)},
        None,
    )
    src_module.SOURCES["custom.test"] = custom_spec
    try:
        out, errs = validate_params("custom.test", {})
        assert errs == []
        assert "opt" not in out
    finally:
        del src_module.SOURCES["custom.test"]

    # 2. _daily_values with value is None (line 340->333)
    rec_none_val = [
        {
            "dailyHeartRateVariability": {
                "date": {"year": 2026, "month": 1, "day": 1},
                "averageHeartRateVariabilityMilliseconds": "invalid",
            }
        }
    ]
    assert (
        src_module._daily_values(
            rec_none_val,
            "dailyHeartRateVariability",
            "averageHeartRateVariabilityMilliseconds",
        )
        == {}
    )

    # 3. Not ready branches for archive is None: 452, 460, 488, 521
    ctx_not_connected = SourceContext(
        user=user, container=app_container, store=miniapp_store
    )
    ctx_not_connected._archive_status = "not_connected"
    assert (
        await resolve(ctx_not_connected, "health.daily", {"metric": "steps", "days": 7})
    )["status"] == "not_connected"
    assert (
        await resolve(ctx_not_connected, "health.daily", {"metric": "hrv", "days": 7})
    )["status"] == "not_connected"
    assert (await resolve(ctx_not_connected, "health.days", {"days": 7}))[
        "status"
    ] == "not_connected"
    assert (await resolve(ctx_not_connected, "body.weighins", {"days": 7}))[
        "status"
    ] == "not_connected"

    # 4. _body_weighins weight outside window: 540->533
    db_file = tmp_path / "archive_out_window.db"
    setup_archive_db(db_file)
    conn = sqlite3.connect(db_file)
    conn.execute(
        "INSERT INTO google_health_archive VALUES (?, ?, ?, ?, ?, ?)",
        (
            "telegram-chat-100",
            "acc1",
            "weight",
            json.dumps(
                {
                    "data": {
                        "name": "wt_old",
                        "weight": {
                            "sampleTime": {
                                "civilTime": {
                                    "date": {"year": 2020, "month": 1, "day": 1}
                                }
                            },
                            "weightGrams": 70000,
                        },
                    }
                }
            ),
            "d_old",
            "2020-01-01",
        ),
    )
    conn.commit()
    conn.close()

    gh_storage: Any = app_container.google_health_storage
    conn_mock = create_autospec(HealthConnection, instance=True)
    conn_mock.status = "connected"
    conn_mock.archive_consent_version = GOOGLE_HEALTH_ARCHIVE_CONSENT_VERSION
    conn_mock.health_user_id = "acc1"
    gh_storage.get_connection.return_value = conn_mock
    gh_storage.archive.database_path.return_value = str(db_file)

    ctx_old = SourceContext(user=user, container=app_container, store=miniapp_store)
    res_weigh = await resolve(ctx_old, "body.weighins", {"days": 7})
    assert res_weigh["status"] == "ok"
    assert len(res_weigh["rows"]) == 0

    # 5. food.days with meal having protein/carbs/fat = None (655->654, 660->659)
    d_str = _today().isoformat()
    await app_container.conn.execute(
        """
        INSERT INTO calorie_logs
        (user_id, logged_date, logged_at, description,
         calories, protein_g, carbs_g, fat_g, meal_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "telegram-chat-100",
            d_str,
            f"{d_str}T09:00:00",
            "Coffee",
            5,
            None,
            None,
            None,
            "breakfast",
        ),
    )
    await app_container.conn.commit()
    ctx_food = SourceContext(user=user, container=app_container, store=miniapp_store)
    res_food = await resolve(ctx_food, "food.days", {"days": 1})
    assert res_food["status"] == "ok"
    assert res_food["rows"][0]["protein"] is None

    # 6. training.weekly with session from a week not in totals: 740->737
    # Monkeypatch _training_sessions to return a session with an arbitrary date
    ctx_train = SourceContext(user=user, container=app_container, store=miniapp_store)
    from unittest.mock import patch

    with patch(
        "blacki.miniapp.sources._training_sessions",
        return_value={"rows": [{"date": "2000-01-01", "minutes": 10, "kcal": 50}]},
    ):
        res_tr = await resolve(ctx_train, "training.weekly", {"weeks": 1})
        assert res_tr["status"] == "ok"
