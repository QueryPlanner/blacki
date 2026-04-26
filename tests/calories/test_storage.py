# mypy: disable-error-code="no-untyped-def"
from unittest.mock import AsyncMock, MagicMock

import pytest

from blacki.calories.storage import (
    CalorieEntry,
    PostgresCalorieStorage,
    close_calorie_storage,
    get_storage,
    init_calorie_storage,
)


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.execute = AsyncMock()
    pool.fetch = AsyncMock()
    pool.fetchval = AsyncMock()
    return pool


@pytest.fixture
async def calorie_storage(mock_pool):
    storage = PostgresCalorieStorage(mock_pool)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(mock_pool) -> None:
    storage = PostgresCalorieStorage(mock_pool)
    await storage.initialize()

    conn = mock_pool.acquire.return_value.__aenter__.return_value
    assert conn.execute.call_count == 2
    assert storage._schema_ready is True


@pytest.mark.asyncio
async def test_add_entry(calorie_storage, mock_pool) -> None:
    mock_pool.fetchval.return_value = 123
    entry = CalorieEntry(
        user_id="user1",
        description="apple",
        calories=95,
        logged_at="2026-04-26T10:00:00",
        logged_date="2026-04-26",
    )

    entry_id = await calorie_storage.add_entry(entry)

    assert entry_id == 123
    mock_pool.fetchval.assert_called_once()
    args = mock_pool.fetchval.call_args[0]
    assert args[1] == "user1"
    assert args[2] == "apple"
    assert args[3] == 95


@pytest.mark.asyncio
async def test_get_daily_summary(calorie_storage, mock_pool) -> None:
    mock_pool.fetch.return_value = [
        {
            "id": 1,
            "user_id": "user1",
            "description": "apple",
            "calories": 100,
            "protein_g": None,
            "carbs_g": 25,
            "fat_g": None,
            "meal_type": "snack",
            "logged_at": "2026-04-26T10:00:00",
            "logged_date": "2026-04-26",
        },
        {
            "id": 2,
            "user_id": "user1",
            "description": "egg",
            "calories": 70,
            "protein_g": 6,
            "carbs_g": None,
            "fat_g": 5,
            "meal_type": "breakfast",
            "logged_at": "2026-04-26T11:00:00",
            "logged_date": "2026-04-26",
        },
    ]

    summary = await calorie_storage.get_daily_summary("user1", "2026-04-26")

    assert summary.date == "2026-04-26"
    assert summary.entry_count == 2
    assert summary.total_calories == 170
    assert summary.total_protein_g == 6
    assert summary.total_carbs_g == 25
    assert summary.total_fat_g == 5
    assert len(summary.entries) == 2


@pytest.mark.asyncio
async def test_get_date_range_summary(calorie_storage, mock_pool) -> None:
    mock_pool.fetch.return_value = [
        {
            "logged_date": "2026-04-26",
            "entry_count": 2,
            "total_calories": 500,
            "total_protein_g": 20,
            "total_carbs_g": 50,
            "total_fat_g": 10,
        },
        {
            "logged_date": "2026-04-25",
            "entry_count": 3,
            "total_calories": 2000,
            "total_protein_g": 100,
            "total_carbs_g": 200,
            "total_fat_g": 50,
        },
    ]

    summaries = await calorie_storage.get_date_range_summary(
        "user1", "2026-04-20", "2026-04-26"
    )

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-26"
    assert summaries[0].total_calories == 500
    assert summaries[1].date == "2026-04-25"
    assert summaries[1].total_calories == 2000


@pytest.mark.asyncio
async def test_update_entry(calorie_storage, mock_pool) -> None:
    mock_pool.execute.return_value = "UPDATE 1"

    result = await calorie_storage.update_entry(
        1, "user1", calories=200, meal_type="lunch"
    )

    assert result is True
    mock_pool.execute.assert_called_once()
    args = mock_pool.execute.call_args[0]
    assert "calories = $3" in args[0]
    assert "meal_type = $4" in args[0]
    assert args[1] == 1
    assert args[2] == "user1"
    assert args[3] == 200
    assert args[4] == "lunch"


@pytest.mark.asyncio
async def test_delete_entry(calorie_storage, mock_pool) -> None:
    mock_pool.execute.return_value = "DELETE 1"

    result = await calorie_storage.delete_entry(1, "user1")

    assert result is True
    mock_pool.execute.assert_called_once_with(
        "DELETE FROM calorie_logs WHERE id = $1 AND user_id = $2",
        1,
        "user1",
    )


@pytest.mark.asyncio
async def test_singleton(mock_pool) -> None:
    import blacki.calories.storage as storage

    storage._storage = None

    with pytest.raises(RuntimeError):
        get_storage()

    instance = await init_calorie_storage(mock_pool)
    assert get_storage() is instance

    await close_calorie_storage()
    with pytest.raises(RuntimeError):
        get_storage()
