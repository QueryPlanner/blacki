# mypy: disable-error-code="no-untyped-def"
"""Unit tests for calorie storage."""

import asyncio

import aiosqlite
import pytest

from blacki.calories.storage import (
    CalorieEntry,
    SqliteCalorieStorage,
)


@pytest.fixture
async def conn():
    """Create an in-memory SQLite connection for testing."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()


@pytest.fixture
def lock():
    """Create a lock for write operations."""
    return asyncio.Lock()


@pytest.fixture
async def storage(conn, lock):
    """Create a storage instance with the test connection."""
    storage = SqliteCalorieStorage(conn, lock)
    await storage.initialize()
    yield storage
    await storage.close()


class TestSqliteCalorieStorage:
    """Tests for SqliteCalorieStorage."""

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, conn, lock) -> None:
        """Should create tables on initialization."""
        storage = SqliteCalorieStorage(conn, lock)
        await storage.initialize()

        assert storage.is_initialized is True

        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='calorie_logs'"
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_add_entry(self, storage) -> None:
        """Should add an entry and return its ID."""
        entry = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=95,
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )

        entry_id = await storage.add_entry(entry)

        assert entry_id == 1

    @pytest.mark.asyncio
    async def test_add_entry_with_macros(self, storage) -> None:
        """Should add an entry with macro nutrients."""
        entry = CalorieEntry(
            user_id="user1",
            description="protein shake",
            calories=200,
            protein_g=30.0,
            carbs_g=10.0,
            fat_g=5.0,
            meal_type="snack",
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )

        entry_id = await storage.add_entry(entry)

        assert entry_id == 1

        summary = await storage.get_daily_summary("user1", "2026-04-26")
        assert summary.total_protein_g == 30.0
        assert summary.total_carbs_g == 10.0
        assert summary.total_fat_g == 5.0

    @pytest.mark.asyncio
    async def test_get_daily_summary(self, storage) -> None:
        """Should get daily summary with entries."""
        entry1 = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=100,
            carbs_g=25.0,
            meal_type="snack",
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )
        entry2 = CalorieEntry(
            user_id="user1",
            description="egg",
            calories=70,
            protein_g=6.0,
            fat_g=5.0,
            meal_type="breakfast",
            logged_at="2026-04-26T11:00:00",
            logged_date="2026-04-26",
        )
        await storage.add_entry(entry1)
        await storage.add_entry(entry2)

        summary = await storage.get_daily_summary("user1", "2026-04-26")

        assert summary.date == "2026-04-26"
        assert summary.entry_count == 2
        assert summary.total_calories == 170
        assert summary.total_protein_g == 6.0
        assert summary.total_carbs_g == 25.0
        assert summary.total_fat_g == 5.0
        assert len(summary.entries) == 2

    @pytest.mark.asyncio
    async def test_get_daily_summary_empty(self, storage) -> None:
        """Should return empty summary for date with no entries."""
        summary = await storage.get_daily_summary("user1", "2026-04-26")

        assert summary.date == "2026-04-26"
        assert summary.entry_count == 0
        assert summary.total_calories == 0
        assert len(summary.entries) == 0

    @pytest.mark.asyncio
    async def test_get_date_range_summary(self, storage) -> None:
        """Should get summaries for date range."""
        for day in range(20, 27):
            entry = CalorieEntry(
                user_id="user1",
                description=f"food day {day}",
                calories=500 + day,
                protein_g=20.0 + day,
                logged_at=f"2026-04-{day:02d}T10:00:00",
                logged_date=f"2026-04-{day:02d}",
            )
            await storage.add_entry(entry)

        summaries = await storage.get_date_range_summary(
            "user1", "2026-04-20", "2026-04-26"
        )

        assert len(summaries) == 7
        assert summaries[0].date == "2026-04-26"
        assert summaries[0].total_calories == 526

    @pytest.mark.asyncio
    async def test_update_entry(self, storage) -> None:
        """Should update an entry."""
        entry = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=100,
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )
        entry_id = await storage.add_entry(entry)

        result = await storage.update_entry(
            entry_id, "user1", calories=200, meal_type="lunch"
        )

        assert result is True
        summary = await storage.get_daily_summary("user1", "2026-04-26")
        assert summary.entries[0].calories == 200
        assert summary.entries[0].meal_type == "lunch"

    @pytest.mark.asyncio
    async def test_update_entry_wrong_user(self, storage) -> None:
        """Should not update entry belonging to different user."""
        entry = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=100,
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )
        entry_id = await storage.add_entry(entry)

        result = await storage.update_entry(entry_id, "user2", calories=200)

        assert result is False
        summary = await storage.get_daily_summary("user1", "2026-04-26")
        assert summary.entries[0].calories == 100

    @pytest.mark.asyncio
    async def test_update_entry_invalid_column(self, storage) -> None:
        """Should raise ValueError for invalid column."""
        entry = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=100,
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )
        entry_id = await storage.add_entry(entry)

        with pytest.raises(ValueError, match="Column 'bogus' is not allowed"):
            await storage.update_entry(entry_id, "user1", bogus="value")

    @pytest.mark.asyncio
    async def test_delete_entry(self, storage) -> None:
        """Should delete an entry."""
        entry = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=100,
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )
        entry_id = await storage.add_entry(entry)

        result = await storage.delete_entry(entry_id, "user1")

        assert result is True
        summary = await storage.get_daily_summary("user1", "2026-04-26")
        assert summary.entry_count == 0

    @pytest.mark.asyncio
    async def test_delete_entry_wrong_user(self, storage) -> None:
        """Should not delete entry belonging to different user."""
        entry = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=100,
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )
        entry_id = await storage.add_entry(entry)

        result = await storage.delete_entry(entry_id, "user2")

        assert result is False
        summary = await storage.get_daily_summary("user1", "2026-04-26")
        assert summary.entry_count == 1

    @pytest.mark.asyncio
    async def test_delete_entry_not_found(self, storage) -> None:
        """Should return False for non-existent entry."""
        result = await storage.delete_entry(999, "user1")

        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_users_isolated(self, storage) -> None:
        """Should isolate data between users."""
        entry1 = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=100,
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )
        entry2 = CalorieEntry(
            user_id="user2",
            description="banana",
            calories=150,
            logged_at="2026-04-26T11:00:00",
            logged_date="2026-04-26",
        )
        await storage.add_entry(entry1)
        await storage.add_entry(entry2)

        summary1 = await storage.get_daily_summary("user1", "2026-04-26")
        summary2 = await storage.get_daily_summary("user2", "2026-04-26")

        assert summary1.entry_count == 1
        assert summary1.total_calories == 100
        assert summary2.entry_count == 1
        assert summary2.total_calories == 150

    @pytest.mark.asyncio
    async def test_update_entry_no_fields_returns_false(self, storage) -> None:
        """Should return False when no fields provided for update."""
        entry = CalorieEntry(
            user_id="user1",
            description="apple",
            calories=100,
            logged_at="2026-04-26T10:00:00",
            logged_date="2026-04-26",
        )
        entry_id = await storage.add_entry(entry)

        result = await storage.update_entry(entry_id, "user1")

        assert result is False


class TestGetStorage:
    """Tests for get_storage function."""

    @pytest.mark.asyncio
    async def test_get_storage_raises_when_not_initialized(self, conn, lock) -> None:
        """Should raise RuntimeError when storage is not initialized."""
        from blacki.calories.storage import get_storage
        from blacki.container import (
            reset_container_for_tests,
            set_container_from_connection,
        )

        set_container_from_connection(conn, lock)

        with pytest.raises(RuntimeError, match="Calorie storage not initialized"):
            get_storage()

        reset_container_for_tests()

    @pytest.mark.asyncio
    async def test_get_storage_returns_storage_when_initialized(
        self, conn, lock
    ) -> None:
        """Should return storage when initialized."""
        from blacki.calories.storage import get_storage
        from blacki.container import (
            reset_container_for_tests,
            set_container_from_connection,
        )

        container = set_container_from_connection(conn, lock)
        await container.calorie_storage.initialize()

        result = get_storage()

        assert result is container.calorie_storage

        reset_container_for_tests()
