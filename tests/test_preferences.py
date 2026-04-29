# mypy: disable-error-code="no-untyped-def"
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blacki.utils.preferences import (
    PostgresPreferencesStorage,
    close_preferences_storage,
    get_preferences_storage,
    init_preferences_storage,
)


@pytest.fixture
def mock_pool():
    pool = MagicMock()

    conn = AsyncMock()
    conn.execute = AsyncMock()

    pool.acquire.return_value.__aenter__.return_value = conn
    pool.execute = AsyncMock()
    pool.fetchrow = AsyncMock()

    return pool


@pytest.fixture
async def preferences_storage(mock_pool):
    storage = PostgresPreferencesStorage(mock_pool)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(mock_pool) -> None:
    storage = PostgresPreferencesStorage(mock_pool)
    await storage.initialize()

    conn = mock_pool.acquire.return_value.__aenter__.return_value
    conn.execute.assert_called()
    assert storage._schema_ready is True


@pytest.mark.asyncio
async def test_get_existing(preferences_storage, mock_pool) -> None:
    mock_pool.fetchrow.return_value = {"value": '{"monday": "push"}'}

    result = await preferences_storage.get("user1", "workout_split")

    assert result == {"monday": "push"}
    mock_pool.fetchrow.assert_called_once_with(
        "SELECT value FROM user_preferences WHERE user_id = $1 AND key = $2",
        "user1",
        "workout_split",
    )


@pytest.mark.asyncio
async def test_get_not_found_returns_default(preferences_storage, mock_pool) -> None:
    mock_pool.fetchrow.return_value = None

    result = await preferences_storage.get("user1", "calorie_goal", 2000)

    assert result == 2000


@pytest.mark.asyncio
async def test_set(preferences_storage, mock_pool) -> None:
    mock_pool.execute.return_value = "INSERT 0 1"

    with patch("blacki.utils.preferences.now_utc") as mock_now:
        mock_now.return_value.isoformat.return_value = "2026-04-26T12:00:00"
        await preferences_storage.set("user1", "calorie_goal", 2500)

    mock_pool.execute.assert_called_once()
    args = mock_pool.execute.call_args[0]
    assert args[1] == "user1"
    assert args[2] == "calorie_goal"
    assert args[3] == "2500"
    assert args[4] == "2026-04-26T12:00:00"


@pytest.mark.asyncio
async def test_delete_success(preferences_storage, mock_pool) -> None:
    mock_pool.execute.return_value = "DELETE 1"

    result = await preferences_storage.delete("user1", "calorie_goal")

    assert result is True
    mock_pool.execute.assert_called_once_with(
        "DELETE FROM user_preferences WHERE user_id = $1 AND key = $2",
        "user1",
        "calorie_goal",
    )


@pytest.mark.asyncio
async def test_delete_not_found(preferences_storage, mock_pool) -> None:
    mock_pool.execute.return_value = "DELETE 0"

    result = await preferences_storage.delete("user1", "calorie_goal")

    assert result is False


@pytest.mark.asyncio
async def test_singleton(mock_pool) -> None:
    # Ensure it raises before init
    # Need to clear global first since other tests might have run
    import blacki.utils.preferences as prefs

    prefs._storage = None

    with pytest.raises(RuntimeError):
        get_preferences_storage()

    storage = await init_preferences_storage(mock_pool)
    assert get_preferences_storage() is storage

    await close_preferences_storage()
    with pytest.raises(RuntimeError):
        get_preferences_storage()


@pytest.mark.asyncio
async def test_reinit_preferences_storage_closes_existing(mock_pool) -> None:
    """init_preferences_storage closes existing storage before replacing."""
    import blacki.utils.preferences as prefs

    existing = PostgresPreferencesStorage(mock_pool)
    existing.close = AsyncMock()  # type: ignore[method-assign]
    prefs._storage = existing

    new = await init_preferences_storage(mock_pool)

    existing.close.assert_awaited_once()
    assert prefs._storage is new

    prefs._storage = None
