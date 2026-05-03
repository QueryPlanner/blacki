# mypy: disable-error-code="no-untyped-def"
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from google.adk.tools import ToolContext

from blacki.calories.storage import DailySummary
from blacki.calories.tools import (
    delete_meal,
    edit_meal,
    get_calorie_summary,
    log_meal,
    set_calorie_goal,
)


@pytest.fixture
def mock_tool_context():
    mock_context = create_autospec(ToolContext, spec_set=True, instance=True)
    mock_context.state = {}
    mock_context.user_id = "user1"
    return mock_context


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
async def test_log_meal_success(
    mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.add_entry.return_value = 1
    mock_storage.get_daily_summary.return_value = DailySummary(
        date="2026-04-26", total_calories=500, entry_count=1
    )

    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = 2000

    result = await log_meal(
        mock_tool_context, description="apple", estimated_calories=95, meal_type="snack"
    )

    assert result["status"] == "success"
    assert result["entry_id"] == 1
    assert result["daily_total"] == 500
    assert result["remaining"] == 1500

    mock_storage.add_entry.assert_called_once()
    entry = mock_storage.add_entry.call_args[0][0]
    assert entry.description == "apple"
    assert entry.calories == 95
    assert entry.meal_type == "snack"


@pytest.mark.asyncio
async def test_log_meal_validation(mock_tool_context) -> None:
    result = await log_meal(mock_tool_context, "apple", -10)
    assert result["status"] == "error"

    result = await log_meal(mock_tool_context, "", 100)
    assert result["status"] == "error"

    result = await log_meal(mock_tool_context, "apple", 100, meal_type="invalid")
    assert result["status"] == "error"


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
async def test_get_calorie_summary_single_day(
    mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.get_daily_summary.return_value = DailySummary(
        date="2026-04-26", total_calories=500, entry_count=1
    )

    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = 2000

    result = await get_calorie_summary(mock_tool_context, date="today", days=1)

    assert result["status"] == "success"
    assert result["calorie_goal"] == 2000
    assert result["summary"]["total_calories"] == 500


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
async def test_get_calorie_summary_multi_day(
    mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.get_date_range_summary.return_value = [
        DailySummary(date="2026-04-26", total_calories=500, entry_count=1),
        DailySummary(date="2026-04-25", total_calories=2000, entry_count=3),
    ]

    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = 2000

    result = await get_calorie_summary(mock_tool_context, date="today", days=7)

    assert result["status"] == "success"
    assert result["calorie_goal"] == 2000
    assert len(result["summaries"]) == 2


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_edit_meal(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.update_entry.return_value = True

    result = await edit_meal(mock_tool_context, entry_id=1, estimated_calories=200)

    assert result["status"] == "success"
    mock_storage.update_entry.assert_called_once_with(1, "user1", calories=200)


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_delete_meal(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.delete_entry.return_value = True

    result = await delete_meal(mock_tool_context, entry_id=1)

    assert result["status"] == "success"
    mock_storage.delete_entry.assert_called_once_with(1, "user1")


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_preferences_storage")
async def test_set_calorie_goal(mock_get_pref, mock_tool_context) -> None:
    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref

    result = await set_calorie_goal(mock_tool_context, daily_calories=2500)

    assert result["status"] == "success"
    assert result["new_goal"] == 2500
    mock_pref.set.assert_called_once_with("user1", "calorie_goal", 2500)


@pytest.mark.asyncio
async def test_set_calorie_goal_validation(mock_tool_context) -> None:
    result = await set_calorie_goal(mock_tool_context, daily_calories=100)
    assert result["status"] == "error"

    result = await set_calorie_goal(mock_tool_context, daily_calories=20000)
    assert result["status"] == "error"


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
async def test_log_meal_with_past_date(
    mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.add_entry.return_value = 1
    mock_storage.get_daily_summary.return_value = DailySummary(
        date="2026-04-25", total_calories=500, entry_count=1
    )

    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = 2000

    result = await log_meal(
        mock_tool_context,
        description="apple",
        estimated_calories=95,
        date="yesterday",
    )

    assert result["status"] == "success"
    entry = mock_storage.add_entry.call_args[0][0]
    assert entry.logged_date != "2026-05-03"


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
async def test_log_meal_with_specific_date(
    mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.add_entry.return_value = 1
    mock_storage.get_daily_summary.return_value = DailySummary(
        date="2026-04-20", total_calories=500, entry_count=1
    )

    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = 2000

    result = await log_meal(
        mock_tool_context,
        description="apple",
        estimated_calories=95,
        date="2026-04-20",
    )

    assert result["status"] == "success"
    entry = mock_storage.add_entry.call_args[0][0]
    assert entry.logged_date == "2026-04-20"


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_edit_meal_with_logged_date(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.update_entry.return_value = True

    result = await edit_meal(mock_tool_context, entry_id=1, logged_date="yesterday")

    assert result["status"] == "success"
    call_kwargs = mock_storage.update_entry.call_args[1]
    assert "logged_date" in call_kwargs
    assert call_kwargs["logged_date"] != "2026-05-03"


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_edit_meal_with_specific_logged_date(
    mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.update_entry.return_value = True

    result = await edit_meal(mock_tool_context, entry_id=1, logged_date="2026-04-15")

    assert result["status"] == "success"
    call_kwargs = mock_storage.update_entry.call_args[1]
    assert call_kwargs["logged_date"] == "2026-04-15"
