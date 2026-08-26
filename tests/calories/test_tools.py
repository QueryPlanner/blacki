# mypy: disable-error-code="no-untyped-def"
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from google.adk.tools import ToolContext
from pydantic import ValidationError

from blacki.calories.storage import DailySummary
from blacki.calories.tools import (
    _is_private_tool_context,
    _meal_saved_message,
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
@pytest.mark.parametrize("operation", ["log", "summary", "edit"])
async def test_invalid_dates_do_not_mutate_or_default_to_today(
    operation: str, mock_tool_context
) -> None:
    """An unparseable explicit date should return an error before storage."""
    invalid_date = "definitely-not-a-real-date"
    if operation == "log":
        result = await log_meal(mock_tool_context, "apple", 95, date=invalid_date)
    elif operation == "summary":
        result = await get_calorie_summary(mock_tool_context, date=invalid_date)
    else:
        result = await edit_meal(mock_tool_context, entry_id=1, date=invalid_date)

    assert result["status"] == "error"
    assert "Could not understand date" in result["message"]


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
        date="2026-04-25",
    )

    assert result["status"] == "success"
    entry = mock_storage.add_entry.call_args[0][0]
    assert entry.logged_date == "2026-04-25"


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
async def test_edit_meal_with_date(mock_get_storage, mock_tool_context) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.update_entry.return_value = True

    result = await edit_meal(mock_tool_context, entry_id=1, date="2026-04-15")

    assert result["status"] == "success"
    call_kwargs = mock_storage.update_entry.call_args[1]
    assert "logged_date" in call_kwargs
    assert call_kwargs["logged_date"] == "2026-04-15"


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_edit_meal_with_specific_date(
    mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.update_entry.return_value = True

    result = await edit_meal(mock_tool_context, entry_id=1, date="2026-04-15")

    assert result["status"] == "success"
    call_kwargs = mock_storage.update_entry.call_args[1]
    assert call_kwargs["logged_date"] == "2026-04-15"


# Tests for exception handling
@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_log_meal_exception(mock_get_storage, mock_tool_context) -> None:
    """Test log_meal handles unexpected exceptions."""
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.add_entry.side_effect = RuntimeError("Database error")

    result = await log_meal(
        mock_tool_context, description="apple", estimated_calories=95
    )

    assert result["status"] == "error"
    assert "unexpected error" in result["message"].lower()


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_edit_meal_exception(mock_get_storage, mock_tool_context) -> None:
    """Test edit_meal handles unexpected exceptions."""
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.update_entry.side_effect = RuntimeError("Database error")

    result = await edit_meal(mock_tool_context, entry_id=1, estimated_calories=200)

    assert result["status"] == "error"
    assert "unexpected error" in result["message"].lower()


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_delete_meal_exception(mock_get_storage, mock_tool_context) -> None:
    """Test delete_meal handles unexpected exceptions."""
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.delete_entry.side_effect = RuntimeError("Database error")

    result = await delete_meal(mock_tool_context, entry_id=1)

    assert result["status"] == "error"
    assert "unexpected error" in result["message"].lower()


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
@patch("blacki.calories.tools.CalorieEntry")
async def test_log_meal_pydantic_validation_error(
    mock_entry_class, mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    """Test log_meal handles ValidationError from CalorieEntry creation."""
    mock_entry_class.side_effect = ValidationError.from_exception_data(
        "CalorieEntry", [{"type": "missing", "loc": ("description",), "input": {}}]
    )

    result = await log_meal(
        mock_tool_context, description="apple", estimated_calories=95
    )

    assert result["status"] == "error"
    assert "validation failed" in result["message"].lower()


# Tests for MealService dispatch (Google Health export enrollment)


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
@patch("blacki.calories.tools._try_get_meal_service")
async def test_log_meal_uses_meal_service_when_available(
    mock_try_get_service, mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    mock_service = AsyncMock()
    mock_service.mutate.return_value = (7, "pending")
    mock_try_get_service.return_value = mock_service

    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.get_daily_summary.return_value = DailySummary(
        date="2026-04-26", total_calories=500, entry_count=1
    )
    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = 2000
    mock_tool_context.state = {"telegram_chat_type": "private"}

    result = await log_meal(
        mock_tool_context, description="apple", estimated_calories=95
    )

    assert result["status"] == "success"
    assert result["entry_id"] == 7
    assert result["google_health_sync"] == "pending"
    assert "sync is pending" in result["message"]
    mock_service.mutate.assert_called_once()
    assert mock_service.mutate.call_args.kwargs["private"] is True


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
async def test_log_meal_summary_read_failure_still_succeeds(
    mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.add_entry.return_value = 1
    mock_storage.get_daily_summary.side_effect = RuntimeError("db unavailable")
    mock_pref = AsyncMock()
    mock_get_pref.return_value = mock_pref
    mock_pref.get.return_value = 2000

    result = await log_meal(
        mock_tool_context, description="apple", estimated_calories=95
    )

    assert result["status"] == "success"
    assert "daily summary is unavailable" in result["message"]
    assert "daily_total" not in result
    assert "remaining" not in result


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
@patch("blacki.calories.tools.get_preferences_storage")
async def test_log_meal_goal_read_failure_still_succeeds(
    mock_get_pref, mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.add_entry.return_value = 1
    mock_storage.get_daily_summary.return_value = DailySummary(
        date="2026-04-26", total_calories=500, entry_count=1
    )
    mock_get_pref.side_effect = RuntimeError("prefs unavailable")

    result = await log_meal(
        mock_tool_context, description="apple", estimated_calories=95
    )

    assert result["status"] == "success"
    assert "calorie_goal" not in result
    assert result["daily_total"] == 500
    assert "remaining" not in result


@pytest.mark.asyncio
async def test_edit_meal_rejects_empty_description(mock_tool_context) -> None:
    result = await edit_meal(mock_tool_context, entry_id=1, description="   ")
    assert result["status"] == "error"
    assert "description cannot be empty" in result["message"]


@pytest.mark.asyncio
async def test_edit_meal_rejects_nonpositive_calories(mock_tool_context) -> None:
    result = await edit_meal(mock_tool_context, entry_id=1, estimated_calories=0)
    assert result["status"] == "error"
    assert "estimated_calories must be > 0" in result["message"]


@pytest.mark.asyncio
async def test_edit_meal_rejects_invalid_macros(mock_tool_context) -> None:
    result = await edit_meal(mock_tool_context, entry_id=1, protein_g=-5.0)
    assert result["status"] == "error"
    assert "macros must be finite and nonnegative" in result["message"]


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_edit_meal_value_error_from_storage(
    mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.update_entry.side_effect = ValueError("owner mismatch")

    result = await edit_meal(mock_tool_context, entry_id=1, estimated_calories=200)

    assert result["status"] == "error"
    assert result["message"] == "owner mismatch"


@pytest.mark.asyncio
@patch("blacki.calories.tools._try_get_meal_service")
async def test_edit_meal_uses_meal_service_when_available(
    mock_try_get_service, mock_tool_context
) -> None:
    mock_service = AsyncMock()
    mock_service.mutate.return_value = (1, "pending")
    mock_try_get_service.return_value = mock_service

    result = await edit_meal(mock_tool_context, entry_id=1, estimated_calories=200)

    assert result["status"] == "success"
    assert result["google_health_sync"] == "pending"
    mock_service.mutate.assert_called_once_with(
        "user1",
        private=False,
        entry_id=1,
        updates={"calories": 200},
    )


@pytest.mark.asyncio
@patch("blacki.calories.tools.get_storage")
async def test_delete_meal_value_error_from_storage(
    mock_get_storage, mock_tool_context
) -> None:
    mock_storage = AsyncMock()
    mock_get_storage.return_value = mock_storage
    mock_storage.delete_entry.side_effect = ValueError("owner mismatch")

    result = await delete_meal(mock_tool_context, entry_id=1)

    assert result["status"] == "error"
    assert result["message"] == "owner mismatch"


@pytest.mark.asyncio
@patch("blacki.calories.tools._try_get_meal_service")
async def test_delete_meal_uses_meal_service_when_available(
    mock_try_get_service, mock_tool_context
) -> None:
    mock_service = AsyncMock()
    mock_service.mutate.return_value = (1, "not_enabled")
    mock_try_get_service.return_value = mock_service

    result = await delete_meal(mock_tool_context, entry_id=1)

    assert result["status"] == "success"
    assert result["google_health_sync"] == "not_enabled"
    mock_service.mutate.assert_called_once_with("user1", private=False, entry_id=1)


def test_is_private_tool_context_true_for_private_chat() -> None:
    ctx = SimpleNamespace(state={"telegram_chat_type": "private"})
    assert _is_private_tool_context(cast(ToolContext, ctx)) is True


def test_is_private_tool_context_false_for_group_chat() -> None:
    ctx = SimpleNamespace(state={"telegram_chat_type": "group"})
    assert _is_private_tool_context(cast(ToolContext, ctx)) is False


def test_is_private_tool_context_false_without_state() -> None:
    ctx = SimpleNamespace()
    assert _is_private_tool_context(cast(ToolContext, ctx)) is False


def test_is_private_tool_context_false_when_state_has_no_getter() -> None:
    ctx = SimpleNamespace(state=object())
    assert _is_private_tool_context(cast(ToolContext, ctx)) is False


def test_meal_saved_message_pending() -> None:
    message = _meal_saved_message("Logged", "pending")
    assert message == "Logged Saved in Blacki; Google Health sync is pending."


def test_meal_saved_message_authorization_required() -> None:
    message = _meal_saved_message("Logged", "authorization_required")
    assert message == "Logged Saved in Blacki; reconnect Google Health to sync it."


def test_meal_saved_message_failed() -> None:
    message = _meal_saved_message("Logged", "failed")
    assert message == (
        "Logged Saved in Blacki; Google Health sync failed and will retry."
    )


def test_meal_saved_message_not_enabled() -> None:
    assert _meal_saved_message("Logged", "not_enabled") == "Logged Saved in Blacki."
