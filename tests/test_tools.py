"""Unit tests for custom tools."""

import logging
import uuid
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import mock classes from conftest
from conftest import MockState, MockToolContext
from google.adk.tools import ToolContext
from pydantic import BaseModel

from blacki.tools import (
    _get_shared_browser_use_client,
    _pending_schemas,
    _pending_tasks,
    _poll_task_output,
    _serialize_browser_output,
    browser_get_task_status,
    browser_list_profiles,
    browser_stop_session,
    browser_task,
    example_tool,
    reset_browser_use_client_cache,
)


@pytest.fixture(autouse=True)
def clear_pending_tasks() -> None:
    """Clear pending task state before each test."""
    _pending_tasks.clear()
    _pending_schemas.clear()


class TestExampleTool:
    """Tests for the example_tool function."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    def test_example_tool_returns_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool returns success status and message."""
        caplog.set_level(logging.INFO)

        state = MockState({"user_id": "test_user", "session_key": "value"})
        tool_context = cast(ToolContext, MockToolContext(state=state))

        result = example_tool(tool_context)

        assert result["status"] == "success"
        assert result["message"] == "Successfully used example_tool."

    def test_example_tool_logs_state_keys(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool logs session state keys."""
        caplog.set_level(logging.INFO)

        state = MockState({"key1": "value1", "key2": "value2"})
        tool_context = cast(ToolContext, MockToolContext(state=state))

        example_tool(tool_context)

        assert "Session state keys:" in caplog.text
        assert "Successfully used example_tool." in caplog.text

        info_records = [r for r in caplog.records if r.levelname == "INFO"]
        assert len(info_records) == 2

    def test_example_tool_with_empty_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that example_tool handles empty state correctly."""
        caplog.set_level(logging.INFO)

        result = example_tool(self._tool_context())

        assert result["status"] == "success"
        assert result["message"] == "Successfully used example_tool."

        assert "Session state keys:" in caplog.text


class _SampleOutputModel(BaseModel):
    """Minimal model for browser_task structured-output tests."""

    title: str


class TestGetSharedBrowserUseClient:
    """Tests for ``_get_shared_browser_use_client``."""

    @pytest.mark.asyncio
    async def test_returns_same_client_for_same_api_key(self) -> None:
        """Same API key should return cached client."""
        await reset_browser_use_client_cache()

        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        with patch(
            "blacki.tools.AsyncBrowserUse", return_value=mock_client
        ) as mock_cls:
            client1 = await _get_shared_browser_use_client("key1")
            mock_cls.assert_called_once_with(api_key="key1")

            mock_cls.reset_mock()
            client2 = await _get_shared_browser_use_client("key1")
            mock_cls.assert_not_called()

            assert client1 is client2

        await reset_browser_use_client_cache()

    @pytest.mark.asyncio
    async def test_creates_new_client_on_api_key_change(self) -> None:
        """Different API key should close old client and create new one."""
        await reset_browser_use_client_cache()

        old_client = MagicMock()
        old_client.close = AsyncMock()
        new_client = MagicMock()
        new_client.close = AsyncMock()

        with patch("blacki.tools.AsyncBrowserUse") as mock_cls:
            mock_cls.side_effect = [old_client, new_client]

            client1 = await _get_shared_browser_use_client("key1")
            assert client1 is old_client

            client2 = await _get_shared_browser_use_client("key2")
            assert client2 is new_client

            old_client.close.assert_awaited_once()

        await reset_browser_use_client_cache()

    @pytest.mark.asyncio
    async def test_handles_close_error_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Error closing old client should be logged, not raised."""
        await reset_browser_use_client_cache()
        caplog.set_level(logging.DEBUG)

        old_client = MagicMock()
        old_client.close = AsyncMock(side_effect=RuntimeError("close failed"))
        new_client = MagicMock()
        new_client.close = AsyncMock()

        with patch("blacki.tools.AsyncBrowserUse") as mock_cls:
            mock_cls.side_effect = [old_client, new_client]

            client = await _get_shared_browser_use_client("key1")
            await _get_shared_browser_use_client("key2")

            assert "Error closing Browser Use client" in caplog.text

        await reset_browser_use_client_cache()


class TestResetBrowserUseClientCache:
    """Tests for ``reset_browser_use_client_cache``."""

    @pytest.mark.asyncio
    async def test_resets_global_state(self) -> None:
        """Cache should be cleared after reset."""
        mock_client = MagicMock()
        mock_client.close = AsyncMock()

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await _get_shared_browser_use_client("key1")

        await reset_browser_use_client_cache()

        from blacki.tools import _browser_use_client, _browser_use_api_key

        assert _browser_use_client is None
        assert _browser_use_api_key is None

    @pytest.mark.asyncio
    async def test_handles_close_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Error during close should be logged, not raised."""
        caplog.set_level(logging.DEBUG)

        mock_client = MagicMock()
        mock_client.close = AsyncMock(side_effect=RuntimeError("close error"))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await _get_shared_browser_use_client("key1")

        await reset_browser_use_client_cache()

        assert "Error while closing shared Browser Use client" in caplog.text

    """Tests for ``_serialize_browser_output`` (Browser Use result shaping)."""

    def test_serializes_list_of_models(self) -> None:
        """Lists of Pydantic models become JSON-ready dict lists."""
        rows = [_SampleOutputModel(title="a"), _SampleOutputModel(title="b")]
        assert _serialize_browser_output(rows) == [
            {"title": "a"},
            {"title": "b"},
        ]

    def test_serializes_dict_with_nested_model(self) -> None:
        """Dict values that are models are serialized recursively."""
        payload = {"item": _SampleOutputModel(title="x"), "n": 1}
        assert _serialize_browser_output(payload) == {"item": {"title": "x"}, "n": 1}

    def test_returns_none_for_none_input(self) -> None:
        """None input should return None."""
        assert _serialize_browser_output(None) is None

    def test_returns_non_json_string_as_is(self) -> None:
        """Non-JSON strings are returned unchanged."""
        assert _serialize_browser_output("plain text") == "plain text"

    def test_parses_json_string(self) -> None:
        """JSON strings are parsed and returned as dict."""
        assert _serialize_browser_output('{"key": "value"}') == {"key": "value"}

    def test_returns_non_serializable_as_is(self) -> None:
        """Non-serializable values like int are returned unchanged."""
        assert _serialize_browser_output(42) == 42


class TestPollTaskOutput:
    """Tests for ``_poll_task_output``."""

    @pytest.mark.asyncio
    async def test_returns_result_on_finished_status(self) -> None:
        """Finished task should return result immediately."""
        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.output = '{"result": "success"}'

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(return_value=mock_task)

        result = await _poll_task_output(
            mock_tasks_client, "task-123", None, timeout=1.0, interval=0.1
        )

        assert result.task is mock_task
        assert result.output == '{"result": "success"}'

    @pytest.mark.asyncio
    async def test_returns_result_on_failed_status(self) -> None:
        """Failed task should return result immediately."""
        mock_task = MagicMock()
        mock_task.status.value = "failed"
        mock_task.output = None

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(return_value=mock_task)

        result = await _poll_task_output(
            mock_tasks_client, "task-123", None, timeout=1.0, interval=0.1
        )

        assert result.task.status.value == "failed"

    @pytest.mark.asyncio
    async def test_returns_result_on_stopped_status(self) -> None:
        """Stopped task should return result immediately."""
        mock_task = MagicMock()
        mock_task.status.value = "stopped"
        mock_task.output = None

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(return_value=mock_task)

        result = await _poll_task_output(
            mock_tasks_client, "task-123", None, timeout=1.0, interval=0.1
        )

        assert result.task.status.value == "stopped"

    @pytest.mark.asyncio
    async def test_parses_json_output(self) -> None:
        """Output should be parsed as JSON when output_schema provided."""
        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.output = '{"title": "test"}'

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(return_value=mock_task)

        result = await _poll_task_output(
            mock_tasks_client,
            "task-123",
            _SampleOutputModel,
            timeout=1.0,
            interval=0.1,
        )

        assert isinstance(result.output, _SampleOutputModel)
        assert result.output.title == "test"

    @pytest.mark.asyncio
    async def test_handles_invalid_json_output(self) -> None:
        """Invalid JSON output should be returned as-is when schema provided."""
        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.output = "not valid json"

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(return_value=mock_task)

        result = await _poll_task_output(
            mock_tasks_client,
            "task-123",
            _SampleOutputModel,
            timeout=1.0,
            interval=0.1,
        )

        assert result.output == "not valid json"

    @pytest.mark.asyncio
    async def test_handles_schema_validation_error(self) -> None:
        """Schema validation failure should return raw output."""
        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.output = '{"wrong_field": "value"}'

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(return_value=mock_task)

        result = await _poll_task_output(
            mock_tasks_client,
            "task-123",
            _SampleOutputModel,
            timeout=1.0,
            interval=0.1,
        )

        assert result.output == '{"wrong_field": "value"}'

    @pytest.mark.asyncio
    async def test_raises_timeout_on_non_terminal_status(self) -> None:
        """TimeoutError should be raised if task doesn't complete in time."""
        mock_task_running = MagicMock()
        mock_task_running.status.value = "created"

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(return_value=mock_task_running)

        with pytest.raises(TimeoutError, match="did not complete within"):
            await _poll_task_output(
                mock_tasks_client,
                "task-123",
                None,
                timeout=0.2,
                interval=0.1,
            )

    @pytest.mark.asyncio
    async def test_polls_until_terminal_status(self) -> None:
        """Should poll until task reaches terminal status."""
        mock_task_created = MagicMock()
        mock_task_created.status.value = "created"

        mock_task_finished = MagicMock()
        mock_task_finished.status.value = "finished"
        mock_task_finished.output = None

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(
            side_effect=[mock_task_created, mock_task_finished]
        )

        result = await _poll_task_output(
            mock_tasks_client, "task-123", None, timeout=1.0, interval=0.01
        )

        assert result.task.status.value == "finished"
        assert mock_tasks_client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_handles_none_output(self) -> None:
        """None output should be returned as-is."""
        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.output = None

        mock_tasks_client = MagicMock()
        mock_tasks_client.get = AsyncMock(return_value=mock_task)

        result = await _poll_task_output(
            mock_tasks_client,
            "task-123",
            _SampleOutputModel,
            timeout=1.0,
            interval=0.1,
        )

        assert result.output is None


class TestBrowserTask:
    """Tests for Browser Use Cloud ``browser_task``."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_browser_task_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return a clear error payload."""
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await browser_task("open example.com", tool_context)

        assert result["status"] == "error"
        assert result["task_id"] is None
        assert "BROWSER_USE_API_KEY" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_browser_task_empty_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace-only task should be rejected."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        result = await browser_task("  \n  ", tool_context)

        assert result["status"] == "error"
        assert "non-empty" in (result.get("error") or "").lower()

    @pytest.mark.asyncio
    async def test_browser_task_invalid_output_schema_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject unsupported ``output_schema`` types."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        result = await browser_task(
            "do something",
            tool_context,
            output_schema="not-valid",
        )

        assert result["status"] == "error"
        assert "output_schema" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_browser_task_returns_pending_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """browser_task should return immediately with pending status."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_session = MagicMock()
        mock_session.live_url = "https://live.example/session"

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=mock_session)

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_task("open example.com", tool_context)

        assert result["status"] == "pending"
        assert result["task_id"] == str(tid)
        assert result["session_id"] == str(sid)
        assert result["live_preview_url"] == "https://live.example/session"
        assert "browser_get_task_status" in result.get("message", "")

        assert str(tid) in _pending_tasks
        assert _pending_tasks[str(tid)]["session_id"] == str(sid)

    @pytest.mark.asyncio
    async def test_browser_task_passes_keep_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """keep_alive=True should be passed to create."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("open example.com", tool_context, keep_alive=True)

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("keepAlive") is True

    @pytest.mark.asyncio
    async def test_browser_task_passes_session_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """session_id should be passed to create for follow-up tasks."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("continue task", tool_context, session_id=str(sid))

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("session_id") == str(sid)

    @pytest.mark.asyncio
    async def test_browser_task_passes_profile_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """profile_id should be passed via session_settings."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()
        profile_uuid = str(uuid.uuid4())

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("login to site", tool_context, profile_id=profile_uuid)

        _args, kwargs = mock_client.tasks.create.call_args
        session_settings = kwargs.get("session_settings")
        assert session_settings is not None
        assert str(session_settings.profile_id) == profile_uuid

    @pytest.mark.asyncio
    async def test_browser_task_passes_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """model parameter should override default LLM."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task(
                "open example.com", tool_context, model="claude-sonnet-4.6"
            )

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("llm") == "claude-sonnet-4.6"

    @pytest.mark.asyncio
    async def test_browser_task_passes_json_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dict output_schema should be sent as structured_output."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("extract x", tool_context, output_schema=schema)

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("structured_output") is not None
        assert "integer" in kwargs["structured_output"]

    @pytest.mark.asyncio
    async def test_browser_task_stores_pydantic_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pydantic model should be stored in _pending_schemas for later polling."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task(
                "get title",
                tool_context,
                output_schema=_SampleOutputModel,
            )

        assert str(tid) in _pending_schemas
        assert _pending_schemas[str(tid)] == _SampleOutputModel

    @pytest.mark.asyncio
    async def test_browser_task_passes_start_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """start_url should be passed to create."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task(
                "open example.com",
                tool_context,
                start_url="https://example.com",
            )

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("start_url") == "https://example.com"

    @pytest.mark.asyncio
    async def test_browser_task_passes_max_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_steps should be passed to create."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("open example.com", tool_context, max_steps=10)

        _args, kwargs = mock_client.tasks.create.call_args
        assert kwargs.get("max_steps") == 10

    @pytest.mark.asyncio
    async def test_browser_task_handles_live_url_exception(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception when getting live URL should be logged, not raised."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        caplog.set_level(logging.DEBUG)
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(side_effect=RuntimeError("session error"))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_task("open example.com", tool_context)

        assert result["status"] == "pending"
        assert result["live_preview_url"] is None
        assert "Failed to load Browser Use session" in caplog.text

    @pytest.mark.asyncio
    async def test_browser_task_handles_create_exception(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception during task creation should return error payload."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        caplog.set_level(logging.DEBUG)
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(side_effect=RuntimeError("create error"))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_task("open example.com", tool_context)

        assert result["status"] == "error"
        assert "create error" in result.get("error", "")
        assert "Browser Use Cloud task creation failed" in caplog.text

    @pytest.mark.asyncio
    async def test_browser_task_passes_proxy_country(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """proxy_country should be passed via session_settings."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        sid = uuid.uuid4()
        tid = uuid.uuid4()

        mock_created = MagicMock()
        mock_created.session_id = sid
        mock_created.id = tid

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock(return_value=mock_created)
        mock_client.sessions.get = AsyncMock(return_value=MagicMock(live_url=None))
        mock_client.close = AsyncMock()

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_task("open example.com", tool_context, proxy_country="us")

        _args, kwargs = mock_client.tasks.create.call_args
        session_settings = kwargs.get("session_settings")
        assert session_settings is not None
        assert session_settings.proxy_country_code.value == "us"


class TestBrowserGetTaskStatus:
    """Tests for ``browser_get_task_status``."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_get_status_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return error."""
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await browser_get_task_status("task-123", tool_context)

        assert result["status"] == "error"
        assert "BROWSER_USE_API_KEY" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_get_status_unknown_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown task_id should return error."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        result = await browser_get_task_status("unknown-task-id", tool_context)

        assert result["status"] == "error"
        assert "Unknown task_id" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_get_status_returns_running_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Timeout during poll should return running status."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        tid = str(uuid.uuid4())
        _pending_tasks[tid] = {
            "api_key": "bu_test",
            "session_id": "session-123",
            "live_preview_url": "https://live.example/session",
            "status": "pending",
        }
        _pending_schemas[tid] = None

        mock_client = MagicMock()
        mock_client.tasks.create = AsyncMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._poll_task_output",
                side_effect=TimeoutError("poll timeout"),
            ),
        ):
            result = await browser_get_task_status(tid, tool_context)

        assert result["status"] == "running"
        assert result["task_id"] == tid
        assert result["output"] is None
        assert "still running" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_get_status_returns_finished_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finished task should return output and clean up pending state."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        tid = str(uuid.uuid4())
        _pending_tasks[tid] = {
            "api_key": "bu_test",
            "session_id": "session-123",
            "live_preview_url": "https://live.example/session",
            "status": "pending",
        }
        _pending_schemas[tid] = None

        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.is_success = True

        mock_result = MagicMock()
        mock_result.task = mock_task
        mock_result.output = '{"ok": true}'

        mock_client = MagicMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._poll_task_output",
                new=AsyncMock(return_value=mock_result),
            ),
        ):
            result = await browser_get_task_status(tid, tool_context)

        assert result["status"] == "finished"
        assert result["output"] == {"ok": True}
        assert result["is_success"] is True
        assert tid not in _pending_tasks
        assert tid not in _pending_schemas

    @pytest.mark.asyncio
    async def test_get_status_uses_stored_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Poll should use stored Pydantic schema for typed output."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        tid = str(uuid.uuid4())
        _pending_tasks[tid] = {
            "api_key": "bu_test",
            "session_id": "session-123",
            "live_preview_url": None,
            "status": "pending",
        }
        _pending_schemas[tid] = _SampleOutputModel

        mock_task = MagicMock()
        mock_task.status.value = "finished"
        mock_task.is_success = True

        mock_result = MagicMock()
        mock_result.task = mock_task
        mock_result.output = _SampleOutputModel(title="test-title")

        mock_client = MagicMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._poll_task_output",
                new=AsyncMock(return_value=mock_result),
            ) as mock_poll,
        ):
            result = await browser_get_task_status(tid, tool_context)

        mock_poll.assert_awaited_once()
        call_args = mock_poll.call_args
        assert call_args[0][2] == _SampleOutputModel
        assert result["output"] == {"title": "test-title"}

    @pytest.mark.asyncio
    async def test_get_status_returns_running_status_for_non_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-terminal status should return output as None with message."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        tid = str(uuid.uuid4())
        _pending_tasks[tid] = {
            "api_key": "bu_test",
            "session_id": "session-123",
            "live_preview_url": None,
            "status": "pending",
        }
        _pending_schemas[tid] = None

        mock_task = MagicMock()
        mock_task.status.value = "started"
        mock_task.is_success = False

        mock_result = MagicMock()
        mock_result.task = mock_task
        mock_result.output = None

        mock_client = MagicMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._poll_task_output",
                new=AsyncMock(return_value=mock_result),
            ),
        ):
            result = await browser_get_task_status(tid, tool_context)

        assert result["status"] == "started"
        assert result["output"] is None
        assert "Task is started" in result.get("message", "")
        assert tid in _pending_tasks

    @pytest.mark.asyncio
    async def test_get_status_handles_poll_exception(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception during polling should return error payload."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        caplog.set_level(logging.DEBUG)
        tool_context = self._tool_context()

        tid = str(uuid.uuid4())
        _pending_tasks[tid] = {
            "api_key": "bu_test",
            "session_id": "session-123",
            "live_preview_url": None,
            "status": "pending",
        }
        _pending_schemas[tid] = None

        mock_client = MagicMock()

        with (
            patch("blacki.tools.AsyncBrowserUse", return_value=mock_client),
            patch(
                "blacki.tools._poll_task_output",
                new=AsyncMock(side_effect=RuntimeError("poll error")),
            ),
        ):
            result = await browser_get_task_status(tid, tool_context)

        assert result["status"] == "error"
        assert "poll error" in result.get("error", "")
        assert "Failed to get browser task status" in caplog.text


class TestBrowserStopSession:
    """Tests for ``browser_stop_session``."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_stop_session_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return error."""
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await browser_stop_session("session-123", tool_context)

        assert result["status"] == "error"
        assert "BROWSER_USE_API_KEY" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_stop_session_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: delete session and return success."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.sessions.delete = AsyncMock(return_value=None)

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_stop_session("session-123", tool_context)

        assert result["status"] == "success"
        assert result["session_id"] == "session-123"
        mock_client.sessions.delete.assert_awaited_once_with("session-123")

    @pytest.mark.asyncio
    async def test_stop_session_handles_exception(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception during session stop should return error payload."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        caplog.set_level(logging.DEBUG)
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.sessions.delete = AsyncMock(
            side_effect=RuntimeError("delete error")
        )

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_stop_session("session-123", tool_context)

        assert result["status"] == "error"
        assert "delete error" in result.get("error", "")
        assert "Failed to stop Browser Use session" in caplog.text


class TestBrowserListProfiles:
    """Tests for ``browser_list_profiles``."""

    @staticmethod
    def _tool_context() -> ToolContext:
        return cast(ToolContext, MockToolContext(state=MockState({})))

    @pytest.mark.asyncio
    async def test_list_profiles_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without API key, return error with empty profiles."""
        monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
        tool_context = self._tool_context()

        result = await browser_list_profiles(tool_context)

        assert result["status"] == "error"
        assert result["profiles"] == []
        assert "BROWSER_USE_API_KEY" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_list_profiles_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: return list of profiles."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        mock_profile = MagicMock()
        mock_profile.id = uuid.uuid4()
        mock_profile.name = "test-profile"

        mock_response = MagicMock()
        mock_response.profiles = [mock_profile]

        mock_client = MagicMock()
        mock_client.profiles.list = AsyncMock(return_value=mock_response)

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_list_profiles(tool_context)

        assert result["status"] == "success"
        assert len(result["profiles"]) == 1
        assert result["profiles"][0]["name"] == "test-profile"

    @pytest.mark.asyncio
    async def test_list_profiles_with_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Query parameter should be passed to API."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        tool_context = self._tool_context()

        mock_response = MagicMock()
        mock_response.profiles = []

        mock_client = MagicMock()
        mock_client.profiles.list = AsyncMock(return_value=mock_response)

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            await browser_list_profiles(tool_context, query="my-profile")

        mock_client.profiles.list.assert_awaited_once_with(query="my-profile")

    @pytest.mark.asyncio
    async def test_list_profiles_handles_exception(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Exception during profile list should return error payload."""
        monkeypatch.setenv("BROWSER_USE_API_KEY", "bu_test")
        caplog.set_level(logging.DEBUG)
        tool_context = self._tool_context()

        mock_client = MagicMock()
        mock_client.profiles.list = AsyncMock(side_effect=RuntimeError("list error"))

        with patch("blacki.tools.AsyncBrowserUse", return_value=mock_client):
            result = await browser_list_profiles(tool_context)

        assert result["status"] == "error"
        assert result["profiles"] == []
        assert "list error" in result.get("error", "")
        assert "Failed to list Browser Use profiles" in caplog.text
