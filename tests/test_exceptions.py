"""Tests for custom exception classes."""

from typing import cast

import pytest

from blacki.utils.exceptions import (
    ConfigurationError,
    InvalidTokenFormatError,
    MissingAPIKeyError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderPermissionError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    TelegramAuthError,
)


class TestConfigurationError:
    def test_basic_error_message(self) -> None:
        error = ConfigurationError("Test error message")
        assert str(error) == "Test error message"
        assert error.docs_url is None

    def test_error_with_docs_url(self) -> None:
        docs_url = "https://example.com/docs"
        error = ConfigurationError("Test error", docs_url=docs_url)
        assert str(error) == "Test error"
        assert error.docs_url == docs_url


class TestTelegramAuthError:
    def test_default_error_message(self) -> None:
        error = TelegramAuthError()
        message = str(error)
        assert "Telegram authentication failed" in message
        assert "@BotFather" in message
        assert error.docs_url is not None

    def test_custom_detail(self) -> None:
        error = TelegramAuthError("Custom error detail")
        message = str(error)
        assert "Custom error detail" in message
        assert "@BotFather" in message


class TestMissingAPIKeyError:
    def test_error_message(self) -> None:
        error = MissingAPIKeyError()
        message = str(error)
        assert "No LLM API key found" in message
        assert "OPENAI_API_KEY" in message
        assert "GOOGLE_API_KEY" in message
        assert error.docs_url is not None


class TestInvalidTokenFormatError:
    def test_error_message(self) -> None:
        error = InvalidTokenFormatError("TEST_TOKEN", "expected-format")
        message = str(error)
        assert "Invalid TEST_TOKEN format" in message
        assert "expected-format" in message


class TestProviderError:
    def test_message_lists_status_code_and_request_id(self) -> None:
        error = ProviderError(status=418, code="teapot", request_id="req_abc-123.v2:x")
        assert error.provider == "openai"
        assert error.status == 418
        assert error.code == "teapot"
        assert error.request_id == "req_abc-123.v2:x"
        assert str(error) == (
            "openai request failed (HTTP 418, code teapot, request req_abc-123.v2:x)."
        )

    def test_message_without_details_has_no_parentheses(self) -> None:
        assert str(ProviderError()) == "openai request failed."

    def test_provider_name_is_configurable(self) -> None:
        error = ProviderError(provider="other")
        assert error.provider == "other"
        assert str(error) == "other request failed."

    @pytest.mark.parametrize(
        "value",
        [
            "invalid value 'PROMPT TEXT'",
            "sk-looks-like-a-key with spaces",
            "x" * 129,
            "",
        ],
    )
    def test_free_text_is_dropped_from_code_and_request_id(self, value: str) -> None:
        error = ProviderError(status=400, code=value, request_id=value)
        assert error.code is None
        assert error.request_id is None
        assert str(error) == "openai request failed (HTTP 400)."

    def test_non_string_code_is_dropped(self) -> None:
        # The SDK can return a non-string code, so the guard is kept at runtime.
        error = ProviderError(status=400, code=cast("str", 500), request_id=None)
        assert error.code is None
        assert str(error) == "openai request failed (HTTP 400)."

    @pytest.mark.parametrize(
        ("error_type", "status", "summary"),
        [
            (ProviderAuthenticationError, 401, "authentication failed"),
            (ProviderPermissionError, 403, "permission denied"),
            (ProviderRateLimitError, 429, "rate limit exceeded"),
            (ProviderRequestError, 400, "rejected the request"),
            (ProviderServerError, 503, "server or network error"),
        ],
    )
    def test_subclasses_share_the_base_contract(
        self, error_type: type[ProviderError], status: int, summary: str
    ) -> None:
        error = error_type(status=status, code="some_code")
        assert isinstance(error, ProviderError)
        assert str(error) == f"openai {summary} (HTTP {status}, code some_code)."
