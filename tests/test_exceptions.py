"""Tests for custom exception classes."""


from blacki.utils.exceptions import (
    ConfigurationError,
    InvalidTokenFormatError,
    MissingAPIKeyError,
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
        assert "OPENROUTER_API_KEY" in message
        assert "GOOGLE_API_KEY" in message
        assert error.docs_url is not None


class TestInvalidTokenFormatError:
    def test_error_message(self) -> None:
        error = InvalidTokenFormatError("TEST_TOKEN", "expected-format")
        message = str(error)
        assert "Invalid TEST_TOKEN format" in message
        assert "expected-format" in message
