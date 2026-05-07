"""Tests for startup validation functions."""

import pytest

from blacki.utils.exceptions import (
    InvalidTokenFormatError,
    MissingAPIKeyError,
    TelegramAuthError,
)
from blacki.utils.validation import (
    validate_api_keys,
    validate_configuration,
    validate_telegram_token,
)


class TestValidateTelegramToken:
    def test_valid_token(self) -> None:
        validate_telegram_token("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")

    def test_missing_token(self) -> None:
        with pytest.raises(TelegramAuthError, match="TELEGRAM_BOT_TOKEN is not set"):
            validate_telegram_token(None)

    def test_empty_token(self) -> None:
        with pytest.raises(TelegramAuthError):
            validate_telegram_token("")

    def test_invalid_format_no_colon(self) -> None:
        with pytest.raises(
            InvalidTokenFormatError, match="Invalid TELEGRAM_BOT_TOKEN format"
        ):
            validate_telegram_token("123456ABCDEF")

    def test_invalid_format_wrong_prefix(self) -> None:
        with pytest.raises(InvalidTokenFormatError):
            validate_telegram_token("ABC:123456")

    def test_invalid_format_special_chars(self) -> None:
        with pytest.raises(InvalidTokenFormatError):
            validate_telegram_token("123456:ABC@DEF")


class TestValidateApiKeys:
    def test_has_openrouter_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        validate_api_keys()

    def test_has_google_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        validate_api_keys()

    def test_has_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        validate_api_keys()

    def test_no_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="No LLM API key found"):
            validate_api_keys()


class TestValidateConfiguration:
    def test_valid_configuration_with_telegram(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ROOT_AGENT_MODEL", raising=False)

        warnings = validate_configuration(
            telegram_enabled=True,
            telegram_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        )
        assert len(warnings) == 1
        assert "ROOT_AGENT_MODEL" in warnings[0]

    def test_valid_configuration_without_telegram(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("ROOT_AGENT_MODEL", "openrouter/google/gemini-2.0-flash-001")

        warnings = validate_configuration(
            telegram_enabled=False,
            telegram_token=None,
        )
        assert len(warnings) == 0

    def test_telegram_enabled_but_no_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(TelegramAuthError, match="TELEGRAM_BOT_TOKEN is not set"):
            validate_configuration(telegram_enabled=True, telegram_token=None)

    def test_telegram_enabled_with_invalid_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(InvalidTokenFormatError):
            validate_configuration(telegram_enabled=True, telegram_token="invalid")

    def test_no_api_keys_with_telegram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError):
            validate_configuration(
                telegram_enabled=True,
                telegram_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            )
