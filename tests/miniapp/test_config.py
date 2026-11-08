from __future__ import annotations

import pytest

from blacki.miniapp.config import MiniAppConfig, MiniAppConfigurationError


def test_from_environment_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIAPP_ENABLED", raising=False)
    assert MiniAppConfig.from_environment() is None

    monkeypatch.setenv("MINIAPP_ENABLED", "false")
    assert MiniAppConfig.from_environment() is None

    monkeypatch.setenv("MINIAPP_ENABLED", "0")
    assert MiniAppConfig.from_environment() is None


def test_from_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIAPP_ENABLED", "true")
    monkeypatch.delenv("MINIAPP_PORT", raising=False)
    monkeypatch.delenv("MINIAPP_HOST", raising=False)
    monkeypatch.delenv("MINIAPP_PUBLIC_URL", raising=False)
    monkeypatch.delenv("MINIAPP_DEV_TELEGRAM_USER_ID", raising=False)
    monkeypatch.delenv("MINIAPP_INIT_DATA_MAX_AGE_SECONDS", raising=False)

    cfg = MiniAppConfig.from_environment()
    assert cfg is not None
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8090
    assert cfg.public_url is None
    assert cfg.dev_telegram_user_id is None
    assert cfg.init_data_max_age_seconds == 900
    assert cfg.session_ttl_seconds == 7200
    assert not cfg.dev_shortcut_enabled
    assert cfg.launch_url is None


def test_from_environment_custom_and_dev_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINIAPP_ENABLED", "1")
    monkeypatch.setenv("MINIAPP_HOST", "0.0.0.0")  # noqa: S104
    monkeypatch.setenv("MINIAPP_PORT", "8888")
    monkeypatch.setenv("MINIAPP_INIT_DATA_MAX_AGE_SECONDS", "300")
    monkeypatch.setenv("MINIAPP_DEV_TELEGRAM_USER_ID", "123456")
    monkeypatch.delenv("MINIAPP_PUBLIC_URL", raising=False)

    cfg = MiniAppConfig.from_environment()
    assert cfg is not None
    assert cfg.host == "0.0.0.0"  # noqa: S104
    assert cfg.port == 8888
    assert cfg.init_data_max_age_seconds == 300
    assert cfg.dev_telegram_user_id == 123456
    assert cfg.dev_shortcut_enabled is True
    assert cfg.launch_url is None


def test_public_url_and_launch_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIAPP_ENABLED", "yes")
    monkeypatch.setenv("MINIAPP_PUBLIC_URL", "https://miniapp.example.com/")
    monkeypatch.setenv("MINIAPP_DEV_TELEGRAM_USER_ID", "123456")

    cfg = MiniAppConfig.from_environment()
    assert cfg is not None
    assert cfg.public_url == "https://miniapp.example.com"
    assert cfg.launch_url == "https://miniapp.example.com/miniapp/"
    # dev shortcut disabled when public_url is configured
    assert cfg.dev_shortcut_enabled is False


@pytest.mark.parametrize(
    "env_vars,err_msg",
    [
        ({"MINIAPP_PORT": "invalid"}, "MINIAPP_PORT must be an integer"),
        ({"MINIAPP_PORT": "80"}, "MINIAPP_PORT must be between 1024 and 65535"),
        ({"MINIAPP_PORT": "70000"}, "MINIAPP_PORT must be between 1024 and 65535"),
        (
            {"MINIAPP_INIT_DATA_MAX_AGE_SECONDS": "50"},
            "MINIAPP_INIT_DATA_MAX_AGE_SECONDS must be between 60 and 86400",
        ),
        (
            {"MINIAPP_INIT_DATA_MAX_AGE_SECONDS": "90000"},
            "MINIAPP_INIT_DATA_MAX_AGE_SECONDS must be between 60 and 86400",
        ),
        (
            {"MINIAPP_PUBLIC_URL": "http://insecure.com"},
            "MINIAPP_PUBLIC_URL must be an https URL",
        ),
        ({"MINIAPP_PUBLIC_URL": "https://"}, "MINIAPP_PUBLIC_URL must be an https URL"),
        (
            {"MINIAPP_PUBLIC_URL": "https://user:pass@example.com"},
            "MINIAPP_PUBLIC_URL can't contain credentials, a query or a fragment",
        ),
        (
            {"MINIAPP_PUBLIC_URL": "https://example.com?query=1"},
            "MINIAPP_PUBLIC_URL can't contain credentials, a query or a fragment",
        ),
        (
            {"MINIAPP_PUBLIC_URL": "https://example.com#frag"},
            "MINIAPP_PUBLIC_URL can't contain credentials, a query or a fragment",
        ),
    ],
)
def test_validation_errors(
    monkeypatch: pytest.MonkeyPatch, env_vars: dict[str, str], err_msg: str
) -> None:
    monkeypatch.setenv("MINIAPP_ENABLED", "true")
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(MiniAppConfigurationError, match=err_msg):
        MiniAppConfig.from_environment()
