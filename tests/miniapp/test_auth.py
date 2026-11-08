from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from blacki.miniapp.auth import (
    MAX_SESSIONS,
    InitDataError,
    SessionStore,
    verify_init_data,
)


def _make_init_data(
    fields: dict[str, str],
    bot_token: str,
    *,
    custom_hash: str | None = None,
    raw_query: str | None = None,
) -> str:
    if raw_query is not None:
        return raw_query
    sorted_fields = sorted(fields.items())
    check_string = "\n".join(f"{k}={v}" for k, v in sorted_fields)
    secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    sig = (
        custom_hash
        or hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    )
    return "&".join(f"{k}={v}" for k, v in sorted_fields) + f"&hash={sig}"


def test_verify_init_data_success() -> None:
    token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"  # noqa: S105
    now = int(time.time())
    user_json = json.dumps({"id": 987654321, "first_name": "Test"})
    fields = {
        "auth_date": str(now),
        "query_id": "AAHdF6IQAAAAAN0XohCQ1T1n",
        "user": user_json,
    }
    init_data = _make_init_data(fields, token)
    user_id = verify_init_data(init_data, token, max_age_seconds=300)
    assert user_id == 987654321


@pytest.mark.parametrize(
    "bad_data,msg",
    [
        ("", "missing launch data"),
        ("a" * 4097, "missing launch data"),
        ("auth_date=123&hash=abc&auth_date=456", "duplicate launch data field"),
    ],
)
def test_verify_init_data_invalid_structure(bad_data: str, msg: str) -> None:
    with pytest.raises(InitDataError, match=msg):
        verify_init_data(bad_data, "token", max_age_seconds=300)


def test_verify_init_data_strict_parse_error() -> None:
    with pytest.raises(InitDataError, match="malformed launch data"):
        # parse_qsl strict_parsing=True raises ValueError on bad format
        verify_init_data("keywithoutvalue", "token", max_age_seconds=300)


def test_verify_init_data_missing_hash() -> None:
    init_data = "auth_date=12345&user=%7B%22id%22%3A1%7D"
    with pytest.raises(InitDataError, match="launch data is not signed"):
        verify_init_data(init_data, "token", max_age_seconds=300)


def test_verify_init_data_signature_mismatch() -> None:
    token = "my_bot_token"  # noqa: S105
    now = int(time.time())
    fields = {"auth_date": str(now), "user": json.dumps({"id": 123})}
    init_data = _make_init_data(fields, token, custom_hash="0" * 64)
    with pytest.raises(InitDataError, match="launch data signature does not match"):
        verify_init_data(init_data, token, max_age_seconds=300)


def test_verify_init_data_bad_auth_date() -> None:
    token = "my_bot_token"  # noqa: S105
    fields = {"auth_date": "not_an_int", "user": json.dumps({"id": 123})}
    init_data = _make_init_data(fields, token)
    with pytest.raises(InitDataError, match="launch data has no date"):
        verify_init_data(init_data, token, max_age_seconds=300)


@pytest.mark.parametrize(
    "auth_date_offset,msg",
    [
        (100, "launch data expired; reopen the app"),  # in the future > now + 60
        (-500, "launch data expired; reopen the app"),  # older than max_age_seconds=300
    ],
)
def test_verify_init_data_expired_or_future(auth_date_offset: int, msg: str) -> None:
    token = "my_bot_token"  # noqa: S105
    now = int(time.time())
    fields = {"auth_date": str(now + auth_date_offset), "user": json.dumps({"id": 123})}
    init_data = _make_init_data(fields, token)
    with pytest.raises(InitDataError, match=msg):
        verify_init_data(init_data, token, max_age_seconds=300)


@pytest.mark.parametrize(
    "user_str",
    [
        "not_json",
        "{}",
        json.dumps({"id": "not_an_id"}),
        json.dumps({"id": 0}),
        json.dumps({"id": -5}),
    ],
)
def test_verify_init_data_invalid_user(user_str: str) -> None:
    token = "my_bot_token"  # noqa: S105
    now = int(time.time())
    fields = {"auth_date": str(now), "user": user_str}
    init_data = _make_init_data(fields, token)
    with pytest.raises(InitDataError, match="launch data has no user"):
        verify_init_data(init_data, token, max_age_seconds=300)


def test_session_store_issue_and_resolve() -> None:
    store = SessionStore(ttl_seconds=100)
    token = store.issue(user_id=42)
    assert isinstance(token, str)
    assert store.resolve(token) == 42


def test_session_store_resolve_invalid_or_expired() -> None:
    store = SessionStore(ttl_seconds=100)
    assert store.resolve("") is None
    assert store.resolve("a" * 129) is None
    assert store.resolve("non_existent_token") is None

    # Expired token
    short_store = SessionStore(ttl_seconds=-10)
    expired_token = short_store.issue(user_id=99)
    assert short_store.resolve(expired_token) is None


def test_session_store_eviction_at_max_sessions() -> None:
    store = SessionStore(ttl_seconds=1000)
    tokens: list[str] = []
    for i in range(MAX_SESSIONS + 5):
        t = store.issue(user_id=1000 + i)
        tokens.append(t)

    assert len(store._sessions) == MAX_SESSIONS
    # The first few sessions should have been evicted
    assert store.resolve(tokens[0]) is None
    # The newest sessions should still exist
    assert store.resolve(tokens[-1]) == 1000 + MAX_SESSIONS + 4
