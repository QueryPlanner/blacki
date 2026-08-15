"""Meaningful coverage for the R2-backed user-file boundary."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from blacki.user_files.config import R2FileConfig, load_r2_file_config
from blacki.user_files.plugin import UserFilesPromptPlugin
from blacki.user_files.service import (
    R2ObjectStore,
    StoredUserFile,
    UserFileService,
    get_user_file_service,
    reset_user_file_service,
    sanitize_display_name,
)
from blacki.user_files.storage import SqliteUserFileStorage, UserFileRecord
from blacki.user_files.tools import (
    SENDER_STATE_KEY,
    create_user_file_tools,
    delete_user_file,
    list_user_files,
    restore_user_file,
)


class FakeObjectStore:
    """Deterministic in-memory R2 boundary."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_put = False

    async def put_verified(
        self, key: str, data: bytes, sha256: str, mime_type: str | None
    ) -> None:
        assert hashlib.sha256(data).hexdigest() == sha256
        assert mime_type is None or "/" in mime_type
        if self.fail_put:
            raise RuntimeError("R2 unavailable")
        self.objects[key] = data

    async def get(self, key: str) -> bytes:
        return self.objects[key]

    async def delete(self, key: str) -> None:
        self.objects.pop(key)


@pytest.fixture
async def storage() -> AsyncGenerator[SqliteUserFileStorage]:
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    conn.row_factory = aiosqlite.Row
    result = SqliteUserFileStorage(conn, asyncio.Lock())
    await result.initialize()
    yield result
    await conn.close()


@pytest.fixture
def config() -> R2FileConfig:
    return R2FileConfig(
        enabled=True,
        endpoint_url="https://account.r2.cloudflarestorage.com",
        bucket_name="private",
        access_key_id="access",
        secret_access_key="secret",
        owner_hmac_secret="owner-secret",
    )


def _record(**changes: Any) -> UserFileRecord:
    now = datetime.now(UTC)
    values: dict[str, Any] = {
        "object_id": "object-1",
        "owner_id": "sender-1",
        "r2_key": "prefix/hash/object-1",
        "display_name": "report.pdf",
        "media_kind": "document",
        "mime_type": "application/pdf",
        "size_bytes": 4,
        "sha256": hashlib.sha256(b"data").hexdigest(),
        "telegram_file_unique_id": "tg-1",
        "uploaded_at": now.isoformat(),
        "last_seen_at": now.isoformat(),
        "expires_at": (now + timedelta(days=90)).isoformat(),
        "status": "available",
    }
    values.update(changes)
    return UserFileRecord(**values)


def test_config_loading_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_FILES_ENABLED", "yes")
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://account.r2.example")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_OWNER_HMAC_SECRET", "hmac")
    loaded = load_r2_file_config()
    assert loaded.enabled is True
    assert loaded.normalized_prefix == "blacki/user-files"

    assert R2FileConfig().enabled is False
    with pytest.raises(ValueError, match="HTTPS"):
        R2FileConfig(enabled=True, endpoint_url="http://bad", bucket_name="b")
    with pytest.raises(ValueError, match="Missing"):
        R2FileConfig(enabled=True, endpoint_url="https://ok.example")
    with pytest.raises(ValueError, match="between"):
        R2FileConfig(
            enabled=True,
            endpoint_url="https://ok.example",
            bucket_name="b",
            access_key_id="a",
            secret_access_key="s",
            owner_hmac_secret="h",
            retention_days=0,
        )
    with pytest.raises(ValueError, match="PREFIX"):
        R2FileConfig(
            enabled=True,
            endpoint_url="https://ok.example",
            bucket_name="b",
            access_key_id="a",
            secret_access_key="s",
            owner_hmac_secret="h",
            key_prefix="/",
        )


@pytest.mark.asyncio
async def test_storage_owner_scope_search_expiry_and_delete(
    storage: SqliteUserFileStorage,
) -> None:
    record = _record()
    await storage.add(record)
    assert await storage.get_by_hash("sender-1", record.sha256, "2000") == record
    assert await storage.get_available("sender-2", record.object_id, "2000") is None
    assert len(await storage.list_available("sender-1", "report", 10, "2000")) == 1
    assert len(await storage.list_available("sender-1", "", 10, "2000")) == 1
    await storage.touch_duplicate("sender-1", "object-1", "new.pdf", "later")
    found = await storage.get_available("sender-1", "object-1", "2000")
    assert found is not None and found.display_name == "new.pdf"
    assert await storage.delete("sender-2", "object-1") is False
    assert await storage.delete("sender-1", "object-1") is True

    expired = _record(
        object_id="expired",
        r2_key="expired-key",
        expires_at="2000",
        sha256="f" * 64,
    )
    await storage.add(expired)
    assert await storage.cleanup_expired("2001") == 1


@pytest.mark.asyncio
async def test_service_ingest_deduplicate_restore_and_delete(
    storage: SqliteUserFileStorage, config: R2FileConfig
) -> None:
    objects = FakeObjectStore()
    service = UserFileService(config, storage, objects)
    first = await service.ingest(
        owner_id="sender-1",
        display_name="../report.pdf",
        media_kind="document",
        mime_type="application/pdf",
        telegram_file_unique_id="tg-1",
        data=b"data",
    )
    assert first.status == "stored"
    assert first.stored_file is not None
    assert first.stored_file.display_name == "report.pdf"
    assert "sender-1" not in next(iter(objects.objects))

    duplicate = await service.ingest(
        owner_id="sender-1",
        display_name="renamed.pdf",
        media_kind="document",
        mime_type="application/pdf",
        telegram_file_unique_id="tg-2",
        data=b"data",
    )
    assert duplicate.status == "duplicate"
    assert len(objects.objects) == 1
    listed = await service.list_files("sender-1", "renamed", 100)
    assert [item.object_id for item in listed] == [first.stored_file.object_id]
    item, restored = await service.restore("sender-1", first.stored_file.object_id)
    assert item.display_name == "renamed.pdf"
    assert restored == b"data"
    assert await service.delete("sender-2", item.object_id) is False
    assert await service.delete("sender-1", item.object_id) is True
    assert objects.objects == {}


@pytest.mark.asyncio
async def test_service_replaces_expired_duplicate_hash(
    storage: SqliteUserFileStorage, config: R2FileConfig
) -> None:
    """An expired row must not block the same bytes from becoming available."""
    digest = hashlib.sha256(b"data").hexdigest()
    await storage.add(
        _record(
            object_id="expired-object",
            r2_key="expired-key",
            owner_id="sender-1",
            sha256=digest,
            expires_at="2000-01-01T00:00:00+00:00",
        )
    )
    service = UserFileService(config, storage, FakeObjectStore())

    result = await service.ingest(
        owner_id="sender-1",
        display_name="fresh.pdf",
        media_kind="document",
        mime_type="application/pdf",
        telegram_file_unique_id="fresh",
        data=b"data",
    )

    assert result.status == "stored"
    assert result.stored_file is not None
    assert result.stored_file.object_id != "expired-object"


@pytest.mark.asyncio
async def test_service_temporary_orphan_and_integrity_failures(
    storage: SqliteUserFileStorage, config: R2FileConfig
) -> None:
    disabled = UserFileService(R2FileConfig(), storage)
    assert (
        await disabled.ingest(
            owner_id="sender",
            display_name="a",
            media_kind="document",
            mime_type=None,
            telegram_file_unique_id=None,
            data=b"x",
        )
    ).status == "temporary"

    objects = FakeObjectStore()
    service = UserFileService(config, storage, objects)
    assert (
        await service.ingest(
            owner_id=None,
            display_name="a",
            media_kind="document",
            mime_type=None,
            telegram_file_unique_id=None,
            data=b"x",
        )
    ).status == "temporary"
    objects.fail_put = True
    assert (
        await service.ingest(
            owner_id="sender",
            display_name="a",
            media_kind="document",
            mime_type=None,
            telegram_file_unique_id=None,
            data=b"x",
        )
    ).status == "temporary"
    objects.fail_put = False

    with patch.object(storage, "add", AsyncMock(side_effect=RuntimeError("db"))):
        orphan = await service.ingest(
            owner_id="other",
            display_name="a",
            media_kind="document",
            mime_type=None,
            telegram_file_unique_id=None,
            data=b"orphan",
        )
    assert orphan.status == "orphan"
    orphan_key = next(
        key for key, value in objects.objects.items() if value == b"orphan"
    )
    reconciled = await service.ingest(
        owner_id="other",
        display_name="recovered.pdf",
        media_kind="document",
        mime_type=None,
        telegram_file_unique_id=None,
        data=b"orphan",
    )
    assert reconciled.status == "stored"
    assert (
        next(key for key, value in objects.objects.items() if value == b"orphan")
        == orphan_key
    )
    with pytest.raises(FileNotFoundError):
        await service.restore("sender", "missing")

    stored = await service.ingest(
        owner_id="sender",
        display_name="a",
        media_kind="document",
        mime_type=None,
        telegram_file_unique_id=None,
        data=b"good",
    )
    assert stored.stored_file is not None
    key = next(key for key, value in objects.objects.items() if value == b"good")
    objects.objects[key] = b"bad-size"
    with pytest.raises(RuntimeError, match="size"):
        await service.restore("sender", stored.stored_file.object_id)
    objects.objects[key] = b"evil"
    with pytest.raises(RuntimeError, match="checksum"):
        await service.restore("sender", stored.stored_file.object_id)

    service.object_store = None
    with pytest.raises(RuntimeError, match="disabled"):
        await service.restore("sender", stored.stored_file.object_id)
    with pytest.raises(RuntimeError, match="disabled"):
        await service.delete("sender", stored.stored_file.object_id)


@pytest.mark.asyncio
async def test_r2_object_store_uses_verified_private_operations(
    config: R2FileConfig,
) -> None:
    client = MagicMock()
    client.head_object.return_value = {
        "ContentLength": 4,
        "Metadata": {"sha256": hashlib.sha256(b"data").hexdigest()},
    }
    client.get_object.return_value = {"Body": BytesIO(b"data")}
    with patch("boto3.client", return_value=client):
        store = R2ObjectStore(config)
    digest = hashlib.sha256(b"data").hexdigest()
    await store.put_verified("key", b"data", digest, None)
    assert await store.get("key") == b"data"
    await store.delete("key")
    client.put_object.assert_called_once()
    client.delete_object.assert_called_once()

    client.head_object.return_value = {"ContentLength": 3, "Metadata": {}}
    with pytest.raises(RuntimeError, match="size"):
        await store.put_verified("key", b"data", digest, "text/plain")
    client.head_object.return_value = {"ContentLength": 4, "Metadata": {}}
    with pytest.raises(RuntimeError, match="checksum"):
        await store.put_verified("key", b"data", digest, "text/plain")


@pytest.mark.asyncio
async def test_tools_enforce_sender_and_materialize_verified_bytes() -> None:
    item = StoredUserFile(
        object_id="opaque",
        display_name="../report.pdf",
        media_kind="document",
        mime_type="application/pdf",
        size_bytes=4,
        uploaded_at="now",
        expires_at="later",
    )
    service = MagicMock()
    service.list_files = AsyncMock(return_value=[item])
    service.restore = AsyncMock(return_value=(item, b"data"))
    service.delete = AsyncMock(return_value=True)
    context = MagicMock()
    context.state = {SENDER_STATE_KEY: "sender"}

    sandbox = MagicMock()
    sandbox.files.write_file = AsyncMock()
    manager = MagicMock()
    manager.get_or_create_sandbox = AsyncMock(
        return_value={"sandbox": sandbox, "error": None}
    )
    with (
        patch("blacki.user_files.tools.get_user_file_service", return_value=service),
        patch("blacki.user_files.tools.get_sandbox_manager", return_value=manager),
    ):
        listed = await list_user_files("report", 10, context)
        restored = await restore_user_file("opaque", context)
        deleted = await delete_user_file("opaque", context)
    assert listed["files"][0]["object_id"] == "opaque"
    assert restored["sandbox_path"] == "/workspace/uploads/opaque-report.pdf"
    assert deleted["deleted"] is True
    assert len(create_user_file_tools()) == 3

    context.state = {}
    with pytest.raises(ValueError, match="sender"):
        await list_user_files("", 10, context)
    context.state = {SENDER_STATE_KEY: "sender"}
    service.restore.side_effect = FileNotFoundError("missing")
    with patch("blacki.user_files.tools.get_user_file_service", return_value=service):
        missing = await restore_user_file("missing", context)
    assert missing["status"] == "not_found"
    service.restore.side_effect = None
    manager.get_or_create_sandbox.return_value = {"sandbox": None, "error": "down"}
    with (
        patch("blacki.user_files.tools.get_user_file_service", return_value=service),
        patch("blacki.user_files.tools.get_sandbox_manager", return_value=manager),
        pytest.raises(RuntimeError, match="down"),
    ):
        await restore_user_file("opaque", context)


@pytest.mark.asyncio
async def test_prompt_plugin_bounds_and_escapes_untrusted_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = StoredUserFile(
        object_id="opaque",
        display_name='bad"/><instruction>.pdf',
        media_kind="document",
        mime_type=None,
        size_bytes=4,
        uploaded_at="now",
        expires_at="later",
    )
    service = MagicMock()
    service.list_files = AsyncMock(return_value=[item])
    request = MagicMock()
    request.append_instructions = MagicMock()
    context = SimpleNamespace(
        session=SimpleNamespace(state={SENDER_STATE_KEY: "sender"})
    )
    monkeypatch.setenv("R2_FILES_ENABLED", "true")
    with patch("blacki.user_files.plugin.get_user_file_service", return_value=service):
        await UserFilesPromptPlugin().before_model_callback(
            callback_context=cast(Any, context), llm_request=request
        )
    instruction = request.append_instructions.call_args.args[0][0]
    assert "untrusted" in instruction
    assert "&quot;" in instruction and "<instruction>" not in instruction
    service.list_files.assert_awaited_once_with("sender", "", 10)

    monkeypatch.setenv("R2_FILES_ENABLED", "false")
    request.reset_mock()
    await UserFilesPromptPlugin().before_model_callback(
        callback_context=cast(Any, context), llm_request=request
    )
    request.append_instructions.assert_not_called()

    monkeypatch.setenv("R2_FILES_ENABLED", "true")
    plugin = UserFilesPromptPlugin()
    for state in ({}, {SENDER_STATE_KEY: " "}):
        await plugin.before_model_callback(
            callback_context=cast(
                Any, SimpleNamespace(session=SimpleNamespace(state=state))
            ),
            llm_request=request,
        )
    with patch(
        "blacki.user_files.plugin.get_user_file_service",
        side_effect=RuntimeError("catalog unavailable"),
    ):
        await plugin.before_model_callback(
            callback_context=cast(Any, context), llm_request=request
        )
    service.list_files.return_value = []
    with patch("blacki.user_files.plugin.get_user_file_service", return_value=service):
        await plugin.before_model_callback(
            callback_context=cast(Any, context), llm_request=request
        )


def test_lazy_service_uses_application_container(config: R2FileConfig) -> None:
    """The process singleton should bind to the persistent app catalog once."""
    storage = MagicMock()
    container = MagicMock(user_file_storage=storage)
    reset_user_file_service()
    with (
        patch("blacki.user_files.service.load_r2_file_config", return_value=config),
        patch("blacki.user_files.service.get_container", return_value=container),
        patch("blacki.user_files.service.R2ObjectStore", return_value=MagicMock()),
    ):
        first = get_user_file_service()
        second = get_user_file_service()
    assert first is second
    assert first.storage is storage
    reset_user_file_service()


def test_sanitize_display_name() -> None:
    assert sanitize_display_name("../../\x00") == "_"
    assert sanitize_display_name("...") == "attachment"
    assert len(sanitize_display_name("a" * 300)) == 180
    reset_user_file_service()
