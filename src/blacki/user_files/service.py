"""R2-backed durable file service."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from blacki.container import get_container

from .config import R2FileConfig, load_r2_file_config
from .storage import SqliteUserFileStorage, UserFileRecord

logger = logging.getLogger(__name__)
_service: UserFileService | None = None
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
MAX_DISPLAY_NAME_CHARS = 180


@dataclass(frozen=True, slots=True)
class StoredUserFile:
    """Safe file metadata exposed to Blacki and the model."""

    object_id: str
    display_name: str
    media_kind: str
    mime_type: str | None
    size_bytes: int
    uploaded_at: str
    expires_at: str


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of one durable storage attempt."""

    stored_file: StoredUserFile | None
    status: str
    warning: str | None = None


def sanitize_display_name(value: str) -> str:
    """Return a bounded filename safe for metadata and sandbox paths."""
    name = _CONTROL_CHARS.sub("_", Path(value).name).strip().strip(".")
    if not name:
        name = "attachment"
    return name[:MAX_DISPLAY_NAME_CHARS]


class R2ObjectStore:
    """Small asynchronous facade over the blocking boto3 S3 client."""

    def __init__(self, config: R2FileConfig) -> None:
        import boto3

        self._bucket = config.bucket_name
        self._client = boto3.client(
            service_name="s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name="auto",
        )

    async def put_verified(
        self, key: str, data: bytes, sha256: str, mime_type: str | None
    ) -> None:
        """Upload an object and verify its size and private checksum metadata."""
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=mime_type or "application/octet-stream",
            Metadata={"sha256": sha256},
        )
        head = await asyncio.to_thread(
            self._client.head_object, Bucket=self._bucket, Key=key
        )
        if int(head.get("ContentLength", -1)) != len(data):
            raise RuntimeError("R2 object size verification failed")
        metadata = head.get("Metadata") or {}
        if not hmac.compare_digest(str(metadata.get("sha256", "")), sha256):
            raise RuntimeError("R2 object checksum verification failed")

    async def get(self, key: str) -> bytes:
        """Download an object fully into host memory."""
        response = await asyncio.to_thread(
            self._client.get_object, Bucket=self._bucket, Key=key
        )
        body = response["Body"]
        return await asyncio.to_thread(body.read)

    async def delete(self, key: str) -> None:
        """Delete one exact object key."""
        await asyncio.to_thread(
            self._client.delete_object, Bucket=self._bucket, Key=key
        )


class ObjectStore(Protocol):
    """Structural boundary implemented by R2 and deterministic test fakes."""

    async def put_verified(
        self, key: str, data: bytes, sha256: str, mime_type: str | None
    ) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...


class UserFileService:
    """Coordinate the SQLite catalog and private R2 object storage."""

    def __init__(
        self,
        config: R2FileConfig,
        storage: SqliteUserFileStorage,
        object_store: ObjectStore | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.object_store = object_store or (
            R2ObjectStore(config) if config.enabled else None
        )
        self._ingest_locks: dict[str, asyncio.Lock] = {}

    async def ingest(
        self,
        *,
        owner_id: str | None,
        display_name: str,
        media_kind: str,
        mime_type: str | None,
        telegram_file_unique_id: str | None,
        data: bytes,
    ) -> IngestResult:
        """Persist bytes for one authenticated Telegram sender."""
        if not self.config.enabled or self.object_store is None:
            return IngestResult(None, "temporary", "R2 file storage is disabled.")
        if owner_id is None or not owner_id.strip():
            return IngestResult(
                None,
                "temporary",
                "This attachment has no Telegram sender identity and was not saved.",
            )
        safe_name = sanitize_display_name(display_name)
        digest = hashlib.sha256(data).hexdigest()
        lock_key = f"{owner_id}:{digest}"
        lock = self._ingest_locks.setdefault(lock_key, asyncio.Lock())
        try:
            async with lock:
                return await self._ingest_locked(
                    owner_id=owner_id,
                    display_name=safe_name,
                    media_kind=media_kind,
                    mime_type=mime_type,
                    telegram_file_unique_id=telegram_file_unique_id,
                    data=data,
                    digest=digest,
                )
        finally:
            if not lock.locked():
                self._ingest_locks.pop(lock_key, None)

    async def _ingest_locked(
        self,
        *,
        owner_id: str,
        display_name: str,
        media_kind: str,
        mime_type: str | None,
        telegram_file_unique_id: str | None,
        data: bytes,
        digest: str,
    ) -> IngestResult:
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        existing = await self.storage.get_by_hash(owner_id, digest, now_iso)
        if existing is not None:
            await self.storage.touch_duplicate(
                owner_id, existing.object_id, display_name, now_iso
            )
            return IngestResult(self._public(existing, display_name), "duplicate")

        object_id = hmac.new(
            self.config.owner_hmac_secret.encode(),
            f"{owner_id}:{digest}".encode(),
            hashlib.sha256,
        ).hexdigest()[:32]
        owner_hash = hmac.new(
            self.config.owner_hmac_secret.encode(),
            owner_id.encode(),
            hashlib.sha256,
        ).hexdigest()
        key = f"{self.config.normalized_prefix}/{owner_hash}/{object_id}"
        expires_at = now + timedelta(days=self.config.retention_days)
        try:
            object_store = self.object_store
            if object_store is None:  # pragma: no cover - guarded by ingest()
                raise RuntimeError("R2 file storage is disabled")
            await object_store.put_verified(key, data, digest, mime_type)
        except Exception:
            logger.exception("Failed to store Telegram attachment in R2")
            return IngestResult(
                None,
                "temporary",
                "R2 storage failed; this attachment is available only temporarily.",
            )

        record = UserFileRecord(
            object_id=object_id,
            owner_id=owner_id,
            r2_key=key,
            display_name=display_name,
            media_kind=media_kind,
            mime_type=mime_type,
            size_bytes=len(data),
            sha256=digest,
            telegram_file_unique_id=telegram_file_unique_id,
            uploaded_at=now_iso,
            last_seen_at=now_iso,
            expires_at=expires_at.isoformat(),
        )
        try:
            await self.storage.add(record)
        except Exception:
            logger.exception("R2 object stored but catalog insertion failed")
            return IngestResult(
                None,
                "orphan",
                "The object reached R2 but could not be added to your file catalog.",
            )
        return IngestResult(self._public(record), "stored")

    async def list_files(
        self, owner_id: str, query: str, limit: int
    ) -> list[StoredUserFile]:
        """List available files for one authenticated owner."""
        bounded_limit = max(1, min(limit, 50))
        now_iso = datetime.now(UTC).isoformat()
        await self.storage.cleanup_expired(now_iso)
        records = await self.storage.list_available(
            owner_id, query, bounded_limit, now_iso
        )
        return [self._public(record) for record in records]

    async def restore(
        self, owner_id: str, object_id: str
    ) -> tuple[StoredUserFile, bytes]:
        """Resolve and verify one owner-scoped object."""
        now_iso = datetime.now(UTC).isoformat()
        record = await self.storage.get_available(owner_id, object_id, now_iso)
        if record is None:
            raise FileNotFoundError("No available file matches that object ID")
        if self.object_store is None:
            raise RuntimeError("R2 file storage is disabled")
        data = await self.object_store.get(record.r2_key)
        if len(data) != record.size_bytes:
            raise RuntimeError("Restored file size does not match the catalog")
        digest = hashlib.sha256(data).hexdigest()
        if not hmac.compare_digest(digest, record.sha256):
            raise RuntimeError("Restored file checksum does not match the catalog")
        return self._public(record), data

    async def delete(self, owner_id: str, object_id: str) -> bool:
        """Delete one exact owner-scoped object and its catalog entry."""
        now_iso = datetime.now(UTC).isoformat()
        record = await self.storage.get_available(owner_id, object_id, now_iso)
        if record is None:
            return False
        if self.object_store is None:
            raise RuntimeError("R2 file storage is disabled")
        await self.object_store.delete(record.r2_key)
        return await self.storage.delete(owner_id, object_id)

    @staticmethod
    def _public(
        record: UserFileRecord, display_name: str | None = None
    ) -> StoredUserFile:
        return StoredUserFile(
            object_id=record.object_id,
            display_name=display_name or record.display_name,
            media_kind=record.media_kind,
            mime_type=record.mime_type,
            size_bytes=record.size_bytes,
            uploaded_at=record.uploaded_at,
            expires_at=record.expires_at,
        )


def get_user_file_service() -> UserFileService:
    """Return the process-wide service backed by the application container."""
    global _service
    if _service is None:
        config = load_r2_file_config()
        storage = get_container().user_file_storage
        _service = UserFileService(config, storage)
    return _service


def reset_user_file_service() -> None:
    """Clear the lazy service between application lifecycles or tests."""
    global _service
    _service = None
