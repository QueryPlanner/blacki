"""Durable, user-scoped Telegram file storage."""

from .config import R2FileConfig, load_r2_file_config, user_files_enabled
from .plugin import UserFilesPromptPlugin
from .service import (
    IngestResult,
    StoredUserFile,
    UserFileService,
    get_user_file_service,
    reset_user_file_service,
)

__all__ = [
    "IngestResult",
    "R2FileConfig",
    "StoredUserFile",
    "UserFileService",
    "UserFilesPromptPlugin",
    "get_user_file_service",
    "load_r2_file_config",
    "reset_user_file_service",
    "user_files_enabled",
]
