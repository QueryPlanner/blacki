"""Sandbox image inspection and model-input integration."""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final

from google.adk.tools import ToolContext
from google.genai import types

from blacki.sandbox.manager import get_sandbox_manager

logger = logging.getLogger(__name__)

SANDBOX_WORKSPACE_ROOT: Final = PurePosixPath("/workspace")
MAX_IMAGE_BYTES: Final = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION: Final = 16_384
MAX_IMAGE_PIXELS: Final = 50_000_000
SUPPORTED_IMAGE_FORMATS: Final = ("PNG", "JPEG", "GIF", "WEBP", "BMP")

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8"
_GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
_WEBP_SIGNATURE = b"RIFF"
_BMP_SIGNATURE = b"BM"
_JPEG_FRAME_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
_JPEG_STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8)})


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Validated metadata for an image that is safe to send to a model."""

    format_name: str
    mime_type: str
    width: int
    height: int


def _validate_dimensions(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise ValueError("Image dimensions are invalid.")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"Image dimensions exceed the {MAX_IMAGE_DIMENSION}px limit.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Image pixel count exceeds the supported limit.")


def _metadata(
    format_name: str, mime_type: str, width: int, height: int
) -> ImageMetadata:
    _validate_dimensions(width, height)
    return ImageMetadata(format_name, mime_type, width, height)


def _parse_png(data: bytes) -> ImageMetadata:
    if len(data) < len(_PNG_SIGNATURE) + 12:
        raise ValueError("PNG header is incomplete.")

    position = len(_PNG_SIGNATURE)
    saw_header = False
    while position < len(data):
        if len(data) - position < 12:
            raise ValueError("PNG chunk is incomplete.")
        chunk_length = int.from_bytes(data[position : position + 4], "big")
        chunk_type_start = position + 4
        chunk_data_start = position + 8
        chunk_end = chunk_data_start + chunk_length + 4
        if chunk_end > len(data):
            raise ValueError("PNG chunk exceeds the file length.")
        chunk_type = data[chunk_type_start:chunk_data_start]
        chunk_data = data[chunk_data_start : chunk_data_start + chunk_length]
        expected_crc = int.from_bytes(
            data[chunk_data_start + chunk_length : chunk_end], "big"
        )
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("PNG checksum is invalid.")

        if not saw_header:
            if chunk_type != b"IHDR" or chunk_length != 13:
                raise ValueError("PNG header is invalid.")
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            image_metadata = _metadata("PNG", "image/png", width, height)
            saw_header = True
        elif chunk_type == b"IHDR":
            raise ValueError("PNG contains more than one header.")

        position = chunk_end
        if chunk_type == b"IEND":
            if chunk_length != 0 or not saw_header or position != len(data):
                raise ValueError("PNG end marker is invalid.")
            return image_metadata

    raise ValueError("PNG end marker is missing.")


def _parse_jpeg(data: bytes) -> ImageMetadata:
    if len(data) < 4:
        raise ValueError("JPEG header is incomplete.")

    position = 2
    image_metadata: ImageMetadata | None = None
    while position < len(data):
        if data[position] != 0xFF:
            raise ValueError("JPEG marker is invalid.")
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            raise ValueError("JPEG marker is incomplete.")
        marker = data[position]
        position += 1

        if marker == 0xD9:
            if image_metadata is None:
                raise ValueError("JPEG frame header is missing.")
            return image_metadata
        if marker == 0xDA:
            if position + 2 > len(data):
                raise ValueError("JPEG scan header is incomplete.")
            scan_length = int.from_bytes(data[position : position + 2], "big")
            if scan_length < 2 or position + scan_length > len(data):
                raise ValueError("JPEG scan header is invalid.")
            position += scan_length
            if image_metadata is None or data.find(b"\xff\xd9", position) < 0:
                raise ValueError("JPEG end marker is missing.")
            return image_metadata
        if marker in _JPEG_STANDALONE_MARKERS:
            continue

        if position + 2 > len(data):
            raise ValueError("JPEG segment length is missing.")
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            raise ValueError("JPEG segment length is invalid.")

        if marker in _JPEG_FRAME_MARKERS:
            if segment_length < 7:
                raise ValueError("JPEG frame header is incomplete.")
            height = int.from_bytes(data[position + 3 : position + 5], "big")
            width = int.from_bytes(data[position + 5 : position + 7], "big")
            image_metadata = _metadata("JPEG", "image/jpeg", width, height)
        position += segment_length

    raise ValueError("JPEG frame is incomplete.")


def _parse_gif(data: bytes) -> ImageMetadata:
    if len(data) < 14 or data[-1:] != b"\x3b":
        raise ValueError("GIF structure is invalid.")
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")
    return _metadata("GIF", "image/gif", width, height)


def _parse_webp(data: bytes) -> ImageMetadata:
    if len(data) < 20 or data[8:12] != b"WEBP":
        raise ValueError("WEBP header is invalid.")
    riff_end = int.from_bytes(data[4:8], "little") + 8
    if riff_end > len(data) or riff_end < 20:
        raise ValueError("WEBP container length is invalid.")

    position = 12
    while position + 8 <= riff_end:
        chunk_type = data[position : position + 4]
        chunk_length = int.from_bytes(data[position + 4 : position + 8], "little")
        chunk_data_start = position + 8
        chunk_end = chunk_data_start + chunk_length
        padded_end = chunk_end + (chunk_length & 1)
        if padded_end > riff_end:
            raise ValueError("WEBP chunk exceeds the file length.")
        chunk_data = data[chunk_data_start:chunk_end]

        if chunk_type == b"VP8X":
            if len(chunk_data) < 10:
                raise ValueError("WEBP extended header is incomplete.")
            width = int.from_bytes(chunk_data[4:7], "little") + 1
            height = int.from_bytes(chunk_data[7:10], "little") + 1
            return _metadata("WEBP", "image/webp", width, height)
        if chunk_type == b"VP8 " and len(chunk_data) >= 10:
            if chunk_data[3:6] != b"\x9d\x01\x2a":
                raise ValueError("WEBP lossy frame header is invalid.")
            width = int.from_bytes(chunk_data[6:8], "little") & 0x3FFF
            height = int.from_bytes(chunk_data[8:10], "little") & 0x3FFF
            return _metadata("WEBP", "image/webp", width, height)
        if chunk_type == b"VP8L" and len(chunk_data) >= 5:
            if chunk_data[0] != 0x2F:
                raise ValueError("WEBP lossless frame header is invalid.")
            dimensions = int.from_bytes(chunk_data[1:5], "little")
            width = (dimensions & 0x3FFF) + 1
            height = ((dimensions >> 14) & 0x3FFF) + 1
            return _metadata("WEBP", "image/webp", width, height)

        position = padded_end

    raise ValueError("WEBP image frame is missing.")


def _parse_bmp(data: bytes) -> ImageMetadata:
    if len(data) < 26:
        raise ValueError("BMP header is incomplete.")
    declared_size = int.from_bytes(data[2:6], "little")
    if declared_size and declared_size > len(data):
        raise ValueError("BMP file length is invalid.")
    pixel_offset = int.from_bytes(data[10:14], "little")
    if pixel_offset >= len(data):
        raise ValueError("BMP pixel data is missing.")

    dib_size = int.from_bytes(data[14:18], "little")
    if dib_size == 12:
        width = int.from_bytes(data[18:20], "little")
        height = int.from_bytes(data[20:22], "little")
    elif dib_size >= 40:
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = abs(int.from_bytes(data[22:26], "little", signed=True))
    else:
        raise ValueError("BMP DIB header is unsupported.")
    return _metadata("BMP", "image/bmp", width, height)


def _inspect_image_bytes(data: bytes) -> ImageMetadata:
    """Validate a bounded image and return its provider-neutral metadata."""
    if not isinstance(data, bytes) or not data:
        raise ValueError("Image file is empty or unreadable.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image file exceeds the {MAX_IMAGE_BYTES} byte limit.")

    if data.startswith(_PNG_SIGNATURE):
        return _parse_png(data)
    if data.startswith(_JPEG_SIGNATURE):
        return _parse_jpeg(data)
    if data.startswith(_GIF_SIGNATURES):
        return _parse_gif(data)
    if data.startswith(_WEBP_SIGNATURE):
        return _parse_webp(data)
    if data.startswith(_BMP_SIGNATURE):
        return _parse_bmp(data)
    formats = ", ".join(SUPPORTED_IMAGE_FORMATS)
    raise ValueError(f"Unsupported image format. Supported formats: {formats}.")


def _normalize_sandbox_path(path: str) -> str:
    """Resolve a user path inside the active sandbox workspace only."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Image path must be a non-empty sandbox path.")
    candidate = path.strip()
    if "\\" in candidate:
        raise ValueError("Image path must use POSIX separators.")

    raw_path = PurePosixPath(candidate)
    if ".." in raw_path.parts:
        raise ValueError("Image path cannot contain parent-directory traversal.")
    normalized = (
        raw_path if raw_path.is_absolute() else SANDBOX_WORKSPACE_ROOT / raw_path
    )
    try:
        relative_path = normalized.relative_to(SANDBOX_WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("Image path must be inside /workspace.") from exc
    if not relative_path.parts:
        raise ValueError("Image path must identify a file inside /workspace.")
    return normalized.as_posix()


def _error_result(error: str, sandbox_path: str | None = None) -> dict[str, Any]:
    return {
        "status": "error",
        "error": error,
        "sandbox_path": sandbox_path,
    }


async def sandbox_view_image(
    path: str, tool_context: ToolContext
) -> dict[str, Any] | list[types.Part]:
    """Read one image from the active sandbox and attach it to the next model turn.

    ``path`` may be relative to ``/workspace`` or an absolute path below
    ``/workspace``. Call this once for each image that needs visual inspection,
    including images restored from a durable Telegram attachment. For a durable
    attachment reference, call ``restore_user_file`` first and pass its returned
    sandbox path. The image is validated and kept as a separate visual input;
    its bytes are not returned as base64 text. This tool is read-only.
    """
    try:
        sandbox_path = _normalize_sandbox_path(path)
    except ValueError as exc:
        return _error_result(str(exc))

    manager = get_sandbox_manager()
    sandbox_result = await manager.get_or_create_sandbox(tool_context.state)
    sandbox = sandbox_result.get("sandbox")
    if sandbox is None or sandbox_result.get("error"):
        return _error_result(
            str(sandbox_result.get("error") or "Sandbox is unavailable.")
        )

    try:
        data = await sandbox.files.read_bytes(sandbox_path)
    except FileNotFoundError:
        return _error_result(
            "Image file was not found in the active sandbox.", sandbox_path
        )
    except Exception as exc:
        logger.warning("Sandbox image read failed (%s)", type(exc).__name__)
        return _error_result(
            "Could not read the image from the active sandbox.", sandbox_path
        )

    try:
        metadata = _inspect_image_bytes(data)
    except ValueError as exc:
        return _error_result(str(exc), sandbox_path)

    return [
        types.Part.from_text(
            text=(
                f"Sandbox image {PurePosixPath(sandbox_path).name}: "
                f"{metadata.format_name}, {metadata.width}x{metadata.height}px"
            )
        ),
        types.Part.from_bytes(data=data, mime_type=metadata.mime_type),
    ]
