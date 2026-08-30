"""Tests for sandbox image validation and multimodal tool transport."""

from __future__ import annotations

import struct
import zlib
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.multimodal_tool_results_plugin import PARTS_RETURNED_BY_TOOLS_ID
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

from blacki.sandbox.images import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    SandboxMultimodalToolResultsPlugin,
    _inspect_image_bytes,
    _normalize_sandbox_path,
    _validate_dimensions,
    sandbox_view_image,
)


def _png_chunk(name: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def _png_bytes(width: int = 2, height: int = 3) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + b"\x00\x00\x00" * width
    pixels = zlib.compress(scanline * height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", pixels)
        + _png_chunk(b"IEND", b"")
    )


def _jpeg_bytes(width: int = 5, height: int = 7) -> bytes:
    frame = (
        b"\xff\xc0"
        + struct.pack(">H", 11)
        + bytes([8])
        + struct.pack(">H", height)
        + struct.pack(">H", width)
        + bytes([1, 1, 0x11, 0])
    )
    scan = b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00"
    return b"\xff\xd8" + frame + scan + b"\x00\xff\xd9"


def _gif_bytes(width: int = 8, height: int = 9) -> bytes:
    return (
        b"GIF89a"
        + struct.pack("<HH", width, height)
        + b"\x80\x00\x00"
        + b"\x00\x00\x00"
        + b"\x3b"
    )


def _webp_bytes(width: int = 10, height: int = 11) -> bytes:
    payload = (
        b"\x00\x00\x00\x00"
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    chunk = b"VP8X" + struct.pack("<I", len(payload)) + payload
    riff_size = 4 + len(chunk)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WEBP" + chunk


def _webp_chunk(name: bytes, data: bytes, *, riff_size: int | None = None) -> bytes:
    chunk = name + struct.pack("<I", len(data)) + data
    if len(data) % 2:
        chunk += b"\x00"
    total_size = riff_size if riff_size is not None else 4 + len(chunk)
    return b"RIFF" + struct.pack("<I", total_size) + b"WEBP" + chunk


def _bmp_bytes(width: int = 12, height: int = 13) -> bytes:
    pixel_data = b"\x00\x00\x00\x00"
    file_size = 54 + len(pixel_data)
    dib = (
        struct.pack("<Iii", 40, width, height)
        + struct.pack("<H", 1)
        + struct.pack("<H", 24)
        + struct.pack("<I", 0)
        + struct.pack("<I", len(pixel_data))
        + struct.pack("<ii", 0, 0)
        + struct.pack("<II", 0, 0)
    )
    return (
        b"BM"
        + struct.pack("<I", file_size)
        + b"\x00\x00\x00\x00"
        + struct.pack("<I", 54)
        + dib
        + pixel_data
    )


@pytest.mark.parametrize(
    ("data", "format_name", "mime_type", "size"),
    [
        (_png_bytes(), "PNG", "image/png", (2, 3)),
        (_jpeg_bytes(), "JPEG", "image/jpeg", (5, 7)),
        (_gif_bytes(), "GIF", "image/gif", (8, 9)),
        (_webp_bytes(), "WEBP", "image/webp", (10, 11)),
        (_bmp_bytes(), "BMP", "image/bmp", (12, 13)),
    ],
)
def test_inspect_image_bytes_supports_common_formats(
    data: bytes, format_name: str, mime_type: str, size: tuple[int, int]
) -> None:
    metadata = _inspect_image_bytes(data)

    assert metadata.format_name == format_name
    assert metadata.mime_type == mime_type
    assert (metadata.width, metadata.height) == size


@pytest.mark.asyncio
async def test_sandbox_view_image_builds_adk_tool_declaration() -> None:
    tool = FunctionTool(func=sandbox_view_image)
    request = LlmRequest()

    await tool.process_llm_request(
        tool_context=cast(Any, SimpleNamespace()),
        llm_request=request,
    )

    assert tool.name == "sandbox_view_image"
    assert tool.name in request.tools_dict


def test_inspect_image_bytes_rejects_invalid_and_unsupported_data() -> None:
    with pytest.raises(ValueError, match="Unsupported image format"):
        _inspect_image_bytes(b"not an image")
    with pytest.raises(ValueError, match="PNG header is incomplete"):
        _inspect_image_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="PNG checksum is invalid"):
        _inspect_image_bytes(_png_bytes()[:-12] + b"\x00" * 12)


@pytest.mark.parametrize(
    "data",
    [
        _png_bytes()[:-12] + b"x",
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 100) + b"IHDR" + b"\x00" * 4,
        b"\x89PNG\r\n\x1a\n" + _png_chunk(b"NOPE", b"\x00" * 13),
        _png_bytes()[:-12]
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 3, 8, 2, 0, 0, 0))
        + _png_bytes()[-12:],
        _png_bytes()[:-12] + _png_chunk(b"IEND", b"x"),
        _png_bytes()[:-12],
    ],
)
def test_inspect_image_bytes_rejects_malformed_pngs(data: bytes) -> None:
    with pytest.raises(ValueError):
        _inspect_image_bytes(data)


@pytest.mark.parametrize(
    "data",
    [
        b"\xff\xd8",
        b"\xff\xd8\x00\x00",
        b"\xff\xd8\xff\xff",
        b"\xff\xd8\xff\xd9",
        b"\xff\xd8\xff\xda\x00",
        b"\xff\xd8\xff\xda\x00\x01",
        b"\xff\xd8\xff\xda\x00\x02\xff\xd9",
        _jpeg_bytes()[:-2],
        b"\xff\xd8\xff\xe0\x00",
        b"\xff\xd8\xff\xe0\x00\x01",
        b"\xff\xd8\xff\xc0\x00\x06" + b"\x00" * 4,
        b"\xff\xd8\xff\xe0\x00\x02",
    ],
)
def test_inspect_image_bytes_rejects_malformed_jpegs(data: bytes) -> None:
    with pytest.raises(ValueError):
        _inspect_image_bytes(data)


def test_inspect_image_bytes_accepts_jpeg_standalone_and_eoi_markers() -> None:
    frame = _jpeg_bytes()[2 : 2 + 13]
    assert _inspect_image_bytes(b"\xff\xd8" + frame + b"\xff\xd9").format_name == "JPEG"
    data = (
        b"\xff\xd8\xff\x01"
        + frame
        + b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\x00\xff\xd9"
    )
    assert _inspect_image_bytes(data).format_name == "JPEG"


def test_inspect_image_bytes_rejects_malformed_gif() -> None:
    with pytest.raises(ValueError, match="GIF structure"):
        _inspect_image_bytes(b"GIF89a" + b"\x00" * 8)


@pytest.mark.parametrize(
    "data",
    [
        b"RIFF" + b"\x00" * 16,
        b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8,
        _webp_chunk(b"JUNK", b"\x00" * 4, riff_size=14),
        _webp_chunk(b"VP8X", b"\x00" * 9),
        _webp_chunk(b"VP8 ", b"\x00" * 10),
        _webp_chunk(b"VP8L", b"\x00" * 5),
        _webp_chunk(b"JUNK", b"\x00"),
    ],
)
def test_inspect_image_bytes_rejects_malformed_webps(data: bytes) -> None:
    with pytest.raises(ValueError):
        _inspect_image_bytes(data)


def test_inspect_image_bytes_supports_vp8_and_vp8l_webp_frames() -> None:
    vp8_payload = b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", 14, 15)
    vp8l_dimensions = (16 | ((17 - 1) << 14)).to_bytes(4, "little")

    assert _inspect_image_bytes(_webp_chunk(b"VP8 ", vp8_payload)).width == 14
    assert (
        _inspect_image_bytes(_webp_chunk(b"VP8L", b"\x2f" + vp8l_dimensions)).height
        == 17
    )


@pytest.mark.parametrize(
    "data",
    [
        b"BM" + b"\x00" * 23,
        b"BM" + struct.pack("<I", 100) + b"\x00" * 20,
        b"BM\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x00\x00\x00" + b"\x00" * 14,
        b"BM"
        + b"\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00"
        + struct.pack("<I", 26)
        + struct.pack("<I", 20)
        + b"\x00" * 8,
    ],
)
def test_inspect_image_bytes_rejects_malformed_bmps(data: bytes) -> None:
    with pytest.raises(ValueError):
        _inspect_image_bytes(data)


def test_inspect_image_bytes_supports_bmp_core_header() -> None:
    data = (
        b"BM"
        + struct.pack("<I", 27)
        + b"\x00\x00\x00\x00"
        + struct.pack("<I", 26)
        + struct.pack("<IHHHH", 12, 3, 4, 1, 24)
        + b"\x00"
    )
    metadata = _inspect_image_bytes(data)
    assert (metadata.width, metadata.height) == (3, 4)


def test_inspect_image_bytes_rejects_empty_and_oversized_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        _inspect_image_bytes(b"")
    with (
        patch("blacki.sandbox.images.MAX_IMAGE_BYTES", 4),
        pytest.raises(ValueError, match="byte limit"),
    ):
        _inspect_image_bytes(b"12345")
    assert MAX_IMAGE_BYTES > 0
    with pytest.raises(ValueError, match="unreadable"):
        _inspect_image_bytes(bytearray(b"data"))


def test_validate_dimensions_enforces_each_limit() -> None:
    with pytest.raises(ValueError, match="invalid"):
        _validate_dimensions(0, 1)
    with pytest.raises(ValueError, match="px limit"):
        _validate_dimensions(MAX_IMAGE_DIMENSION + 1, 1)
    with pytest.raises(ValueError, match="pixel count"):
        _validate_dimensions(10_000, MAX_IMAGE_PIXELS // 10_000 + 1)


def _sandbox_manager(data: bytes) -> tuple[MagicMock, MagicMock]:
    sandbox = MagicMock()
    sandbox.files.read_bytes = AsyncMock(return_value=data)
    manager = MagicMock()
    manager.get_or_create_sandbox = AsyncMock(
        return_value={"sandbox": sandbox, "error": None}
    )
    return manager, sandbox


@pytest.mark.asyncio
async def test_sandbox_view_image_returns_separate_visual_parts() -> None:
    manager, sandbox = _sandbox_manager(_png_bytes())
    context = SimpleNamespace(state={})

    with patch("blacki.sandbox.images.get_sandbox_manager", return_value=manager):
        result = await sandbox_view_image("uploads/photo.png", cast(Any, context))

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], types.Part)
    assert result[0].text is not None and "photo.png" in result[0].text
    assert result[1].inline_data is not None
    assert result[1].inline_data.mime_type == "image/png"
    assert result[1].inline_data.data == _png_bytes()
    sandbox.files.read_bytes.assert_awaited_once_with("/workspace/uploads/photo.png")


@pytest.mark.asyncio
async def test_sandbox_view_image_uses_reconnected_sandbox_for_restored_file() -> None:
    manager, sandbox = _sandbox_manager(_png_bytes())
    state = {"__sandbox_id__": "restored-sandbox"}
    context = SimpleNamespace(state=state)

    with patch("blacki.sandbox.images.get_sandbox_manager", return_value=manager):
        result = await sandbox_view_image(
            "/workspace/uploads/restored-photo.png", cast(Any, context)
        )

    assert isinstance(result, list)
    manager.get_or_create_sandbox.assert_awaited_once_with(state)
    sandbox.files.read_bytes.assert_awaited_once_with(
        "/workspace/uploads/restored-photo.png"
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../photo.png",
        "/tmp/photo.png",
        "uploads\\photo.png",
        "/workspace/../photo.png",
    ],
)
async def test_sandbox_view_image_rejects_paths_outside_workspace(path: str) -> None:
    manager = MagicMock()
    context = SimpleNamespace(state={})

    with patch("blacki.sandbox.images.get_sandbox_manager", return_value=manager):
        result = await sandbox_view_image(path, cast(Any, context))

    assert isinstance(result, dict)
    assert result["status"] == "error"
    manager.get_or_create_sandbox.assert_not_called()


def test_normalize_sandbox_path_rejects_workspace_root() -> None:
    with pytest.raises(ValueError, match="identify a file"):
        _normalize_sandbox_path("/workspace")


@pytest.mark.asyncio
async def test_sandbox_view_image_handles_sandbox_and_file_errors() -> None:
    context = SimpleNamespace(state={})
    disabled_manager = MagicMock()
    disabled_manager.get_or_create_sandbox = AsyncMock(
        return_value={"sandbox": None, "error": "Sandbox is disabled"}
    )
    with patch(
        "blacki.sandbox.images.get_sandbox_manager", return_value=disabled_manager
    ):
        disabled = await sandbox_view_image("photo.png", cast(Any, context))
    assert isinstance(disabled, dict)
    assert disabled == {
        "status": "error",
        "error": "Sandbox is disabled",
        "sandbox_path": None,
    }

    missing_manager, missing_sandbox = _sandbox_manager(_png_bytes())
    missing_sandbox.files.read_bytes.side_effect = FileNotFoundError
    with patch(
        "blacki.sandbox.images.get_sandbox_manager", return_value=missing_manager
    ):
        missing = await sandbox_view_image("photo.png", cast(Any, context))
    assert isinstance(missing, dict)
    assert missing["error"] == "Image file was not found in the active sandbox."

    failed_manager, failed_sandbox = _sandbox_manager(_png_bytes())
    failed_sandbox.files.read_bytes.side_effect = RuntimeError("secret")
    with patch(
        "blacki.sandbox.images.get_sandbox_manager", return_value=failed_manager
    ):
        failed = await sandbox_view_image("photo.png", cast(Any, context))
    assert isinstance(failed, dict)
    assert failed["error"] == "Could not read the image from the active sandbox."
    assert "secret" not in failed["error"]


@pytest.mark.asyncio
async def test_sandbox_view_image_reports_invalid_image() -> None:
    manager, _ = _sandbox_manager(b"not an image")
    context = SimpleNamespace(state={})

    with patch("blacki.sandbox.images.get_sandbox_manager", return_value=manager):
        result = await sandbox_view_image("photo.png", cast(Any, context))

    assert isinstance(result, dict)
    assert result["status"] == "error"
    assert "Unsupported image format" in result["error"]
    assert result["sandbox_path"] == "/workspace/photo.png"


@pytest.mark.asyncio
async def test_multimodal_plugin_keeps_multiple_sandbox_images_independent() -> None:
    first = _png_bytes(2, 3)
    second = _png_bytes(4, 5)
    manager, sandbox = _sandbox_manager(first)
    sandbox.files.read_bytes.side_effect = [first, second]
    state: dict[str, Any] = {}
    tool_context = SimpleNamespace(state=state)
    tool = SimpleNamespace(name="sandbox_view_image")

    with patch("blacki.sandbox.images.get_sandbox_manager", return_value=manager):
        first_result = await sandbox_view_image(
            "uploads/first.png", cast(Any, tool_context)
        )
        second_result = await sandbox_view_image(
            "uploads/second.png", cast(Any, tool_context)
        )
    assert isinstance(first_result, list)
    assert isinstance(second_result, list)

    bridge = SandboxMultimodalToolResultsPlugin()
    for result in (first_result, second_result):
        assert await bridge.after_tool_callback(
            tool=cast(Any, tool),
            tool_args={},
            tool_context=cast(Any, tool_context),
            result=cast(Any, result),
        ) == {
            "status": "success",
            "message": "Image attached as a separate visual input.",
        }

    first_function_response = types.Part.from_function_response(
        name="sandbox_view_image",
        response={"result": first_result},
    ).function_response
    second_function_response = types.Part.from_function_response(
        name="sandbox_view_image",
        response={"result": second_result},
    ).function_response
    assert first_function_response is not None
    assert second_function_response is not None
    event = SimpleNamespace(
        content=types.Content(
            role="user",
            parts=[
                types.Part(function_response=first_function_response),
                types.Part(function_response=second_function_response),
            ],
        )
    )
    await bridge.on_event_callback(
        invocation_context=cast(Any, SimpleNamespace()),
        event=cast(Any, event),
    )
    assert first_function_response.response == {
        "status": "success",
        "message": "Image attached as a separate visual input.",
    }
    assert second_function_response.response == {
        "status": "success",
        "message": "Image attached as a separate visual input.",
    }
    serialized_event = event.content.model_dump(mode="json")
    assert "Sandbox image first.png" not in str(serialized_event)
    assert "Sandbox image second.png" not in str(serialized_event)

    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="sandbox_view_image",
                        response={
                            "status": "success",
                            "message": "Image attached as a separate visual input.",
                        },
                    )
                ],
            )
        ]
    )
    await bridge.before_model_callback(
        callback_context=cast(Any, SimpleNamespace(state=state)),
        llm_request=request,
    )

    parts = request.contents[-1].parts or []
    inline_parts = [part for part in parts if part.inline_data is not None]
    text_parts = [part.text for part in parts if part.text]
    assert len(inline_parts) == 2
    inline_data = [part.inline_data for part in inline_parts]
    assert all(data is not None for data in inline_data)
    assert [data.data for data in inline_data if data is not None] == [first, second]
    assert text_parts == [
        "Sandbox image first.png: PNG, 2x3px",
        "Sandbox image second.png: PNG, 4x5px",
    ]
    assert (
        PARTS_RETURNED_BY_TOOLS_ID not in state
        or state[PARTS_RETURNED_BY_TOOLS_ID] == []
    )


@pytest.mark.asyncio
async def test_multimodal_plugin_ignores_non_part_results() -> None:
    plugin = SandboxMultimodalToolResultsPlugin()
    state: dict[str, Any] = {}
    tool_context = SimpleNamespace(state=state)
    result: dict[str, Any] = {"status": "error"}

    altered = await plugin.after_tool_callback(
        tool=cast(Any, SimpleNamespace(name="other_tool")),
        tool_args={},
        tool_context=cast(Any, tool_context),
        result=result,
    )

    assert altered is None

    image_error = await plugin.after_tool_callback(
        tool=cast(Any, SimpleNamespace(name="sandbox_view_image")),
        tool_args={},
        tool_context=cast(Any, tool_context),
        result=result,
    )
    assert image_error is None
    assert state == {}


@pytest.mark.asyncio
async def test_multimodal_plugin_ignores_unrelated_events() -> None:
    plugin = SandboxMultimodalToolResultsPlugin()

    assert (
        await plugin.on_event_callback(
            invocation_context=cast(Any, SimpleNamespace()),
            event=cast(Any, SimpleNamespace(content=None)),
        )
        is None
    )
    assert (
        await plugin.on_event_callback(
            invocation_context=cast(Any, SimpleNamespace()),
            event=cast(Any, SimpleNamespace(content=types.Content(parts=[]))),
        )
        is None
    )

    unrelated = types.Part.from_function_response(
        name="other_tool", response={"result": []}
    )
    non_dict = SimpleNamespace(
        function_response=SimpleNamespace(name="sandbox_view_image", response="text")
    )
    no_result = types.Part.from_function_response(
        name="sandbox_view_image", response={"status": "success"}
    )
    event = SimpleNamespace(
        content=SimpleNamespace(
            parts=[unrelated, non_dict, no_result],
        )
    )
    await plugin.on_event_callback(
        invocation_context=cast(Any, SimpleNamespace()),
        event=cast(Any, event),
    )
    assert no_result.function_response is not None
    assert no_result.function_response.response == {"status": "success"}
