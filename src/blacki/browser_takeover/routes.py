"""FastAPI routes for private Agent Browser pair browsing."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .service import BrowserTakeoverService, get_browser_takeover_service

COOKIE_NAME = "blacki_browser_takeover"
_MAX_INPUT_BYTES = 16 * 1024
_MAX_TEXT_BYTES = 4096
_ALLOWED_MOUSE_EVENTS = frozenset(
    {"mousePressed", "mouseReleased", "mouseMoved", "mouseWheel"}
)
_ALLOWED_MOUSE_BUTTONS = frozenset({"left", "middle", "right", "none"})
_ALLOWED_KEYBOARD_EVENTS = frozenset({"keyDown", "keyUp", "char"})
_ALLOWED_TOUCH_EVENTS = frozenset({"touchStart", "touchMove", "touchEnd"})
_FRAME_METADATA_KEYS = frozenset(
    {
        "deviceWidth",
        "deviceHeight",
        "pageScaleFactor",
        "offsetTop",
        "scrollOffsetX",
        "scrollOffsetY",
        "timestamp",
    }
)
_SECURITY_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Content-Security-Policy": (
        "default-src 'none'; connect-src 'self' ws: wss:; img-src data:; "
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}

_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Private browser control</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111; color: #eee; display: grid;
           min-height: 100vh; grid-template-rows: auto 1fr auto; }
    header, footer { padding: 12px; display: flex; gap: 12px; align-items: center;
                     background: #191919; }
    header { justify-content: space-between; }
    #site-info { min-width: 0; }
    #site-info strong { display: block; }
    #expected, #current { display: block; margin-top: 4px; color: #bbb;
                          font-size: 12px; overflow-wrap: anywhere; }
    #current.warning { color: #ffcf70; font-weight: 650; }
    #viewport { width: 100%; height: 100%; object-fit: contain; touch-action: none;
                background: #080808; }
    #keyboard { flex: 1; min-width: 0; padding: 12px; border-radius: 8px;
                border: 1px solid #555; background: #222; color: #fff; }
    button { padding: 11px 16px; border: 0; border-radius: 8px; font-weight: 650; }
    #done { background: #b7f7c2; color: #102414; }
    #status { color: #bbb; font-size: 14px; white-space: nowrap; }
  </style>
</head>
<body>
  <header>
    <div id="site-info">
      <strong>Private browser control</strong>
      <span id="expected">Expected site: verifying...</span>
      <span id="current">Current site: waiting for browser...</span>
    </div>
    <span id="status">Connecting...</span>
  </header>
  <canvas id="viewport" tabindex="0" aria-label="Remote browser viewport"></canvas>
  <footer>
    <input id="keyboard" type="password" autocomplete="off" autocapitalize="none"
           spellcheck="false" placeholder="Tap here for the mobile keyboard">
    <button id="done" type="button">Done</button>
  </footer>
<script>
(() => {
  const status = document.querySelector('#status');
  const expected = document.querySelector('#expected');
  const current = document.querySelector('#current');
  const canvas = document.querySelector('#viewport');
  const keyboard = document.querySelector('#keyboard');
  const context = canvas.getContext('2d');
  let socket;
  let expectedOrigin = '';
  let currentOrigin = '';
  let metadata = {deviceWidth: 1280, deviceHeight: 720};
  const activeTouches = new Set();

  const send = value => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(value));
    }
  };
  const key = (eventType, value, code = value) =>
    send({type: 'input_keyboard', eventType, key: value, code});
  const press = (value, code = value) => {
    key('keyDown', value, code); key('keyUp', value, code);
  };

  function renderOrigins() {
    expected.textContent = expectedOrigin
      ? `Expected site: ${expectedOrigin}`
      : 'Expected site: verifying...';
    if (!currentOrigin) {
      current.textContent = 'Current site: waiting for browser...';
      current.classList.remove('warning');
      return;
    }
    current.textContent = `Current site: ${currentOrigin}`;
    const changed = Boolean(expectedOrigin && currentOrigin !== expectedOrigin);
    current.classList.toggle('warning', changed);
    if (changed) {
      current.textContent += ' - verify this site before typing sensitive data';
    }
  }

  async function redeem() {
    const token = location.hash.slice(1);
    history.replaceState(null, '', location.pathname);
    if (!token) {
      throw new Error('This takeover link is missing or has already been used.');
    }
    const response = await fetch(`${location.pathname}/redeem`, {
      method: 'POST', credentials: 'same-origin',
      headers: {'content-type': 'application/json'}, body: JSON.stringify({token})
    });
    if (!response.ok) throw new Error('This takeover link is invalid or expired.');
  }

  function connectStream() {
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${scheme}//${location.host}${location.pathname}/ws`);
    socket.onopen = () => {
      status.textContent = 'Connected';
      send({type: 'config', maxFps: 12, pacing: 'ack'});
    };
    socket.onclose = () => { status.textContent = 'Disconnected'; };
    socket.onmessage = event => {
      if (typeof event.data !== 'string') return;
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (message.type === 'takeover_meta') {
        expectedOrigin = message.expectedOrigin || '';
        renderOrigins();
        return;
      }
      if (message.type === 'navigation') {
        currentOrigin = message.origin || '';
        renderOrigins();
        return;
      }
      if (message.type === 'frame') {
        metadata = message.metadata || metadata;
        const image = new Image();
        image.onload = () => {
          canvas.width = image.width; canvas.height = image.height;
          context.drawImage(image, 0, 0);
          if (message.seq !== undefined) send({type: 'ack', seq: message.seq});
        };
        image.src = `data:image/jpeg;base64,${message.data}`;
      }
    };
  }

  function displayedContentRect() {
    const rect = canvas.getBoundingClientRect();
    const sourceWidth = canvas.width || metadata.deviceWidth;
    const sourceHeight = canvas.height || metadata.deviceHeight;
    if (!rect.width || !rect.height || !sourceWidth || !sourceHeight) return null;
    const scale = Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    return {
      left: rect.left + (rect.width - width) / 2,
      top: rect.top + (rect.height - height) / 2,
      width,
      height
    };
  }

  function point(event) {
    const rect = displayedContentRect();
    if (!rect) return null;
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) {
      return null;
    }
    return {
      x: localX * metadata.deviceWidth / rect.width,
      y: localY * metadata.deviceHeight / rect.height
    };
  }

  canvas.addEventListener('pointerdown', event => {
    const p = point(event);
    if (!p) return;
    event.preventDefault();
    if (event.pointerType === 'touch') {
      activeTouches.add(event.pointerId);
      canvas.setPointerCapture?.(event.pointerId);
      send({type: 'input_touch', eventType: 'touchStart',
            touchPoints: [{x: p.x, y: p.y, id: event.pointerId}]});
      return;
    }
    send({type: 'input_mouse', eventType: 'mousePressed', x: p.x, y: p.y,
          button: 'left', clickCount: 1});
  });

  canvas.addEventListener('pointermove', event => {
    if (event.pointerType !== 'touch' || !activeTouches.has(event.pointerId)) return;
    const p = point(event);
    if (!p) return;
    event.preventDefault();
    send({type: 'input_touch', eventType: 'touchMove',
          touchPoints: [{x: p.x, y: p.y, id: event.pointerId}]});
  });

  canvas.addEventListener('pointerup', event => {
    event.preventDefault();
    if (event.pointerType === 'touch') {
      if (activeTouches.delete(event.pointerId)) {
        send({type: 'input_touch', eventType: 'touchEnd', touchPoints: []});
      }
      return;
    }
    const p = point(event);
    if (!p) return;
    send({type: 'input_mouse', eventType: 'mouseReleased', x: p.x, y: p.y,
          button: 'left', clickCount: 1});
    canvas.focus();
  });

  canvas.addEventListener('pointercancel', event => {
    if (event.pointerType === 'touch' && activeTouches.delete(event.pointerId)) {
      send({type: 'input_touch', eventType: 'touchEnd', touchPoints: []});
    }
  });

  canvas.addEventListener('wheel', event => {
    const p = point(event);
    if (!p) return;
    event.preventDefault();
    send({type: 'input_mouse', eventType: 'mouseWheel', x: p.x, y: p.y,
          deltaX: event.deltaX, deltaY: event.deltaY});
  }, {passive: false});

  canvas.addEventListener('keydown', event => {
    event.preventDefault(); press(event.key, event.code);
  });
  keyboard.addEventListener('beforeinput', event => {
    event.preventDefault();
    if (event.inputType === 'deleteContentBackward') press('Backspace', 'Backspace');
    else if (event.data) for (const char of event.data) press(char, char);
    keyboard.value = '';
  });
  keyboard.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === 'Tab') {
      event.preventDefault(); press(event.key, event.code);
    }
  });
  document.querySelector('#done').addEventListener('click', async () => {
    const response = await fetch(`${location.pathname}/complete`, {
      method: 'POST', credentials: 'same-origin'
    });
    if (response.ok) {
      status.textContent = 'Control returned. You can close this page.';
      keyboard.disabled = true; socket?.close();
    } else status.textContent = 'Session expired.';
  });

  redeem().then(connectStream).catch(error => { status.textContent = error.message; });
})();
</script>
</body>
</html>"""


def _service() -> BrowserTakeoverService | None:
    return get_browser_takeover_service()


def _same_origin(request: Request, service: BrowserTakeoverService) -> bool:
    return request.headers.get("origin") == service.config.public_origin


def _private_response(content: Any, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content, status_code=status_code, headers=_SECURITY_HEADERS)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _bounded_number(value: Any, *, limit: float = 100_000.0) -> bool:
    return _is_number(value) and -limit <= value <= limit


def _only_keys(payload: dict[str, Any], allowed: frozenset[str]) -> bool:
    return all(key in allowed for key in payload)


def _valid_ack(payload: dict[str, Any]) -> bool:
    if not _only_keys(payload, frozenset({"type", "seq"})):
        return False
    seq = payload.get("seq")
    return _is_int(seq) and 0 <= seq <= 2**53 - 1


def _valid_config(payload: dict[str, Any]) -> bool:
    if not _only_keys(payload, frozenset({"type", "maxFps", "pacing"})):
        return False
    if "maxFps" not in payload and "pacing" not in payload:
        return False
    max_fps = payload.get("maxFps")
    if max_fps is not None and (not _is_int(max_fps) or not 0 <= max_fps <= 120):
        return False
    pacing = payload.get("pacing")
    return pacing is None or pacing == "ack"


def _short_string(value: Any, *, max_bytes: int = 128) -> bool:
    return isinstance(value, str) and len(value.encode()) <= max_bytes


def _valid_keyboard(payload: dict[str, Any]) -> bool:
    allowed = frozenset({"type", "eventType", "key", "code", "text", "modifiers"})
    if not _only_keys(payload, allowed):
        return False
    event_type = payload.get("eventType")
    if event_type not in _ALLOWED_KEYBOARD_EVENTS:
        return False
    modifiers = payload.get("modifiers")
    if modifiers is not None and (not _is_int(modifiers) or not 0 <= modifiers <= 15):
        return False
    if event_type == "char":
        text = payload.get("text")
        return (
            isinstance(text, str)
            and bool(text)
            and len(text.encode()) <= _MAX_TEXT_BYTES
        )
    key = payload.get("key")
    code = payload.get("code")
    return _short_string(key) and bool(key) and (code is None or _short_string(code))


def _valid_mouse(payload: dict[str, Any]) -> bool:
    allowed = frozenset(
        {
            "type",
            "eventType",
            "x",
            "y",
            "button",
            "clickCount",
            "deltaX",
            "deltaY",
            "modifiers",
        }
    )
    if not _only_keys(payload, allowed):
        return False
    event_type = payload.get("eventType")
    if event_type not in _ALLOWED_MOUSE_EVENTS:
        return False
    if not _bounded_number(payload.get("x")) or not _bounded_number(payload.get("y")):
        return False
    modifiers = payload.get("modifiers")
    if modifiers is not None and (not _is_int(modifiers) or not 0 <= modifiers <= 15):
        return False
    button = payload.get("button")
    if button is not None and button not in _ALLOWED_MOUSE_BUTTONS:
        return False
    click_count = payload.get("clickCount")
    if click_count is not None and (
        not _is_int(click_count) or not 0 <= click_count <= 3
    ):
        return False
    if event_type == "mouseWheel":
        if "deltaX" not in payload and "deltaY" not in payload:
            return False
        return ("deltaX" not in payload or _bounded_number(payload.get("deltaX"))) and (
            "deltaY" not in payload or _bounded_number(payload.get("deltaY"))
        )
    return True


def _valid_touch_point(point: Any) -> bool:
    if not isinstance(point, dict):
        return False
    if not _only_keys(point, frozenset({"x", "y", "id"})):
        return False
    if not _bounded_number(point.get("x")) or not _bounded_number(point.get("y")):
        return False
    touch_id = point.get("id")
    return touch_id is None or (_is_int(touch_id) and 0 <= touch_id <= 2**31 - 1)


def _valid_touch(payload: dict[str, Any]) -> bool:
    if not _only_keys(payload, frozenset({"type", "eventType", "touchPoints"})):
        return False
    event_type = payload.get("eventType")
    if event_type not in _ALLOWED_TOUCH_EVENTS:
        return False
    points = payload.get("touchPoints")
    if not isinstance(points, list) or len(points) > 10:
        return False
    if event_type != "touchEnd" and not points:
        return False
    return all(_valid_touch_point(point) for point in points)


def _valid_client_message(message: str) -> bool:
    if len(message.encode()) > _MAX_INPUT_BYTES:
        return False
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    message_type = payload.get("type")
    if message_type == "ack":
        return _valid_ack(payload)
    if message_type == "config":
        return _valid_config(payload)
    if message_type == "input_keyboard":
        return _valid_keyboard(payload)
    if message_type == "input_mouse":
        return _valid_mouse(payload)
    if message_type == "input_touch":
        return _valid_touch(payload)
    return False


def _normalized_web_origin(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or host is None:
        return None
    host = host.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = (parsed.scheme == "https" and port == 443) or (
        parsed.scheme == "http" and port == 80
    )
    suffix = "" if port is None or default_port else f":{port}"
    return f"{parsed.scheme.lower()}://{host}{suffix}"


def _safe_server_message(message: str) -> str | None:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    message_type = payload.get("type")
    if message_type == "frame":
        data = payload.get("data")
        if not isinstance(data, str):
            return None
        safe: dict[str, Any] = {"type": "frame", "data": data}
        seq = payload.get("seq")
        if _is_int(seq) and seq >= 0:
            safe["seq"] = seq
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            safe_metadata = {
                key: value
                for key, value in metadata.items()
                if key in _FRAME_METADATA_KEYS and _is_number(value)
            }
            if safe_metadata:
                safe["metadata"] = safe_metadata
        return json.dumps(safe, separators=(",", ":"))
    if message_type == "status":
        connected = payload.get("connected")
        if not isinstance(connected, bool):
            return None
        return json.dumps(
            {"type": "status", "connected": connected},
            separators=(",", ":"),
        )
    if message_type == "url":
        url = payload.get("url")
        origin = _normalized_web_origin(url) if isinstance(url, str) else None
        if origin is None:
            return None
        return json.dumps(
            {"type": "navigation", "origin": origin},
            separators=(",", ":"),
        )
    return None


def create_browser_takeover_router() -> APIRouter:
    """Create routes without exposing upstream sandbox endpoints to the client."""
    router = APIRouter(prefix="/browser-takeover")

    @router.get("", response_class=HTMLResponse)
    async def takeover_page() -> HTMLResponse:
        if _service() is None:
            return HTMLResponse("Not found", status_code=404)
        return HTMLResponse(_PAGE, headers=_SECURITY_HEADERS)

    @router.post("/redeem")
    async def redeem(request: Request) -> Response:
        service = _service()
        if service is None or not _same_origin(request, service):
            return _private_response({"status": "invalid"}, status_code=404)
        try:
            raw_body = await request.body()
            if len(raw_body) > 4096:
                return _private_response({"status": "invalid"}, status_code=404)
            body = json.loads(raw_body)
            token = body.get("token", "") if isinstance(body, dict) else ""
            if not isinstance(token, str):
                token = ""
        except (json.JSONDecodeError, UnicodeDecodeError):
            token = ""
        browser_token = await service.redeem(token)
        if browser_token is None:
            return _private_response({"status": "invalid"}, status_code=404)
        response = _private_response({"status": "ready"})
        response.set_cookie(
            COOKIE_NAME,
            browser_token,
            httponly=True,
            secure=service.config.secure_cookie,
            samesite="strict",
            max_age=service.config.ttl_seconds,
            path="/browser-takeover",
        )
        return response

    @router.post("/complete")
    async def complete(request: Request) -> Response:
        service = _service()
        token = request.cookies.get(COOKIE_NAME)
        if (
            service is None
            or not _same_origin(request, service)
            or not await service.complete(token)
        ):
            return _private_response({"status": "invalid"}, status_code=404)
        response = _private_response({"status": "complete"})
        response.delete_cookie(COOKIE_NAME, path="/browser-takeover")
        return response

    @router.websocket("/ws")
    async def browser_stream(websocket: WebSocket) -> None:
        service = _service()
        token = websocket.cookies.get(COOKIE_NAME)
        same_origin = (
            service is not None
            and websocket.headers.get("origin") == service.config.public_origin
        )
        if service is None or not same_origin:
            await websocket.close(code=4401)
            return
        session = await service.authorize(token)
        if session is None:
            await websocket.close(code=4401)
            return

        await websocket.accept()
        await websocket.send_json(
            {
                "type": "takeover_meta",
                "expectedOrigin": session.expected_origin,
            }
        )
        try:
            async with connect(
                session.upstream_url,
                additional_headers=session.upstream_headers,
                max_size=8 * 1024 * 1024,
            ) as upstream:

                async def from_browser() -> None:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                            continue
                        safe_message = _safe_server_message(message)
                        if safe_message is not None:
                            await websocket.send_text(safe_message)

                async def from_user() -> None:
                    while True:
                        message = await websocket.receive_text()
                        if _valid_client_message(message):
                            await upstream.send(message)

                tasks = {
                    asyncio.create_task(from_browser()),
                    asyncio.create_task(from_user()),
                }
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*done, *pending, return_exceptions=True)
        except (ConnectionClosed, WebSocketDisconnect):
            return
        except Exception:
            await websocket.close(code=1011)

    return router
