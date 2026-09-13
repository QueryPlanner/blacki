"""FastAPI routes for private Agent Browser pair browsing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .service import BrowserTakeoverService, get_browser_takeover_service

COOKIE_NAME = "blacki_browser_takeover"
_MAX_INPUT_BYTES = 16 * 1024
_ALLOWED_INPUT_TYPES = frozenset(
    {
        "ack",
        "config",
        "input_keyboard",
        "input_mouse",
        "input_touch",
        "status",
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
    header, footer { padding: 12px; display: flex; gap: 10px; align-items: center;
                     background: #191919; }
    header { justify-content: space-between; }
    #viewport { width: 100%; height: 100%; object-fit: contain; touch-action: none;
                background: #080808; }
    #keyboard { flex: 1; min-width: 0; padding: 12px; border-radius: 8px;
                border: 1px solid #555; background: #222; color: #fff; }
    button { padding: 11px 16px; border: 0; border-radius: 8px; font-weight: 650; }
    #done { background: #b7f7c2; color: #102414; }
    #status { color: #bbb; font-size: 14px; }
  </style>
</head>
<body>
  <header>
    <strong>Private browser control</strong><span id="status">Connecting...</span>
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
  const canvas = document.querySelector('#viewport');
  const keyboard = document.querySelector('#keyboard');
  const context = canvas.getContext('2d');
  let socket;
  let metadata = {deviceWidth: 1280, deviceHeight: 720};

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
      const message = JSON.parse(event.data);
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

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * metadata.deviceWidth / rect.width,
      y: (event.clientY - rect.top) * metadata.deviceHeight / rect.height
    };
  }
  canvas.addEventListener('pointerdown', event => {
    const p = point(event);
    send({type: 'input_mouse', eventType: 'mousePressed', x: p.x, y: p.y,
          button: 'left', clickCount: 1});
  });
  canvas.addEventListener('pointerup', event => {
    const p = point(event);
    send({type: 'input_mouse', eventType: 'mouseReleased', x: p.x, y: p.y,
          button: 'left', clickCount: 1});
    canvas.focus();
  });
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


def _valid_client_message(message: str) -> bool:
    if len(message.encode()) > _MAX_INPUT_BYTES:
        return False
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("type") in _ALLOWED_INPUT_TYPES


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
                        else:
                            await websocket.send_text(message)

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
        finally:
            await service.complete(token)

    return router
