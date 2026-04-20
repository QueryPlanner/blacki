"""WebSocket-based real-time browser system for true non-blocking operations."""

import asyncio
import json
import logging
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Set, Optional, Callable
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from browser_use_sdk import AsyncBrowserUse
from browser_use_sdk.v2.client import SessionSettings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class WebSocketBrowserSession:
    """WebSocket session for real-time browser control."""
    session_id: str
    websocket: WebSocket
    browser_client: AsyncBrowserUse
    active_tasks: Dict[str, Any] = field(default_factory=dict)
    
class WebSocketBrowserManager:
    """Manager for WebSocket-based browser operations."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.sessions: Dict[str, WebSocketBrowserSession] = {}
        self.task_listeners: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None) -> str:
        """Accept WebSocket connection and create browser session."""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Create or reuse session
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            logger.info(f"Reusing browser session: {session_id}")
        else:
            # Create new browser client
            api_key = os.environ.get("BROWSER_USE_API_KEY")
            if not api_key:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "BROWSER_USE_API_KEY not configured"
                }))
                await websocket.close()
                return None
            
            if not session_id:
                session_id = str(uuid4())
            
            browser_client = AsyncBrowserUse(api_key=api_key)
            session = WebSocketBrowserSession(
                session_id=session_id,
                websocket=websocket,
                browser_client=browser_client
            )
            self.sessions[session_id] = session
            logger.info(f"Created new browser session: {session_id}")
        
        # Send session info
        await websocket.send_text(json.dumps({
            "type": "session_info",
            "session_id": session_id,
            "message": "Browser session established"
        }))
        
        return session_id
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        
        # Clean up sessions that reference this websocket
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session.websocket == websocket:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self._cleanup_session(session_id)
    
    async def handle_message(self, websocket: WebSocket, message: dict):
        """Handle incoming WebSocket messages."""
        try:
            message_type = message.get("type")
            
            if message_type == "create_task":
                await self._create_browser_task(websocket, message)
            elif message_type == "get_task_status":
                await self._get_task_status(websocket, message)
            elif message_type == "cancel_task":
                await self._cancel_task(websocket, message)
            elif message_type == "create_session":
                await self._create_session(websocket, message)
            elif message_type == "close_session":
                await self._close_session(websocket, message)
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }))
        
        except Exception as e:
            logger.exception("Error handling WebSocket message")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def _create_browser_task(self, websocket: WebSocket, message: dict):
        """Create and start monitoring a browser task."""
        # Find session for this websocket
        session = self._get_session_for_websocket(websocket)
        if not session:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No active browser session"
            }))
            return
        
        task_id = str(uuid4())
        task_description = message.get("task", "")
        
        if not task_description:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Task description is required"
            }))
            return
        
        # Store task listener
        if task_id not in self.task_listeners:
            self.task_listeners[task_id] = set()
        self.task_listeners[task_id].add(websocket)
        
        # Start task monitoring in background
        asyncio.create_task(self._monitor_task_realtime(session, task_id, task_description, message))
        
        await websocket.send_text(json.dumps({
            "type": "task_created",
            "task_id": task_id,
            "message": "Task started, monitoring for updates..."
        }))
    
    async def _monitor_task_realtime(
        self, 
        session: WebSocketBrowserSession, 
        task_id: str, 
        task_description: str, 
        message: dict
    ):
        """Monitor task and send real-time updates."""
        try:
            # Prepare task creation
            create_kwargs = {
                "llm": message.get("model") or "browser-use-llm",
                "keepAlive": message.get("keep_alive", False),
            }
            
            if message.get("start_url"):
                create_kwargs["start_url"] = message["start_url"]
            if message.get("max_steps"):
                create_kwargs["max_steps"] = message["max_steps"]
            
            # Session settings
            session_settings = {}
            if message.get("profile_id"):
                session_settings["profileId"] = message["profile_id"]
            if message.get("proxy_country"):
                session_settings["proxyCountryCode"] = message["proxy_country"]
            if session_settings:
                create_kwargs["session_settings"] = SessionSettings(**session_settings)
            
            # Create task
            created = await session.browser_client.tasks.create(
                task_description.strip(), 
                **create_kwargs
            )
            
            # Send task started
            await self._broadcast_to_listeners(task_id, {
                "type": "task_started",
                "task_id": task_id,
                "browser_task_id": str(created.id),
                "session_id": session.session_id,
                "message": "Browser task started"
            })
            
            # Monitor completion
            result = await self._monitor_task_completion(
                session.browser_client, 
                str(created.id), 
                message.get("timeout", 300)
            )
            
            # Send completion
            await self._broadcast_to_listeners(task_id, {
                "type": "task_completed",
                "task_id": task_id,
                "result": result,
                "message": "Task completed"
            })
        
        except Exception as e:
            logger.exception(f"Task {task_id} failed")
            await self._broadcast_to_listeners(task_id, {
                "type": "task_error",
                "task_id": task_id,
                "error": str(e),
                "message": "Task failed"
            })
        finally:
            # Cleanup
            self.task_listeners.pop(task_id, None)
    
    async def _monitor_task_completion(
        self, 
        client: AsyncBrowserUse, 
        task_id: str, 
        timeout: int
    ) -> dict:
        """Monitor task completion."""
        import time
        
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            try:
                task_view = await client.tasks.get(task_id)
                status = task_view.status.value
                
                if status in ("finished", "failed", "stopped"):
                    return {
                        "status": status,
                        "is_success": task_view.is_success,
                        "output": task_view.output,
                        "completed_at": time.monotonic()
                    }
                
                # Send progress updates for long-running tasks
                elapsed = time.monotonic() - start_time
                if elapsed > 10:  # After 10 seconds
                    await asyncio.sleep(2)
                else:
                    await asyncio.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error monitoring task {task_id}: {e}")
                await asyncio.sleep(2)
        
        return {
            "status": "timeout",
            "error": f"Task timed out after {timeout} seconds"
        }
    
    async def _get_task_status(self, websocket: WebSocket, message: dict):
        """Get status of a specific task."""
        task_id = message.get("task_id")
        if not task_id:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "task_id is required"
            }))
            return
        
        # Check if this task has listeners
        if task_id in self.task_listeners:
            await websocket.send_text(json.dumps({
                "type": "task_info",
                "task_id": task_id,
                "status": "monitoring",
                "message": "Task is being monitored in real-time"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "task_info",
                "task_id": task_id,
                "status": "not_found",
                "message": "Task not found or not being monitored"
            }))
    
    async def _cancel_task(self, websocket: WebSocket, message: dict):
        """Cancel a running task."""
        task_id = message.get("task_id")
        if not task_id:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "task_id is required"
            }))
            return
        
        # Send cancellation confirmation
        await self._broadcast_to_listeners(task_id, {
            "type": "task_cancelled",
            "task_id": task_id,
            "message": "Task cancelled by user"
        })
        
        # Cleanup
        self.task_listeners.pop(task_id, None)
    
    async def _create_session(self, websocket: WebSocket, message: dict):
        """Create a new browser session."""
        session_id = await self.connect(websocket, message.get("session_id"))
        if session_id:
            await websocket.send_text(json.dumps({
                "type": "session_created",
                "session_id": session_id
            }))
    
    async def _close_session(self, websocket: WebSocket, message: dict):
        """Close a browser session."""
        session = self._get_session_for_websocket(websocket)
        if session:
            self._cleanup_session(session.session_id)
            await websocket.send_text(json.dumps({
                "type": "session_closed",
                "session_id": session.session_id
            }))
    
    def _get_session_for_websocket(self, websocket: WebSocket) -> Optional[WebSocketBrowserSession]:
        """Find session for a WebSocket connection."""
        for session in self.sessions.values():
            if session.websocket == websocket:
                return session
        return None
    
    def _cleanup_session(self, session_id: str):
        """Clean up a browser session."""
        session = self.sessions.pop(session_id, None)
        if session:
            try:
                # Close browser client
                asyncio.create_task(session.browser_client.close())
                
                # Remove from listeners
                task_ids_to_remove = []
                for task_id, listeners in self.task_listeners.items():
                    listeners.discard(session.websocket)
                    if not listeners:
                        task_ids_to_remove.append(task_id)
                
                for task_id in task_ids_to_remove:
                    self.task_listeners.pop(task_id, None)
            
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
    
    async def _broadcast_to_listeners(self, task_id: str, message: dict):
        """Broadcast message to all listeners of a task."""
        listeners = self.task_listeners.get(task_id, set())
        disconnected = set()
        
        for websocket in listeners.copy():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected websockets
        listeners.difference_update(disconnected)

# Global WebSocket manager
ws_browser_manager = WebSocketBrowserManager()

# FastAPI WebSocket endpoint handler
async def browser_websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for browser operations."""
    session_id = None
    try:
        # Initial connection
        session_id = await ws_browser_manager.connect(websocket)
        if not session_id:
            return
        
        # Handle messages
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            await ws_browser_manager.handle_message(websocket, data)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        ws_browser_manager.disconnect(websocket)

import os