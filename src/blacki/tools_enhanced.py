"""Enhanced browser tools with non-blocking patterns."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import weakref
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import httpx
from browser_use_sdk import AsyncBrowserUse
from browser_use_sdk.v2.client import SessionSettings
from google.adk.tools import ToolContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_BROWSER_USE_DEFAULT_MODEL = "browser-use-llm"

@dataclass
class TaskState:
    """Track state of browser tasks."""
    task_id: str
    session_id: str
    created_at: float
    callback: Optional[callable] = None
    status: str = "pending"
    result: Optional[dict] = None
    
@dataclass
class BrowserPool:
    """Pool of browser clients for concurrent operations."""
    clients: Dict[str, AsyncBrowserUse] = field(default_factory=dict)
    active_tasks: Dict[str, TaskState] = field(default_factory=dict)
    task_monitors: Dict[str, asyncio.Task] = field(default_factory=dict)
    
    def __post_init__(self):
        # Cleanup on exit
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self._cleanup)
    
    async def _cleanup(self, signum=None, frame=None):
        """Clean up resources on shutdown."""
        logger.info("Cleaning up browser pool resources...")
        for client in self.clients.values():
            with suppress(Exception):
                await client.close()
        for task in self.task_monitors.values():
            task.cancel()
        
_browser_pool = BrowserPool()

class NonBlockingBrowserUse:
    """Non-blocking browser use wrapper with proper async task monitoring."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncBrowserUse(api_key=api_key)
        
    async def create_and_monitor_task(
        self, 
        task_description: str,
        output_schema: Optional[type[BaseModel]] = None,
        callback: Optional[callable] = None,
        **kwargs
    ) -> str:
        """Create a task and start monitoring without blocking.
        
        Returns:
            Task ID immediately, starts monitoring in background.
        """
        # Create task
        created = await self.client.tasks.create(task_description, **kwargs)
        task_id = str(created.id)
        session_id = str(created.session_id)
        
        # Register task for monitoring
        task_state = TaskState(
            task_id=task_id,
            session_id=session_id,
            created_at=asyncio.get_event_loop().time(),
            callback=callback
        )
        
        _browser_pool.active_tasks[task_id] = task_state
        
        # Start background monitoring
        monitor_task = asyncio.create_task(
            self._monitor_task(task_state, output_schema)
        )
        _browser_pool.task_monitors[task_id] = monitor_task
        
        logger.info(f"Started non-blocking task monitoring: {task_id}")
        return task_id
    
    async def _monitor_task(
        self, 
        task_state: TaskState, 
        output_schema: Optional[type[BaseModel]] = None
    ):
        """Monitor task completion asynchronously."""
        try:
            while True:
                # Get task status
                task_view = await self.client.tasks.get(task_state.task_id)
                status = task_view.status.value
                
                # Update state
                task_state.status = status
                
                if status in ("finished", "failed", "stopped"):
                    # Prepare result
                    output = task_view.output
                    if output and output_schema:
                        try:
                            parsed = json.loads(output)
                            output = output_schema.model_validate(parsed)
                        except Exception as e:
                            logger.warning(f"Failed to parse structured output: {e}")
                    
                    result = {
                        "status": status,
                        "task_id": task_state.task_id,
                        "session_id": task_state.session_id,
                        "is_success": task_view.is_success,
                        "output": _serialize_browser_output(output),
                    }
                    
                    task_state.result = result
                    
                    # Notify callback if provided
                    if task_state.callback:
                        try:
                            await task_state.callback(result)
                        except Exception as e:
                            logger.error(f"Task callback failed: {e}")
                    
                    break
                
                # Wait before next check with exponential backoff
                await asyncio.sleep(min(2 ** (asyncio.get_event_loop().time() - task_state.created_at) * 0.1, 30))
                
        except asyncio.CancelledError:
            logger.info(f"Task monitoring cancelled: {task_state.task_id}")
            raise
        except Exception as e:
            logger.error(f"Task monitoring error: {e}")
            task_state.status = "error"
            task_state.result = {
                "status": "error",
                "error": str(e),
                "task_id": task_state.task_id,
                "session_id": task_state.session_id,
                "is_success": False,
            }
        finally:
            # Cleanup
            _browser_pool.task_monitors.pop(task_state.task_id, None)

# Enhanced async browser task with non-blocking behavior
async def browser_task_async(
    task: str,
    tool_context: ToolContext,
    output_schema: Optional[type[BaseModel]] = None,
    keep_alive: bool = False,
    session_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    model: Optional[str] = None,
    start_url: Optional[str] = None,
    max_steps: Optional[int] = None,
    proxy_country: Optional[str] = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Non-blocking browser task that returns immediately with a task ID."""
    
    api_key = os.environ.get("BROWSER_USE_API_KEY", "").strip()
    if not api_key:
        return {"status": "error", "error": "BROWSER_USE_API_KEY not set", "output": None}
    
    # Validate task
    if not task.strip():
        return {"status": "error", "error": "Task description must be non-empty", "output": None}
    
    # Prepare task creation kwargs
    create_kwargs = {
        "llm": model or _BROWSER_USE_DEFAULT_MODEL,
        "keepAlive": keep_alive,
    }
    
    if session_id:
        create_kwargs["session_id"] = session_id
    if start_url:
        create_kwargs["start_url"] = start_url
    if max_steps:
        create_kwargs["max_steps"] = max_steps
    
    # Session settings
    session_settings = {}
    if profile_id:
        session_settings["profileId"] = profile_id
    if proxy_country:
        session_settings["proxyCountryCode"] = proxy_country
    if session_settings:
        create_kwargs["session_settings"] = SessionSettings(**session_settings)
    
    # Handle structured output
    if output_schema:
        if isinstance(output_schema, dict):
            create_kwargs["structured_output"] = json.dumps(output_schema)
        elif isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            create_kwargs["structured_output"] = json.dumps(output_schema.model_json_schema())
    
    try:
        # Create browser instance and start monitoring
        browser = NonBlockingBrowserUse(api_key)
        
        # Create task and get ID immediately
        task_id = await browser.create_and_monitor_task(
            task.strip(),
            output_schema=output_schema,
            **create_kwargs
        )
        
        return {
            "status": "started",
            "task_id": task_id,
            "message": f"Task started. Monitor with browser_get_task_status('{task_id}')",
            "output": None,
        }
        
    except Exception as e:
        logger.exception("Failed to start browser task")
        return {
            "status": "error", 
            "error": str(e),
            "output": None,
        }

async def browser_get_task_status(
    task_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Get status of a browser task without blocking."""
    
    task_state = _browser_pool.active_tasks.get(task_id)
    if not task_state:
        return {"status": "error", "error": "Task not found", "output": None}
    
    # If task is completed, return result
    if task_state.result:
        # Remove from active tasks if completed
        if task_state.status in ("finished", "failed", "stopped", "error"):
            _browser_pool.active_tasks.pop(task_id, None)
        return task_state.result
    
    # Return current status
    return {
        "status": task_state.status,
        "task_id": task_id,
        "session_id": task_state.session_id,
        "message": "Task still running...",
    }

async def browser_cancel_task(
    task_id: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Cancel a running browser task."""
    
    # Cancel monitoring task
    monitor_task = _browser_pool.task_monitors.get(task_id)
    if monitor_task:
        monitor_task.cancel()
    
    # Update task state
    task_state = _browser_pool.active_tasks.get(task_id)
    if task_state:
        task_state.status = "cancelled"
        task_state.result = {
            "status": "cancelled",
            "task_id": task_id,
            "message": "Task cancelled by user"
        }
    
    return {"status": "success", "message": f"Task {task_id} cancelled"}

def _serialize_browser_output(output: Any) -> Any:
    """Serialize browser output for JSON."""
    if output is None:
        return None
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    if isinstance(output, (list, tuple)):
        return [_serialize_browser_output(item) for item in output]
    if isinstance(output, dict):
        return {key: _serialize_browser_output(value) for key, value in output.items()}
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output
    return output