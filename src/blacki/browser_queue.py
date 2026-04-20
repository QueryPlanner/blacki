"""Browser task queue system for concurrent non-blocking operations."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

import httpx
from browser_use_sdk import AsyncBrowserUse
from browser_use_sdk.v2.client import SessionSettings
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BrowserTaskRequest:
    """Request for browser task execution."""
    task_id: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    output_schema: Optional[type[BaseModel]] = None
    keep_alive: bool = False
    session_id: Optional[str] = None
    profile_id: Optional[str] = None
    model: Optional[str] = None
    start_url: Optional[str] = None
    max_steps: Optional[int] = None
    proxy_country: Optional[str] = None
    callback: Optional[Callable] = None
    timeout: int = 300
    
    # Queue metadata
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None

@dataclass
class BrowserTaskResult:
    """Result of browser task execution."""
    task_id: str
    status: str  # pending, running, finished, failed, cancelled
    result: Optional[dict] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None

class BrowserTaskQueue:
    """Asynchronous queue for managing browser tasks concurrently."""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, BrowserTaskRequest] = {}
        self.task_results: Dict[str, BrowserTaskResult] = {}
        self.browser_clients: Dict[str, AsyncBrowserUse] = {}
        self.queue_processor_task: Optional[asyncio.Task] = None
        self._stopped = False
        
    async def start(self):
        """Start the task queue processor."""
        if not self.queue_processor_task:
            self.queue_processor_task = asyncio.create_task(self._process_queue())
            logger.info("Browser task queue started")
    
    async def stop(self):
        """Stop the task queue processor."""
        self._stopped = True
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            await asyncio.gather(self.queue_processor_task, return_exceptions=True)
        
        # Close all browser clients
        for client in self.browser_clients.values():
            try:
                await client.close()
            except Exception as e:
                logger.error(f"Error closing browser client: {e}")
    
    async def submit_task(self, request: BrowserTaskRequest) -> str:
        """Submit a browser task to the queue."""
        await self.task_queue.put((request.priority.value, request.created_at, request))
        
        # Initialize result tracking
        self.task_results[request.task_id] = BrowserTaskResult(
            task_id=request.task_id,
            status="pending"
        )
        
        logger.info(f"Task {request.task_id} submitted to queue")
        return request.task_id
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> BrowserTaskResult:
        """Get result of a specific task."""
        if task_id not in self.task_results:
            raise ValueError(f"Task {task_id} not found")
        
        # Wait for completion if not finished
        result = self.task_results[task_id]
        if result.status not in ("finished", "failed", "cancelled"):
            start_time = asyncio.get_event_loop().time()
            while result.status not in ("finished", "failed", "cancelled"):
                if timeout and (asyncio.get_event_loop().time() - start_time) >= timeout:
                    raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
                await asyncio.sleep(0.1)
        
        return result
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        # Mark as cancelled
        if task_id in self.task_results:
            self.task_results[task_id].status = "cancelled"
            self.task_results[task_id].completed_at = asyncio.get_event_loop().time()
        
        # Remove from running tasks
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        logger.info(f"Task {task_id} cancelled")
        return True
    
    async def _process_queue(self):
        """Process tasks from the queue concurrently."""
        try:
            while not self._stopped:
                # Check if we can start more tasks
                if len(self.running_tasks) >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next task
                try:
                    priority, created_at, request = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Start task processing
                asyncio.create_task(self._execute_task(request))
        
        except asyncio.CancelledError:
            logger.info("Task queue processor cancelled")
        except Exception as e:
            logger.error(f"Task queue processor error: {e}")
    
    async def _execute_task(self, request: BrowserTaskRequest):
        """Execute a single browser task."""
        api_key = os.environ.get("BROWSER_USE_API_KEY")
        if not api_key:
            self._complete_task(request, None, "BROWSER_USE_API_KEY not set")
            return
        
        try:
            # Mark as running
            request.started_at = asyncio.get_event_loop().time()
            self.task_results[request.task_id].status = "running"
            
            # Get or create browser client
            if api_key not in self.browser_clients:
                self.browser_clients[api_key] = AsyncBrowserUse(api_key=api_key)
            client = self.browser_clients[api_key]
            
            # Track running task
            self.running_tasks[request.task_id] = request
            
            # Prepare task creation arguments
            create_kwargs = {
                "llm": request.model or "browser-use-llm",
                "keepAlive": request.keep_alive,
            }
            
            if request.session_id:
                create_kwargs["session_id"] = request.session_id
            if request.start_url:
                create_kwargs["start_url"] = request.start_url
            if request.max_steps:
                create_kwargs["max_steps"] = request.max_steps
            
            # Session settings
            session_settings = {}
            if request.profile_id:
                session_settings["profileId"] = request.profile_id
            if request.proxy_country:
                session_settings["proxyCountryCode"] = request.proxy_country
            if session_settings:
                create_kwargs["session_settings"] = SessionSettings(**session_settings)
            
            # Structured output
            if request.output_schema:
                if isinstance(request.output_schema, dict):
                    create_kwargs["structured_output"] = json.dumps(request.output_schema)
                elif isinstance(request.output_schema, type) and issubclass(request.output_schema, BaseModel):
                    create_kwargs["structured_output"] = json.dumps(request.output_schema.model_json_schema())
            
            # Create browser task
            created = await client.tasks.create(request.description.strip(), **create_kwargs)
            session_id = str(created.session_id)
            
            # Monitor task completion
            result = await self._monitor_task_completion(
                client, str(created.id), request, session_id
            )
            
            self._complete_task(request, result)
        
        except Exception as e:
            logger.exception(f"Task {request.task_id} execution failed")
            self._complete_task(request, None, str(e))
        finally:
            # Remove from running tasks
            self.running_tasks.pop(request.task_id, None)
    
    async def _monitor_task_completion(
        self, 
        client: AsyncBrowserUse, 
        task_id: str, 
        request: BrowserTaskRequest,
        session_id: str
    ) -> dict:
        """Monitor task completion with timeout."""
        import time
        
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < request.timeout:
            try:
                task_view = await client.tasks.get(task_id)
                status = task_view.status.value
                
                if status in ("finished", "failed", "stopped"):
                    output = task_view.output
                    if output and request.output_schema:
                        try:
                            parsed = json.loads(output)
                            output = request.output_schema.model_validate(parsed)
                        except Exception:
                            pass
                    
                    return {
                        "status": status,
                        "task_id": task_id,
                        "session_id": session_id,
                        "is_success": task_view.is_success,
                        "output": _serialize_browser_output(output),
                    }
                
                # Exponential backoff
                await asyncio.sleep(min(2 ** (time.monotonic() - start_time) * 0.1, 30))
            
            except Exception as e:
                logger.error(f"Error monitoring task {task_id}: {e}")
                await asyncio.sleep(5)
        
        # Timeout
        return {
            "status": "timeout",
            "error": f"Task timed out after {request.timeout} seconds",
            "task_id": task_id,
            "session_id": session_id,
            "is_success": False,
        }
    
    def _complete_task(self, request: BrowserTaskRequest, result: Optional[dict], error: Optional[str] = None):
        """Mark task as completed."""
        completed_time = asyncio.get_event_loop().time()
        
        self.task_results[request.task_id].status = "finished" if result else "failed"
        self.task_results[request.task_id].completed_at = completed_time
        self.task_results[request.task_id].result = result
        self.task_results[request.task_id].error = error
        
        # Call callback if provided
        if request.callback:
            try:
                asyncio.create_task(request.callback(request.task_id, result, error))
            except Exception as e:
                logger.error(f"Task callback failed: {e}")
        
        logger.info(f"Task {request.task_id} completed: {self.task_results[request.task_id].status}")

# Global task queue instance
browser_queue = BrowserTaskQueue(max_concurrent=5)

# Helper functions for easy integration
async def queue_browser_task(
    description: str,
    priority: TaskPriority = TaskPriority.NORMAL,
    output_schema: Optional[type[BaseModel]] = None,
    **kwargs
) -> str:
    """Queue a browser task for asynchronous execution."""
    await browser_queue.start()
    
    task_id = str(uuid4())
    request = BrowserTaskRequest(
        task_id=task_id,
        description=description,
        priority=priority,
        output_schema=output_schema,
        **kwargs
    )
    
    return await browser_queue.submit_task(request)

async def get_queued_task_result(task_id: str, timeout: Optional[float] = None) -> dict:
    """Get result of a queued browser task."""
    result = await browser_queue.get_task_result(task_id, timeout)
    return {
        "task_id": result.task_id,
        "status": result.status,
        "result": result.result,
        "error": result.error,
    }

# Auto-start queue
import os
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