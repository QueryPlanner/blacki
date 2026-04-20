"""
Integration guide for non-blocking browser tools in blacki repository.

This file shows how to integrate the enhanced browser tools into your existing
blacki project.
"""

import asyncio
from typing import Optional

# Import the enhanced tools
from .tools_enhanced import browser_task_async, browser_get_task_status, browser_cancel_task
from .browser_queue import browser_queue, queue_browser_task, get_queued_task_result
from .browser_websocket import browser_websocket_endpoint

# Integration patterns for different use cases

class NonBlockingBrowserIntegration:
    """
    Integration class that provides different non-blocking patterns
    based on your use case.
    """
    
    def __init__(self):
        self.mode = "enhanced"  # "enhanced", "queue", or "websocket"
        self.queue_started = False
    
    async def start_queue_mode(self, max_concurrent: int = 5):
        """Start queue mode for concurrent browser operations."""
        await browser_queue.start()
        self.queue_started = True
        self.mode = "queue"
    
    async def execute_browser_task(
        self, 
        task: str, 
        tool_context, 
        output_schema: Optional = None,
        mode: str = "enhanced",
        **kwargs
    ):
        """
        Execute browser task using the specified non-blocking mode.
        
        Args:
            task: Browser task description
            tool_context: ADK tool context
            output_schema: Optional output schema
            mode: "enhanced", "queue", or "websocket"
            **kwargs: Additional browser parameters
        """
        
        if mode == "enhanced":
            return await browser_task_async(
                task, tool_context, output_schema=output_schema, **kwargs
            )
        
        elif mode == "queue":
            if not self.queue_started:
                await self.start_queue_mode()
            
            return await queue_browser_task(
                task, 
                output_schema=output_schema, 
                **kwargs
            )
        
        elif mode == "websocket":
            # Return connection info for WebSocket usage
            return {
                "status": "websocket_required",
                "message": "Connect to WebSocket endpoint for real-time browser control",
                "websocket_url": "/browser-ws",
                "task": task
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    async def get_task_result(
        self, 
        task_id: str, 
        tool_context,
        mode: str = "enhanced",
        timeout: Optional[float] = None
    ):
        """Get result of a browser task."""
        
        if mode == "enhanced":
            return await browser_get_task_status(task_id, tool_context)
        
        elif mode == "queue":
            return await get_queued_task_result(task_id, timeout)
        
        else:
            return {"status": "error", "message": f"Mode {mode} not supported for result retrieval"}

# Global integration instance
browser_integration = NonBlockingBrowserIntegration()

# Update your existing tools.py to use non-blocking versions
async def enhanced_browser_task(*args, **kwargs):
    """Drop-in replacement for your existing browser_task function."""
    return await browser_integration.execute_browser_task(*args, **kwargs)

async def enhanced_browser_get_status(*args, **kwargs):
    """Drop-in replacement for getting task status."""
    return await browser_integration.get_task_result(*args, **kwargs)

# Example usage in your existing agent code:
"""
# In your agent.py or tools.py, replace blocking calls:

# OLD (blocking):
result = await browser_task("Navigate to example.com", tool_context)
if result["status"] == "success":
    # Process result...

# NEW (non-blocking):
task_response = await enhanced_browser_task("Navigate to example.com", tool_context)
if task_response["status"] == "started":
    task_id = task_response["task_id"]
    # Continue with other work...
    
    # Later, check result:
    result = await enhanced_browser_get_status(task_id, tool_context)
    if result["status"] == "finished":
        # Process result...
"""

# WebSocket endpoint integration (add to your FastAPI server):
"""
from fastapi import FastAPI
from .browser_websocket import browser_websocket_endpoint

app = FastAPI()

@app.websocket("/browser-ws")
async def browser_websocket(websocket: WebSocket):
    await browser_websocket_endpoint(websocket)

# Frontend usage:
# const ws = new WebSocket('ws://localhost:8000/browser-ws');
# ws.send(JSON.stringify({
#   type: 'create_task',
#   task: 'Navigate to example.com and click the login button'
# }));
# ws.onmessage = (event) => {
#   const update = JSON.parse(event.data);
#   if (update.type === 'task_started') {
#     console.log('Task started:', update.task_id);
#   } else if (update.type === 'task_completed') {
#     console.log('Task completed:', update.result);
#   }
# };
"""

# Production configuration recommendations:
"""
PRODUCTION_SETTINGS = {
    "max_concurrent_browser_tasks": 10,  # Adjust based on your server capacity
    "default_timeout": 300,  # 5 minutes
    "websocket_enabled": True,  # For real-time monitoring
    "queue_mode_for_batch": True,  # For batch operations
    "enhanced_mode_for_single": True,  # For single tasks
    "task_cleanup_interval": 3600,  # Cleanup old tasks every hour
}

# Environment variables:
# BROWSER_USE_API_KEY=your_api_key
# MAX_CONCURRENT_TASKS=5
# ENABLE_WEBSOCKET=true
# QUEUE_MODE_ENABLED=true
"""