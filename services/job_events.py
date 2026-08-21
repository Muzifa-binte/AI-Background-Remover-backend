"""
Job Events Service.

Provides an asynchronous Pub/Sub event broadcaster for batch job lifecycle events:
  - job_started:       Batch job execution has begun
  - file_processing:   Individual file inference started
  - file_done:         Individual file completed successfully
  - file_error:        Individual file failed
  - job_done:          All files in the batch have been processed

Connected clients subscribe via Server-Sent Events (SSE) at /api/batch/{job_id}/events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Set

logger = logging.getLogger(__name__)


class JobEventBroadcaster:
    """Manages event subscriptions per job_id and broadcasts events to connected clients."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Register a new subscriber queue for a specific job."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = set()
            self._subscribers[job_id].add(queue)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue when client disconnects."""
        async with self._lock:
            if job_id in self._subscribers:
                self._subscribers[job_id].discard(queue)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    async def publish(self, job_id: str, event_type: str, data: dict) -> None:
        """Broadcast an event payload to all active subscribers for job_id."""
        async with self._lock:
            queues = list(self._subscribers.get(job_id, set()))

        if not queues:
            return

        message = {
            "event": event_type,
            "data": data,
        }

        for q in queues:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # Discard oldest if consumer is lagging
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except Exception:
                    pass

    async def event_generator(
        self,
        job_id: str,
        initial_data: dict | None = None,
        heartbeat_interval: float = 15.0,
    ) -> AsyncGenerator[str, None]:
        """
        Yield SSE formatted messages:
          event: <event_type>\n
          data: <json_string>\n\n
        """
        queue = await self.subscribe(job_id)
        try:
            # Emit initial snapshot event if available
            if initial_data:
                yield f"event: snapshot\ndata: {json.dumps(initial_data)}\n\n"

            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
                    event_name = msg.get("event", "message")
                    data_str = json.dumps(msg.get("data", {}))
                    yield f"event: {event_name}\ndata: {data_str}\n\n"

                    # If job has completed, stop stream after sending final event
                    if event_name in ("job_done", "job_error"):
                        break
                except asyncio.TimeoutError:
                    # Send periodic keep-alive comment to prevent proxy timeouts
                    yield ": ping\n\n"
        finally:
            await self.unsubscribe(job_id, queue)


# Global singleton instance
job_events = JobEventBroadcaster()
