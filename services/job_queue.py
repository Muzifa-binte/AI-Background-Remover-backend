"""
Async Job Queue Service.

Provides a bounded worker pool for executing heavy AI inference tasks (e.g. batch background removal).
Features:
  - Bounded concurrency: Prevents CPU/RAM exhaustion when multiple users submit batches.
  - Non-blocking execution: Runs CPU-bound inference in thread pool executors.
  - Real-time updates: Notifies subscribers via job_events.
  - Graceful lifecycle: Starts workers on FastAPI startup and cleanly shuts down on termination.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default to 2 concurrent workers for CPU safety; can be scaled up on high-core / GPU servers
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_BATCH_WORKERS", "2"))


class JobQueue:
    """Bounded async task queue and worker pool for batch jobs."""

    def __init__(self, concurrency: int = MAX_CONCURRENT_WORKERS) -> None:
        self.concurrency = max(1, concurrency)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start background worker tasks."""
        if self._running:
            return
        self._running = True
        logger.info(f"[JobQueue] Starting {self.concurrency} batch inference workers...")
        for i in range(self.concurrency):
            worker_task = asyncio.create_task(self._worker_loop(worker_id=i + 1))
            self._workers.append(worker_task)

    async def stop(self) -> None:
        """Gracefully stop all worker tasks."""
        if not self._running:
            return
        logger.info("[JobQueue] Stopping batch inference workers...")
        self._running = False
        for worker_task in self._workers:
            worker_task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("[JobQueue] All workers stopped.")

    async def enqueue(self, job_id: str) -> None:
        """Enqueue a job_id for background processing."""
        await self._queue.put(job_id)
        logger.info(f"[JobQueue] Job {job_id} enqueued (queue size: {self._queue.qsize()})")

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop processing jobs from the queue."""
        logger.info(f"[JobQueue] Worker #{worker_id} ready.")
        while self._running:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[JobQueue] Worker #{worker_id} error fetching task: {e}")
                continue

            try:
                logger.info(f"[JobQueue] Worker #{worker_id} started processing job {job_id}")
                from services.batch import process_batch_job
                await process_batch_job(job_id)
                logger.info(f"[JobQueue] Worker #{worker_id} finished processing job {job_id}")
            except asyncio.CancelledError:
                logger.warning(f"[JobQueue] Worker #{worker_id} cancelled while processing job {job_id}")
                break
            except Exception as e:
                logger.error(f"[JobQueue] Worker #{worker_id} failed on job {job_id}: {e}", exc_info=True)
            finally:
                self._queue.task_done()


# Global singleton instance
job_queue = JobQueue()
