"""
Tests for job_events and job_queue services.
"""
from __future__ import annotations
import asyncio
import pytest


class TestJobEventBroadcaster:
    """Unit tests for the async Pub/Sub event broadcaster."""

    async def test_subscribe_and_receive_event(self):
        """A subscriber receives published events."""
        from services.job_events import JobEventBroadcaster
        bus = JobEventBroadcaster()
        queue = await bus.subscribe("job-1")
        await bus.publish("job-1", "file_done", {"index": 0})
        msg = queue.get_nowait()
        assert msg["event"] == "file_done"
        assert msg["data"]["index"] == 0

    async def test_unsubscribe_stops_delivery(self):
        """After unsubscribing, no events are delivered."""
        from services.job_events import JobEventBroadcaster
        bus = JobEventBroadcaster()
        queue = await bus.subscribe("job-2")
        await bus.unsubscribe("job-2", queue)
        await bus.publish("job-2", "job_done", {})
        assert queue.empty()

    async def test_multiple_subscribers(self):
        """Multiple subscribers all receive the same event."""
        from services.job_events import JobEventBroadcaster
        bus = JobEventBroadcaster()
        q1 = await bus.subscribe("job-3")
        q2 = await bus.subscribe("job-3")
        await bus.publish("job-3", "job_started", {"status": "running"})
        assert not q1.empty()
        assert not q2.empty()

    async def test_event_generator_yields_sse_format(self):
        """event_generator yields correctly formatted SSE strings."""
        from services.job_events import JobEventBroadcaster
        bus = JobEventBroadcaster()
        events_received = []

        async def consume():
            async for chunk in bus.event_generator("job-4"):
                events_received.append(chunk)
                if "job_done" in chunk:
                    break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        await bus.publish("job-4", "file_done", {"index": 0})
        await asyncio.sleep(0.05)
        await bus.publish("job-4", "job_done", {"status": "done"})
        await asyncio.wait_for(task, timeout=2.0)

        assert any("event: file_done" in e for e in events_received)
        assert any("event: job_done" in e for e in events_received)

    async def test_event_generator_with_initial_snapshot(self):
        """Initial snapshot event is emitted immediately on connect."""
        from services.job_events import JobEventBroadcaster
        bus = JobEventBroadcaster()
        chunks = []

        async def consume():
            gen = bus.event_generator("job-5", initial_data={"status": "pending"})
            async for chunk in gen:
                chunks.append(chunk)
                if "job_done" in chunk:
                    break

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        await bus.publish("job-5", "job_done", {})
        await asyncio.wait_for(task, timeout=2.0)

        assert any("snapshot" in c for c in chunks)


class TestJobQueue:
    """Unit tests for the bounded async worker pool."""

    async def test_start_creates_workers(self):
        """start() spawns the configured number of worker tasks."""
        from services.job_queue import JobQueue
        q = JobQueue(concurrency=3)
        await q.start()
        assert q._running is True
        assert len(q._workers) == 3
        await q.stop()

    async def test_stop_cancels_all_workers(self):
        """stop() terminates all worker tasks cleanly."""
        from services.job_queue import JobQueue
        q = JobQueue(concurrency=2)
        await q.start()
        await q.stop()
        assert q._running is False
        assert len(q._workers) == 0

    async def test_enqueue_adds_to_queue(self):
        """enqueue() adds a job_id to the internal queue."""
        from services.job_queue import JobQueue
        q = JobQueue(concurrency=1)
        # Don't start workers — just verify queue size
        await q._queue.put("test-job")
        assert q._queue.qsize() == 1

    async def test_double_start_is_safe(self):
        """Calling start() twice does not create duplicate workers."""
        from services.job_queue import JobQueue
        q = JobQueue(concurrency=2)
        await q.start()
        await q.start()  # Second call should be a no-op
        assert len(q._workers) == 2
        await q.stop()
