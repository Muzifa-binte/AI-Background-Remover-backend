"""
Tests for batch processing routes:
  POST /api/batch/start
  GET  /api/batch/{job_id}/status
  GET  /api/batch/{job_id}/events  (SSE)
  GET  /api/batch/{job_id}/download

run_inference is mocked so no AI model weights are loaded.
"""
from __future__ import annotations

import io
import json
import zipfile
import httpx
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _png_upload(name: str = "img.png"):
    from tests.conftest import MINIMAL_PNG
    return ("files", (name, io.BytesIO(MINIMAL_PNG), "image/png"))


async def _start_job(client, auth_headers, n_files=2, quality="fast"):
    """Helper: POST /api/batch/start and return the response."""
    files = [_png_upload(f"img{i}.png") for i in range(n_files)]
    with patch("services.batch.run_inference"):
        resp = await client.post(
            "/api/batch/start",
            headers=auth_headers,
            files=files,
            data={"quality": quality},
        )
    return resp


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBatchStart:
    async def test_batch_start_success(self, client: httpx.AsyncClient, auth_headers: dict):
        """Starting a batch job returns job_id and total_files."""
        with patch("services.job_queue.job_queue.enqueue"), \
             patch("inference.run_inference"):
            resp = await _start_job(client, auth_headers, n_files=2)

        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["total_files"] == 2
        assert data["quality"] == "fast"
        assert data["status"] == "pending"

    async def test_batch_start_all_quality_modes(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """All quality modes are accepted."""
        for quality in ("fast", "standard", "quality"):
            with patch("services.job_queue.job_queue.enqueue"), \
                 patch("services.batch.run_inference"):
                resp = await client.post(
                    "/api/batch/start",
                    headers=auth_headers,
                    files=[_png_upload("a.png")],
                    data={"quality": quality},
                )
            assert resp.status_code == 200, f"Quality '{quality}' failed: {resp.text}"

    async def test_batch_start_invalid_quality(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Unknown quality string returns 400."""
        resp = await client.post(
            "/api/batch/start",
            headers=auth_headers,
            files=[_png_upload()],
            data={"quality": "turbo"},
        )
        assert resp.status_code == 400

    async def test_batch_start_no_files(self, client: httpx.AsyncClient, auth_headers: dict):
        """Submitting no files returns 400."""
        resp = await client.post(
            "/api/batch/start",
            headers=auth_headers,
            data={"quality": "fast"},
        )
        assert resp.status_code in (400, 422)

    async def test_batch_start_too_many_files(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Submitting more than 20 files returns 400."""
        files = [_png_upload(f"img{i}.png") for i in range(21)]
        resp = await client.post(
            "/api/batch/start",
            headers=auth_headers,
            files=files,
            data={"quality": "fast"},
        )
        assert resp.status_code == 400

    async def test_batch_start_unauthenticated(self, client: httpx.AsyncClient):
        """Requests without token return 401."""
        from tests.conftest import MINIMAL_PNG
        resp = await client.post(
            "/api/batch/start",
            files=[("files", ("a.png", io.BytesIO(MINIMAL_PNG), "image/png"))],
            data={"quality": "fast"},
        )
        assert resp.status_code == 401

    async def test_batch_start_unsupported_file_type(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """GIF files are rejected with 400."""
        resp = await client.post(
            "/api/batch/start",
            headers=auth_headers,
            files=[("files", ("anim.gif", io.BytesIO(b"GIF89a"), "image/gif"))],
            data={"quality": "fast"},
        )
        assert resp.status_code == 400


class TestBatchStatus:
    async def test_batch_status_pending(self, client: httpx.AsyncClient, auth_headers: dict):
        """A freshly submitted job starts as pending."""
        with patch("services.job_queue.job_queue.enqueue"):
            start_resp = await _start_job(client, auth_headers)
        job_id = start_resp.json()["job_id"]

        status_resp = await client.get(
            f"/api/batch/{job_id}/status", headers=auth_headers
        )
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("pending", "running", "done")
        assert "files" in data
        assert "total" in data
        assert "completed" in data
        assert "failed" in data

    async def test_batch_status_not_found(self, client: httpx.AsyncClient, auth_headers: dict):
        """Requesting status for a non-existent job returns 404."""
        resp = await client.get("/api/batch/nonexistent-id/status", headers=auth_headers)
        assert resp.status_code == 404

    async def test_batch_status_unauthenticated(self, client: httpx.AsyncClient):
        """Status without auth returns 401."""
        resp = await client.get("/api/batch/some-job-id/status")
        assert resp.status_code == 401


class TestBatchSSEEvents:
    async def test_batch_events_returns_stream(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """SSE endpoint responds with text/event-stream content-type."""
        with patch("services.job_queue.job_queue.enqueue"):
            start_resp = await _start_job(client, auth_headers, n_files=1)
        job_id = start_resp.json()["job_id"]

        async def fake_generator(*args, **kwargs):
            yield "event: snapshot\ndata: {}\n\n"

        with patch("routes.batch.job_events.event_generator", side_effect=fake_generator):
            resp = await client.get(f"/api/batch/{job_id}/events", headers=auth_headers)
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    async def test_batch_events_not_found(self, client: httpx.AsyncClient, auth_headers: dict):
        """SSE for non-existent job returns 404."""
        resp = await client.get("/api/batch/missing-job/events", headers=auth_headers)
        assert resp.status_code == 404


class TestBatchDownload:
    async def test_batch_download_not_done(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Downloading a still-pending job returns 409."""
        with patch("services.job_queue.job_queue.enqueue"):
            start_resp = await _start_job(client, auth_headers)
        job_id = start_resp.json()["job_id"]

        resp = await client.get(f"/api/batch/{job_id}/download", headers=auth_headers)
        # Job is pending/running so download is premature → 409
        assert resp.status_code == 409

    async def test_batch_download_not_found(self, client: httpx.AsyncClient, auth_headers: dict):
        """Downloading a non-existent job returns 404."""
        resp = await client.get("/api/batch/missing-job/download", headers=auth_headers)
        assert resp.status_code == 404


class TestBatchJobProcessing:
    async def test_process_batch_job_updates_status(self):
        """process_batch_job marks files done and publishes events."""
        import os
        import asyncio
        from tests.conftest import get_memory_collection, TEST_USER_ID, MINIMAL_PNG
        from unittest.mock import patch as _patch

        # Seed a job in the in-memory DB
        jobs_col = get_memory_collection("batch_jobs")
        job_id = "test-job-abc"
        await jobs_col.insert_one({
            "job_id": job_id,
            "user_id": TEST_USER_ID,
            "quality": "fast",
            "status": "pending",
            "created_at": "2026-01-01T00:00:00+00:00",
            "files": [{
                "original_name": "a.png",
                "upload_path": "uploads/fake.png",
                "output_filename": None,
                "status": "queued",
                "error": None,
            }],
            "total": 1,
            "completed": 0,
            "failed": 0,
        })

        with _patch("services.database.get_collection", side_effect=get_memory_collection), \
             _patch("services.batch.run_inference"), \
             _patch("services.job_events.job_events.publish") as mock_publish:

            from services.batch import process_batch_job
            await process_batch_job(job_id)

            # job_done event must have been published
            published_events = [call.args[1] for call in mock_publish.call_args_list]
            assert "job_done" in published_events

        # Verify DB state updated to done
        updated = await jobs_col.find_one({"job_id": job_id})
        assert updated["status"] == "done"
        assert updated["completed"] == 1
