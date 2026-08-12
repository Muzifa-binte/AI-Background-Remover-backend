"""
Batch Processing Service.

Manages batch background-removal jobs, persisted to MongoDB so job state
survives server restarts and horizontal scaling is possible.

Job states
──────────
  pending   → queued, not yet started
  running   → actively processing
  done      → all files complete (or errored individually)

Per-file states
───────────────
  queued    → waiting
  processing→ inference running
  done      → output PNG saved
  error     → inference failed for this file
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Literal

# Add AI module to path (mirrors bg_removal.py pattern)
_AI_DIR = Path(__file__).resolve().parents[2] / "AI-Background-Remover-AI"
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from inference import run_inference  # noqa: E402

from services.database import get_collection  # noqa: E402


# ── Type definitions ───────────────────────────────────────────────────────

FileStatus = Literal["queued", "processing", "done", "error"]
JobStatus  = Literal["pending", "running", "done"]


class FileEntry(TypedDict):
    original_name:   str
    upload_path:     str
    output_filename: str | None
    status:          FileStatus
    error:           str | None


class Job(TypedDict):
    job_id:     str
    user_id:    str
    status:     JobStatus
    created_at: str
    files:      list[FileEntry]
    total:      int
    completed:  int
    failed:     int


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Internal helpers ───────────────────────────────────────────────────────

def _collection():
    """Return the Motor collection for batch jobs."""
    return get_collection("batch_jobs")


async def _upsert_job(job: Job) -> None:
    """Persist the full job document to MongoDB (upsert by job_id)."""
    try:
        col = _collection()
        await col.replace_one(
            {"job_id": job["job_id"]},
            job,
            upsert=True,
        )
    except Exception:
        pass  # persistence failure should not crash the processing loop


# ── Public helpers ─────────────────────────────────────────────────────────

async def create_job(file_entries: list[dict], user_id: str) -> str:
    """
    Register a new batch job, persist it to MongoDB, and return its job_id.

    Args:
        file_entries: list of {"original_name": str, "upload_path": str}
        user_id:      ID of the user who owns this job
    """
    job_id = str(uuid.uuid4())
    files: list[FileEntry] = [
        {
            "original_name":   e["original_name"],
            "upload_path":     e["upload_path"],
            "output_filename": None,
            "status":          "queued",
            "error":           None,
        }
        for e in file_entries
    ]
    job: Job = {
        "job_id":     job_id,
        "user_id":    user_id,
        "status":     "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files":      files,
        "total":      len(files),
        "completed":  0,
        "failed":     0,
    }
    await _upsert_job(job)
    return job_id


async def get_job(job_id: str, user_id: str | None = None) -> Job | None:
    """
    Retrieve job from MongoDB.

    Args:
        job_id:  The job identifier.
        user_id: When provided, the job must belong to this user (authorization).

    Returns:
        Job dict or None if not found / not owned by user.
    """
    try:
        query: dict = {"job_id": job_id}
        if user_id:
            query["user_id"] = user_id
        col = _collection()
        doc = await col.find_one(query, {"_id": 0})
        return dict(doc) if doc else None  # type: ignore[arg-type]
    except Exception:
        return None


def process_batch(job_id: str) -> None:
    """
    Background task entry point. Runs in FastAPI's thread-pool executor.

    Iterates over queued files, runs bg removal on each, and persists
    per-file and overall job status to MongoDB after every file.
    """
    # We're in a sync thread — run async DB calls via a new event loop
    loop = asyncio.new_event_loop()

    try:
        job = loop.run_until_complete(_fetch_job_sync(job_id))
        if job is None:
            return

        job["status"] = "running"
        loop.run_until_complete(_upsert_job(job))

        for entry in job["files"]:
            entry["status"] = "processing"
            loop.run_until_complete(_upsert_job(job))

            stem = Path(entry["upload_path"]).stem
            output_filename = f"{stem}_result.png"
            output_path     = os.path.join(OUTPUT_DIR, output_filename)

            try:
                run_inference(entry["upload_path"], output_path)
                entry["output_filename"] = output_filename
                entry["status"]          = "done"
                job["completed"]        += 1
            except Exception as exc:
                entry["status"] = "error"
                entry["error"]  = str(exc)
                job["failed"]  += 1

            loop.run_until_complete(_upsert_job(job))

        job["status"] = "done"
        loop.run_until_complete(_upsert_job(job))

    finally:
        loop.close()


async def _fetch_job_sync(job_id: str) -> Job | None:
    """Fetch a job by ID without user check (internal use by process_batch)."""
    try:
        col = _collection()
        doc = await col.find_one({"job_id": job_id}, {"_id": 0})
        return dict(doc) if doc else None  # type: ignore[arg-type]
    except Exception:
        return None


def build_zip(job: Job) -> tuple[bytes, str]:
    """
    Build an in-memory ZIP of all successfully processed output files.

    Args:
        job: The job dict (already retrieved from MongoDB).

    Returns:
        (zip_bytes, zip_filename)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in job["files"]:
            if entry["status"] == "done" and entry["output_filename"]:
                file_path = os.path.join(OUTPUT_DIR, entry["output_filename"])
                if os.path.isfile(file_path):
                    zf.write(file_path, arcname=entry["output_filename"])

    buf.seek(0)
    zip_filename = f"batch_{job['job_id'][:8]}_results.zip"
    return buf.read(), zip_filename
