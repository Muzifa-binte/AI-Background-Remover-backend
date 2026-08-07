"""
Batch Processing Service.

Manages in-memory job state for batch background-removal jobs.
Each job tracks per-file status so the frontend can poll for live progress.

Design
──────
- Jobs are stored in a module-level dict (sufficient for single-process
  deployments; swap for Redis if you later scale horizontally).
- process_batch() is called from a FastAPI BackgroundTask — it runs the
  existing run_inference() pipeline sequentially per file.
- The zip download streams all completed output files on demand.

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
    status:     JobStatus
    created_at: str
    files:      list[FileEntry]
    total:      int
    completed:  int
    failed:     int


# ── In-memory store ────────────────────────────────────────────────────────

_jobs: dict[str, Job] = {}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Public helpers ─────────────────────────────────────────────────────────

def create_job(file_entries: list[dict]) -> str:
    """
    Register a new batch job and return its job_id.

    Args:
        file_entries: list of {"original_name": str, "upload_path": str}
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
    _jobs[job_id] = {
        "job_id":     job_id,
        "status":     "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files":      files,
        "total":      len(files),
        "completed":  0,
        "failed":     0,
    }
    return job_id


def get_job(job_id: str) -> Job | None:
    """Return the job state dict or None if not found."""
    return _jobs.get(job_id)


def process_batch(job_id: str) -> None:
    """
    Background task entry point.

    Iterates over queued files, runs bg removal on each, and updates
    per-file and overall job status in place.

    This function is intentionally synchronous — it runs inside a
    FastAPI BackgroundTask thread.
    """
    job = _jobs.get(job_id)
    if job is None:
        return

    job["status"] = "running"

    for entry in job["files"]:
        entry["status"] = "processing"

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

    job["status"] = "done"


def build_zip(job_id: str) -> tuple[bytes, str] | None:
    """
    Build an in-memory ZIP of all successfully processed output files.

    Returns:
        (zip_bytes, zip_filename) or None if job not found / no files done.
    """
    job = _jobs.get(job_id)
    if job is None:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in job["files"]:
            if entry["status"] == "done" and entry["output_filename"]:
                file_path = os.path.join(OUTPUT_DIR, entry["output_filename"])
                if os.path.isfile(file_path):
                    zf.write(file_path, arcname=entry["output_filename"])

    buf.seek(0)
    zip_filename = f"batch_{job_id[:8]}_results.zip"
    return buf.read(), zip_filename


def cleanup_job(job_id: str) -> None:
    """Remove job from in-memory store (call after download if desired)."""
    _jobs.pop(job_id, None)
