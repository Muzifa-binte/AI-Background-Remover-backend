"""
Batch Processing Routes.

POST /api/batch/start
─────────────────────
Accepts up to 20 image files, saves them, creates a job record in MongoDB,
fires a BackgroundTask to process them, and returns the job_id immediately.

GET /api/batch/{job_id}/status
───────────────────────────────
Returns the full job state (per-file statuses, counts, overall status).
Poll this every 1–2 seconds from the frontend.

GET /api/batch/{job_id}/download
──────────────────────────────────
Streams a ZIP of all successfully processed output PNGs.
Only available when job status is "done".
"""

import os
import uuid

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response, JSONResponse

from models.user    import UserOut
from services.auth  import get_current_user
from services.quota import check_and_increment_quota
from services.batch import create_job, get_job, process_batch, build_zip

router = APIRouter(tags=["Batch Processing"])

UPLOAD_DIR    = "uploads"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10
MAX_FILES     = 20

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/batch/start")
async def batch_start(
    background_tasks: BackgroundTasks,
    files:            list[UploadFile] = File(...),
    current_user:     UserOut          = Depends(get_current_user),
):
    """
    Start a batch background-removal job.

    - **files** Up to 20 JPEG/PNG/WebP images (each ≤ 10 MB)

    Returns a **job_id** immediately. Poll `/api/batch/{job_id}/status`
    for progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES} files per batch.",
        )

    # Charge quota upfront for all files in the batch
    for _ in files:
        await check_and_increment_quota(current_user.user_id)

    file_entries: list[dict] = []

    for upload in files:
        if upload.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File '{upload.filename}' has unsupported type '{upload.content_type}'. "
                       "Use JPEG, PNG, or WebP.",
            )

        contents = await upload.read()
        if len(contents) > MAX_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename}' exceeds {MAX_SIZE_MB} MB limit.",
            )

        file_id     = str(uuid.uuid4())
        safe_name   = os.path.basename(upload.filename or "upload")
        upload_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_name}")

        async with aiofiles.open(upload_path, "wb") as f:
            await f.write(contents)

        file_entries.append({
            "original_name": safe_name,
            "upload_path":   upload_path,
        })

    job_id = await create_job(file_entries, user_id=current_user.user_id)

    # Fire-and-forget: runs in FastAPI's thread pool
    background_tasks.add_task(process_batch, job_id)

    return JSONResponse({
        "job_id":      job_id,
        "total_files": len(file_entries),
        "status":      "pending",
    })


@router.get("/batch/{job_id}/status")
async def batch_status(
    job_id:       str,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Get the current status of a batch job.

    Returns per-file statuses and overall progress counts.
    Poll every 1–2 seconds while status is "pending" or "running".
    Users can only access their own jobs.
    """
    job = await get_job(job_id, user_id=current_user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    # Strip server-side paths from the client response
    files_out = [
        {
            "original_name":   entry["original_name"],
            "output_filename": entry["output_filename"],
            "download_url":    f"/api/download/{entry['output_filename']}"
                               if entry["output_filename"] else None,
            "status":          entry["status"],
            "error":           entry["error"],
        }
        for entry in job["files"]
    ]

    return JSONResponse({
        "job_id":     job["job_id"],
        "status":     job["status"],
        "created_at": job["created_at"],
        "total":      job["total"],
        "completed":  job["completed"],
        "failed":     job["failed"],
        "files":      files_out,
    })


@router.get("/batch/{job_id}/download")
async def batch_download(
    job_id:       str,
    current_user: UserOut = Depends(get_current_user),
):
    """
    Download all successfully processed images as a ZIP archive.

    Only available once the job status is **"done"**.
    Users can only download their own jobs.
    """
    job = await get_job(job_id, user_id=current_user.user_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is still '{job['status']}'. Wait until status is 'done'.",
        )

    zip_bytes, zip_filename = build_zip(job)
    if not zip_bytes:
        raise HTTPException(
            status_code=404,
            detail="No completed files available for download.",
        )

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{zip_filename}"',
            "Content-Length":      str(len(zip_bytes)),
        },
    )
