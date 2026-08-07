"""
Smart Crop Route.

POST /api/smart-crop
────────────────────
Accepts a raw image upload, runs the full bg-removal pipeline to produce
a transparent PNG, then auto-detects the subject bounding box from the
alpha mask and returns a tightly-cropped result.

Form fields
───────────
  file           File   (required) JPEG / PNG / WebP, max 10 MB
  padding_pct    float  Extra padding as fraction of bbox (0.0–0.5, default 0.05)
  aspect_ratio   str    "free"|"1:1"|"4:3"|"3:4"|"16:9"|"9:16"|... (default "free")
  min_size       int    Minimum side length in px (default 64)
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.database  import get_collection
from services.bg_removal import remove_background
from services.smart_crop import smart_crop, get_aspect_ratio_keys

router = APIRouter(tags=["Smart Crop"])

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_TYPES  = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB    = 10


@router.post("/smart-crop")
async def smart_crop_endpoint(
    file:         UploadFile = File(...),
    padding_pct:  float      = Form(0.05),
    aspect_ratio: str        = Form("free"),
    min_size:     int        = Form(64),
):
    """
    Remove background and auto-crop to the detected subject.

    - **file**          Source image (JPEG, PNG, WebP ≤ 10 MB)
    - **padding_pct**   Padding around subject as fraction of bbox (0.0–0.5)
    - **aspect_ratio**  Output aspect ratio (free / 1:1 / 4:3 / 3:4 / 16:9 / 9:16 / …)
    - **min_size**      Minimum output side length in pixels
    """
    # ── Validate ──────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    valid_ratios = get_aspect_ratio_keys()
    if aspect_ratio not in valid_ratios:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aspect_ratio '{aspect_ratio}'. Choose from: {valid_ratios}",
        )

    padding_pct = max(0.0, min(0.5, padding_pct))
    min_size    = max(16, min(2048, min_size))

    # ── Save upload ───────────────────────────────────────────────────────
    upload_id  = str(uuid.uuid4())
    safe_name  = os.path.basename(file.filename or "upload")
    upload_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{safe_name}")

    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(contents)

    # ── Step 1: remove background ─────────────────────────────────────────
    removed_filename = f"{upload_id}_removed.png"
    removed_path     = os.path.join(OUTPUT_DIR, removed_filename)

    try:
        await remove_background(upload_path, removed_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}")

    # ── Step 2: smart crop ────────────────────────────────────────────────
    cropped_filename = f"{upload_id}_cropped.png"
    cropped_path     = os.path.join(OUTPUT_DIR, cropped_filename)

    try:
        loop = asyncio.get_event_loop()
        crop_meta = await loop.run_in_executor(
            None,
            lambda: smart_crop(
                fg_path       = removed_path,
                output_path   = cropped_path,
                original_path = upload_path,
                padding_pct   = padding_pct,
                aspect_ratio  = aspect_ratio,
                min_size      = min_size,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Smart crop failed: {exc}")

    # ── Persist metadata ──────────────────────────────────────────────────
    try:
        collection = get_collection("smart_crop_history")
        await collection.insert_one({
            "upload_id":          upload_id,
            "original_name":      safe_name,
            "removed_filename":   removed_filename,
            "cropped_filename":   cropped_filename,
            "settings": {
                "padding_pct":  padding_pct,
                "aspect_ratio": aspect_ratio,
                "min_size":     min_size,
            },
            "crop_meta":          crop_meta,
            "created_at":         datetime.now(timezone.utc),
        })
    except Exception:
        pass

    return JSONResponse({
        "upload_id":          upload_id,
        "removed_filename":   removed_filename,
        "cropped_filename":   cropped_filename,
        "removed_url":        f"/api/download/{removed_filename}",
        "cropped_url":        f"/api/download/{cropped_filename}",
        "crop_meta":          crop_meta,
    })
