"""
Image Enhancement Route.

POST /api/enhance
  Accepts a multipart image upload plus JSON-encoded enhancement parameters,
  runs the enhancement pipeline, persists metadata to MongoDB, and returns
  the output filename with a download URL.

Query / form parameters (all optional — defaults = no change):
  brightness       float  0.0–3.0  (1.0 = unchanged)
  contrast         float  0.0–3.0  (1.0 = unchanged)
  saturation       float  0.0–3.0  (1.0 = unchanged)
  sharpness        float  0.0–3.0  (1.0 = unchanged)
  denoise          bool   false
  auto_wb          bool   false
  denoise_strength int    5–15     (default 9)
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.database import get_collection
from services.enhancement import enhance_image

router = APIRouter(tags=["Enhancement"])

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10


@router.post("/enhance")
async def enhance_endpoint(
    file: UploadFile = File(...),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    saturation: float = Form(1.0),
    sharpness: float = Form(1.0),
    denoise: bool = Form(False),
    auto_wb: bool = Form(False),
    denoise_strength: int = Form(9),
):
    """
    Enhance an uploaded image and return a download URL for the result.

    - **file**             Image file (JPEG, PNG, or WebP, max 10 MB)
    - **brightness**       Brightness multiplier (0.0–3.0, default 1.0)
    - **contrast**         Contrast multiplier (0.0–3.0, default 1.0)
    - **saturation**       Saturation multiplier (0.0–3.0, default 1.0)
    - **sharpness**        Sharpness multiplier (0.0–3.0, default 1.0)
    - **denoise**          Apply bilateral noise reduction (default false)
    - **auto_wb**          Apply grey-world white balance (default false)
    - **denoise_strength** Bilateral filter size 5–15 (default 9)
    """
    # --- Validate file type ------------------------------------------------
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPEG, PNG, or WebP.",
        )

    # --- Validate file size ------------------------------------------------
    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_SIZE_MB} MB limit.",
        )

    # --- Clamp parameters to safe ranges -----------------------------------
    brightness      = max(0.0, min(3.0, brightness))
    contrast        = max(0.0, min(3.0, contrast))
    saturation      = max(0.0, min(3.0, saturation))
    sharpness       = max(0.0, min(3.0, sharpness))
    denoise_strength = max(5, min(15, denoise_strength))

    # --- Persist upload to disk --------------------------------------------
    upload_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename or "upload")
    upload_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{safe_name}")

    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(contents)

    # --- Run enhancement pipeline in thread pool (CPU-bound) ---------------
    output_filename = f"{upload_id}_enhanced.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        loop = asyncio.get_event_loop()
        image_meta = await loop.run_in_executor(
            None,
            lambda: enhance_image(
                input_path=upload_path,
                output_path=output_path,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                sharpness=sharpness,
                denoise=denoise,
                auto_wb=auto_wb,
                denoise_strength=denoise_strength,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {exc}")

    # --- Persist metadata to MongoDB (non-fatal) ---------------------------
    try:
        collection = get_collection("enhance_history")
        await collection.insert_one(
            {
                "upload_id":        upload_id,
                "original_name":    safe_name,
                "output_filename":  output_filename,
                "settings": {
                    "brightness":        brightness,
                    "contrast":          contrast,
                    "saturation":        saturation,
                    "sharpness":         sharpness,
                    "denoise":           denoise,
                    "auto_wb":           auto_wb,
                    "denoise_strength":  denoise_strength,
                },
                "image_meta":       image_meta,
                "created_at":       datetime.now(timezone.utc),
            }
        )
    except Exception:
        pass  # Non-fatal — still return result even if DB write fails

    return JSONResponse(
        {
            "upload_id":       upload_id,
            "output_filename": output_filename,
            "download_url":    f"/api/download/{output_filename}",
            "image_meta":      image_meta,
        }
    )
