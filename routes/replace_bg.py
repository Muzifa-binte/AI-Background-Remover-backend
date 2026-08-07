"""
Background Replacement Route.

Two-step flow
─────────────
Step 1  POST /api/remove-background          (existing route — no changes)
        → returns { output_filename, download_url }

Step 2  POST /api/replace-background
        → accepts the transparent PNG filename from step 1 (or a fresh upload)
          plus background options, runs compositing, returns a new download URL.

Form fields
───────────
  fg_filename      str   (required) — output_filename from step 1
                         e.g. "abc123_result.png"
  bg_type          str   "solid" | "gradient" | "image"   (default "solid")

  # solid
  solid_color      str   CSS hex, e.g. "#ffffff"           (default "#ffffff")

  # gradient
  gradient_start   str   CSS hex start colour              (default "#e8336d")
  gradient_end     str   CSS hex end colour                (default "#2fbfb0")
  gradient_dir     str   "horizontal"|"vertical"|"diagonal" (default "vertical")

  # image
  bg_file          File  (required when bg_type="image")
  bg_fit           str   "cover"|"contain"|"stretch"       (default "cover")
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.database    import get_collection
from services.compositing import composite_background

router = APIRouter(tags=["Background Replacement"])

OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_BG_TYPES   = {"image/jpeg", "image/png", "image/webp"}
MAX_BG_SIZE_MB     = 20          # bg images can be larger than subject uploads
ALLOWED_BG_MODES   = {"solid", "gradient", "image"}
ALLOWED_DIRECTIONS = {"horizontal", "vertical", "diagonal"}
ALLOWED_FITS       = {"cover", "contain", "stretch"}


@router.post("/replace-background")
async def replace_background_endpoint(
    fg_filename:     str  = Form(...),
    bg_type:         str  = Form("solid"),
    solid_color:     str  = Form("#ffffff"),
    gradient_start:  str  = Form("#e8336d"),
    gradient_end:    str  = Form("#2fbfb0"),
    gradient_dir:    str  = Form("vertical"),
    bg_fit:          str  = Form("cover"),
    bg_file: UploadFile   = File(None),
):
    """
    Composite a previously bg-removed PNG over a new background.

    - **fg_filename**    Transparent PNG filename returned by /api/remove-background
    - **bg_type**        "solid" | "gradient" | "image"
    - **solid_color**    Hex colour (solid mode)
    - **gradient_start** Hex start colour (gradient mode)
    - **gradient_end**   Hex end colour (gradient mode)
    - **gradient_dir**   Gradient direction (gradient mode)
    - **bg_file**        Background image upload (image mode)
    - **bg_fit**         Resize strategy for image mode
    """
    # ── Validate bg_type ──────────────────────────────────────────────────
    if bg_type not in ALLOWED_BG_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"bg_type must be one of: {', '.join(sorted(ALLOWED_BG_MODES))}.",
        )
    if gradient_dir not in ALLOWED_DIRECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"gradient_dir must be one of: {', '.join(sorted(ALLOWED_DIRECTIONS))}.",
        )
    if bg_fit not in ALLOWED_FITS:
        raise HTTPException(
            status_code=400,
            detail=f"bg_fit must be one of: {', '.join(sorted(ALLOWED_FITS))}.",
        )

    # ── Locate foreground file ────────────────────────────────────────────
    safe_fg = os.path.basename(fg_filename)
    fg_path = os.path.join(OUTPUT_DIR, safe_fg)
    if not os.path.isfile(fg_path):
        raise HTTPException(
            status_code=404,
            detail=f"Foreground file '{safe_fg}' not found. Run /api/remove-background first.",
        )

    # ── Handle background image upload (image mode only) ─────────────────
    bg_image_path: str | None = None
    if bg_type == "image":
        if bg_file is None or bg_file.filename in (None, ""):
            raise HTTPException(
                status_code=400,
                detail="bg_file is required when bg_type='image'.",
            )
        if bg_file.content_type not in ALLOWED_BG_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Background image must be JPEG, PNG, or WebP.",
            )
        bg_contents = await bg_file.read()
        if len(bg_contents) > MAX_BG_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Background image exceeds {MAX_BG_SIZE_MB} MB limit.",
            )
        bg_id       = str(uuid.uuid4())
        safe_bg_name = os.path.basename(bg_file.filename or "bg")
        bg_image_path = os.path.join(UPLOAD_DIR, f"{bg_id}_{safe_bg_name}")
        async with aiofiles.open(bg_image_path, "wb") as f:
            await f.write(bg_contents)

    # ── Run compositing in thread pool (CPU-bound) ────────────────────────
    result_id       = str(uuid.uuid4())
    output_filename = f"{result_id}_composited.png"
    output_path     = os.path.join(OUTPUT_DIR, output_filename)

    try:
        loop = asyncio.get_event_loop()
        image_meta = await loop.run_in_executor(
            None,
            lambda: composite_background(
                foreground_path      = fg_path,
                output_path          = output_path,
                bg_type              = bg_type,          # type: ignore[arg-type]
                solid_color          = solid_color,
                gradient_color_start = gradient_start,
                gradient_color_end   = gradient_end,
                gradient_direction   = gradient_dir,     # type: ignore[arg-type]
                bg_image_path        = bg_image_path,
                bg_fit               = bg_fit,           # type: ignore[arg-type]
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compositing failed: {exc}")

    # ── Persist metadata (non-fatal) ──────────────────────────────────────
    try:
        collection = get_collection("replace_bg_history")
        await collection.insert_one({
            "result_id":       result_id,
            "fg_filename":     safe_fg,
            "output_filename": output_filename,
            "bg_type":         bg_type,
            "settings": {
                "solid_color":      solid_color,
                "gradient_start":   gradient_start,
                "gradient_end":     gradient_end,
                "gradient_dir":     gradient_dir,
                "bg_fit":           bg_fit,
            },
            "image_meta":      image_meta,
            "created_at":      datetime.now(timezone.utc),
        })
    except Exception:
        pass

    return JSONResponse({
        "result_id":       result_id,
        "output_filename": output_filename,
        "download_url":    f"/api/download/{output_filename}",
        "image_meta":      image_meta,
    })
