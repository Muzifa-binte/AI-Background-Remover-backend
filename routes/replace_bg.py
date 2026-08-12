import os, uuid, asyncio
from datetime import datetime, timezone
import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse
from services.database    import get_collection
from services.compositing import composite_background
from services.auth        import get_current_user
from services.quota       import check_and_increment_quota
from services.storage     import save_file, get_download_url
from models.user          import UserOut

router = APIRouter(tags=["Background Replacement"])

OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_BG_TYPES   = {"image/jpeg", "image/png", "image/webp"}
MAX_BG_SIZE_MB     = 20
ALLOWED_BG_MODES   = {"solid", "gradient", "image"}
ALLOWED_DIRECTIONS = {"horizontal", "vertical", "diagonal"}
ALLOWED_FITS       = {"cover", "contain", "stretch"}


@router.post("/replace-background")
async def replace_background_endpoint(
    fg_filename:    str        = Form(...),
    bg_type:        str        = Form("solid"),
    solid_color:    str        = Form("#ffffff"),
    gradient_start: str        = Form("#e8336d"),
    gradient_end:   str        = Form("#2fbfb0"),
    gradient_dir:   str        = Form("vertical"),
    bg_fit:         str        = Form("cover"),
    bg_file:        UploadFile = File(None),
    current_user:   UserOut    = Depends(get_current_user),
):
    await check_and_increment_quota(current_user.user_id)

    if bg_type not in ALLOWED_BG_MODES:
        raise HTTPException(status_code=400, detail=f"bg_type must be one of: {', '.join(sorted(ALLOWED_BG_MODES))}.")
    if gradient_dir not in ALLOWED_DIRECTIONS:
        raise HTTPException(status_code=400, detail=f"gradient_dir must be one of: {', '.join(sorted(ALLOWED_DIRECTIONS))}.")
    if bg_fit not in ALLOWED_FITS:
        raise HTTPException(status_code=400, detail=f"bg_fit must be one of: {', '.join(sorted(ALLOWED_FITS))}.")

    safe_fg = os.path.basename(fg_filename)
    fg_path = os.path.join(OUTPUT_DIR, safe_fg)
    if not os.path.isfile(fg_path):
        raise HTTPException(status_code=404, detail=f"Foreground file '{safe_fg}' not found.")

    bg_image_path: str | None = None
    if bg_type == "image":
        if bg_file is None or bg_file.filename in (None, ""):
            raise HTTPException(status_code=400, detail="bg_file is required when bg_type='image'.")
        if bg_file.content_type not in ALLOWED_BG_TYPES:
            raise HTTPException(status_code=400, detail="Background image must be JPEG, PNG, or WebP.")
        bg_contents = await bg_file.read()
        if len(bg_contents) > MAX_BG_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Background image exceeds {MAX_BG_SIZE_MB} MB limit.")
        bg_id         = str(uuid.uuid4())
        safe_bg_name  = os.path.basename(bg_file.filename or "bg")
        bg_image_path = os.path.join(UPLOAD_DIR, f"{bg_id}_{safe_bg_name}")
        async with aiofiles.open(bg_image_path, "wb") as f:
            await f.write(bg_contents)

    result_id       = str(uuid.uuid4())
    output_filename = f"{result_id}_composited.png"
    output_path     = os.path.join(OUTPUT_DIR, output_filename)

    try:
        loop = asyncio.get_event_loop()
        image_meta = await loop.run_in_executor(
            None,
            lambda: composite_background(
                foreground_path=fg_path, output_path=output_path,
                bg_type=bg_type, solid_color=solid_color,  # type: ignore
                gradient_color_start=gradient_start, gradient_color_end=gradient_end,
                gradient_direction=gradient_dir, bg_image_path=bg_image_path,  # type: ignore
                bg_fit=bg_fit,  # type: ignore
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compositing failed: {exc}")

    download_url = await save_file(output_path, output_filename)

    try:
        collection = get_collection("replace_bg_history")
        await collection.insert_one({
            "result_id": result_id, "user_id": current_user.user_id,
            "fg_filename": safe_fg, "output_filename": output_filename,
            "download_url": download_url, "bg_type": bg_type,
            "settings": {
                "solid_color": solid_color, "gradient_start": gradient_start,
                "gradient_end": gradient_end, "gradient_dir": gradient_dir, "bg_fit": bg_fit,
            },
            "image_meta": image_meta, "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass

    return JSONResponse({
        "result_id": result_id, "output_filename": output_filename,
        "download_url": download_url, "image_meta": image_meta,
    })
