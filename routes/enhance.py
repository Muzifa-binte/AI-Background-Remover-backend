import os, uuid, asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse
import aiofiles
from services.database    import get_collection
from services.enhancement import enhance_image
from services.auth        import get_current_user
from services.quota       import check_and_increment_quota
from services.storage     import save_file
from services.tracking    import track_usage, track_action
from models.user          import UserOut

router = APIRouter(tags=["Enhancement"])

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10


@router.post("/enhance")
async def enhance_endpoint(
    file:             UploadFile = File(...),
    brightness:       float      = Form(1.0),
    contrast:         float      = Form(1.0),
    saturation:       float      = Form(1.0),
    sharpness:        float      = Form(1.0),
    denoise:          bool       = Form(False),
    auto_wb:          bool       = Form(False),
    denoise_strength: int        = Form(9),
    current_user:     UserOut    = Depends(get_current_user),
):
    await check_and_increment_quota(current_user.user_id)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    brightness       = max(0.0, min(3.0, brightness))
    contrast         = max(0.0, min(3.0, contrast))
    saturation       = max(0.0, min(3.0, saturation))
    sharpness        = max(0.0, min(3.0, sharpness))
    denoise_strength = max(5, min(15, denoise_strength))

    upload_id   = str(uuid.uuid4())
    safe_name   = os.path.basename(file.filename or "upload")
    upload_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{safe_name}")
    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(contents)

    output_filename = f"{upload_id}_enhanced.png"
    output_path     = os.path.join(OUTPUT_DIR, output_filename)

    try:
        loop = asyncio.get_event_loop()
        image_meta = await loop.run_in_executor(
            None,
            lambda: enhance_image(
                input_path=upload_path, output_path=output_path,
                brightness=brightness, contrast=contrast,
                saturation=saturation, sharpness=sharpness,
                denoise=denoise, auto_wb=auto_wb,
                denoise_strength=denoise_strength,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {exc}")

    download_url = await save_file(output_path, output_filename)

    try:
        collection = get_collection("enhance_history")
        await collection.insert_one({
            "upload_id": upload_id, "user_id": current_user.user_id,
            "original_name": safe_name, "output_filename": output_filename,
            "download_url": download_url,
            "settings": {
                "brightness": brightness, "contrast": contrast,
                "saturation": saturation, "sharpness": sharpness,
                "denoise": denoise, "auto_wb": auto_wb,
                "denoise_strength": denoise_strength,
            },
            "image_meta": image_meta, "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass

    await track_usage(
        user_id=current_user.user_id, feature="enhance", image_id=upload_id,
        metadata={"denoise": denoise, "auto_wb": auto_wb},
    )
    await track_action(
        user_id=current_user.user_id, image_id=upload_id, action_type="enhance",
        suggestion="Applied brightness/contrast/saturation/sharpness enhancement",
        applied=True,
    )

    return JSONResponse({
        "upload_id": upload_id, "output_filename": output_filename,
        "download_url": download_url, "image_meta": image_meta,
    })
