from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from services.bg_removal import remove_background
from services.database   import get_collection
from services.auth       import get_current_user
from services.quota      import check_and_increment_quota
from services.storage    import save_file
from models.user         import UserOut
import aiofiles
import os
import uuid
from datetime import datetime, timezone

router = APIRouter(tags=["Background Removal"])

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10


@router.post("/remove-background")
async def remove_bg_endpoint(
    file:         UploadFile = File(...),
    current_user: UserOut    = Depends(get_current_user),
):
    await check_and_increment_quota(current_user.user_id)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    upload_id   = str(uuid.uuid4())
    safe_name   = os.path.basename(file.filename or "upload")
    upload_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{safe_name}")
    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(contents)

    output_filename = f"{upload_id}_result.png"
    output_path     = os.path.join(OUTPUT_DIR, output_filename)

    try:
        await remove_background(upload_path, output_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    download_url = await save_file(output_path, output_filename)

    try:
        collection = get_collection("history")
        await collection.insert_one({
            "upload_id":       upload_id,
            "user_id":         current_user.user_id,
            "original_name":   safe_name,
            "output_filename": output_filename,
            "download_url":    download_url,
            "created_at":      datetime.now(timezone.utc),
        })
    except Exception:
        pass

    return JSONResponse({
        "output_filename": output_filename,
        "download_url":    download_url,
    })
