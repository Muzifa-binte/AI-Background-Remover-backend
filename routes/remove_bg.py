from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from services.bg_removal import remove_background
import aiofiles
import os
import uuid

router = APIRouter(tags=["Background Removal"])

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10


@router.post("/remove-background")
async def remove_bg_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, removes its background, and returns a
    transparent PNG.

    - **file**: Image file (JPEG, PNG, or WebP, max 10 MB)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    # Persist the upload
    upload_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(contents)

    # Run AI background removal
    output_filename = f"{uuid.uuid4()}_result.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    await remove_background(upload_path, output_path)

    return FileResponse(output_path, media_type="image/png", filename=output_filename)
