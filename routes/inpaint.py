import os
import uuid
import asyncio
from datetime import datetime, timezone
import aiofiles
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from typing import Optional
from PIL import Image, ImageDraw
import json

from services.database import get_collection
from services.auth import get_current_user
from services.quota import check_and_increment_quota
from services.storage import save_file
from services.tracking import track_usage, track_action
from models.user import UserOut
from services.inpainting import inpaint_image

router = APIRouter(tags=["Inpainting"])

OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10

@router.post("/inpaint")
async def inpaint_endpoint(
    file: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    mask_points: Optional[str] = Form(None),
    current_user: UserOut = Depends(get_current_user),
):
    await check_and_increment_quota(current_user.user_id)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or WebP.")
    
    if not mask and not mask_points:
        raise HTTPException(status_code=400, detail="Either mask file or mask_points must be provided.")

    file_contents = await file.read()
    if len(file_contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")
    
    upload_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename or "upload")
    image_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{safe_name}")
    
    async with aiofiles.open(image_path, "wb") as f:
        await f.write(file_contents)

    mask_path = os.path.join(UPLOAD_DIR, f"{upload_id}_mask.png")

    if mask:
        if mask.content_type not in ALLOWED_TYPES:
             raise HTTPException(status_code=400, detail="Unsupported mask type.")
        mask_contents = await mask.read()
        async with aiofiles.open(mask_path, "wb") as f:
            await f.write(mask_contents)
    elif mask_points:
        try:
            points = json.loads(mask_points)
            img = Image.open(image_path)
            mask_img = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask_img)
            for p in points:
                x, y, r = p['x'], p['y'], p['radius']
                draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
            mask_img.save(mask_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid mask_points format: {e}")

    output_filename = f"{upload_id}_inpainted.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        loop = asyncio.get_event_loop()
        image_meta = await loop.run_in_executor(
            None,
            lambda: inpaint_image(image_path, mask_path, output_path)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {exc}")

    download_url = await save_file(output_path, output_filename)

    try:
        collection = get_collection("inpaint_history")
        await collection.insert_one({
            "upload_id": upload_id,
            "user_id": current_user.user_id,
            "original_name": safe_name,
            "output_filename": output_filename,
            "download_url": download_url,
            "image_meta": image_meta,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception:
        pass

    await track_usage(
        user_id=current_user.user_id, feature="inpaint", image_id=upload_id,
        metadata={"method": "mask_file" if mask else "mask_points"}
    )
    await track_action(
        user_id=current_user.user_id, image_id=upload_id, action_type="inpaint",
        suggestion="Removed object using LaMa inpainting",
        applied=True,
    )

    return JSONResponse({
        "result_id": upload_id,
        "output_filename": output_filename,
        "download_url": download_url,
        "image_meta": image_meta,
    })
