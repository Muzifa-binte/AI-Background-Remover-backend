import os
import uuid
import asyncio
import aiofiles
from datetime import datetime, timezone
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from services.auth import get_current_user
from services.quota import check_and_increment_quota
from services.tracking import track_usage
from models.user import UserOut

router = APIRouter(tags=["Vectorize"])

OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 15


def _run_vectorize(image_path: str, output_path: str) -> None:
    """Run vtracer vectorization synchronously (called in executor)."""
    import vtracer
    vtracer.convert_image_to_svg_py(
        image_path,
        output_path,
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=4,
        color_precision=6,
        layer_difference=16,
        corner_threshold=60,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=8,
    )


@router.post("/vectorize")
async def vectorize_endpoint(
    file: UploadFile = File(...),
    current_user: UserOut = Depends(get_current_user),
):
    """Convert a PNG/JPG image to an SVG vector using vtracer."""
    await check_and_increment_quota(current_user.user_id)

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    file_contents = await file.read()
    if len(file_contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    upload_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename or "upload")
    image_path = os.path.join(UPLOAD_DIR, f"{upload_id}_{safe_name}")

    async with aiofiles.open(image_path, "wb") as f:
        await f.write(file_contents)

    output_filename = f"{upload_id}_vectorized.svg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: _run_vectorize(image_path, output_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {exc}")
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass

    if not os.path.exists(output_path):
        raise HTTPException(status_code=500, detail="SVG output file was not generated.")

    download_url = f"/api/download/{output_filename}"

    try:
        await track_usage(
            user_id=current_user.user_id,
            feature="vectorize",
            image_id=upload_id,
            metadata={"source_name": safe_name},
        )
    except Exception:
        pass

    return JSONResponse({
        "result_id": upload_id,
        "output_filename": output_filename,
        "download_url": download_url,
    })
