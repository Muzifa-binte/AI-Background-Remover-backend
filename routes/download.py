from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter(tags=["Download"])

OUTPUT_DIR = "output"


@router.get("/download/{filename}")
async def download_image(filename: str):
    """
    Downloads a previously processed image by filename.

    - **filename**: Name of the output PNG file
    """
    # Prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(file_path, media_type="image/png", filename=safe_filename)
