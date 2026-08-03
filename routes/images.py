from fastapi import APIRouter, HTTPException
from services.database import get_collection
import os

router = APIRouter(tags=["Images"])

OUTPUT_DIR = "output"


@router.delete("/image/{image_id}")
async def delete_image(image_id: str):
    """
    Deletes a processed image from storage and removes its history record.

    - **image_id**: The upload UUID (stem of the output filename, without '_result.png')
    """
    safe_id = os.path.basename(image_id)
    output_filename = f"{safe_id}_result.png"
    file_path = os.path.join(OUTPUT_DIR, output_filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    # Remove the file from disk
    os.remove(file_path)

    # Remove the corresponding MongoDB record
    try:
        collection = get_collection("history")
        await collection.delete_one({"upload_id": safe_id})
    except Exception:
        # Non-fatal — file is already gone
        pass

    return {"message": f"Image {safe_id} deleted successfully."}
