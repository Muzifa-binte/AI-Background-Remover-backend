from fastapi import APIRouter, HTTPException
import os

router = APIRouter(tags=["Images"])

OUTPUT_DIR = "output"


@router.delete("/image/{image_id}")
async def delete_image(image_id: str):
    """
    Deletes a processed image from storage by its ID/filename.

    - **image_id**: The unique identifier (filename stem) of the image to delete
    """
    # TODO: also remove the corresponding record from MongoDB
    safe_id = os.path.basename(image_id)
    file_path = os.path.join(OUTPUT_DIR, f"{safe_id}.png")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found.")

    os.remove(file_path)
    return {"message": f"Image {safe_id} deleted successfully."}
