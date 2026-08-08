from fastapi import APIRouter, HTTPException, Depends
from services.database import get_collection
from services.auth     import get_current_user
from models.user       import UserOut
import os

router = APIRouter(tags=["Images"])

OUTPUT_DIR = "output"


@router.delete("/image/{image_id}")
async def delete_image(
    image_id:     str,
    current_user: UserOut = Depends(get_current_user),
):
    safe_id         = os.path.basename(image_id)
    output_filename = f"{safe_id}_result.png"
    file_path       = os.path.join(OUTPUT_DIR, output_filename)

    # Verify the record belongs to this user before deleting
    try:
        collection = get_collection("history")
        record = await collection.find_one({"upload_id": safe_id})
        if record and record.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=403, detail="Not authorised to delete this image.")
    except HTTPException:
        raise
    except Exception:
        pass

    if os.path.isfile(file_path):
        os.remove(file_path)

    try:
        collection = get_collection("history")
        await collection.delete_one({"upload_id": safe_id, "user_id": current_user.user_id})
    except Exception:
        pass

    return {"message": f"Image {safe_id} deleted successfully."}
