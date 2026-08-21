import os
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from models.user import UserOut
from services.auth import get_current_user
from services.database import get_collection
from services.quota import get_quota_status

router = APIRouter(prefix="/auth", tags=["Auth"])

OUTPUT_DIR = "output"

@router.get("/stats")
async def get_user_stats(current_user: UserOut = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Return aggregated usage statistics for the dashboard:
    - Quota usage (used, limit)
    - Operation counts (remove_bg, enhance, replace_bg, smart_crop, recolor)
    - Storage used (in bytes)
    """
    uid = current_user.user_id

    # 1. Quota
    try:
        quota_data = await get_quota_status(uid)
    except Exception:
        quota_data = {"used": 0, "limit": 0, "remaining": 0}

    # 2. History Counts & Storage
    sources = [
        ("history",            "remove_bg",   "output_filename"),
        ("enhance_history",    "enhance",     "output_filename"),
        ("replace_bg_history", "replace_bg",  "output_filename"),
        ("smart_crop_history", "smart_crop",  "output_filename"),
        ("recolor_history",    "recolor",     "output_filename"),
    ]

    operations = {
        "remove_bg": 0,
        "enhance": 0,
        "replace_bg": 0,
        "smart_crop": 0,
        "recolor": 0,
    }
    
    total_images = 0
    storage_bytes = 0

    for col_name, op_type, filename_field in sources:
        try:
            collection = get_collection(col_name)
            # Find all docs for this user
            cursor = collection.find({"user_id": uid}, {"_id": 0, filename_field: 1})
            docs = await cursor.to_list(length=None)
            
            count = len(docs)
            operations[op_type] = count
            total_images += count
            
            # Calculate storage
            for doc in docs:
                filename = doc.get(filename_field)
                if filename:
                    # Resolve path
                    safe_filename = os.path.basename(filename)
                    file_path = os.path.join(OUTPUT_DIR, safe_filename)
                    if os.path.exists(file_path):
                        storage_bytes += os.path.getsize(file_path)
                        
        except Exception as e:
            # Skip if collection doesn't exist
            pass

    return {
        "quota": quota_data,
        "operations": operations,
        "total_images": total_images,
        "storage_bytes": storage_bytes
    }
