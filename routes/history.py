from fastapi import APIRouter, HTTPException, Depends
from typing import List
import asyncio
from services.database import get_collection
from services.auth     import get_current_user
from models.user       import UserOut

router = APIRouter(tags=["History"])


@router.get("/history", response_model=List[dict])
async def get_history(current_user: UserOut = Depends(get_current_user)):
    """Returns the current user's bg-removal history (most recent first, capped at 50)."""
    try:
        collection = get_collection("history")
        cursor  = (
            collection
            .find({"user_id": current_user.user_id}, {"_id": 0})
            .sort("created_at", -1)
            .limit(50)
        )
        results = await cursor.to_list(length=50)
        for record in results:
            if "created_at" in record:
                record["created_at"] = record["created_at"].isoformat()
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch history: {exc}")


async def _fetch_collection(collection_name: str, user_id: str, op_type: str, limit: int) -> list:
    """Fetch records from one history collection and normalise them into a common shape."""
    try:
        col = get_collection(collection_name)
        cursor = (
            col
            .find({"user_id": user_id}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        records = await cursor.to_list(length=limit)
    except Exception:
        return []

    normalised = []
    for r in records:
        # Serialise datetime
        if "created_at" in r:
            r["created_at"] = r["created_at"].isoformat()

        # Normalise primary key → always "upload_id"
        if "result_id" in r and "upload_id" not in r:
            r["upload_id"] = r.pop("result_id")

        # Normalise display name → always "original_name"
        if "original_name" not in r:
            r["original_name"] = r.get("fg_filename", "Unknown")

        # Normalise output filename → always "output_filename"
        # smart_crop stores the final file as "cropped_filename"
        if "output_filename" not in r and "cropped_filename" in r:
            r["output_filename"] = r["cropped_filename"]

        # Attach operation type discriminator
        r["operation_type"] = op_type

        normalised.append(r)

    return normalised


@router.get("/history/all", response_model=List[dict])
async def get_history_all(current_user: UserOut = Depends(get_current_user)):
    """
    Returns the current user's unified history across all four operation types
    (remove_bg, enhance, replace_bg, smart_crop), merged and sorted by date,
    capped at 100 records total.
    """
    PER_COLLECTION = 100  # fetch up to 100 from each, then trim after merge

    results = await asyncio.gather(
        _fetch_collection("history",           current_user.user_id, "remove_bg",  PER_COLLECTION),
        _fetch_collection("enhance_history",   current_user.user_id, "enhance",    PER_COLLECTION),
        _fetch_collection("replace_bg_history",current_user.user_id, "replace_bg", PER_COLLECTION),
        _fetch_collection("smart_crop_history",current_user.user_id, "smart_crop", PER_COLLECTION),
    )

    merged = [item for bucket in results for item in bucket]

    # Sort merged list by created_at descending (ISO strings sort correctly)
    merged.sort(key=lambda r: r.get("created_at", ""), reverse=True)

    return merged[:100]
