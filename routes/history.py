from fastapi import APIRouter, HTTPException
from typing import List
from services.database import get_collection

router = APIRouter(tags=["History"])


@router.get("/history", response_model=List[dict])
async def get_history():
    """
    Returns the image processing history (most recent first, capped at 50).
    """
    try:
        collection = get_collection("history")
        cursor = collection.find({}, {"_id": 0}).sort("created_at", -1).limit(50)
        results = await cursor.to_list(length=50)
        # Serialise datetime to ISO string
        for record in results:
            if "created_at" in record:
                record["created_at"] = record["created_at"].isoformat()
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch history: {exc}")
