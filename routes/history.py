from fastapi import APIRouter, HTTPException, Depends
from typing import List
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
