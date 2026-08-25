import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from services.database import get_collection
from services.auth import get_current_user
from models.user import UserOut
from models.collaboration import ActionLogCreate, ActionLogOut

router = APIRouter(prefix="/actions", tags=["AI Action History"])


@router.post("", response_model=ActionLogOut)
async def log_action(
    payload: ActionLogCreate,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("action_history")
        doc = {
            "action_id": str(uuid.uuid4()),
            "user_id": current_user.user_id,
            "image_id": payload.image_id,
            "action_type": payload.action_type,
            "suggestion": payload.suggestion,
            "applied": payload.applied,
            "created_at": datetime.now(timezone.utc),
        }
        await collection.insert_one(doc)
        doc.pop("_id", None)
        return ActionLogOut(**doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not log action: {exc}")


@router.get("", response_model=List[ActionLogOut])
async def list_actions(
    image_id: Optional[str] = Query(None, description="Filter by image"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("action_history")
        query: dict = {"user_id": current_user.user_id}
        if image_id:
            query["image_id"] = image_id
        if action_type:
            query["action_type"] = action_type

        cursor = collection.find(query, {"_id": 0}).sort("created_at", -1).limit(200)
        results = await cursor.to_list(length=200)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch action history: {exc}")
