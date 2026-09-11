import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from services.database import get_collection
from services.auth import get_current_user
from models.user import UserOut
from models.collaboration import FavoriteCreate, FavoriteOut

router = APIRouter(prefix="/favorites", tags=["Favorites"])


@router.post("", response_model=FavoriteOut)
async def create_favorite(
    payload: FavoriteCreate,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("favorites")
        doc = {
            "favorite_id": str(uuid.uuid4()),
            "user_id": current_user.user_id,
            "content": payload.content,
            "source": payload.source,
            "created_at": datetime.now(timezone.utc),
        }
        await collection.insert_one(doc)
        doc.pop("_id", None)
        return FavoriteOut(**doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save favorite: {exc}")


@router.get("", response_model=List[FavoriteOut])
async def list_favorites(current_user: UserOut = Depends(get_current_user)):
    try:
        collection = get_collection("favorites")
        cursor = (
            collection
            .find({"user_id": current_user.user_id}, {"_id": 0})
            .sort("created_at", -1)
        )
        results = await cursor.to_list(length=200)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch favorites: {exc}")


@router.delete("/{favorite_id}")
async def delete_favorite(
    favorite_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("favorites")
        result = await collection.delete_one(
            {"favorite_id": favorite_id, "user_id": current_user.user_id}
        )
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Favorite not found.")
        return {"status": "deleted", "favorite_id": favorite_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete favorite: {exc}")
