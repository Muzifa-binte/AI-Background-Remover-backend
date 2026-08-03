from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter(tags=["History"])


@router.get("/history", response_model=List[dict])
async def get_history():
    """
    Returns the authenticated user's image processing history.

    Requires JWT authentication (to be wired in once auth is implemented).
    """
    # TODO: query MongoDB via the database service and filter by current user
    return []
