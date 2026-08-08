"""
Authentication Routes.

POST /api/auth/register  — create account
POST /api/auth/login     — exchange credentials for JWT
GET  /api/auth/me        — return current user (requires token)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from models.user   import UserCreate, UserLogin, UserOut, TokenResponse
from services.auth import (
    hash_password, verify_password,
    create_access_token, get_current_user,
)
from services.database import get_collection

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: UserCreate):
    """
    Create a new user account.

    - **name**     Display name (2–80 chars)
    - **email**    Unique email address
    - **password** Min 8 characters
    """
    collection = get_collection("users")

    # Reject duplicate email
    existing = await collection.find_one({"email": body.email.lower()})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    user_id = str(uuid.uuid4())
    now     = datetime.now(timezone.utc)

    user_doc = {
        "user_id":          user_id,
        "name":             body.name.strip(),
        "email":            body.email.lower(),
        "hashed_password":  hash_password(body.password),
        "created_at":       now,
    }
    await collection.insert_one(user_doc)

    token = create_access_token({"sub": user_id, "email": body.email.lower()})
    user_out = UserOut(
        user_id    = user_id,
        name       = user_doc["name"],
        email      = user_doc["email"],
        created_at = now,
    )
    return TokenResponse(access_token=token, user=user_out)


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    """
    Exchange email + password for a JWT access token.
    """
    collection = get_collection("users")
    doc = await collection.find_one({"email": body.email.lower()})

    if doc is None or not verify_password(body.password, doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": doc["user_id"], "email": doc["email"]})
    user_out = UserOut(
        user_id    = doc["user_id"],
        name       = doc["name"],
        email      = doc["email"],
        created_at = doc["created_at"],
    )
    return TokenResponse(access_token=token, user=user_out)


@router.get("/me", response_model=UserOut)
async def me(current_user: UserOut = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user
