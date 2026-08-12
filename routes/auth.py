"""
Authentication Routes.

POST /api/auth/register  — create account, returns access token + sets refresh cookie
POST /api/auth/login     — exchange credentials for tokens
POST /api/auth/refresh   — exchange refresh cookie for a new access token
POST /api/auth/logout    — clear the refresh token cookie
GET  /api/auth/me        — return current user (requires access token)
GET  /api/auth/quota     — return daily quota usage for current user
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Cookie, HTTPException, Response, status, Depends
from pymongo.errors import ServerSelectionTimeoutError, NetworkTimeout, ConnectionFailure

from models.user   import UserCreate, UserLogin, UserOut, TokenResponse
from services.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user, REFRESH_COOKIE_NAME, REFRESH_TOKEN_EXPIRE_DAYS,
)
from services.database import get_collection
from services.quota    import get_quota_status

router = APIRouter(prefix="/auth", tags=["Auth"])

# Derive cookie security flags from environment so dev (HTTP) works without
# warnings while production (HTTPS) enforces Secure + SameSite=None.
_COOKIE_SECURE   = os.getenv("COOKIE_SECURE",    "false").lower() == "true"
_COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE",  "lax")   # "none" in production behind HTTPS

_REFRESH_MAX_AGE = REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60  # seconds


def _set_refresh_cookie(response: Response, token: str) -> None:
    """Attach the refresh token as an httpOnly cookie to the response."""
    response.set_cookie(
        key      = REFRESH_COOKIE_NAME,
        value    = token,
        httponly = True,
        secure   = _COOKIE_SECURE,
        samesite = _COOKIE_SAMESITE,
        max_age  = _REFRESH_MAX_AGE,
        path     = "/api/auth",   # cookie is only sent to auth endpoints
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Remove the refresh token cookie."""
    response.delete_cookie(
        key      = REFRESH_COOKIE_NAME,
        httponly = True,
        secure   = _COOKIE_SECURE,
        samesite = _COOKIE_SAMESITE,
        path     = "/api/auth",
    )


# ── Register ───────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: UserCreate, response: Response):
    """
    Create a new user account.

    Returns a short-lived access token in the body and sets a long-lived
    refresh token in an httpOnly cookie.

    - **name**     Display name (2–80 chars)
    - **email**    Unique email address
    - **password** Min 8 characters
    """
    try:
        collection = get_collection("users")

        existing = await collection.find_one({"email": body.email.lower()})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with that email already exists.",
            )

        user_id = str(uuid.uuid4())
        now     = datetime.now(timezone.utc)

        user_doc = {
            "user_id":         user_id,
            "name":            body.name.strip(),
            "email":           body.email.lower(),
            "hashed_password": hash_password(body.password),
            "created_at":      now,
        }
        await collection.insert_one(user_doc)

        claims = {"sub": user_id, "email": body.email.lower()}
        access_token  = create_access_token(claims)
        refresh_token = create_refresh_token(claims)
        _set_refresh_cookie(response, refresh_token)

        user_out = UserOut(
            user_id    = user_id,
            name       = user_doc["name"],
            email      = user_doc["email"],
            created_at = now,
        )
        return TokenResponse(access_token=access_token, user=user_out)

    except HTTPException:
        raise
    except (ServerSelectionTimeoutError, NetworkTimeout, ConnectionFailure) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unreachable. Please try again later.",
        ) from exc


# ── Login ──────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin, response: Response):
    """
    Exchange email + password for a short-lived access token (body) and
    a long-lived refresh token (httpOnly cookie).
    """
    try:
        collection = get_collection("users")
        doc = await collection.find_one({"email": body.email.lower()})

        if doc is None or not verify_password(body.password, doc["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        claims = {"sub": doc["user_id"], "email": doc["email"]}
        access_token  = create_access_token(claims)
        refresh_token = create_refresh_token(claims)
        _set_refresh_cookie(response, refresh_token)

        user_out = UserOut(
            user_id    = doc["user_id"],
            name       = doc["name"],
            email      = doc["email"],
            created_at = doc["created_at"],
        )
        return TokenResponse(access_token=access_token, user=user_out)

    except HTTPException:
        raise
    except (ServerSelectionTimeoutError, NetworkTimeout, ConnectionFailure) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unreachable. Please try again later.",
        ) from exc


# ── Refresh ────────────────────────────────────────────────────────────────

@router.post("/refresh")
async def refresh(
    response: Response,
    bgr_refresh: str | None = Cookie(default=None),
):
    """
    Exchange a valid refresh token cookie for a new short-lived access token.

    The refresh cookie is rotated on every call (new refresh token issued).
    Returns: {"access_token": str, "token_type": "bearer"}
    """
    if not bgr_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No refresh token provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate the refresh token — raises 401 on failure
    token_data = decode_token(bgr_refresh, expected_type="refresh")

    # Verify the user still exists in the database
    collection = get_collection("users")
    doc = await collection.find_one({"user_id": token_data.user_id}, {"_id": 0})
    if doc is None:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = {"sub": doc["user_id"], "email": doc["email"]}

    # Rotate: issue new access token + new refresh token
    new_access  = create_access_token(claims)
    new_refresh = create_refresh_token(claims)
    _set_refresh_cookie(response, new_refresh)

    return {"access_token": new_access, "token_type": "bearer"}


# ── Logout ─────────────────────────────────────────────────────────────────

@router.post("/logout", status_code=204)
async def logout(response: Response):
    """
    Clear the refresh token cookie. The client should discard its access token.
    """
    _clear_refresh_cookie(response)


# ── Me ─────────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserOut)
async def me(current_user: UserOut = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user


# ── Quota ──────────────────────────────────────────────────────────────────

@router.get("/quota")
async def quota(current_user: UserOut = Depends(get_current_user)):
    """
    Return the authenticated user's current daily quota usage.

    Response fields:
    - **used**      Operations consumed today (UTC day)
    - **limit**     Daily limit (0 means disabled)
    - **remaining** Operations left today (null when disabled)
    - **resets_at** ISO-8601 timestamp of next quota reset (midnight UTC)
    - **disabled**  True when quota enforcement is turned off
    """
    return await get_quota_status(current_user.user_id)
