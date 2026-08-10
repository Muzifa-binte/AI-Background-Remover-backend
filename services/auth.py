"""
Authentication Service.

Responsibilities
────────────────
  - Password hashing / verification (bcrypt via passlib)
  - JWT creation and verification (python-jose)
  - FastAPI dependency: get_current_user — injects authenticated user into routes
  - Quota check helper — raises 429 when user exceeds daily limit
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import bcrypt as _bcrypt
from jose import JWTError, jwt

from models.user import TokenData, UserOut
from services.database import get_collection

# ── Config from environment ────────────────────────────────────────────────

SECRET_KEY  = os.getenv("SECRET_KEY", "change-me-in-production-use-a-long-random-string")
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ── Helpers ────────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email:   str = payload.get("email")
        if user_id is None:
            raise ValueError("missing sub")
        return TokenData(user_id=user_id, email=email)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency ─────────────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    """
    Resolve a Bearer JWT to a UserOut model.
    Raises 401 if the token is invalid or the user no longer exists.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = decode_token(token)

    collection = get_collection("users")
    doc = await collection.find_one({"user_id": token_data.user_id}, {"_id": 0})
    if doc is None:
        raise credentials_exc

    return UserOut(
        user_id    = doc["user_id"],
        name       = doc["name"],
        email      = doc["email"],
        created_at = doc["created_at"],
    )
