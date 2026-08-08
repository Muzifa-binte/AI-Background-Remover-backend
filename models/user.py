"""
User Pydantic schemas.

UserCreate   — payload for POST /api/auth/register
UserLogin    — payload for POST /api/auth/login
UserOut      — safe public representation (no password hash)
UserInDB     — full internal representation (includes hashed_password)
TokenResponse— JWT response body
TokenData    — decoded JWT claims
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    name:     str       = Field(..., min_length=2,  max_length=80)
    email:    EmailStr
    password: str       = Field(..., min_length=8,  max_length=128)


class UserLogin(BaseModel):
    email:    EmailStr
    password: str       = Field(..., min_length=1)


class UserOut(BaseModel):
    user_id:    str
    name:       str
    email:      str
    created_at: datetime


class UserInDB(UserOut):
    hashed_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         UserOut


class TokenData(BaseModel):
    user_id:  Optional[str] = None
    email:    Optional[str] = None
