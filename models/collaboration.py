"""
Collaboration & Sharing Pydantic schemas.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class ChatMessageIn(BaseModel):
    conversation_id: str = Field(..., description="Groups messages into one conversation")
    role: Literal["user", "assistant"]
    content: str
    image_id: Optional[str] = None


class ChatMessageOut(BaseModel):
    message_id: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    image_id: Optional[str] = None
    created_at: datetime


class ExportFormat(BaseModel):
    format: Literal["pdf", "text"] = "text"


class PromptTemplateCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    prompt_text: str = Field(..., min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=list)


class PromptTemplateOut(BaseModel):
    template_id: str
    user_id: str
    title: str
    prompt_text: str
    tags: list[str]
    use_count: int = 0
    created_at: datetime


class ActionLogCreate(BaseModel):
    image_id: str
    action_type: str = Field(..., description="e.g. remove_bg, enhance, replace_bg, smart_crop, recolor")
    suggestion: str = Field(..., description="What the AI suggested")
    applied: bool = True


class ActionLogOut(BaseModel):
    action_id: str
    user_id: str
    image_id: str
    action_type: str
    suggestion: str
    applied: bool
    created_at: datetime


class CommentCreate(BaseModel):
    image_id: str
    text: str = Field(..., min_length=1, max_length=2000)


class CommentOut(BaseModel):
    comment_id: str
    image_id: str
    user_id: str
    user_name: str
    text: str
    created_at: datetime
