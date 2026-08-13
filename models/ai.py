from pydantic import BaseModel
from typing import Optional


class ImageAnalysisResponse(BaseModel):
    subject: str
    image_type: str
    background_description: str
    suggested_use: str
    editing_recommendations: list[str]


class CaptionResponse(BaseModel):
    caption: str
    style: str


class CaptionsResponse(BaseModel):
    captions: list[str]
    style: str


class BackgroundSuggestionsResponse(BaseModel):
    suggestions: list[str]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    thinking: Optional[str] = None
