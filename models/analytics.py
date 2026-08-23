"""
Analytics & Insights Pydantic schemas.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UsageEventCreate(BaseModel):
    feature: str = Field(..., description="e.g. remove_bg, enhance, chat, replace_bg, smart_crop, recolor")
    image_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class UsageEventOut(BaseModel):
    event_id: str
    user_id: str
    feature: str
    image_id: Optional[str] = None
    metadata: dict
    created_at: datetime


class FeatureUsageCount(BaseModel):
    feature: str
    count: int


class UsageSummary(BaseModel):
    total_events: int
    by_feature: list[FeatureUsageCount]
    period_days: int


class CostLogCreate(BaseModel):
    feature: str
    provider: str = Field(default="anthropic", description="anthropic | openai | other")
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class CostLogOut(BaseModel):
    cost_id: str
    user_id: str
    feature: str
    provider: str
    model: Optional[str] = None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: datetime


class CostSummary(BaseModel):
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    by_feature: dict[str, float]
    period_days: int


class FeedbackCreate(BaseModel):
    image_id: Optional[str] = None
    action_type: str = Field(..., description="Which AI feature/suggestion is being rated")
    rating: int = Field(..., ge=1, le=5, description="1 (bad) to 5 (great); use 1/5 for thumbs down/up")
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackOut(BaseModel):
    feedback_id: str
    user_id: str
    image_id: Optional[str] = None
    action_type: str
    rating: int
    comment: Optional[str] = None
    created_at: datetime


class SuccessMetricRow(BaseModel):
    action_type: str
    suggested_count: int
    applied_count: int
    apply_rate: float


class SuccessMetrics(BaseModel):
    overall_apply_rate: float
    by_action_type: list[SuccessMetricRow]
    period_days: int
