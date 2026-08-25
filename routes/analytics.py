import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from services.database import get_collection
from services.auth import get_current_user
from models.user import UserOut
from models.analytics import (
    UsageEventCreate,
    UsageEventOut,
    UsageSummary,
    FeatureUsageCount,
    CostLogCreate,
    CostLogOut,
    CostSummary,
    FeedbackCreate,
    FeedbackOut,
    SuccessMetrics,
    SuccessMetricRow,
)

router = APIRouter(prefix="/analytics", tags=["Analytics & Insights"])


@router.post("/usage", response_model=UsageEventOut)
async def log_usage_event(
    payload: UsageEventCreate,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("usage_events")
        doc = {
            "event_id": str(uuid.uuid4()),
            "user_id": current_user.user_id,
            "feature": payload.feature,
            "image_id": payload.image_id,
            "metadata": payload.metadata,
            "created_at": datetime.now(timezone.utc),
        }
        await collection.insert_one(doc)
        doc.pop("_id", None)
        return UsageEventOut(**doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not log usage event: {exc}")


@router.get("/usage", response_model=UsageSummary)
async def get_usage_summary(
    days: int = Query(30, ge=1, le=365),
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("usage_events")
        since = datetime.now(timezone.utc) - timedelta(days=days)
        pipeline = [
            {"$match": {"user_id": current_user.user_id, "created_at": {"$gte": since}}},
            {"$group": {"_id": "$feature", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        rows = await collection.aggregate(pipeline).to_list(length=100)
        by_feature = [FeatureUsageCount(feature=r["_id"], count=r["count"]) for r in rows]
        total = sum(r.count for r in by_feature)
        return UsageSummary(total_events=total, by_feature=by_feature, period_days=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch usage summary: {exc}")


@router.get("/success", response_model=SuccessMetrics)
async def get_success_metrics(
    days: int = Query(30, ge=1, le=365),
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("action_history")
        since = datetime.now(timezone.utc) - timedelta(days=days)
        pipeline = [
            {"$match": {"user_id": current_user.user_id, "created_at": {"$gte": since}}},
            {
                "$group": {
                    "_id": "$action_type",
                    "suggested_count": {"$sum": 1},
                    "applied_count": {"$sum": {"$cond": ["$applied", 1, 0]}},
                }
            },
        ]
        rows = await collection.aggregate(pipeline).to_list(length=100)

        by_action_type = []
        total_suggested = 0
        total_applied = 0
        for r in rows:
            suggested = r["suggested_count"]
            applied = r["applied_count"]
            total_suggested += suggested
            total_applied += applied
            rate = round(applied / suggested, 4) if suggested else 0.0
            by_action_type.append(
                SuccessMetricRow(
                    action_type=r["_id"],
                    suggested_count=suggested,
                    applied_count=applied,
                    apply_rate=rate,
                )
            )

        overall_rate = round(total_applied / total_suggested, 4) if total_suggested else 0.0
        return SuccessMetrics(
            overall_apply_rate=overall_rate,
            by_action_type=by_action_type,
            period_days=days,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch success metrics: {exc}")


@router.post("/cost", response_model=CostLogOut)
async def log_cost(
    payload: CostLogCreate,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("cost_logs")
        doc = {
            "cost_id": str(uuid.uuid4()),
            "user_id": current_user.user_id,
            "feature": payload.feature,
            "provider": payload.provider,
            "model": payload.model,
            "input_tokens": payload.input_tokens,
            "output_tokens": payload.output_tokens,
            "cost_usd": payload.cost_usd,
            "created_at": datetime.now(timezone.utc),
        }
        await collection.insert_one(doc)
        doc.pop("_id", None)
        return CostLogOut(**doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not log cost: {exc}")


@router.get("/cost", response_model=CostSummary)
async def get_cost_summary(
    days: int = Query(30, ge=1, le=365),
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("cost_logs")
        since = datetime.now(timezone.utc) - timedelta(days=days)
        pipeline = [
            {"$match": {"user_id": current_user.user_id, "created_at": {"$gte": since}}},
            {
                "$group": {
                    "_id": "$feature",
                    "cost": {"$sum": "$cost_usd"},
                    "input_tokens": {"$sum": "$input_tokens"},
                    "output_tokens": {"$sum": "$output_tokens"},
                }
            },
        ]
        rows = await collection.aggregate(pipeline).to_list(length=100)

        by_feature = {r["_id"]: round(r["cost"], 6) for r in rows}
        total_cost = round(sum(r["cost"] for r in rows), 6)
        total_input = sum(r["input_tokens"] for r in rows)
        total_output = sum(r["output_tokens"] for r in rows)

        return CostSummary(
            total_cost_usd=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            by_feature=by_feature,
            period_days=days,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch cost summary: {exc}")


@router.post("/feedback", response_model=FeedbackOut)
async def submit_feedback(
    payload: FeedbackCreate,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("feedback")
        doc = {
            "feedback_id": str(uuid.uuid4()),
            "user_id": current_user.user_id,
            "image_id": payload.image_id,
            "action_type": payload.action_type,
            "rating": payload.rating,
            "comment": payload.comment,
            "created_at": datetime.now(timezone.utc),
        }
        await collection.insert_one(doc)
        doc.pop("_id", None)
        return FeedbackOut(**doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not submit feedback: {exc}")


@router.get("/feedback/summary")
async def get_feedback_summary(
    days: int = Query(30, ge=1, le=365),
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("feedback")
        since = datetime.now(timezone.utc) - timedelta(days=days)
        pipeline = [
            {"$match": {"user_id": current_user.user_id, "created_at": {"$gte": since}}},
            {
                "$group": {
                    "_id": "$action_type",
                    "avg_rating": {"$avg": "$rating"},
                    "count": {"$sum": 1},
                }
            },
        ]
        rows = await collection.aggregate(pipeline).to_list(length=100)
        return {
            "period_days": days,
            "by_action_type": [
                {
                    "action_type": r["_id"],
                    "avg_rating": round(r["avg_rating"], 2),
                    "count": r["count"],
                }
                for r in rows
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch feedback summary: {exc}")
