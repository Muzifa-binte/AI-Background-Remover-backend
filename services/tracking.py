"""
Internal tracking helpers.

These write directly to MongoDB (no HTTP round-trip) so existing routes and
services can log usage / actions / cost with a single awaited call, right
after a feature successfully runs.

All functions are best-effort: failures are swallowed (logged) so a tracking
bug never breaks the actual user-facing feature.
"""

from __future__ import annotations
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from services.database import get_collection

logger = logging.getLogger(__name__)


async def track_usage(
    user_id: str,
    feature: str,
    image_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Log that a user hit an AI feature. Feeds GET /api/analytics/usage."""
    try:
        collection = get_collection("usage_events")
        await collection.insert_one({
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "feature": feature,
            "image_id": image_id,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as exc:
        logger.warning(f"[tracking] track_usage failed: {exc}")


async def track_action(
    user_id: str,
    image_id: str,
    action_type: str,
    suggestion: str,
    applied: bool = True,
) -> None:
    """Log an AI suggestion + whether it was applied. Feeds GET /api/analytics/success."""
    try:
        collection = get_collection("action_history")
        await collection.insert_one({
            "action_id": str(uuid.uuid4()),
            "user_id": user_id,
            "image_id": image_id,
            "action_type": action_type,
            "suggestion": suggestion,
            "applied": applied,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as exc:
        logger.warning(f"[tracking] track_action failed: {exc}")


_DEFAULT_RATES = {
    "gemini": {"input": 0.000075, "output": 0.0003},
    "groq":   {"input": 0.00005,  "output": 0.00008},
}


def estimate_cost_usd(provider: str, input_tokens: int, output_tokens: int) -> float:
    rates = _DEFAULT_RATES.get(provider, _DEFAULT_RATES["gemini"])
    return round(
        (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"],
        6,
    )


async def track_cost(
    user_id: str,
    feature: str,
    provider: str,
    model: Optional[str],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: Optional[float] = None,
) -> None:
    """Log token usage / cost of one AI API call. Feeds GET /api/analytics/cost."""
    try:
        if cost_usd is None:
            cost_usd = estimate_cost_usd(provider, input_tokens, output_tokens)
        collection = get_collection("cost_logs")
        await collection.insert_one({
            "cost_id": str(uuid.uuid4()),
            "user_id": user_id,
            "feature": feature,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as exc:
        logger.warning(f"[tracking] track_cost failed: {exc}")
